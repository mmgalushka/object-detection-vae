"""
A module for converting a data source to TFRecords.
"""
from __future__ import annotations

from os import X_OK

import json
import glob
from pathlib import Path

from tensorflow.python.framework.ops import Tensor

import yaml
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from .config import BATCH_SIZE
from .dataset import DataFormat

tf.compat.v1.enable_eager_execution()


def create_tfrecords(dataset_dir: str,
                     dataset_format: DataFormat,
                     dataset_categories: list,
                     tfrecords_dir: str = None,
                     tfrecords_size: int = 256,
                     image_width: int = None,
                     image_height: int = None,
                     verbose: bool = False):
    # Gets input directory, containing dataset files that need to be
    # transformed to TFRecords.
    input_dir = Path(dataset_dir)
    if not input_dir.exists():
        raise FileExistsError(f'Input directory not found at: {input_dir}')

    # Creates the output directory, where TFRecords should be stored.
    if tfrecords_dir is None:
        output_dir = input_dir.parent / (input_dir.name + '-tfrecords')
    else:
        output_dir = Path(tfrecords_dir)
    output_dir.mkdir(exist_ok=True)

    # Creates a map for mapping categories
    # full_dataset_categories = dataset_categories.copy()
    # full_dataset_categories.insert(0, None)
    # categories_map = {
    #     category: code for code, category in enumerate(full_dataset_categories)
    # }

    if dataset_format == DataFormat.CSV:
        _csv_to_tfrecords(input_dir, output_dir, dataset_categories,
                          tfrecords_size, image_width, image_height, verbose)
    else:
        raise ValueError('invalid ')


def _csv_to_tfrecords(input_dir: Path, output_dir: Path, categories: list,
                      tfrecords_size: int, image_width: int, image_height: int,
                      verbose: bool):

    # --- Internal function ----------------------------------------------------
    # The function receives a CSV row and converts it into an example.
    def get_example(row):
        image_file = Path(row['image'])
        if image_file.is_absolute():
            fp = image_file
        else:
            fp = input_dir / image_file

        feature = {
            **_image_feature(fp, image_width, image_height),
            **_bboxes_feature(json.loads(row['bboxes'])),
            **_segments_feature(json.loads(row['segments'])),
            **_categories_feature(eval(row['categories']))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    # --- Internal function ----------------------------------------------------
    # This function transforms an individual CSV file into the TFRecords.
    def transform_file(csv_file):
        # Loads the content of all CSV files into the Dataframe.
        df = pd.read_csv(csv_file, index_col=False)

        # Check the CSV file columns. The expected (default) structure of
        # the CSV file should include the following columns:
        # file_name | bbox | segment | category
        if list(df.columns) == ['image', 'bboxes', 'segments', 'categories']:
            pass
        else:
            raise ValueError(
                f'Invalid structure of the CSV file: {csv_file};\n'
                'The expected CSV file must contain the following columns:\n'
                '  - image\n'
                '  - bboxes\n'
                '  - segments\n'
                '  - categories\n'
                'The This column order must be preserved.')

        # Makes a directory where TFRecords files will be stored. For example
        #    output_dir -> /x/y/z
        #    csv_file   -> train.csv
        #
        # the TFRecords directory will be
        #    tfrecords_dir ->  /x/y/z/train
        tfrecords_dir = output_dir / csv_file.stem
        tfrecords_dir.mkdir(exist_ok=True)

        # The TFRecords writer.
        writer = None
        # The index for the next TFRecords partition.
        part_index = -1
        # The count of how many records stored in the TFRecords files. It
        # is set here to maximum capacity (as a trick) to make the "if"
        # condition in the loop equals to True and start 0 - partition.
        part_count = tfrecords_size

        # Initializes the progress bar of verbose mode is on.
        if verbose:
            pbar = tqdm(total=len(df))

        for _, row in df.iterrows():
            if part_count >= tfrecords_size:
                # The current partition has been reached the maximum capacity,
                # so we need to start a new one.
                if writer is not None:
                    # Closes the existing TFRecords writer.
                    writer.close()
                part_index += 1
                writer = tf.io.TFRecordWriter(
                    str(tfrecords_dir / f'part-{part_index}.tfrecord'))
                part_count = 0

            example = get_example(row)
            writer.write(example.SerializeToString())
            part_count += 1

            # Updates the progress bar of verbose mode is on.
            if verbose:
                pbar.update(1)

        # Closes the existing TFRecords writer after the last row.
        writer.close()

    # Processes all CSV files in the input directory.
    partitions = ['train', 'val', 'test']
    for partition in partitions:
        transform_file(Path(input_dir / f'{partition}.csv'))


def _image_feature(fp: Path, width: int, height: int):
    """Returns a bytes_list from a string / byte."""
    image = Image.open(fp)
    if isinstance(width, int) and isinstance(height, int):
        array = np.array(image.resize((width, height)))
    elif width is None and height is None:
        array = np.array(image)
    else:
        raise ValueError(
            'Invalid arguments for resizing an image. Both arguments '
            'representing image width and height must be either integer '
            f'or None. But received width is {type(width)} and height is '
            f'{type(height)}.')
    return {
        'image/shape':
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(array.shape))),
        'image/content':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[array.tostring()]))
    }


def _bboxes_feature(bboxes: list):
    """Returns an int64_list from a bool / enum / int / uint."""
    data = []
    for bbox in bboxes:
        data.extend(bbox)

    return {
        'bboxes/number':
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=[len(bboxes)])),
        'bboxes/data':
            tf.train.Feature(int64_list=tf.train.Int64List(value=data))
    }


def _segments_feature(segments: list):
    """Returns an int64_list from a bool / enum / int / uint."""
    schema = []
    data = []
    for segment in segments:
        schema.append(len(segment))
        data.extend(segment)

    return {
        'segments/schema':
            tf.train.Feature(int64_list=tf.train.Int64List(value=schema)),
        'segments/data':
            tf.train.Feature(float_list=tf.train.FloatList(value=data))
    }


def _categories_feature(categories: list):
    """Returns an int64_list from a bool / enum / int / uint."""
    return {
        'categories/number':
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=[len(categories)])),
        'categories/data':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[category.encode('utf-8') for category in categories])
            )
    }


def count_records(tfrecords_dir: Path, verbose: bool) -> int:
    records_count = 0

    tfrecord_files = glob.glob(str(tfrecords_dir / 'part-*.tfrecord'))

    # Initializes the progress bar of verbose mode is on.
    if verbose:
        pbar = tqdm(total=len(tfrecord_files))

    for tfrecord_files in tfrecord_files:
        # Counts number of record per TFRecords file.
        records_count += sum(1 for _ in tf.data.TFRecordDataset(tfrecord_files))
        # Updates the progress bar of verbose mode is on.
        if verbose:
            pbar.update(1)

    return records_count


def create_generator(tfrecords_dir: Path,
                     num_detecting_objects,
                     detecting_categories: list,
                     batch_size: int = BATCH_SIZE):

    def _parse_function(proto):
        keys_to_features = {
            'image/shape':
                tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'image/content':
                tf.io.FixedLenSequenceFeature([], tf.string,
                                              allow_missing=True),
            'bboxes/number':
                tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'bboxes/data':
                tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'categories/number':
                tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'categories/data':
                tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True)
        }
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)

        X = tf.io.decode_raw(parsed_features['image/content'], tf.uint8)
        X = tf.reshape(X, parsed_features['image/shape'])
        X = tf.cast(X, tf.float32) / 255.0

        bboxes = feature_to_bboxes(parsed_features, num_detecting_objects)
        categories = feature_to_categories(parsed_features,
                                           detecting_categories,
                                           num_detecting_objects)
        y = tf.concat([bboxes, categories], axis=1)

        return X, (X, y)

    tfrecord_files = glob.glob(str(tfrecords_dir / 'part-*.tfrecord'))

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset


def feature_to_bboxes(parsed_features, num_detecting_objects: int) -> tf.Tensor:
    # Loads all binding boxes defined in a TFRecord.

    n = parsed_features['bboxes/number'][0]
    # 2

    bboxes = parsed_features['bboxes/data']
    bboxes = tf.reshape(bboxes, (-1, 4))
    # [[1 2 3 4]
    #  [5 6 7 8]], shape=(2, 4), dtype=int32

    if n < num_detecting_objects:
        # If the obtained number of record binding boxes is less than the
        # detection capacity, then the binding boxes must be padded with
        # [0, 0, 0, 0].
        bboxes = tf.pad(
            bboxes, [[0, num_detecting_objects - n], [0, 0]], constant_values=0)
        # [[1 2 3 4]
        #  [5 6 7 8]
        #  [0 0 0 0]
        #  [0 0 0 0]], shape=(4, 4), dtype=int32)
        # values = [b'rectangle' b'triangle' b'[UNK]' b'[UNK]'], shape=(4,)
    elif n > 4:
        # If the obtained number of record binding boxes is greater than the
        # detection capacity, then the  binding boxes list must be sliced.
        bboxes = tf.slice(bboxes, [0, 0], [num_detecting_objects, 4])

    bboxes = tf.cast(bboxes, dtype=tf.float32)
    # [[1. 2. 3. 4.]
    #  [5. 6. 7. 8.]
    #  [0. 0. 0. 0.]
    #  [0. 0. 0. 0.]], shape=(4, 4), dtype=float32
    return bboxes


def feature_to_categories(parsed_features, detecting_categories: list,
                          num_detecting_objects: int) -> tf.Tensor:
    # Loads all categories values defined in a TFRecord.

    n = parsed_features['categories/number'][0]
    # 2

    categories = parsed_features['categories/data']
    # [b'rectangle' b'triangle'], shape=(4,), dtype=string

    if n < num_detecting_objects:
        # If the obtained number of record categories less than the
        # detection capacity, then the categories list must be padded with
        # "unknown category".
        categories = tf.pad(
            categories, [[0, num_detecting_objects - n]],
            constant_values='[UNK]')
        # [b'rectangle' b'triangle' b'[UNK]' b'[UNK]'], shape=(4,), dtype=string
    elif n > num_detecting_objects:
        # If the obtained number of record categories greater than the
        # detection capacity, then the categories list must be sliced.
        categories = tf.slice(categories, [0], [num_detecting_objects])

    categories = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=detecting_categories)(
            categories)
    # [2 3 1 1], shape=(4,), dtype=int64

    categories = categories - tf.constant(1, dtype=tf.int64)
    # [1 2 0 0], shape=(4,), dtype=int64

    categories = tf.one_hot(categories, len(detecting_categories) + 1)
    # [[0. 1. 0.]
    #  [0. 0. 1.]
    #  [1. 0. 0.]
    #  [1. 0. 0.]], shape=(4, 3), dtype=float32
    return categories