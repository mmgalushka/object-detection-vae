"""
A module for converting a data source to TFRecords.
"""

import pathlib
import json
import glob

import yaml
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from .dataset import DATASET_DIR, DATASET_FORMAT, DATASET_CATEGORIES, DataFormat
from .image import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, IMAGE_CAPACITY

TFRECORDS_CAPACITY = 256


def create_tfrecords(dataset_dir: str = DATASET_DIR,
                     dataset_format: DataFormat = DATASET_FORMAT,
                     dataset_categories: list = DATASET_CATEGORIES,
                     tfrecords_dir: str = None,
                     tfrecords_capacity: int = TFRECORDS_CAPACITY,
                     image_width: int = IMAGE_WIDTH,
                     image_height: int = IMAGE_HEIGHT,
                     image_channels: int = IMAGE_CHANNELS,
                     verbose: bool = False):
    # Gets input directory, containing dataset files that need to be
    # transformed to TFRecords.
    input_dir = pathlib.Path(dataset_dir)
    if not input_dir.exists():
        raise FileExistsError(f'Input directory not found at: {input_dir}')

    # Creates the output directory, where TFRecords should be stored.
    if tfrecords_dir is None:
        output_dir = input_dir.parent / (input_dir.name + '-tfrecords')
    else:
        output_dir = pathlib.Path(tfrecords_dir)
    output_dir.mkdir(exist_ok=True)

    # Creates a map for mapping categories
    categories_map = {
        category: code for code, category in enumerate(dataset_categories)
    }

    if dataset_format == DataFormat.CSV:
        _csv_to_tfrecords(input_dir, output_dir, categories_map,
                          tfrecords_capacity, image_width, image_height,
                          image_channels, verbose)
    else:
        raise ValueError('invalid ')


def _csv_to_tfrecords(input_dir: pathlib.Path, output_dir: pathlib.Path,
                      categories_map: dict, tfrecords_capacity: int,
                      image_width: int, image_height: int, image_channels: int,
                      verbose: bool):

    # --- Internal function ----------------------------------------------------
    # The function receives a CSV row and converts it into an example.
    def get_example(row):
        image_file = pathlib.Path(row['image'])
        if image_file.is_absolute():
            fp = image_file
        else:
            fp = input_dir / image_file

        feature = {
            **_image_feature(fp, image_width, image_height),
            **_bboxes_feature(json.loads(row['bboxes'])),
            **_segments_feature(json.loads(row['segments'])),
            **_categories_feature(eval(row['categories']), categories_map)
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
        part_count = tfrecords_capacity

        # Initializes the progress bar of verbose mode is on.
        if verbose:
            pbar = tqdm(total=len(df))

        for _, row in df.iterrows():
            if part_count >= tfrecords_capacity:
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
        transform_file(pathlib.Path(input_dir / f'{partition}.csv'))


def _image_feature(fp: pathlib.Path, width: int, height: int):
    """Returns a bytes_list from a string / byte."""
    image = Image.open(fp)
    array = np.array(image.resize((width, height)))
    return {
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


def _categories_feature(categories: list, categories_map: dict):
    """Returns an int64_list from a bool / enum / int / uint."""
    one_hot = np.zeros((len(categories), len(categories_map)), dtype=int)
    for i, category in enumerate(categories):
        one_hot[i, categories_map[category]] = 1
    one_hot = one_hot.flatten()

    return {
        'categories':
            tf.train.Feature(int64_list=tf.train.Int64List(value=one_hot))
    }


def create_generator(tfrecords_dir):

    def _parse_function(proto):
        keys_to_features = {
            'image/content':
                tf.io.FixedLenSequenceFeature([], tf.string,
                                              allow_missing=True),
            'bboxes/data':
                tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'categories':
                tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        parsed_features['image/content'] = tf.io.decode_raw(
            parsed_features['image/content'], tf.uint8)
        x = tf.reshape(parsed_features['image/content'], (64, 64, 3))
        x = tf.cast(x, tf.float32) / 255.0

        b = parsed_features['bboxes/data']
        b = tf.reshape(b, (2, 4))
        b = tf.cast(b, dtype=tf.float32)

        c = parsed_features['categories']
        c = tf.reshape(c, (2, 2))
        c = tf.cast(c, dtype=tf.float32)

        y = tf.concat([b, c], axis=1)

        return x, (x, y)

    tfrecord_files = glob.glob(f'{tfrecords_dir}/part-*.tfrecord')

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(64)

    return dataset