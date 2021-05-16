"""
A module for converting a data source to TFRecords.
"""

import pathlib
import json
import glob

import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image

TFRECORDS_DIR = 'dataset/tfrecords'
TFRECORDS_SIZE = 3


def _numeric_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _image_feature(row):
    """Returns a bytes_list from a string / byte."""
    image = Image.open('dataset/synthetic/' + row['image'])
    array = np.array(image.resize((64, 64)))
    return {
        'image/content':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[array.tostring()]))
    }


def _bbox_feature(row):
    """Returns an int64_list from a bool / enum / int / uint."""
    bboxes = json.loads(row['bbox'])

    data = []
    for bbox in bboxes:
        data.extend(bbox)

    return {
        'bbox/data':
            tf.train.Feature(int64_list=tf.train.Int64List(value=data))
    }


def _segment_feature(row):
    """Returns an int64_list from a bool / enum / int / uint."""
    segments = json.loads(row['segment'])

    schema = []
    data = []
    for segment in segments:
        schema.append(len(segment))
        data.extend(segment)

    return {
        'segment/schema':
            tf.train.Feature(int64_list=tf.train.Int64List(value=schema)),
        'segment/data':
            tf.train.Feature(float_list=tf.train.FloatList(value=data))
    }


def _category_feature(row, config):
    """Returns an int64_list from a bool / enum / int / uint."""
    catmap = {'rectangle': 0, 'triangle': 1}

    categories = eval(row['category'])
    one_hot = np.zeros((len(categories), 2), dtype=int)

    for i, category in enumerate(categories):
        one_hot[i, catmap[category]] = 1
    one_hot = one_hot.flatten()

    return {
        'category':
            tf.train.Feature(int64_list=tf.train.Int64List(value=one_hot))
    }


def transform_csv_to_tfrecords(dataset_dir: str, tfrecords_dir: str,
                               tfrecords_size: int, verbose: bool):

    # Gets input directory, containing dataset CSV files that need to be
    # transformed to TFRecords.
    input_dir = pathlib.Path(dataset_dir)
    if not input_dir.exists():
        raise FileExistsError(f'Input directory not found at: {input_dir}')

    # Creates the output directory, where TFRecords should be stored.
    output_dir = pathlib.Path(tfrecords_dir)
    output_dir.mkdir(exist_ok=True)

    # --- Internal function ----------------------------------------------------
    # The function receives a CSV row and converts it into an example.
    def get_example(row):
        feature = {
            **_image_feature(row),
            **_bbox_feature(row)
            # **_segment_feature(row),
            # **_category_feature(row)
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
        if list(df.columns) == ['image', 'bbox', 'segment', 'category']:
            pass
        else:
            raise ValueError(
                f'Invalid structure of the CSV file: {csv_file};\n'
                'The expected CSV file must contain the following columns:\n'
                '  - image\n'
                '  - bbox\n'
                '  - segment\n'
                '  - category\n'
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
    for csv_file in glob.glob(str(input_dir / '*.csv')):
        transform_file(pathlib.Path(csv_file))
