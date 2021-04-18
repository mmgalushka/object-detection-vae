"""
A collection of transformers to convert a data source to TFRecords.
"""

import pathlib
import json
import glob

import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image


def _numeric_feature(value, config):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _categorical_feature(value, config):
    """Returns an int64_list from a bool / enum / int / uint."""
    one_hot = np.zeros(2, dtype=int)
    one_hot[value] = 1
    return tf.train.Feature(int64_list=tf.train.Int64List(value=one_hot))


def _image_feature(filename, config):
    """Returns a bytes_list from a string / byte."""
    image = Image.open('dataset/synthetic/' + filename)
    array = np.array(image.resize((224, 224)))
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[array.tostring()]))


FN_FEATURE = {
    'numeric': _numeric_feature,
    'categorical': _categorical_feature,
    'image': _image_feature,
}


def transform_CSV_to_TFRecords(config: dict):
    # Makes sure the configuration is defined.
    if not isinstance(config, dict):
        raise TypeError(f'config must be {dict} but got {type(config)};')

    # Gets input directory, containing dataset CSV files that need to be
    # transformed to TFRecords.
    input_dir = pathlib.Path(config['input']['directory'])
    if not input_dir.exists():
        raise FileExistsError(f'Input directory not found at: {input_dir}')

    # Creates the output directory, where TFRecords should be stored.
    output_dir = pathlib.Path(config['output']['directory'])
    output_dir.mkdir(exist_ok=True)

    # The data schema represents instructions about CSV data that are used
    # in the transformation process. The availability of this file is
    # mandatory.
    schema_file = input_dir / 'schema.json'
    if not schema_file.exists():
        raise FileExistsError(f'CSV schema file not found at: {input_dir}')
    else:
        with open(schema_file) as schema_fp:
            schema = json.load(schema_fp)

    # --- Internal function ----------------------------------------------------
    # The function receives a CSV row (a record) and using the preloaded
    # schema transforms it into an example.
    def _get_example(row):
        feature = {}
        for name, properties in schema['features'].items():
            fn_feature = FN_FEATURE[properties['type']]
            feature[name] = fn_feature(row[name], properties)
        return tf.train.Example(features=tf.train.Features(feature=feature))

    # --- Internal function ----------------------------------------------------
    # This function transforms an individual CSV file into the TFRecords.
    def _transform_file(csv_file):
        # Loads the content of all CSV files into the Dataframe.
        df = pd.read_csv(csv_file)

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
        part_count = config['output']['capacity']

        pbar = tqdm(total=len(df))
        for _, row in df.iterrows():
            if part_count >= config['output']['capacity']:
                # The current partition has been reached the maximum capacity,
                # so we need to start a new one.
                if writer is not None:
                    # Closes the existing TFRecords writer.
                    writer.close()
                part_index += 1
                writer = tf.io.TFRecordWriter(
                    str(tfrecords_dir / f'part-{part_index}.tfrecord'))
                part_count = 0

            example = _get_example(row)
            writer.write(example.SerializeToString())
            part_count += 1
            pbar.update(1)
        # Closes the existing TFRecords writer after the last row.
        writer.close()

    # Processes all CSV files in the input directory.
    for csv_file in glob.glob(str(input_dir / '*.csv')):
        _transform_file(pathlib.Path(csv_file))
