"""
A generator.
"""
from __future__ import annotations

import glob

import tensorflow as tf


def _parse_function(proto):
    keys_to_features = {
        'figure':
            tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'shape':
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    parsed_features['figure'] = tf.io.decode_raw(parsed_features['figure'],
                                                 tf.uint8)
    return tf.reshape(parsed_features['figure'],
                      (224, 224, 3)), parsed_features['shape']


def create_dataset(tfrecords_dir):
    tfrecord_files = glob.glob(f'{tfrecords_dir}/part-*.tfrecord')
    print(tfrecord_files)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    # dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(32)
    return dataset
