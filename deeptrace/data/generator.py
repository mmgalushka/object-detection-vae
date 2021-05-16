"""
A generator.
"""
from __future__ import annotations

import glob
import random

import tensorflow as tf


def _parse_function(proto):
    keys_to_features = {
        'image/content':
            tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'bbox/data':
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    parsed_features['image/content'] = tf.io.decode_raw(
        parsed_features['image/content'], tf.uint8)
    x = tf.reshape(parsed_features['image/content'], (64, 64, 3))
    x = tf.cast(x, tf.float32) * (1. / 255)

    b = parsed_features['bbox/data']
    return x, (x, tf.reshape(b, (2, 4)))
    # return x, x


def create_generator(tfrecords_dir):
    tfrecord_files = glob.glob(f'{tfrecords_dir}/part-*.tfrecord')
    print(tfrecord_files)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(64)
    return dataset
