"""
A module for defining custom losses. 
"""

from logging import log
import sys
import tensorflow as tf
from tensorflow.python.eager.def_function import run_functions_eagerly
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.losses import KLDivergence
import tensorflow_addons as tfa
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Input, BatchNormalization, Conv2D, Flatten,
                                     Lambda, Dense, Reshape, Conv2DTranspose)
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from scipy.optimize import linear_sum_assignment


def hungarian_dist(y_true, y_pred):
    A = tf.cast(y_true, dtype=tf.float32)
    B = tf.cast(y_pred, dtype=tf.float32)

    N = len(y_true)
    K = 2

    def ha(x, seq):

        def body(i, seq, x):
            temp = tf.identity(x)
            idx = tf.gather(seq, indices=[i])

            activation = tf.sparse.to_dense(
                tf.SparseTensor(indices=idx, values=[1], dense_shape=[K, K]))

            probe = tf.math.add(temp, activation)

            row = tf.reduce_sum(probe, axis=0)
            row = (tf.where(tf.greater(row, 1)))
            row_mask = tf.equal(tf.size(row), 0)

            col = tf.reduce_sum(probe, axis=1)
            col = (tf.where(tf.greater(col, 1)))
            col_mask = tf.equal(tf.size(col), 0)

            mask = tf.math.logical_and(row_mask, col_mask)

            return tf.cond(mask, lambda: [i + 1, seq, probe],
                           lambda: [i + 1, seq, x])

        def condition(i, seq, x):
            return tf.less_equal(i, 3)

        output = tf.while_loop(
            condition, body, [0, seq, tf.zeros((K, K), dtype=tf.int32)])
        return output[2]

    def mask_maker(x):
        F = tf.reshape(x, [-1])
        idx = tf.argsort(F)

        # ii, jj = tf.meshgrid(tf.range(K), tf.range(K), indexing='ij')
        # ij = tf.reshape(tf.stack([ii, jj], axis=-1), (K * K, 2))
        ij = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]])
        pairs = tf.cast(tf.gather(ij, idx), dtype=tf.int64)

        return tf.cast(ha(x, pairs), dtype=tf.int32)

    row_norms_A = tf.reduce_sum(tf.square(A), axis=2)
    row_norms_A = tf.reshape(row_norms_A, [N, -1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=2)
    row_norms_B = tf.reshape(row_norms_B, [N, 1, -1])  # Row vector.

    ssd = row_norms_A - 2 * tf.matmul(A, tf.transpose(B, perm=[0, 2, 1
                                                              ])) + row_norms_B
    rsd = tf.sqrt(tf.cast(ssd, dtype=tf.float32))

    mask = tf.map_fn(mask_maker, rsd, dtype=tf.int32)
    mask = tf.cast(mask, tf.float32)
    result = tf.math.multiply(rsd, mask)
    result = tf.reduce_sum(result, (1, 2))

    return result