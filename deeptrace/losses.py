"""
A module for defining custom losses. 
"""

import tensorflow as tf
from tensorflow.python.keras.losses import mean_absolute_error


def euclidean_distance(a, b):
    """
        dist = sqrt((a - b)^2) = sqrt(a^2 - 2ab.T - b^2)
    """
    # Example of input data, both tensors have shape=(1, 2, 4):
    #
    # a = [
    #   [[1. 2. 3. 4.]
    #    [5. 6. 7. 8.]]
    # ]
    #
    # b = [
    #   [[1. 1. 1. 1.]
    #    [2. 2. 2. 2.]]
    # ]

    # N = a.shape[0]
    N = len(a)
    # Batch size: N = 1

    a2 = tf.reshape(tf.reduce_sum(tf.square(a), axis=2), [N, -1, 1])
    # a2 = [[[ 30.]
    #        [175.]]]

    b2 = tf.reshape(tf.reduce_sum(tf.square(b), axis=2), [N, 1, -1])

    # b2 = [[[4. 16.]]]

    dist = tf.sqrt(a2 - 2 * tf.matmul(a, tf.transpose(b, perm=[0, 2, 1])) + b2)
    # dist = [[[ 3.7416575  2.4494898]
    #          [11.224972   9.273619 ]]]
    return dist


def pairs_mesh(n):
    # n == 2

    r = tf.range(n)
    # r = [0 1],   shape=(2,)

    a = tf.expand_dims(tf.tile(tf.expand_dims(r, 1), [1, tf.shape(r)[0]]), 2)
    # a = [[[0]
    #       [0]]
    #      [[1]
    #       [1]]], shape=(2, 2, 1)

    b = tf.expand_dims(tf.tile(tf.expand_dims(r, 0), [tf.shape(r)[0], 1]), 2)
    # b= [[[0]
    #      [1]]
    #     [[0]
    #      [1]]], shape=(2, 2, 1)

    pairs = tf.reshape(tf.concat([a, b], axis=2), [-1, 2])
    # pairs = [[0 0]
    #          [0 1]
    #          [1 0]
    #          [1 1]], shape=(4, 2)
    return pairs


def hungarian_mask(cost):
    # cost = [[[ 3.7416575  2.4494898]
    #          [11.224972   9.273619 ]]], shape=(1, 2, 2)

    n = cost.shape[1]
    # n = 4

    mesh = pairs_mesh(n)
    # mesh = [[0 0]
    #         [0 1]
    #         [1 0]
    #         [1 1]], shape=(4, 2)

    order = tf.argsort(tf.reshape(cost, [-1]))
    # order = [1 0 3 2], shape=(4,)

    pairs = tf.cast(tf.gather(mesh, order), dtype=tf.int64)

    # pairs = [[0 1]
    #          [0 0]
    #          [1 1]
    #          [1 0]], shape=(4, 2)

    def body(i, pairs, mask):
        pair = tf.cast(tf.gather(pairs, indices=[i]), dtype=tf.int64)
        # pair = [0 1] -> [0 0] -> [1 1] -> [1 0]

        activation = tf.sparse.to_dense(
            tf.SparseTensor(indices=pair, values=[1], dense_shape=[n, n]))
        # activation = [0 1] -> [1 0] -> [0 0] -> [0 0]
        #              [0 0]    [0 0]    [0 1]    [1 0]

        probe = tf.math.add(mask, activation)
        # probe = [0 1] -> [1 1] -> [0 1] -> [0 1]
        #         [0 0]    [0 0]    [0 1]    [1 0]

        row = tf.reduce_sum(probe, axis=0)
        row = (tf.where(tf.greater(row, 1)))
        row = tf.equal(tf.size(row), 0)
        # row = True -> True -> False -> True

        col = tf.reduce_sum(probe, axis=1)
        col = (tf.where(tf.greater(col, 1)))
        col = tf.equal(tf.size(col), 0)
        # col = True -> False -> True -> True

        conjunction = tf.math.logical_and(row, col)
        # conjunction = True -> False -> False -> True

        return tf.cond(conjunction, lambda: [i + 1, pairs, probe],
                       lambda: [i + 1, pairs, mask])

    def condition(i, pairs, mask):
        return tf.less_equal(i, len(pairs) - 1)

    output = tf.while_loop(
        condition, body, [0, pairs, tf.zeros((n, n), dtype=tf.int32)])
    # output = [
    #   4,
    #   [[0, 1],
    #    [0, 0],
    #    [1, 1],
    #    [1, 0]],
    #   [[0, 1],    | this is a computing
    #    [1, 0]]    | mask (second element)
    # ]
    return output[2]


def hungarian_dist(y_true, y_pred):
    v_true = tf.cast(y_true, dtype=tf.float32)
    v_pred = tf.cast(y_pred, dtype=tf.float32)

    # v_true = [
    #   [[1. 1. 1. 1.]
    #    [3. 3. 3. 3.]]
    # ], shape=(1, 2, 4)
    #
    # v_pred = [
    #   [[4. 4. 4. 4.]
    #    [2. 2. 2. 2.]]
    # ] , shape=(1, 2, 4)

    # dist = euclidean_distance(v_true, v_pred)
    # mask = tf.cast(tf.map_fn(hungarian_mask, dist, dtype=tf.int32), tf.float32)
    # return tf.reduce_sum(tf.math.multiply(dist, mask), (1, 2))

    anchor_true = tf.slice(v_true, [0, 0, 0], [-1, 2, 2])
    anchor_pred = tf.slice(v_pred, [0, 0, 0], [-1, 2, 2])

    anchor_dist = euclidean_distance(anchor_true, anchor_pred)
    mask = tf.cast(
        tf.map_fn(hungarian_mask, anchor_dist, dtype=tf.int32), tf.float32)

    bbox_dist = euclidean_distance(v_true, v_pred)
    return tf.reduce_sum(tf.math.multiply(bbox_dist, mask), (1, 2))


def pair_categorical_crossentropy(a, b):

    n = a.shape[1]
    k = b.shape[2]

    a = tf.tile(a, tf.constant([1, 1, n], tf.int32))
    a = tf.expand_dims(a, 2)
    a = tf.reshape(a, [-1, n * n, k])

    b = tf.tile(b, tf.constant([1, n, 1], tf.int32))

    entropy = tf.losses.categorical_crossentropy(a, b)
    entropy = tf.reshape(entropy, (-1, n, n))
    return entropy


def total_dist(y_true, y_pred):
    v_true = tf.cast(y_true, dtype=tf.float32)
    v_pred = tf.cast(y_pred, dtype=tf.float32)

    # v_true = [
    #   [[1. 1. 1. 1.]
    #    [3. 3. 3. 3.]]
    # ], shape=(1, 2, 4)
    #
    # v_pred = [
    #   [[4. 4. 4. 4.]
    #    [2. 2. 2. 2.]]
    # ] , shape=(1, 2, 4)

    # dist = euclidean_distance(v_true, v_pred)
    # mask = tf.cast(tf.map_fn(hungarian_mask, dist, dtype=tf.int32), tf.float32)
    # return tf.reduce_sum(tf.math.multiply(dist, mask), (1, 2))

    bbox_n = 4
    label_n = 2

    bbox_true = tf.slice(v_true, [0, 0, 0], [-1, 2, bbox_n])
    bbox_pred = tf.slice(v_pred, [0, 0, 0], [-1, 2, bbox_n])

    label_true = tf.slice(v_true, [0, 0, bbox_n], [-1, 2, label_n])
    label_pred = tf.slice(v_pred, [0, 0, bbox_n], [-1, 2, label_n])

    bbox_dist = euclidean_distance(bbox_true, bbox_pred)
    lable_entropy = pair_categorical_crossentropy(label_true, label_pred)

    mask = tf.cast(
        tf.map_fn(hungarian_mask, bbox_dist, dtype=tf.int32), tf.float32)

    bbox_loss = tf.reduce_sum(tf.math.multiply(bbox_dist, mask), (1, 2))
    label_loss = tf.reduce_sum(tf.math.multiply(lable_entropy, mask), (1, 2))

    return bbox_loss + label_loss


# def hungarian_dist_old(y_true, y_pred):
#     A = tf.cast(y_true, dtype=tf.float32)
#     B = tf.cast(y_pred, dtype=tf.float32)

#     N = len(y_true)
#     K = 2

#     def ha(x, seq):

#         def body(i, seq, x):
#             temp = tf.identity(x)
#             idx = tf.gather(seq, indices=[i])

#             activation = tf.sparse.to_dense(
#                 tf.SparseTensor(indices=idx, values=[1], dense_shape=[K, K]))

#             probe = tf.math.add(temp, activation)

#             row = tf.reduce_sum(probe, axis=0)
#             row = (tf.where(tf.greater(row, 1)))
#             row_mask = tf.equal(tf.size(row), 0)

#             col = tf.reduce_sum(probe, axis=1)
#             col = (tf.where(tf.greater(col, 1)))
#             col_mask = tf.equal(tf.size(col), 0)

#             mask = tf.math.logical_and(row_mask, col_mask)

#             return tf.cond(mask, lambda: [i + 1, seq, probe],
#                            lambda: [i + 1, seq, x])

#         def condition(i, seq, x):
#             return tf.less_equal(i, 3)

#         output = tf.while_loop(
#             condition, body, [0, seq, tf.zeros((K, K), dtype=tf.int32)])
#         return output[2]

#     def mask_maker(x):
#         F = tf.reshape(x, [-1])
#         idx = tf.argsort(F)

#         # ii, jj = tf.meshgrid(tf.range(K), tf.range(K), indexing='ij')
#         # ij = tf.reshape(tf.stack([ii, jj], axis=-1), (K * K, 2))
#         ij = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]])
#         pairs = tf.cast(tf.gather(ij, idx), dtype=tf.int64)

#         return tf.cast(ha(x, pairs), dtype=tf.int32)

#     row_norms_A = tf.reduce_sum(tf.square(A), axis=2)
#     row_norms_A = tf.reshape(row_norms_A, [N, -1, 1])  # Column vector.

#     row_norms_B = tf.reduce_sum(tf.square(B), axis=2)
#     row_norms_B = tf.reshape(row_norms_B, [N, 1, -1])  # Row vector.

#     ssd = row_norms_A - 2 * tf.matmul(A, tf.transpose(B, perm=[0, 2, 1
#                                                               ])) + row_norms_B
#     rsd = tf.sqrt(tf.cast(ssd, dtype=tf.float32))

#     mask = tf.map_fn(mask_maker, rsd, dtype=tf.int32)
#     mask = tf.cast(mask, tf.float32)
#     result = tf.math.multiply(rsd, mask)
#     result = tf.reduce_sum(result, (1, 2))

#     return result