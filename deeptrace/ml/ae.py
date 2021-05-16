import sys
import tensorflow as tf
from tensorflow.python.eager.def_function import run_functions_eagerly
import tensorflow_addons as tfa
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Input, BatchNormalization, Conv2D, Flatten,
                                     Lambda, Dense, Reshape, Conv2DTranspose)
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from scipy.optimize import linear_sum_assignment


class SequenceEncoder(Model):

    def __init__(self, input_shape, latent_size):
        super(SequenceEncoder, self).__init__()

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(
                shape=(batch_size, latent_size), mean=0., stddev=1.)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        encoder_conv_1 = Conv2D(
            filters=8,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='encoder_conv_1')
        encoder_norm_1 = BatchNormalization(name='encoder_norm_1')
        encoder_conv_2 = Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='encoder_conv_2')
        encoder_norm_2 = BatchNormalization(name='encoder_norm_2')
        encoder_flatten = Flatten(name='encoder_flatten')
        encoder_dense_latent = Dense(
            latent_size, activation='relu', name='encoder_latent')
        encoder_z_mean = Dense(
            latent_size, activation='linear', name='encoder_z_mean')
        encoder_z_log_var = Dense(
            latent_size, activation='linear', name='encoder_z_log_var')
        encoder_z = Lambda(
            sampling, output_shape=(latent_size,), name='encoder_z')

        encoder_input = Input(shape=input_shape, name='encoder_input')
        hidden = encoder_conv_1(encoder_input)
        # hidden = encoder_norm_1(hidden)
        hidden = encoder_conv_2(hidden)
        # hidden = encoder_norm_2(hidden)
        hidden = encoder_flatten(hidden)
        latent = encoder_dense_latent(hidden)
        z_mean = encoder_z_mean(latent)
        z_log_var = encoder_z_log_var(latent)
        encoder_output = encoder_z([z_mean, z_log_var])

        super(SequenceEncoder, self).__init__(encoder_input, encoder_output)

        # def vae_loss(x, x_mean):
        #     x = K.flatten(x)
        #     x_mean = K.flatten(x_mean)
        #     xent_loss = input_shape[0] * input_shape[1] * tf.reduce_mean(
        #         binary_crossentropy(x, x_mean))
        #     kl_loss = -0.5 * K.mean(
        #         1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #     return xent_loss + kl_loss

        # self.__vae_loss = vae_loss

        def kl_reconstruction_loss(true, pred):
            # Reconstruction loss
            # reconstruction_loss = mean_absolute_error(
            #     K.flatten(true),
            #     K.flatten(pred)) * input_shape[0] * input_shape[1]

            # reconstruction_loss = mean_absolute_error(
            #     K.flatten(true), K.flatten(pred))

            reconstruction_loss = binary_crossentropy(
                K.flatten(true), K.flatten(pred))

            # KL divergence loss
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            # Total loss = 50% rec + 50% KL divergence loss
            # return K.mean(reconstruction_loss + kl_loss)
            # return reconstruction_loss + kl_loss
            return reconstruction_loss
            # return kl_loss

        self.__vae_loss = kl_reconstruction_loss

    def get_vae_loss(self):
        return self.__vae_loss


class SequenceDecoder(Model):

    def __init__(self, input_shape, latent_size):
        super(SequenceDecoder, self).__init__()

        decoder_dense_latent = Dense(
            64 * 8 * 8, activation='relu', name='decoder_latent')
        decoder_reshape = Reshape((8, 8, 64))

        decoder_conv_1 = Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='decoder_conv_1')
        decoder_norm_1 = BatchNormalization(momentum=0.8, name='decoder_norm_1')
        decoder_conv_2 = Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='decoder_conv_2')
        decoder_norm_2 = BatchNormalization(momentum=0.8, name='decoder_norm_2')
        decoder_conv_3 = Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='decoder_conv_3')
        decoder_norm_3 = BatchNormalization(momentum=0.8, name='decoder_norm_3')
        decoder_output = Conv2DTranspose(
            filters=3,  #channels,
            kernel_size=3,
            activation='sigmoid',
            padding='same',
            name='decoder_output')

        decoder_input = Input(shape=(latent_size,), name='decoder_input')
        hidden = decoder_dense_latent(decoder_input)
        hidden = decoder_reshape(hidden)
        hidden = decoder_conv_1(hidden)
        # hidden = decoder_norm_1(hidden)
        hidden = decoder_conv_2(hidden)
        # hidden = decoder_norm_2(hidden)
        hidden = decoder_conv_3(hidden)
        # hidden = decoder_norm_3(hidden)
        decoder_output = decoder_output(hidden)

        super(SequenceDecoder, self).__init__(decoder_input, decoder_output)


class ObjectLocalizer(Model):

    def __init__(self, latent_size):
        super(ObjectLocalizer, self).__init__()
        local_input = Input(shape=(latent_size,), name='local_input')
        x = Dense(1024, activation='relu')(local_input)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(8)(x)
        local_output = Reshape((2, 4))(x)

        super(ObjectLocalizer, self).__init__(local_input, local_output)


class SequenceAutoencoder(Model):

    def __init__(self, input_shape, latent_size):
        super(SequenceAutoencoder, self).__init__()

        self.__encoder = SequenceEncoder(input_shape, latent_size)
        self.__decoder = SequenceDecoder(input_shape, latent_size)
        self.__local = ObjectLocalizer(latent_size)

        auto_input = self.__encoder.input
        hidden = self.__encoder(auto_input)
        auto_output = self.__decoder(hidden)
        bbox_output = self.__local(hidden)

        super(SequenceAutoencoder, self).__init__(auto_input,
                                                  [auto_output, bbox_output])

        # super(SequenceAutoencoder, self).__init__(auto_input, auto_output)

    def compile(self):
        # self.load_weights('model.h5')
        def hungarian_dist(y_true, y_pred):
            A = tf.cast(y_true, dtype=tf.float32)
            B = tf.cast(y_pred, dtype=tf.float32)

            N = 64
            K = 2

            def ha(x, seq):

                def body(i, seq, x):
                    temp = tf.identity(x)
                    idx = tf.gather(seq, indices=[i])

                    activation = tf.sparse.to_dense(
                        tf.SparseTensor(
                            indices=idx, values=[1], dense_shape=[K, K]))

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
                    condition, body,
                    [0, seq, tf.zeros((K, K), dtype=tf.int32)])
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

            ssd = row_norms_A - 2 * tf.matmul(A, tf.transpose(
                B, perm=[0, 2, 1])) + row_norms_B
            rsd = tf.sqrt(tf.cast(ssd, dtype=tf.float32))

            mask = tf.map_fn(mask_maker, rsd, dtype=tf.int32)
            mask = tf.cast(mask, tf.float32)
            result = tf.math.multiply(rsd, mask)
            result = tf.reduce_sum(result, (1, 2))

            return result

        losses = {
            "sequence_decoder_1": self.__encoder.get_vae_loss(),
            "object_localizer_1": hungarian_dist
        }
        lossWeights = {"sequence_decoder_1": 1, "object_localizer_1": 0.01}

        super(SequenceAutoencoder, self).compile(
            optimizer='adam', loss=losses, loss_weights=lossWeights)

        # super(SequenceAutoencoder, self).compile(
        #     optimizer='adam', loss=self.__encoder.get_vae_loss())

    def summary(self):
        print('Encoder:')
        self.__encoder.summary()
        print('Decoder:')
        self.__decoder.summary()
        print('Local:')
        self.__local.summary()
