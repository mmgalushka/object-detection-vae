"""
A module for handling models. 
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

from .layers import KLDivergence


def create_model(input_shape, latent_size):
    encoder = DefaultEncoder(input_shape, latent_size)
    decoder = DefaultDecoder(input_shape, latent_size)
    sampler = KLSampler(code_size=latent_size, beta=0.01)

    localizer = ObjectLocalizer(latent_size)

    auto_input = encoder.input
    hidden = encoder(auto_input)
    hidden = sampler(hidden)
    auto_output = decoder(hidden)
    bbox_output = localizer(hidden)

    return Model(auto_input, [auto_output, bbox_output])


class KLSampler(Model):

    def __init__(self, code_size, batch_size=None, beta=1.0):
        super().__init__()

        self._beta = beta
        self._code_size = code_size
        self._batch_size = batch_size

        in_mean = Input((self._code_size,))
        in_logvar = Input((self._code_size,))
        kld = KLDivergence(
            code_size=self._code_size,
            batch_size=self._batch_size)([in_mean, in_logvar])
        super().__init__(
            inputs=[in_mean, in_logvar], outputs=kld, name='sampler')


class DefaultEncoder(Model):

    def __init__(self, input_shape, latent_size):
        super().__init__()

        encoder_conv_1 = Conv2D(
            filters=8,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='encoder_conv_1')
        encoder_conv_2 = Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='encoder_conv_2')
        encoder_flatten = Flatten(name='encoder_flatten')
        encoder_z_mean = Dense(
            latent_size, activation='linear', name='encoder_z_mean')
        encoder_z_log_var = Dense(
            latent_size, activation='linear', name='encoder_z_log_var')

        encoder_input = Input(shape=input_shape, name='encoder_input')
        hidden = encoder_conv_1(encoder_input)
        hidden = encoder_conv_2(hidden)
        hidden = encoder_flatten(hidden)
        z_mean = encoder_z_mean(hidden)
        z_log_var = encoder_z_log_var(hidden)

        super().__init__(encoder_input, [z_mean, z_log_var])


class DefaultDecoder(Model):

    def __init__(self, input_shape, latent_size):
        super().__init__()

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
        decoder_conv_2 = Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='decoder_conv_2')
        decoder_conv_3 = Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='decoder_conv_3')
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
        hidden = decoder_conv_2(hidden)
        hidden = decoder_conv_3(hidden)
        decoder_output = decoder_output(hidden)

        super().__init__(decoder_input, decoder_output)


class ObjectLocalizer(Model):

    def __init__(self, latent_size):
        super().__init__()
        local_input = Input(shape=(latent_size,), name='local_input')
        x = Dense(1024, activation='relu')(local_input)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(8)(x)
        local_output = Reshape((2, 4))(x)

        super().__init__(local_input, local_output)