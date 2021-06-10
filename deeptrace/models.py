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
                                     Lambda, Dense, Reshape, Conv2DTranspose,
                                     Concatenate, Activation, MaxPooling2D)
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from .layers import KLDivergence
from .config import Config
from .losses import total_dist


def create_model(config: Config, verbose: bool = False) -> Model:
    input_shape = (config['input/image/width'], config['input/image/height'],
                   config['input/image/channels'])
    latent_size = config['fitting/model/latent_size']
    num_detecting_objects = config['output/detecting/capacity']
    num_detecting_categories = len(config['output/detecting/categories']) + 1

    encoder = DefaultEncoder(input_shape, latent_size)
    decoder = DefaultDecoder(input_shape, latent_size)
    sampler = KLSampler(code_size=latent_size, beta=0.01)

    localizer = ObjectLocalizer(latent_size, num_detecting_objects)
    classifier = ObjectClassifier(latent_size, num_detecting_objects,
                                  num_detecting_categories)

    if verbose:
        localizer.summary()
        classifier.summary()

    auto_input = encoder.input
    x = encoder(auto_input)
    x = sampler(x)
    auto_output = decoder(x)
    bbox_output = localizer(x)
    label_output = classifier(x)

    combine_output = Concatenate(axis=2)([bbox_output, label_output])

    model = Model(auto_input, [auto_output, combine_output])
    losses = {"DefaultDecoder": 'bce', "concatenate": total_dist}
    lossWeights = {"DefaultDecoder": 1, "concatenate": 1}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights)

    return model


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
        encoder_input = Input(shape=input_shape, name='encoder_input')
        encoder = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name='encoder_conv_1')(
                encoder_input)
        encoder = MaxPooling2D(
            pool_size=(2, 2), name='encoder_max_pool_1')(
                encoder)
        encoder = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name='encoder_conv_2')(
                encoder)
        encoder = MaxPooling2D(
            pool_size=(2, 2), name='encoder_max_pool_2')(
                encoder)
        encoder = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name='encoder_conv_3')(
                encoder)
        encoder = MaxPooling2D(
            pool_size=(2, 2), name='encoder_max_pool_3')(
                encoder)
        encoder = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name='encoder_conv_4')(
                encoder)
        encoder = MaxPooling2D(
            pool_size=(2, 2), name='encoder_max_pool_4')(
                encoder)
        encoder = Flatten(name='encoder_flatten')(encoder)
        mean = Dense(latent_size, name='encoder_mean')(encoder)
        logvar = Dense(latent_size, name='logvar_mean')(encoder)
        super().__init__(encoder_input, [mean, logvar], name='DefaultEncoder')


class DefaultDecoder(Model):

    def __init__(self, input_shape, latent_size):
        super().__init__()
        decoder_input = Input(shape=(latent_size,), name='decoder_input')
        decoder = Dense(2 * 2 * 256, name='decoder_dense')(decoder_input)
        decoder = Reshape((2, 2, 256), name='decoder_reshape')(decoder)
        decoder = Conv2DTranspose(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            activation='relu',
            padding='same')(
                decoder)
        decoder = Conv2DTranspose(
            filters=64,
            kernel_size=(3, 3),
            strides=2,
            activation='relu',
            padding='same')(
                decoder)
        decoder = Conv2DTranspose(
            filters=32,
            kernel_size=(3, 3),
            strides=2,
            activation='relu',
            padding='same')(
                decoder)
        decoder = Conv2DTranspose(
            filters=16,
            kernel_size=(3, 3),
            strides=2,
            activation='relu',
            padding='same')(
                decoder)
        decoder_output = Conv2DTranspose(
            filters=input_shape[-1],
            kernel_size=(3, 3),
            strides=2,
            activation='sigmoid',
            padding='same')(
                decoder)
        super().__init__(decoder_input, decoder_output, name='DefaultDecoder')


class ObjectLocalizer(Model):

    def __init__(self, latent_size: int, num_detecting_objects: int):
        super().__init__()
        local_input = Input(shape=(latent_size,), name='localizer_input')
        x_stop_grad = Lambda(lambda x: K.stop_gradient(x))(local_input)
        x = Dense(1024, activation='relu')(x_stop_grad)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        # 4 is a number of parameters used to define the binding box.
        x = Dense(num_detecting_objects * 4)(x)
        local_output = Reshape((num_detecting_objects, 4))(x)

        super().__init__(local_input, local_output)


class ObjectClassifier(Model):

    def __init__(self, latent_size: int, num_detecting_objects: int,
                 num_detecting_categories: int):
        super().__init__()

        class_input = Input(shape=(latent_size,), name='classifier_input')
        x_stop_grad = Lambda(lambda x: K.stop_gradient(x))(class_input)
        x = Dense(1024, activation='relu')(x_stop_grad)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(num_detecting_objects * num_detecting_categories)(x)
        x = Reshape((num_detecting_objects, num_detecting_categories))(x)
        class_output = tf.keras.activations.softmax(x, axis=-1)

        super().__init__(class_input, class_output)