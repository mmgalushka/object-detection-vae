"""
A module for handling custom model layers. 
"""

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow.keras.backend as K
from hungarian_loss import hungarian_loss

class KLDivergence(Layer):

    def __init__(self, code_size, beta=1.0, **kwargs):
        self._beta = beta
        self._code_size = code_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, layer_inputs, **kwargs):
        if len(layer_inputs) != 2:
            raise ValueError('must be mean and logvar')
        mean = layer_inputs[0]
        logvar = layer_inputs[1]
        if len(mean.shape) != 2 and len(logvar.shape) != 2:
            raise ValueError('(b,c))')

        batch = K.shape(mean)[0]
        dim = K.shape(mean)[1]

        latent_loss = -0.5 * (1 + logvar - K.square(mean) - K.exp(logvar))
        latent_loss = K.sum(latent_loss, axis=1, keepdims=True)
        latent_loss = K.mean(latent_loss, axis=0)

        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
        stddev = K.exp(logvar * 0.5)

        gamma = (0.0 if self._beta == 0 else 1.0)

        return K.in_train_phase(mean + gamma * stddev * epsilon,
                                mean + 0 * logvar)

    def get_config(self):
        config = {'beta': self._beta, 'code_size': self._code_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]




class EntitiesAssignment(Layer):
    def __init__(self):
        super(EntitiesAssignment, self).__init__()

    def call(self, inputs):
        return hungarian_loss(inputs[0], inputs[1])


class EntitiesRearrangement(Layer):
    def __init__(self):
        super(EntitiesRearrangement, self).__init__()

    def call(self, inputs):
        entities = inputs[0]
        assignments = inputs[1]

        order = tf.gather(tf.where(assignments), indices=[0,2], axis=1)
        return tf.reshape(tf.gather_nd(entities, order), entities.shape)
