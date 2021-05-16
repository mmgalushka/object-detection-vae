import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, GlobalAveragePooling2D,
                                     Dropout, Flatten, Input, MaxPooling2D,
                                     Activation)
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.callbacks import TensorBoard
from .ae import SequenceAutoencoder


def create_model():
    # input_shape = (64, 64, 3)

    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(8))

    # model.compile('adadelta', 'mse')

    # return model

    # model = VisionTransformer(
    #     image_size=64,
    #     patch_size=8,
    #     num_layers=8,
    #     num_classes=8,  # not relevent
    #     d_model=64,
    #     num_heads=8,
    #     mlp_dim=1024,
    #     channels=3,
    #     dropout=0.1,
    # )
    # model.compile('adadelta', 'mse')
    # # model.summary()
    # return model

    print("[INFO] building autoencoder...")
    # (encoder, decoder, autoencoder) = ConvAutoencoder.build(64, 64, 3)
    # encoder.summary()
    # opt = Adam(lr=1e-3)

    # autoencoder.compile(loss="mse", optimizer='adam')

    # Compile the model using KL loss
    autoencoder = SequenceAutoencoder((64, 64, 3), 1024)
    autoencoder.compile()
    autoencoder.summary()

    return autoencoder
