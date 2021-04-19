import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, GlobalAveragePooling2D,
                                     Dropout, Flatten, Input, MaxPooling2D,
                                     Activation)


def create_model():
    input_shape = (224, 224, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    return model
