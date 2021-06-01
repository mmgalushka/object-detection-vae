"""
A module for orchestrating model training and prediction.
"""

import math
import pathlib

import tensorflow as tf
from tensorflow.keras import callbacks

from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import math

from .tfrecords import create_generator, count_records
from .models import create_model
from .losses import hungarian_dist, total_dist

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.compat.v1.enable_eager_execution()

NUM_DETECTING_OBJECTS = 100
BATCH_SIZE = 64
EPOCHS = 100


def create_estimator(input_shape: tuple, latent_size: int,
                     detecting_categories: list):
    model = create_model(input_shape, latent_size)

    losses = {"DefaultDecoder": 'mae', "concatenate": total_dist}
    lossWeights = {"DefaultDecoder": 1, "concatenate": 0.01}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights)

    return Estimator(model, detecting_categories)


class Estimator:

    def __init__(self, model, detecting_categories):
        self.model = model
        self.detecting_categories = detecting_categories

    def train(self, train_dir: str, val_dir: str, num_detecting_objects: int,
              batch_size: int, train_steps_per_epoch: int,
              val_steps_per_epoch: int, epochs: int, verbose: bool):
        checkpoint = ModelCheckpoint(
            'model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1)

        self.model.summary()

        train_tfrecords_dir = pathlib.Path(train_dir)
        train_data = create_generator(
            tfrecords_dir=train_tfrecords_dir,
            detecting_categories=self.detecting_categories,
            num_detecting_objects=num_detecting_objects,
            batch_size=batch_size)
        # Counts the number of training steps per epoch if it number
        # is not specified.
        if train_steps_per_epoch == 0:
            train_steps_per_epoch = math.ceil(
                count_records(train_tfrecords_dir, verbose) / batch_size)

        val_tfrecords_dir = pathlib.Path(val_dir)
        validation_data = create_generator(
            tfrecords_dir=val_tfrecords_dir,
            detecting_categories=self.detecting_categories,
            num_detecting_objects=num_detecting_objects,
            batch_size=batch_size)
        # Counts the number of validation steps per epoch if it number
        # is not specified.
        if val_steps_per_epoch == 0:
            val_steps_per_epoch = math.ceil(
                count_records(val_tfrecords_dir, verbose) / batch_size)

        self.model.fit(
            train_data,
            steps_per_epoch=train_steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=val_steps_per_epoch,
            workers=4,
            verbose=int(verbose),
            callbacks=[checkpoint])

    def predict(self, fp: str):
        print('================================================')
        print(fp)
        self.model.load_weights('model.h5')
        image = Image.open(fp)
        array = np.array(image) * (1. / 255)
        # print(array.shape)

        batch = self.model.predict(np.array([array]))
        output = Image.fromarray((batch[0][0] * 255).astype(np.uint8))
        # print(batch[0])
        # output = Image.fromarray((batch[0] * 255).astype(np.uint8))

        coordinates = batch[1][0]
        # print(coordinates)

        font = ImageFont.load_default()

        draw = ImageDraw.Draw(output)
        for (x, y, w, h, _, r, t) in coordinates:
            probas = [_, r, t]
            idx = np.argmax(probas)
            if idx > 0:
                if probas[idx] > 0.8:
                    print('%.2f\t%.2f\t%.2f' % (_, r, t))

                    bbox = (x, y, x + w, y + h)
                    draw.rectangle(bbox, outline='red')
                    draw.text((x + 3, y),
                              'R' if r > t else 'T',
                              'red',
                              font=font)
        output.save('output.jpg', 'JPEG', quality=100, subsampling=0)

        draw1 = ImageDraw.Draw(image)
        for (x, y, w, h, _, r, t) in coordinates:
            idx = np.argmax([_, r, t])
            if idx > 0:
                bbox = (x, y, x + w, y + h)
                draw1.rectangle(bbox, outline='red')
                draw1.text((x + 3, y), 'R' if r > t else 'T', 'red', font=font)
        image.save('real.jpg', 'JPEG', quality=100, subsampling=0)