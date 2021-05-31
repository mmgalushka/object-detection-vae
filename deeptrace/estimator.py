"""
A module for orchestrating model training and prediction.
"""

import pathlib

import tensorflow as tf
from tensorflow.keras import callbacks

from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import math

from .tfrecords import create_generator
from .models import create_model
from .losses import hungarian_dist, total_dist

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.compat.v1.enable_eager_execution()


def create_estimator(input_shape: tuple, latent_size: int,
                     detecting_categories: list):
    model = create_model(input_shape, latent_size)

    losses = {"default_decoder_1": 'mae', "concatenate": total_dist}
    lossWeights = {"default_decoder_1": 1, "concatenate": 0.01}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights)

    return Estimator(model, detecting_categories)


class Estimator:

    def __init__(self, model, detecting_categories):
        self.model = model
        self.detecting_categories = detecting_categories

    def train(self, train_dir: str, val_dir: str, batch_size: int,
              verbose: bool):
        checkpoint = ModelCheckpoint(
            'model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1)

        self.model.summary()

        train_data = create_generator(
            tfrecords_dir=pathlib.Path(train_dir),
            detecting_categories=self.detecting_categories,
            num_detecting_objects=100,
            batch_size=batch_size)
        validation_data = create_generator(
            tfrecords_dir=pathlib.Path(val_dir),
            detecting_categories=self.detecting_categories,
            num_detecting_objects=100,
            batch_size=batch_size)

        self.model.fit(
            train_data,
            steps_per_epoch=int(7000 / batch_size),
            epochs=300,
            validation_data=validation_data,
            validation_steps=int(2000 / batch_size),
            workers=4,
            verbose=1,
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