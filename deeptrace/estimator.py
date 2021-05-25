"""
A module for orchestrating model training and prediction.
"""

import tensorflow as tf
from tensorflow.keras import callbacks

from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import math

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


def create_estimator(model):
    return Estimator(model)


class Estimator:

    def __init__(self, model):
        self.model = model

    def train(self, train_data, validation_data):
        checkpoint = ModelCheckpoint(
            'model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1)

        self.model.summary()

        self.model.fit(
            train_data,
            steps_per_epoch=int(7000 / 64),
            epochs=300,
            validation_data=validation_data,
            validation_steps=int(2000 / 64),
            workers=4,
            verbose=1,
            callbacks=[checkpoint])

    def predict(self, fp: str):
        print('================================================')
        print(fp)
        self.model.load_weights('model.h5')
        image = Image.open(fp)
        array = np.array(image) * (1. / 255)
        print(array.shape)

        batch = self.model.predict(np.array([array]))
        output = Image.fromarray((batch[0][0] * 255).astype(np.uint8))
        # print(batch[0])
        # output = Image.fromarray((batch[0] * 255).astype(np.uint8))

        coordinates = batch[1][0]
        print(coordinates)

        font = ImageFont.load_default()

        draw = ImageDraw.Draw(output)
        for (x, y, w, h, r, t) in coordinates:
            bbox = (x, y, x + w, y + h)
            draw.rectangle(bbox, outline='red')
            draw.text((x + 3, y), 'R' if r > t else 'T', 'red', font=font)
        output.save('output.jpg', 'JPEG', quality=100, subsampling=0)

        draw1 = ImageDraw.Draw(image)
        for (x, y, w, h, r, t) in coordinates:
            bbox = (x, y, x + w, y + h)
            draw1.rectangle(bbox, outline='red')
            draw1.text((x + 3, y), 'R' if r > t else 'T', 'red', font=font)
        image.save('real.jpg', 'JPEG', quality=100, subsampling=0)