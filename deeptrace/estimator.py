"""
A module for orchestrating model training and prediction.
"""

import math
from pathlib import Path

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
from .experiment import Experiment


def train(experiment, verbose: bool = False):
    config = experiment.config
    model = create_model(config, verbose)

    checkpoint = ModelCheckpoint(
        str(experiment.directory / 'model.h5'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1)

    batch_size = config['fitting/batch/size']

    train_path = Path(config['input/train/dir'])
    train_data = create_generator(
        tfrecords_dir=train_path,
        num_detecting_objects=config['output/detecting/capacity'],
        detecting_categories=config['output/detecting/categories'],
        batch_size=batch_size)
    train_steps = config['input/train/steps']
    # Counts the number of training steps per epoch if it number
    # is not specified.
    if train_steps == 0:
        train_steps = math.ceil(count_records(train_path, verbose) / batch_size)

    val_path = Path(config['input/train/dir'])
    val_data = create_generator(
        tfrecords_dir=val_path,
        num_detecting_objects=config['output/detecting/capacity'],
        detecting_categories=config['output/detecting/categories'],
        batch_size=batch_size)
    val_steps = config['input/train/steps']
    # Counts the number of validation steps per epoch if it number
    # is not specified.
    if val_steps == 0:
        val_steps = math.ceil(count_records(val_path, verbose) / batch_size)

    model.fit(
        train_data,
        steps_per_epoch=train_steps,
        epochs=config['fitting/epochs'],
        validation_data=val_data,
        validation_steps=val_steps,
        workers=4,
        verbose=int(verbose),
        callbacks=[checkpoint])


def predict(experiment, image_file: str, verbose: bool = False):
    print('================================================')
    print(image_file)

    config = experiment.config
    model = create_model(config)
    if verbose:
        model.summary()
    model.load_weights('model.h5')

    image = Image.open(image_file)
    array = np.array(image) * (1. / 255)
    # print(array.shape)

    batch = model.predict(np.array([array]))
    output = Image.fromarray((batch[0][0] * 255).astype(np.uint8))
    # print(batch[0])
    # output = Image.fromarray((batch[0] * 255).astype(np.uint8))

    coordinates = batch[1][0]
    print('----->>>', len(coordinates))
    print(coordinates)

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
                draw.text((x + 3, y), 'R' if r > t else 'T', 'red', font=font)
    output.save('output.jpg', 'JPEG', quality=100, subsampling=0)

    draw1 = ImageDraw.Draw(image)
    for (x, y, w, h, _, r, t) in coordinates:
        idx = np.argmax([_, r, t])
        if idx > 0:
            bbox = (x, y, x + w, y + h)
            draw1.rectangle(bbox, outline='red')
            draw1.text((x + 3, y), 'R' if r > t else 'T', 'red', font=font)
    image.save('real.jpg', 'JPEG', quality=100, subsampling=0)
