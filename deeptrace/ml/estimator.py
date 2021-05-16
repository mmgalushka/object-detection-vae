import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import math

tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()


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

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            verbose=0,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0)

        # Define configuration parameters
        start_lr = 0.001
        exp_decay = 0.1

        # Define the scheduling function
        def schedule(epoch):

            def lr(epoch, start_lr, exp_decay):
                if epoch > 25:
                    return start_lr * math.exp(-exp_decay * epoch)
                return 0.001

            return 1.0 * lr(epoch, start_lr, exp_decay)

        lr_callback = tf.keras.callbacks.LearningRateScheduler(
            schedule, verbose=True)

        early_stop = EarlyStopping(patience=10)

        # self.model.fit(
        #     train_data,
        #     steps_per_epoch=None,
        #     epochs=300,
        #     validation_data=validation_data,
        #     validation_steps=None,
        #     workers=4,
        #     verbose=1,
        #     callbacks=[reduce_lr, early_stop])

        # train the convolutional autoencoder

        # self.model.summary()

        self.model.fit(
            train_data,
            steps_per_epoch=108,
            epochs=300,
            validation_data=validation_data,
            validation_steps=33,
            workers=4,
            verbose=1,
            callbacks=[checkpoint])

        # self.model.fit(
        #     trainX, trainX,
        #     validation_data=(testX, testX),
        #     epochs=EPOCHS,
        #     batch_size=BS)

        # self.model.save_weights('model', save_format='tf')

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

        draw = ImageDraw.Draw(image)
        for (x, y, w, h) in coordinates:
            bbox = (x, y, x + w, y + h)
            draw.rectangle(bbox, outline='red')
        image.save('output.jpg', 'JPEG', quality=100, subsampling=0)
