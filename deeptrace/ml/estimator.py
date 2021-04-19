def create_estimator(model):
    return Estimator(model)


class Estimator:

    def __init__(self, model):
        self.model = model

    def train(self, train_data, validation_data):
        self.model.fit(
            train_data,
            steps_per_epoch=None,
            epochs=10,
            validation_data=validation_data,
            validation_steps=None,
            workers=4,
            verbose=1)
