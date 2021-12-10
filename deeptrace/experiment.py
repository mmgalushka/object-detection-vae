"""
A module for handling machine learning experiment.
"""

from __future__ import annotations

from typing import overload
from enum import Enum
from pathlib import Path
from logging import getLogger

from tensorflow.keras.callbacks import ModelCheckpoint

import yaml

from .config import save_config, load_config, Config
from .estimator import train, predict
from .tfrecords import create_generator
from .models import create_model

LOG = getLogger(__name__)

__all__ = [
    'make_experiment', 'ExperimentNotFoundException', 'ExperimentStatus',
    'Experiment'
]


def make_experiment(experiment_dir: str = None,
                    must_exist: bool = False) -> Experiment:
    """Creates a new or gets an existing experiment.

    Args:
        experiment_dir (srt): The experiment directory.
        not_exist_ok (bool): The flag for raising an error if the requested
            experiment does not exist. If the flag equals `False`, this method
            will consider that not existing experiment directory is ok,
            otherwise, if the flag equals `True,` this method will consider
            that not existing experiment directory is an error and will raise
            `ExperimentNotFoundException`. 

    Returns:
        Experiment: The created experiment object.

    Raises:
        ExperimentNotFoundException: If experiment directory does not exist
            and `must_exist` ia set to `True`
    """
    experiment_path = Path.cwd() if experiment_dir is None else Path(
        experiment_dir)

    if experiment_path.exists():
        LOG.info('Retrieved an existing experiment;')
    else:
        if must_exist:
            raise ExperimentNotFoundException(experiment_path)
        experiment_path.mkdir(parents=True)
        LOG.info('Creating a new experiment;')

    return Experiment(experiment_path)


class ExperimentNotFoundException(Exception):
    """This exception is raised when the requested experiment has not found.

    Args:
        experiment_path (Path): The requested experiment path.
    """

    def __init__(self, experiment_path) -> None:
        super().__init__(str(experiment_path))


class ExperimentStatus(int, Enum):
    """The enumeration of experiment statuses"""
    VOID = 0
    """The status indicates an empty experiment."""
    INPROGRESS = 1
    """The status indicates an experiment model under training."""
    COMPLETED = 2
    """The status indicates an experiment model training has completed."""
    FAILED = 3
    """The status indicates an experiment model training has failed."""

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'{self.__class__.__name__}({str(self)})'

    @staticmethod
    def by(value: int) -> ExperimentStatus:
        """Creates the experiment status by the value.

        Args:
            value (str): The experiment status value.

        Returns:
            status (ExperimentStatus) The created experiment status.

        Raises:
            ValueError: If the experiment status value is not supported.
        """
        for e in ExperimentStatus:
            if e == value:
                return e
        raise ValueError(value)

    @staticmethod
    def values() -> list:
        """Returns a list of experiment status values."""
        return list(map(str, ExperimentStatus))


class Experiment:
    """A machine learning experiment.

    The experiment object combines and provides unified access to all
    artefacts that had been created during the model training, such as
    training log, model weights and etc.

    Args:
        path (Path): The experiment location.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        LOG.info('The experiment location: %s;', self.path)

    @property
    def path(self) -> Path:
        """The experiment path."""
        return self._path

    @property
    def config(self) -> dict:
        """Gets the experiment configuration."""
        fp = self.path / 'config.yaml'
        try:
            return load_config(fp)
        except FileNotFoundError:
            return None

    @config.setter
    def config(self, value):
        """Sets the experiment configuration."""
        fp = self.path / 'config.yaml'
        save_config(fp, value)

    @property
    def status(self) -> ExperimentStatus:
        """Gets the experiment status."""
        fp = self.path / '.status'
        with fp.open('r') as f:
            return ExperimentStatus.by(int(f.read().strip()))

    @status.setter
    def status(self, value: ExperimentStatus):
        """Sets the experiment status."""
        fp = self.path / '.status'
        with fp.open('w') as f:
            f.write(str(value))

    def train(self,
              config: Config,
              retrain: bool = False,
              verbose: bool = False):
        """Trains a model associated with this experiment."""

        # Lets check and overwrite an experiment configuration if needed.
        if self.config is not None:
            # If a user has specified an experiment with an already existing
            # configuration it usually means that training of the associated
            # machine learning model has been terminated or completed
            # (successfully or unsuccessfully). Later, we will check the
            # experiment status and select appropriate actions. But if a user
            # has also specified the "re-train" flag, then the previous
            # experiment configuration must be overwritten and experiment
            # status reset to VOID.
            LOG.debug('The experiment already has a configuration;')
            if retrain:
                self.config = config
                self.status = ExperimentStatus.VOID
                LOG.info('Overwrote the experiment configuration (since '
                         'training has been launched with the retrain '
                         'flag); ')
        else:
            # If no configuration has associated with the specified experiment,
            # this means a new experiment. In this case, we need to save the
            # configuration and set the experiment status to VOID
            self.config = config
            self.status = ExperimentStatus.VOID
            LOG.info('Saved a new experiment configuration;')

        # Lets check and the experiment status.
        # if self.status > ExperimentStatus.INPROGRESS:
            # Any experiment status above the in-progress indicates the
            # raining of the associate machine learning model has completed
            # (successfully or unsuccessfully). In this case, we need to
            # terminate this function.
            # LOG.info('The experiment has been completed, with status: %s',
                    #  self.status)
            # return
            # Note: If a user wants to rerun a model training under this
            # experiment, it will require to set
            # "re-train" option/flag.
        LOG.info('The experiment current status: %s;', self.status)

        # The model training block.
        try:
            # This will save disc access operation (by loading config from
            # file to memory)
            exp_config = self.config

            # --- Create data generators ---------------------------------------

            # Gets the batch size use during training and validation
            batch_size = exp_config['fitting/batch/size']
            LOG.debug('Selected batch size: %d;', batch_size)

            # Gets the list of detecting categories.
            detecting_categories = exp_config['output/detecting/categories']
            LOG.debug('Selected list for detecting categories: %s;',
                      detecting_categories)

            # Ges the maximum number of detecting objects per image.
            detecting_capacity = exp_config['output/detecting/capacity']
            LOG.debug(
                'Selected the maximum number of detecting objects per '
                'image: %s;', detecting_capacity)

            # Creates generators for loading training data.
            LOG.debug('Creating training data generator;')
            train_data, train_steps = create_generator(
                tfrecords_path=Path(exp_config['input/train/dir']),
                detecting_categories=detecting_categories,
                num_detecting_objects=detecting_capacity,
                batch_size=batch_size,
                steps_per_epoch=exp_config['input/train/steps'],
                verbose=verbose)

            # Creates generators for loading validation data.
            LOG.debug('Creating validation data generator;')
            val_data, val_steps = create_generator(
                tfrecords_path=Path(exp_config['input/val/dir']),
                detecting_categories=detecting_categories,
                num_detecting_objects=detecting_capacity,
                batch_size=batch_size,
                steps_per_epoch=exp_config['input/val/steps'],
                verbose=verbose)

            # --- Create machine learning model --------------------------------

            model = create_model(exp_config, verbose)

            # --- Define model training callbacks ------------------------------

            checkpoint = ModelCheckpoint(
                str(self.path / 'model.h5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1)
            callbacks = [checkpoint]

            # --- Run model training callbacks ---------------------------------

            model.fit(
                train_data,
                steps_per_epoch=train_steps,
                epochs=exp_config['fitting/epochs'],
                validation_data=val_data,
                validation_steps=val_steps,
                workers=4,
                verbose=int(verbose),
                callbacks=callbacks)

            # Update the experiment status to indicate that model training
            # has been completed.
            self.status = ExperimentStatus.COMPLETED

            LOG.info('Completed model training;')
        except Exception as err:
            # Update the experiment status to indicate that model training
            # has failed.
            self.status = ExperimentStatus.FAILED

            LOG.critical('Model training has filed:', exc_info=err)

    def predict(self, image_file: str, verbose: bool = False):
        """Predicts using a model associated with this experiment."""
        # Runs prediction function defined in the estimator.
        predict(self, image_file, verbose)