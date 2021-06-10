"""
A module for handling machine learning experiment.
"""

from __future__ import annotations
from typing import overload
from deeptrace.tfrecords import count_records

from enum import Enum
from pathlib import Path

import yaml

from .config import save_config, load_config, Config
from .estimator import train, predict


def create_experiment(directory: str = None) -> Experiment:
    path = Path.cwd() if directory is None else Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return Experiment(path)


def get_experiment(directory: str = None) -> Experiment:
    path = Path.cwd() if directory is None else Path(directory)
    return Experiment(path)


class ExperimentStatus(str, Enum):
    VOID = 'void'
    INPROGRESS = 'inprogress'
    COMPLETED = 'completed'
    FAILED = 'failed'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'{self.__class__.__name__}({str(self)})'

    @staticmethod
    def by(value) -> ExperimentStatus:
        """Creates the experiment status by the value.

        Args:
            value (str): The experiment status value.

        Returns:
            status (ExperimentStatus) The created experiment status.

        Raises:
            ValueError: If the experiment status value is not supported.
        """
        for e in ExperimentStatus:
            if str(e) == value:
                return e
        raise ValueError(value)

    @staticmethod
    def values() -> list:
        """Returns a list of experiment status values."""
        return list(map(str, ExperimentStatus))


class Experiment:

    def __init__(self, directory):
        self.__directory = directory

    @property
    def directory(self):
        return self.__directory

    @property
    def config(self) -> dict:
        fp = self.directory / 'config.yaml'
        try:
            return load_config(fp)
        except FileNotFoundError:
            return None

    @config.setter
    def config(self, value):
        fp = self.directory / 'config.yaml'
        save_config(fp, value)

    @property
    def status(self) -> ExperimentStatus:
        fp = self.directory / '.status'
        with fp.open('r') as f:
            return ExperimentStatus.by(f.read().strip())

    @status.setter
    def status(self, value: ExperimentStatus):
        fp = self.directory / '.status'
        with fp.open('w') as f:
            f.write(str(value))

    def train(self,
              config: Config,
              retrain: bool = False,
              verbose: bool = False):
        # Lets check and overwrite an experiment configuration in needed.
        has_config = self.config is not None
        if (has_config and retrain) or (not has_config):
            self.config = config

        # Runs training function defined in the estimator.
        train(self, verbose)

    def predict(self, image_file: str, verbose: bool = False):
        # Runs prediction function defined in the estimator.
        predict(self, image_file, verbose)