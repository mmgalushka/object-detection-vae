"""
A module for setting up a logger. 
"""

import os
import logging
import logging.config
import pathlib

from functools import wraps
from datetime import datetime

import yaml


def setup_logging(config_file: str = 'logging.yaml',
                  default_level: int = logging.INFO):
    config_path = pathlib.Path(config_file)
    if config_path.exists():
        with config_path.open('r') as f:
            config = yaml.safe_load(f.read())

        logger_parh = pathlib.Path(config['handlers']['file']['filename'])
        logger_parh.parent.mkdir(exist_ok=True)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


class TimeIt:
    """Times a code snipper.

    Args:
        description (str): The code snipper description.
        logger (Logger): The logger to use for reporting (default=None).
    """

    def __init__(self, description, logger):
        self.__description = description
        self.__logger = logger
        self.__start = None
        self.__finish = None

    def __enter__(self):
        self.__start = datetime.now()

    def __exit__(self, exc_type, exc_value, traceback):
        # Adding this function allows prettifying the log-output.
        def timer(duration):
            if exc_type is None:
                self.__logger.debug('%s execution time %s;', self.__description,
                                    duration)
            else:
                self.__logger.debug('%s execution has failed after %s;',
                                    self.__description, duration)

        self.__finish = datetime.now()
        timer(self.__finish - self.__start)


def timeit(logger: logging.Logger):
    """Times a function.

    Args:
        logger (Logger): The logger to use for reporting (default=None).
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            with TimeIt("The \'%s\'" % func.__name__, logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator