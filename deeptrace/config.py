"""
A module for handling configuration. 
"""

from pathlib import Path

import yaml

IMAGE_WIDTH = 64
"""A default image width."""
IMAGE_HEIGHT = 64
"""A default image height."""
IMAGE_CHANNELS = 3
"""A default number of image channels."""

DATASET_DIR = 'dataset/synthetic'
TRAIN_DIR = f'{DATASET_DIR}-tfrecords/train'
VAL_DIR = f'{DATASET_DIR}-tfrecords/val'

DETECTING_CATEGORIES = ['rectangle', 'triangle']
NUM_DETECTING_OBJECTS = 10

LATENT_SIZE = 1024

BATCH_SIZE = 64
EPOCHS = 100


class Config(dict):
    """An experiment configuration.

    Args:
        source (dict): The source for creating an experiment configuration.
    """

    def __init__(self, source: dict = None):
        super().__init__({} or source)

    def __getitem__(self, key):
        path = key.split('/')
        branch = self.copy()

        for p in path:
            if isinstance(branch, list):
                try:
                    branch = branch[int(p)]
                except IndexError:
                    raise KeyError('%s in %s;' % (p, key))
            else:
                branch = branch[p]

        return branch

    def __setitem__(self, key, value):
        path = key.split('/')
        branch = self.copy()

        if len(path) == 1:
            super().__setitem__(key, value)
        elif len(path) > 1:
            for p in path[:-1]:
                if isinstance(branch, list):
                    try:
                        branch = branch[int(p)]
                    except IndexError:
                        raise KeyError('%s in %s;' % (p, key))
                else:
                    branch = branch[p]

            if isinstance(branch, list):
                try:
                    branch[int(path[-1])] = value
                except IndexError:
                    raise KeyError('%s in %s;' % (path[-1], key))
            else:
                branch[path[-1]] = value
        else:
            raise KeyError('empty key;')

    def __delitem__(self, key):
        path = key.split('/')
        branch = self.copy()

        if len(path) == 1:
            super().__delitem__(key)
        elif len(path) > 1:
            for p in path[:-1]:
                if isinstance(branch, list):
                    try:
                        branch = branch[int(p)]
                    except IndexError:
                        raise KeyError('%s in %s;' % (p, key))
                else:
                    branch = branch[p]

            if isinstance(branch, list):
                try:
                    del branch[int(path[-1])]
                except IndexError:
                    raise KeyError('%s in %s;' % (path[-1], key))
            else:
                del branch[path[-1]]
        else:
            raise KeyError('empty key;')

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False


def create_config(image_width: int = IMAGE_WIDTH,
                  image_height: int = IMAGE_HEIGHT,
                  image_channels: int = IMAGE_CHANNELS,
                  train_dir: str = TRAIN_DIR,
                  train_steps_per_epoch: int = 0,
                  val_dir: str = VAL_DIR,
                  val_steps_per_epoch: int = 0,
                  num_detecting_objects: int = NUM_DETECTING_OBJECTS,
                  detecting_categories: list = BATCH_SIZE,
                  latent_size: int = LATENT_SIZE,
                  batch_size: int = BATCH_SIZE,
                  epochs: int = EPOCHS):
    source = {
        'input': {
            'image': {
                'width': image_width,
                'height': image_height,
                'channels': image_channels
            },
            'train': {
                'dir': train_dir,
                'steps': train_steps_per_epoch
            },
            'val': {
                'dir': val_dir,
                'steps': val_steps_per_epoch
            }
        },
        'output': {
            'detecting': {
                'capacity': num_detecting_objects,
                'categories': detecting_categories
            }
        },
        'fitting': {
            'model': {
                'latent_size': latent_size
            },
            'batch': {
                'size': batch_size
            },
            'epochs': epochs
        }
    }
    return Config(source)


def load_config(fp: Path) -> Config:
    with fp.open('r') as f:
        return Config(yaml.safe_load(f))


def save_config(fp: Path, config: Config):
    with fp.open('w') as f:
        yaml.dump(dict(config), f)