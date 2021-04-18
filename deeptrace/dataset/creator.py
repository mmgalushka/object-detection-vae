"""
A synthetic dataset creator.
"""

import random
import pathlib
import json

from tqdm import tqdm

from .image import (create_synthetic_image, IMAGE_WIDTH, IMAGE_HEIGHT)
from .geometry import Rectangle, Triangle


def create_csv_dataset(config: dict):
    """Generates CSV dataset according to the specified configuration.

    Args:
        config (dict): The configuration to use for generating dataset.
    """
    # Makes sure the configuration is defined.
    if not isinstance(config, dict):
        raise TypeError(f'config must be {dict} but got {type(config)};')

    # Creates the dataset directory if it does not exist.
    dataset_dir = pathlib.Path('dataset' or config.get('output'))
    dataset_dir.mkdir(exist_ok=True)

    # Creates subdirectory for storing synthetic data if it does not exist.
    synthetic_dir = dataset_dir / 'synthetic'
    synthetic_dir.mkdir(exist_ok=True)

    # Creates subdirectory for storing synthetic images if it does not exist.
    images_dir = synthetic_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    # Defines CSV files to populate with synthetic data.
    train_file = pathlib.Path(synthetic_dir / 'train.csv')
    val_file = pathlib.Path(synthetic_dir / 'val.csv')
    test_file = pathlib.Path(synthetic_dir / 'test.csv')

    # Creates synthetic data.
    toggle = True
    with open(train_file, 'w') as train,\
        open(val_file,'w') as val,\
        open(test_file, 'w') as test:

        # Adds a CSV header.
        train.write('figure,shape\n')
        val.write('figure,shape\n')
        test.write('figure,shape\n')

        # Adds CSV records.
        pbar = tqdm(range(1000))
        for image_id in range(1000):
            if toggle:
                filename = f'rectangle{image_id}.jpg'
                image = create_synthetic_image({}, Rectangle)
                target = 0
            else:
                filename = f'triangle{image_id}.jpg'
                image = create_synthetic_image({}, Triangle)
                target = 1
            path = images_dir / filename
            image.save(path)

            partition = random.uniform(0, 1)
            record = f'images/{filename},{target}\n'
            if partition <= 0.7:
                train.write(record)
            elif partition <= 0.9:
                val.write(record)
            else:
                test.write(record)

            pbar.update(1)

            toggle = not toggle

    # Creates schema for synthetic data.
    schema_file = pathlib.Path(synthetic_dir / 'schema.json')
    with open(schema_file, 'w') as schema:
        json.dump(
            {
                'features': {
                    'figure': {
                        'type': 'image',
                        'transforme': {
                            'size': {
                                'width': IMAGE_WIDTH,
                                'height': IMAGE_HEIGHT
                            }
                        }
                    },
                    'shape': {
                        'type': 'categorical',
                        'transforme': {
                            'encode': 'one-hot',
                            'map': {
                                0: 0,
                                1: 1
                            }
                        }
                    }
                }
            },
            schema,
            indent=4)
