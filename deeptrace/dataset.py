"""
A module for creating and transforming dataset to TFRecords.
"""

from __future__ import annotations

import glob
import random

import tensorflow as tf

import pathlib
import random
from enum import Enum

from tqdm import tqdm

from .image import Palette, Background, create_synthetic_image

__all__ = ['DATASET_DIR', 'DataFormat', 'create_csv_dataset']

DATASET_DIR = 'dataset/synthetic'
"""A default dataset directory."""
DATASET_SIZE = 10000


class DataFormat(str, Enum):
    CSV = 'csv'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'{self.__class__.__name__}({str(self)})'

    @staticmethod
    def values():
        """Returns a list of data format values."""
        return list(map(str, DataFormat))

    @staticmethod
    def default():
        """Returns a default data format value."""
        return DataFormat.CSV


def create_csv_dataset(dataset_dir: str, dataset_size: int, image_width: int,
                       image_height: int, image_palette: Palette,
                       image_background: Background, image_capacity: int,
                       verbose: bool):
    """Generates CSV dataset.

    Args:
        dataset_dir (str): The directory where generated data will be stored.
        dataset_size (int): The number of generated samples.
        image_width (int): The image width in pixels.
        image_height (int): The image height in pixels.
        image_palette (Palette): The palette for generating images.
        image_background (Background): The palette for generating images.
        image_capacity (int): The number of geometrical shapes per image.
        verbose (bool): The flag to set verbose mode.
    """
    # Creates the dataset output directory if it does not exist.
    output_dir = pathlib.Path(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Creates subdirectory for storing synthetic images if it does not exist.
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    # Defines CSV files to populate with synthetic data.
    train_file = pathlib.Path(output_dir / 'train.csv')
    val_file = pathlib.Path(output_dir / 'val.csv')
    test_file = pathlib.Path(output_dir / 'test.csv')

    # Creates synthetic data.
    with open(train_file, 'w') as train,\
        open(val_file,'w') as val,\
        open(test_file, 'w') as test:

        # Adds a CSV header.
        header = 'image,bbox,segment,category\n'
        train.write(header)
        val.write(header)
        test.write(header)

        # Initializes the progress bar of verbose mode is on.
        if verbose:
            pbar = tqdm(range(dataset_size))

        # Adds CSV records.
        for image_id in range(dataset_size):
            image_file = f'image{image_id}.jpg'
            image, shapes = create_synthetic_image(image_width, image_height,
                                                   image_palette,
                                                   image_background,
                                                   image_capacity)
            path = images_dir / image_file
            image.save(path, 'JPEG', quality=100, subsampling=0)

            # Collects binding boxes and polygons.
            bboxes = []
            segments = []
            categories = []
            for shape in shapes:
                bboxes.append(shape.bbox.flatten())
                segments.append(shape.polygon.flatten())
                categories.append(shape.category)

            # Assembles a dataset record.
            record = ','.join([
                f'images/{image_file}', f'"{str(bboxes)}"',
                f'"{str(segments)}"', f'"{categories}"', '\n'
            ])

            partition = random.uniform(0, 1)
            if partition <= 0.7:
                train.write(record)
            elif partition <= 0.9:
                val.write(record)
            else:
                test.write(record)

            # Updates the progress bar of verbose mode is on.
            if verbose:
                pbar.update(1)


# def _create_coco_dataset(config: dict):
#     """Generates COCO dataset according to the specified configuration.

#     Args:
#         config (dict): The configuration to use for generating dataset.
#     """
#     # Makes sure the configuration is defined.
#     if not isinstance(config, dict):
#         raise TypeError(f'config must be {dict} but got {type(config)};')

#     # Creates the dataset directory if it does not exist.
#     dataset_dir = pathlib.Path('dataset' or config.get('output'))
#     dataset_dir.mkdir(exist_ok=True)

#     # Creates subdirectory for storing synthetic data if it does not exist.
#     synthetic_dir = dataset_dir / 'synthetic'
#     synthetic_dir.mkdir(exist_ok=True)

#     # Creates subdirectory for storing synthetic images if it does not exist.
#     images_dir = synthetic_dir / 'images'
#     images_dir.mkdir(exist_ok=True)

#     # Defines JSON files to populate with synthetic data.
#     train_file = pathlib.Path(synthetic_dir / 'train.json')
#     val_file = pathlib.Path(synthetic_dir / 'val.json')
#     test_file = pathlib.Path(synthetic_dir / 'test.json')

#     # Creates a generic COCO template.
#     template = {
#         'info': {
#             'description': 'COCO 2017 Dataset',
#             'url': 'http://cocodataset.org',
#             'version': '1.0',
#             'year': 2017,
#             'contributor': 'COCO Consortium',
#             'date_created': '2017/09/01'
#         },
#         'licenses': [{
#             'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
#             'id': 1,
#             'name': 'Attribution-NonCommercial-ShareAlike License'
#         }],
#         'images': [],
#         'categories': [{
#             'supercategory': 'geometry',
#             'id': 0,
#             'name': 'rectangle'
#         }, {
#             'supercategory': 'geometry',
#             'id': 1,
#             'name': 'triangle'
#         }],
#         'annotations': []
#     }

#     # Defines train/validation/test data aggregators.
#     train = copy.deepcopy(template)
#     val = copy.deepcopy(template)
#     test = copy.deepcopy(template)

#     # Creates COCO records.
#     annotation_id = 0
#     toggle = 0
#     pbar = tqdm(range(1000))
#     for image_id in range(1000):
#         if toggle:
#             filename = f'rectangle{image_id}.jpg'
#             image = create_synthetic_image({})
#             target = 'rectangle'
#         else:
#             filename = f'triangle{image_id}.jpg'
#             image = create_synthetic_image({})
#             target = 'triangle'
#         path = images_dir / filename
#         image.save(path)

#         # Collects binding boxes and polygons.
#         bbox = []
#         segmentation = []
#         for shape in image.shapes:
#             bbox.append(shape.bbox.flatten())
#             segmentation.append(shape.polygon.flatten())

#         # Assembles a dataset record.
#         images_record = {
#             'file_name': f'images/{filename}',
#             'coco_url': f'images/{filename}',
#             'height': 64,
#             'width': 64,
#             'date_captured': '2013-11-14 17:02:52',
#             'id': image_id
#         }
#         annotations_record = {
#             'segmentation': segmentation,
#             'area': 702.1057499999998,
#             'iscrowd': 0,
#             'image_id': image_id,
#             'bbox': bbox,
#             'category_id': int(toggle),
#             'id': annotation_id
#         }
#         annotation_id += 1

#         partition = random.uniform(0, 1)
#         if partition <= 0.7:
#             train['images'].append(images_record)
#             train['annotations'].append(annotations_record)
#         elif partition <= 0.9:
#             val['images'].append(images_record)
#             val['annotations'].append(annotations_record)
#         else:
#             test['images'].append(images_record)
#             test['annotations'].append(annotations_record)

#         pbar.update(1)

#         toggle = not toggle

#     with open(train_file, 'w') as fp:
#         json.dump(train, fp, indent=4)
#     with open(val_file, 'w') as fp:
#         json.dump(val, fp, indent=4)
#     with open(test_file, 'w') as fp:
#         json.dump(test, fp, indent=4)


def create_generator(tfrecords_dir):

    def _parse_function(proto):
        keys_to_features = {
            'image/content':
                tf.io.FixedLenSequenceFeature([], tf.string,
                                              allow_missing=True),
            'bbox/data':
                tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        parsed_features['image/content'] = tf.io.decode_raw(
            parsed_features['image/content'], tf.uint8)
        x = tf.reshape(parsed_features['image/content'], (64, 64, 3))
        x = tf.cast(x, tf.float32) / 255.0

        b = parsed_features['bbox/data']
        return x, (x, tf.reshape(b, (2, 4)))

    tfrecord_files = glob.glob(f'{tfrecords_dir}/part-*.tfrecord')
    print(tfrecord_files)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(64)

    return dataset