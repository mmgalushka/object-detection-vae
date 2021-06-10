"""
A module for handling user actions. 
"""

import os

from .image import IMAGE_CAPACITY, Palette, Background
from .dataset import DATASET_DIR, DATASET_SIZE, DataFormat, create_csv_dataset
from .tfrecords import create_tfrecords
from .experiment import create_experiment, get_experiment
from .config import (create_config, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT,
                     TRAIN_DIR, VAL_DIR, DETECTING_CATEGORIES,
                     NUM_DETECTING_OBJECTS, LATENT_SIZE, BATCH_SIZE, EPOCHS)


def dataset(subparsers):

    def run(args):
        create_csv_dataset(
            dataset_dir=args.output,
            dataset_size=args.size,
            image_width=args.image_width,
            image_height=args.image_height,
            image_palette=args.image_palette,
            image_background=args.image_background,
            image_capacity=args.image_capacity,
            verbose=args.verbose)

    # ---------------------------------
    # Sets "dataset" command options
    # ---------------------------------
    parser = subparsers.add_parser('dataset')
    parser.set_defaults(func=run)

    # --- I/O options -----------------
    parser.add_argument(
        '-o',
        '--output',
        metavar='DIR',
        type=str,
        default=DATASET_DIR,
        help=f'an output directory (default="{DATASET_DIR}")')
    parser.add_argument(
        '-f',
        '--format',
        choices=DataFormat.values(),
        type=str,
        default=DataFormat.default(),
        help=f'a format of creating dataset (default="{DataFormat.default()}")')
    parser.add_argument(
        '-s',
        '--size',
        metavar='NUMBER',
        type=int,
        default=DATASET_SIZE,
        help=f'a number of generated data samples (default={DATASET_SIZE})')

    # --- image options ---------------
    parser.add_argument(
        '--image-width',
        metavar='PIXELS',
        type=int,
        nargs=1,
        default=IMAGE_WIDTH,
        help=f'a generated image width (default={IMAGE_WIDTH})')
    parser.add_argument(
        '--image-height',
        metavar='PIXELS',
        type=int,
        nargs=1,
        default=IMAGE_HEIGHT,
        help=f'a generated image height (default={IMAGE_HEIGHT})')
    parser.add_argument(
        '--image-palette',
        choices=Palette.values(),
        type=str,
        default=Palette.default(),
        help=f'an image palette (default="{Palette.default()}")')
    parser.add_argument(
        '--image-background',
        choices=Background.values(),
        type=str,
        default=Background.default(),
        help=f'an image background (default="{Background.default()}")')
    parser.add_argument(
        '--image-capacity',
        metavar='NUMBER',
        type=int,
        default=IMAGE_CAPACITY,
        help=f'a number of shapes per image (default={IMAGE_CAPACITY})')

    # --- system options --------------
    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')


def tfrecords(subparsers):

    def run(args):
        create_tfrecords(
            dataset_dir=args.input,
            dataset_format=args.format,
            dataset_categories=['rectangle', 'triangle'],
            tfrecords_dir=args.output,
            tfrecords_size=args.size,
            image_width=args.image_width,
            image_height=args.image_height,
            verbose=args.verbose)

    # ---------------------------------
    # Sets "tfrecords" command options
    # ---------------------------------
    parser = subparsers.add_parser('tfrecords')
    parser.set_defaults(func=run)

    # --- input options ---------------
    parser.add_argument(
        '-i',
        '--input',
        metavar='DIR',
        type=str,
        default=DATASET_DIR,
        help=f'an input directory with source data (default="{DATASET_DIR}")')
    parser.add_argument(
        '-f',
        '--format',
        choices=DataFormat.values(),
        type=str,
        default=DataFormat.default(),
        help=f'a format of source dataset (default="{DataFormat.default()}")')

    # --- output options --------------
    parser.add_argument(
        '-o',
        '--output',
        metavar='DIR',
        type=str,
        default=None,
        help=f'an output directory for TFRecords (default=None)')
    parser.add_argument(
        '-s',
        '--size',
        metavar='NUMBER',
        type=int,
        default=256,
        help=f'a number of records per partion (default=256)')

    # --- image options ---------------
    parser.add_argument(
        '--image-width',
        metavar='PIXELS',
        type=int,
        default=None,
        help=f'an image width resize to (default=None)')
    parser.add_argument(
        '--image-height',
        metavar='PIXELS',
        type=int,
        default=None,
        help=f'an image height resize to (default=None)')

    # --- system options --------------
    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')


def train(subparsers):

    def run(args):
        config = create_config(args.image_width, args.image_height,
                               args.image_channels, args.train_dir,
                               args.train_steps_per_epoch, args.val_dir,
                               args.val_steps_per_epoch,
                               args.num_detecting_objects,
                               args.detecting_categories, args.latent_size,
                               args.batch_size, args.epochs)
        experiment = create_experiment(args.experiment_dir)
        experiment.train(config, args.retrain, args.verbose)

    # -----------------------------
    # Sets "train" command options
    # -----------------------------
    parser = subparsers.add_parser('train')
    parser.set_defaults(func=run)

    parser.add_argument(
        'experiment_dir',
        metavar='EXPERIMENT_DIR',
        type=str,
        nargs='?',
        default=None,
        help='an experiment directory')

    # --- input options ---------------
    parser.add_argument(
        '--image-width',
        metavar='PIXELS',
        type=int,
        nargs=1,
        default=IMAGE_WIDTH,
        help=f'a generated image width (default={IMAGE_WIDTH})')
    parser.add_argument(
        '--image-height',
        metavar='PIXELS',
        type=int,
        nargs=1,
        default=IMAGE_HEIGHT,
        help=f'a generated image height (default={IMAGE_HEIGHT})')
    parser.add_argument(
        '--image-channels',
        metavar='NUMBER',
        type=int,
        default=IMAGE_CHANNELS,
        help=f'a number of image channels (default={IMAGE_CHANNELS})')

    parser.add_argument(
        '--train-dir',
        metavar='DIR',
        type=str,
        default=TRAIN_DIR,
        help=f'an input directory with training data (default={TRAIN_DIR})')
    parser.add_argument(
        '--train-steps-per-epoch',
        metavar='NUMBER',
        type=int,
        default=0,
        help='a number of trining steps per epoch (default=0)')
    parser.add_argument(
        '--val-dir',
        metavar='DIR',
        type=str,
        default=VAL_DIR,
        help=f'an input directory with validation data (default={VAL_DIR})')
    parser.add_argument(
        '--val-steps-per-epoch',
        metavar='NUMBER',
        type=int,
        default=0,
        help='a number of validation steps per epoch (default=0)')

    # --- output options ---------------
    parser.add_argument(
        '--detecting-categories',
        metavar='LABEL',
        type=str,
        nargs='+',
        default=DETECTING_CATEGORIES,
        help=f'a list of detecting categories (default={DETECTING_CATEGORIES})')
    parser.add_argument(
        '--num-detecting-objects',
        metavar='NUMBER',
        type=int,
        default=NUM_DETECTING_OBJECTS,
        help=f'a number of detecting objects per image (default={NUM_DETECTING_OBJECTS})'
    )

    # --- training options ------------
    parser.add_argument(
        '-r',
        '--retrain',
        help='the flag forcing model to retrain',
        action='store_true')
    parser.add_argument(
        '--latent-size',
        metavar='SIZE',
        type=int,
        default=LATENT_SIZE,
        help=f'a latent space size (default={LATENT_SIZE})')
    parser.add_argument(
        '--batch-size',
        metavar='SIZE',
        type=int,
        default=BATCH_SIZE,
        help=f'a batch size (default={BATCH_SIZE})')
    parser.add_argument(
        '--epochs',
        metavar='NUMBER',
        type=int,
        default=EPOCHS,
        help=f'a number of training epochs (default={EPOCHS})')

    # --- system options --------------
    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')


def predict(subparsers):

    def run(args):
        experiment = get_experiment(args.experiment_dir)
        experiment.predict(args.input, args.verbose)

    # -----------------------------
    # Sets "train" command options
    # -----------------------------
    parser = subparsers.add_parser('predict')
    parser.set_defaults(func=run)

    parser.add_argument(
        'experiment_dir',
        metavar='EXPERIMENT_DIR',
        type=str,
        nargs='?',
        default=None,
        help='an experiment directory')

    parser.add_argument(
        '-i', '--input', metavar='FILE', type=str, help='an input directory')

    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')