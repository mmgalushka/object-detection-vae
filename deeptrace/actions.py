"""
A module for handling user actions. 
"""

from .tfrecords import create_generator
from .models import create_model
from .estimator import create_estimator
from .losses import hungarian_dist, total_dist
from .image import IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CAPACITY, Palette, Background
from .dataset import DATASET_DIR, DATASET_SIZE, DataFormat, create_csv_dataset
from .tfrecords import TFRECORDS_CAPACITY, create_tfrecords


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
            tfrecords_dir=args.output,
            tfrecords_capacity=args.capacity,
            image_width=args.image_width,
            image_height=args.image_height,
            image_channels=args.image_channels,
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
        '-c',
        '--capacity',
        metavar='NUMBER',
        type=int,
        default=TFRECORDS_CAPACITY,
        help=f'a number of records per partion (default="{TFRECORDS_CAPACITY}")'
    )

    # --- image options ---------------
    parser.add_argument(
        '--image-width',
        metavar='PIXELS',
        type=int,
        default=IMAGE_WIDTH,
        help=f'a recording image width (default={IMAGE_WIDTH})')
    parser.add_argument(
        '--image-height',
        metavar='PIXELS',
        type=int,
        default=IMAGE_HEIGHT,
        help=f'a recording image height (default={IMAGE_HEIGHT})')
    parser.add_argument(
        '--image-channels',
        metavar='NUMBER',
        type=int,
        default=IMAGE_CHANNELS,
        help=f'a recording image channels (default={IMAGE_CHANNELS})')

    # --- system options --------------
    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')


def train(subparsers):

    def run(args):

        input_shape = (64, 64, 3)
        latent_size = 1024

        losses = {"default_decoder_1": 'mae', "concatenate": total_dist}
        lossWeights = {"default_decoder_1": 1, "concatenate": 0.01}

        model = create_model(input_shape, latent_size)

        model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights)

        estimator = create_estimator(model)

        train_data = create_generator('dataset/synthetic-tfrecords/train')
        validation_data = create_generator('dataset/synthetic-tfrecords/val')
        estimator.train(train_data, validation_data)

    # -----------------------------
    # Sets "train" command options
    # -----------------------------
    parser = subparsers.add_parser('train')
    parser.set_defaults(func=run)

    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')


def predict(subparsers):

    def run(args):
        input_shape = (64, 64, 3)
        latent_size = 1024

        model = create_model(input_shape, latent_size)
        estimator = create_estimator(model)

        estimator.predict(args.input)

    # -----------------------------
    # Sets "train" command options
    # -----------------------------
    parser = subparsers.add_parser('predict')
    parser.set_defaults(func=run)

    parser.add_argument(
        '-i', '--input', metavar='FILE', type=str, help='an input directory')

    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')