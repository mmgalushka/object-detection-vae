"""
An action for converting a data source to TFRecords
"""

from .dataset import create_dataset
from .ml import create_model, create_estimator


def train(subparsers):
    parser = subparsers.add_parser('train')
    parser.set_defaults(func=run)

    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')


def run(args):
    model = create_model()
    estimator = create_estimator(model)

    train_data = create_dataset('dataset/tfrecords/train')
    validation_data = create_dataset('dataset/tfrecords/val')
    estimator.train(train_data, validation_data)
