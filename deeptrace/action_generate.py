"""
An action for generating synthetic dataset.
"""

from .dataset import create_csv_dataset


def generate(subparsers):
    parser = subparsers.add_parser('generate')
    parser.set_defaults(func=run)

    parser.add_argument(
        '-o',
        '--output',
        metavar='DIR',
        type=str,
        nargs=1,
        help='an output directory')

    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')


def run(args):
    create_csv_dataset(dict())