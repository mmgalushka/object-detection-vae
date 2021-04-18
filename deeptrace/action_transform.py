"""
An action for converting a data source to TFRecords
"""

from .dataset import transform_CSV_to_TFRecords


def transform(subparsers):
    parser = subparsers.add_parser('transform')
    parser.set_defaults(func=run)

    parser.add_argument(
        '-i',
        '--input',
        metavar='DIR',
        nargs=1,
        type=str,
        help='an input directory')
    parser.add_argument(
        '-o',
        '--output',
        metavar='DIR',
        nargs=1,
        type=str,
        help='an output directory')
    parser.add_argument(
        '-f',
        '--format',
        choices=['csv', 'coco'],
        type=str,
        help='a format od source dataset')
    parser.add_argument(
        '-c',
        '--capacity',
        metavar='NUMBER',
        nargs=1,
        type=int,
        help='a number of records sorted in each TFRecord part')

    parser.add_argument(
        '-v',
        '--verbose',
        help='the flag to set verbose mode',
        action='store_true')


def run(args):
    transform_CSV_to_TFRecords({
        'input': {
            'directory': 'dataset/synthetic' or args.input
        },
        'output': {
            'directory': 'dataset/tfrecords' or args.output,
            'capacity': 200 or args.capacity,
        },
        'verbose': args.verbose
    })
