"""
The DeepTrace main module.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import argparse
import logging

import tensorflow as tf
import deeptrace as dt

import warnings

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

dt.setup_logging()
LOG = logging.getLogger(__name__)
LOG.info('The application logger has been configured;')

parser = argparse.ArgumentParser(prog='./helper.sh', usage='%(prog)s')
subparsers = parser.add_subparsers()

# ---------------------------------
# Initializes application commands
# ---------------------------------
dt.dataset(subparsers)
dt.tfrecords(subparsers)
dt.train(subparsers)
dt.predict(subparsers)


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'help':
            parser.print_help()
        else:
            args = parser.parse_args(sys.argv[1:])
            args.func(args)
    else:
        parser.print_help()
    exit(os.EX_OK)


if __name__ == '__main__':
    main()
