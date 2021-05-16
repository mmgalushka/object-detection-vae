"""
The DeepTrace main module.
"""

import os
import sys
import argparse

import deeptrace as dt

parser = argparse.ArgumentParser(prog='./helper.sh', usage='%(prog)s')
subparsers = parser.add_subparsers()

# --------------------------------------
# Initializes dataset handling commands
# --------------------------------------
dt.dataset(subparsers)
dt.tfrecords(subparsers)

# --------------------------------------
# Initializes ML handling commands
# --------------------------------------
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
