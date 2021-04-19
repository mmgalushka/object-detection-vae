"""
The module for manipulating real and synthetic data.
"""

from .creator import create_csv_dataset
from .transformer import transform_CSV_to_TFRecords
from .generator import create_dataset
