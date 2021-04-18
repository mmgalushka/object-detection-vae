"""
An image.
"""

from __future__ import annotations

import random

import PIL.Image
import PIL.ImageDraw

from .point import Point
from .bbox import BBox

IMAGE_PADDING = 2
"""An area around the image perimeter does not accessible for drawing."""
IMAGE_WIDTH = 224
"""A default image width."""
IMAGE_HEIGHT = 224
"""A default image height."""


class Image:

    def __init__(self, image, *shapes):
        self.image = image
        self.shapes = shapes

    @property
    def mode(self):
        return self.image.mode

    @property
    def im(self):
        return self.image.im

    @property
    def readonly(self):
        return self.image.readonly

    def load(self):
        return self.image.load()

    def save(self, fp):
        self.image.save(fp, 'JPEG')

    @staticmethod
    def new(size: tuple, *shapes) -> Image:
        image = Image(PIL.Image.new('RGB', size, (255, 255, 255)), shapes)

        draw = PIL.ImageDraw.Draw(image)
        for shape in shapes:
            draw.polygon(shape.polygon, fill='black')

        return image


def create_synthetic_image(config: dict, shape: type) -> Image:
    """Returns a generated image with a drawing of a rectangle.

    Args:
        config (dict): The configuration for generating an image.
        shape (type): One of the following classes `Rectangle` or `Triangle`.
    Returns:
        The generate image.
    """
    # Get image properties from the configuration. If this information is
    # not available then use the default one.
    padding = config.get('padding') or IMAGE_PADDING
    width = config.get('width') or IMAGE_WIDTH
    height = config.get('height') or IMAGE_HEIGHT
    size = (width, height)

    # Generates a coordinate point in the top-left quadrant.
    tl = Point(
        random.randrange(padding, width / 2),
        random.randrange(padding, height / 2))

    # Generates a coordinate point in the Bottom-right quadrant.
    br = Point(
        random.randrange(width / 2, width - padding),
        random.randrange(height / 2, height - padding))

    # Creates a binding box around a figure.
    bbox = BBox(tl, br)

    # Creates a synthetic image and projecting a specified shape.
    return Image.new(size, shape(bbox))
