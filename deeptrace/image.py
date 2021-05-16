"""
A module for manipulating images.
"""

from __future__ import annotations

from typing import List
from enum import Enum

import numpy as np

import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps

IMAGE_WIDTH = 64
"""A default image width."""
IMAGE_HEIGHT = 64
"""A default image height."""
IMAGE_CAPACITY = 2
"""A default number of geometrical shapes per image."""


class Palette(str, Enum):
    COLOR = 'color'
    GRAY = 'gray'
    BINARY = 'binary'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'{self.__class__.__name__}({str(self)})'

    @staticmethod
    def values():
        """Returns a list of palette values."""
        return list(map(str, Palette))

    @staticmethod
    def default():
        """Returns a default palette value."""
        return Palette.COLOR


class Background(str, Enum):
    WHITE = 'white'
    BLACK = 'black'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'{self.__class__.__name__}({str(self)})'

    @staticmethod
    def values():
        """Returns a list of background values."""
        return list(map(str, Background))

    @staticmethod
    def default():
        """Returns a default background value."""
        return Background.WHITE


class Point:
    """A point.
    
    Args:
        x (int): The `x` coordinate.
        y (int): The `y` coordinate.

    Attributes:
        x (int): The `x` coordinate.
        y (int): The `y` coordinate.  
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def flatten(self) -> list:
        """Returns a flattened representation of a point.

        Returns:
            The array `[x, y]` where `x, y` the point coordinates.
        """
        return [self.x, self.y]


class Polygon(List[Point]):
    """A polygon"""

    def __str__(self):
        return f'[{", ".join(map(str,self))}]'

    def flatten(self) -> list:
        """Returns a flattened representation of a polygon.

        Returns:
            The array `[x1, y1, x2, y2, ...]` where `x_, y_` the polygon
            vertices.
        """
        output = []
        for point in self:
            output.extend(point.flatten())
        return output


class BBox(Polygon):
    """A binding box.

    Args:
        anchor (Point): The top-left point of the binding box.
        width (int): The binding box width.
        height (int): The binding box height.

    Attributes:
        anchor (Point): The top-left point of the binding box.
        width (int): The binding box width.
        height (int): The binding box height.
    """

    def __init__(self, anchor: Point, width: int, height: int):
        super().__init__([anchor, Point(anchor.x + width, anchor.y + height)])

    def __str__(self):
        return f'[{self.anchor}, {self.width}, {self.height}]'

    @property
    def anchor(self):
        return self[0]

    @property
    def width(self):
        return self[-1].x - self[0].x

    @property
    def height(self):
        return self[-1].y - self[0].y

    def flatten(self) -> list:
        """Returns a flattened representation of a binding box.

        Returns:
            The array `[x, y, width, height]` where `x, y` the binding box
            anchor coordinates and `width, height` its width and height.
        """
        return [*self.anchor.flatten(), self.width, self.height]


class Shape:
    """A base shape to define geometrical figure.
    
    Args:
        polygon (Polygon): The collection of vertices for drawing a 
            geometrical figure.
        bbox (BBox): The binding box for encapsulating a geometric
            figure.

    Attributes:
        polygon (Polygon): The collection of vertices for drawing a 
            geometrical figure.
        bbox (BBox): The binding box for encapsulating a geometric
            figure. 
    """

    def __init__(self, polygon: Polygon, bbox: BBox, color: tuple,
                 category: str):
        self.polygon = polygon
        self.bbox = bbox
        self.color = color
        self.category = category

    def __str__(self):
        return f'(BBox{str(self.bbox)}, Polygon{str(self.polygon)})'


class Rectangle(Shape):
    """A rectangle shape."""

    def __init__(self, bbox: BBox, color: tuple):
        polygon = Polygon([
            bbox.anchor,
            Point(bbox.anchor.x + bbox.width, bbox.anchor.y),
            Point(bbox.anchor.x + bbox.width, bbox.anchor.y + bbox.height),
            Point(bbox.anchor.x, bbox.anchor.y + bbox.height)
        ])
        super().__init__(polygon, bbox, color, 'rectangle')


class Triangle(Shape):
    """A triangle shape."""

    def __init__(self, bbox: BBox, color: tuple):
        shift = np.random.randint(1, bbox.width)
        polygon = Polygon([
            Point(bbox.anchor.x + shift, bbox.anchor.y),
            Point(bbox.anchor.x + bbox.width, bbox.anchor.y + bbox.height),
            Point(bbox.anchor.x, bbox.anchor.y + bbox.height)
        ])
        super().__init__(polygon, bbox, color, 'triangle')


def create_synthetic_image(image_width: int, image_height: int,
                           image_palette: Palette, image_background: Background,
                           image_capacity: int) -> tuple:
    """Returns a generated image with binding boxes and segments.

    Args:
        image_width (int): The image width in pixels.
        image_height (int): The image height in pixels.
        image_palette (Palette): The palette for generating images.
        image_background (Background): The palette for generating images.
        image_capacity (int): The number of geometrical shapes per image.

    Returns:
        image (PIL.Image): The generate image.
        shapes (list): The geometrical shapes the image consists of.
    """
    # Creates synthetic image od appropriate capacity
    shapes = []
    colors = list(range(25, 255, 25))
    for _ in range(image_capacity):
        # Picks a random shape.
        shape = np.random.choice([Rectangle, Triangle])

        # Picks a random shape color.
        if image_palette in [Palette.COLOR, Palette.GRAY]:
            color = tuple(np.random.choice(colors, size=3))
        else:
            if image_background == Background.WHITE:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)

        min_object_size = int(min(image_width, image_height) * 0.1)
        max_object_size = int(min(image_width, image_height) * 0.5)
        w, h = np.random.randint(min_object_size, max_object_size, size=2)

        # Generates the first coordinates of a binding box.
        x = np.random.randint(0, image_width - w)
        y = np.random.randint(0, image_height - h)

        # Creates a binding box.
        bbox = BBox(Point(x, y), w, h)

        # Collects the object shape (inside the binding box).
        shapes.append(shape(bbox, color))

    # Creates a synthetic image and projecting specified shapes.
    image = PIL.Image.new('RGB', (image_width, image_height), image_background)
    draw = PIL.ImageDraw.Draw(image)
    for shape in shapes:
        draw.polygon(shape.polygon.flatten(), fill=shape.color)

    return image, shapes