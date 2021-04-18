"""
A collection of geometrical figures.
"""

from .point import Point
from .bbox import BBox
from .polygon import Polygon


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

    def __init__(self, polygon: list, bbox: BBox):
        self._polygon = polygon
        self.bbox = bbox


class Rectangle(Shape):
    """A rectangle shape."""

    def __init__(self, bbox: BBox):
        polygon = Polygon([
            bbox.anchor,
            bbox.anchor.right(bbox.width),
            bbox.anchor.right(bbox.width).down(bbox.height),
            bbox.anchor.down(bbox.height)
        ])
        super().__init__(polygon, bbox)

    @property
    def polygon(self):
        return list(map(Point.as_tuple, self._polygon))

    def __str__(self):
        return ', '.join(self.polygon)


class Triangle(Shape):
    """A triangle shape."""

    def __init__(self, bbox: BBox):
        polygon = Polygon([
            bbox.anchor.right(bbox.width / 2),
            bbox.anchor.right(bbox.width).down(bbox.height),
            bbox.anchor.down(bbox.height),
        ])
        super().__init__(polygon, bbox)

    @property
    def polygon(self):
        return list(map(Point.as_tuple, self._polygon))

    def __str__(self):
        return ', '.join(self.polygon)