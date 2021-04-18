"""
A binding box is rectangle around an object of interest on an image.
"""

from .point import Point


class BBox:
    """A binding box.

    Args:
        a (Point): The `a` corner point. 
        b (Point): The `b` corner point. 

    Attributes:
        anchor (Point): The top-left point of the binding box.
        width (int): The binding box width.
        height (int): The binding box height.
    """

    def __init__(self, a: Point, b: Point):
        self.anchor = min(a, b)
        self.width, self.height = a - b
