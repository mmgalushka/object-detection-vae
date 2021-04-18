"""
A polygon is a multi-section line around an object of interest on an image.
"""

from typing import List

from .point import Point


class Polygon(List[Point]):
    """A polygon"""

    def as_tuples(self) -> List[tuple]:
        """Returns a polygon as a list of tuples.

        Returns:
            The list of tuples where each tuple represents point coordinates
            `(x, y)`.
        """
        return list(map(Point.as_tuple, self))
