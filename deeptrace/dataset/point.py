"""
A point in a two-dimensional space defined by `(x, y)` coordinates.
"""

from __future__ import annotations


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

    def __lt__(self, other):
        return self.x < other.x and self.y < other.y

    def __le__(self, other):
        return self.x <= other.x and self.y <= other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x and self.y != other.y

    def __gt__(self, other):
        return self.x > other.x and self.y > other.y

    def __ge__(self, other):
        return self.x >= other.x and self.y >= other.y

    def __sub__(self, other):
        return abs(self.x - other.x), abs(self.y - other.y)

    def right(self, delta: int) -> Point:
        """Moves the point right on the specified number of pixels. 
        
        Args:
            delta (int):  The number of pixels to move the point on.

        Returns:
            The new point `(x + delta, y)`
        """
        return Point(self.x + delta, self.y)

    def left(self, delta: int) -> Point:
        """Moves the point left on the specified number of pixels. 
        
        Args:
            delta (int):  The number of pixels to move the point on.

        Returns:
            The new point `(x - delta, y)`
        """
        return Point(self.x - delta, self.y)

    def up(self, delta: int) -> Point:
        """Moves the point up on the specified number of pixels. 
        
        Args:
            delta (int):  The number of pixels to move the point on.

        Returns:
            The new point `(x, y - delta)`
        """
        return Point(self.x, self.y - delta)

    def down(self, delta: int) -> Point:
        """Moves the point down on the specified number of pixels. 
        
        Args:
            delta (int):  The number of pixels to move the point on.

        Returns:
            The new point `(x, y + delta)`
        """
        return Point(self.x, self.y + delta)

    def as_tuple(self) -> tuple:
        """Returns a point as a tuples.

        Returns:
            The tuples, which represents point coordinates `(x, y)`.
        """
        return (self.x, self.y)
