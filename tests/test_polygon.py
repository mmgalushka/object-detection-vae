"""
Test for deeptrace/data/polygon.py
"""

from deeptrace import Point
from deeptrace import Polygon


def test_polygon_constructor():
    v1 = Point(1, 2)
    v2 = Point(3, 4)
    p = Polygon([v1, v2])
    assert len(p) == 2
    assert p[0] == Point(1, 2)
    assert p[1] == Point(3, 4)


def test_polygon_flatten():
    v1 = Point(1, 2)
    v2 = Point(3, 4)
    p = Polygon([v1, v2])
    assert p.flatten() == [1, 2, 3, 4]


def test_polygon_str():
    v1 = Point(1, 2)
    v2 = Point(3, 4)
    p = Polygon([v1, v2])
    assert str(p) == '[(1, 2), (3, 4)]'
