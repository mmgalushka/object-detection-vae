"""
Test for deeptrace/data/bbox.py
"""

from deeptrace import Point
from deeptrace import BBox


def test_bbox_constructor():
    b = BBox(Point(1, 2), 3, 4)
    assert b.anchor == Point(1, 2)
    assert b.width == 3
    assert b.height == 4


def test_bbox_flatten():
    b = BBox(Point(1, 2), 3, 4)
    assert b.flatten() == [1, 2, 3, 4]


def test_bbox_str():
    b = BBox(Point(1, 2), 3, 4)
    assert str(b) == '[(1, 2), 3, 4]'
