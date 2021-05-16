"""
Test for deeptrace/data/point.py
"""

from deeptrace import Point


def test_point_constructor():
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2


def test_point_comparison():
    p = Point(1, 2)
    assert p == Point(1, 2)
    assert p != Point(0, 2)
    assert p != Point(1, 0)
    assert p != Point(0, 0)


def test_point_movers():
    p = Point(1, 1)

    p1 = p.right(1)
    assert p1 == Point(2, 1)

    p2 = p1.down(1)
    assert p2 == Point(2, 2)

    p3 = p2.left(1)
    assert p3 == Point(1, 2)

    p4 = p3.up(1)
    assert p4 == Point(1, 1)


def test_point_flatten():
    p = Point(1, 2)
    assert p.flatten() == [1, 2]


def test_point_str():
    p = Point(1, 2)
    assert str(p) == '(1, 2)'