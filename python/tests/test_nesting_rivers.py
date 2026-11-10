"""Rivers in a nest: a river belongs to the level whose computed region holds it
(pygetm.nesting.runtime._river_in; the assignment itself, NestingRun._assign_rivers,
runs in the nested IDEAL setups)."""

import types
import unittest

import numpy as np

from pygetm.constants import CoordinateType
from pygetm.nesting.runtime import _river_in


def domain(nx=4, ny=3, d=1000.0):
    """A regular grid of d m cells on the supergrid (x east, y north)."""
    x = np.arange(2 * nx + 1) * d / 2
    y = np.arange(2 * ny + 1) * d / 2
    X, Y = np.meshgrid(x, y)
    return types.SimpleNamespace(_x=X, _y=Y, _lon=None, _lat=None)


def river(x, y, ct=CoordinateType.XY):
    return types.SimpleNamespace(x=x, y=y, coordinate_type=ct)


class TestRiverIn(unittest.TestCase):
    def setUp(self):
        self.dom = domain()
        self.computed = np.ones((3, 4), bool)
        self.computed[1, 2] = False  # a hole cell

    def test_inside_and_hole(self):
        self.assertTrue(_river_in(self.dom, river(500.0, 500.0), self.computed))
        self.assertFalse(_river_in(self.dom, river(2500.0, 1500.0), self.computed))  # in the hole

    def test_edge_between_a_hole_and_a_computed_cell(self):
        # on the edge between the hole cell and its computed neighbour: this level's
        self.assertTrue(_river_in(self.dom, river(2000.0, 1500.0), self.computed))

    def test_outside(self):
        self.assertFalse(_river_in(self.dom, river(9000.0, 500.0), self.computed))

    def test_by_index(self):
        self.assertTrue(_river_in(self.dom, river(0, 0, CoordinateType.IJ), self.computed))
        self.assertFalse(_river_in(self.dom, river(2, 1, CoordinateType.IJ), self.computed))
        self.assertFalse(_river_in(self.dom, river(9, 1, CoordinateType.IJ), self.computed))


if __name__ == "__main__":
    unittest.main()
