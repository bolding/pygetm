"""A curvilinear child level (``grid``): its coordinates, and the start-up check
that it coincides with the refined parent grid where water crosses the interface
(pygetm.nesting.masks.check_conforming)."""

import os
import tempfile
import unittest

import numpy as np

from pygetm.nesting.geometry import load_levels
from pygetm.nesting import masks

R, BAND = 2, 2
WINDOW = dict(istart=8, istop=10, jstart=5, jstop=15)  # 2 x 10 parent cells: a strait north from j = 5


def layout(grid):
    return {
        "root": dict(nx=20, ny=20, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
        "child": dict(parent="root", ratio=R, time_ratio=R, grid=grid, **WINDOW),
    }


def child_grid(path, bend_from):
    """The refined parent grid, bent sideways from child row ``bend_from`` on."""
    nx, ny = (WINDOW["istop"] - WINDOW["istart"]) * R, (WINDOW["jstop"] - WINDOW["jstart"]) * R
    ii = np.arange(2 * (nx + 2 * BAND) + 1) / 2.0 - BAND
    jj = np.arange(2 * (ny + 2 * BAND) + 1) / 2.0 - BAND
    I, J = np.meshgrid(ii, jj)
    x = WINDOW["istart"] + I / R
    y = WINDOW["jstart"] + J / R
    x = x + 0.02 * np.clip(J - bend_from, 0.0, None) ** 2 / R  # a bend (staying over the root's land)
    np.savez(path, x=x, y=y)


def wet_masks(levels):
    """The root is land around the child's window except south of it; the child is water."""
    wets = {}
    for lev in levels:
        w = np.ones(lev.extended_shape, dtype=bool)
        if lev.parent is None:
            w[5:, 6:12] = False  # the coast around the strait
        else:
            h = lev.band
            w[:h, :] = True
            w[:, :h] = w[:, -h:] = False
            w[-h:, :] = False
        wets[lev.name] = w
    return wets


def check(bend_from):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "child.npz")
        child_grid(path, bend_from)
        cfg = layout(path)
        levels = load_levels(cfg, halo=BAND, root_band=0, active=False)
        wets = wet_masks(levels)
        H = {k: np.where(v, 10.0, np.nan) for k, v in wets.items()}
        levels = load_levels(cfg, halo=BAND, root_band=0, wet=wets)
        w, _, _ = masks.make_consistent(levels, wets, H)
        child = next(l for l in levels if l.name == "child")
        xs, ys = child.supergrid()
        masks.check_conforming(levels, w)
        return child, xs


class TestCurvilinearLevel(unittest.TestCase):
    def test_conforming_band(self):
        # rim 2 r + ring/halo 2 = 6 child rows must conform; bending from row 8 is fine
        child, xs = check(bend_from=8)
        self.assertEqual(xs.shape, (2 * (20 + 2 * BAND) + 1, 2 * (4 + 2 * BAND) + 1))
        xc, yc = child.centres()
        self.assertEqual(xc.shape, (child.ny, child.nx))
        self.assertGreater(np.ptp(xc[-1]), 0.0)
        with self.assertRaises(ValueError):
            child.edges()

    def test_bend_inside_band_refused(self):
        with self.assertRaisesRegex(ValueError, "refined parent grid"):
            check(bend_from=3)

    def test_regular_levels_unchanged(self):
        levels = load_levels(layout(None), halo=BAND, root_band=0, active=False)
        child = next(l for l in levels if l.name == "child")
        x, y = child.supergrid()
        xe, ye = child.edges()
        np.testing.assert_array_equal(x[0, ::2], xe)
        np.testing.assert_array_equal(y[::2, 0], ye)


if __name__ == "__main__":
    unittest.main()
