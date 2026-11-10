"""pygetm.dams: a line of zero width that closes the faces it crosses. A closed
basin split by a dam, a river and a wind in the west half: nothing crosses the dam,
each half keeps its volume (the west half gains the river's), and with a gap in the
dam water does cross."""

import datetime
import unittest

import cftime
import numpy as np

import pygetm
import pygetm.dams

START = cftime.datetime(2000, 1, 1)
STOP = cftime.datetime(2000, 1, 1, 6)
FLOW = 100.0


def run(lines, nz=None):
    domain = pygetm.domain.create_cartesian(
        np.linspace(0.0, 40e3, 41), np.linspace(0.0, 20e3, 21), H=20.0, interfaces=True, f=1e-4,
        logger=pygetm.parallel.get_logger(level="ERROR"))
    closed = sum(pygetm.dams.close(domain, line) for line in lines)
    domain.rivers.add_by_location("river", 5e3, 10e3)
    kw = {} if nz is None else dict(vertical_coordinates=pygetm.vertical_coordinates.Sigma(nz))
    sim = pygetm.Simulation(domain, runtype=pygetm.RunType.BAROTROPIC_2D if nz is None else pygetm.RunType.BAROTROPIC_3D,
                            airsea=pygetm.airsea.Fluxes(taux=0.2, tauy=0.05), **kw)
    sim.rivers["river"].flow.set(FLOW)
    if nz is not None:
        dye = sim.tracers.add("dye")
        dye.fill(0.0)
        dye.all_values[:, :, : sim.T.halox + 20] = 1.0  # the west half
        sim.rivers["river"]["dye"].set(1.0)
    T = sim.T
    h = T.halox
    west = np.zeros(T.H.all_values.shape, bool)
    west[h:-h, h : h + 20] = True
    east = np.zeros_like(west)
    east[h:-h, h + 20 : -h] = True

    def volumes():
        v = (T.H.all_values + T.z.all_values) * T.area.all_values
        return v[west].sum(), v[east].sum()

    sim.start(START, 10.0, 10, report=datetime.timedelta(days=1))
    v0 = volumes()
    umax = 0.0
    while sim.time < STOP:
        sim.advance()
        umax = max(umax, float(np.abs(sim.momentum.U.all_values[h:-h, h + 19]).max()))  # the faces at x = 20 km
    dyeast = None if nz is None else float((dye.all_values * T.hn.all_values)[:, east].max())
    sim.finish()
    return closed, v0, volumes(), umax, dyeast, float(np.abs(T.z.all_values[h:-h, h:-h]).max())


class TestDams(unittest.TestCase):
    def test_full_dam_2d(self):
        closed, (w0, e0), (w1, e1), umax, _, zmax = run([[(20e3, -1e3), (20e3, 21e3)]])
        self.assertEqual(closed, 20)
        self.assertEqual(umax, 0.0)
        self.assertLess(abs(e1 / e0 - 1.0), 1e-14)
        self.assertLess(abs(w1 / (w0 + FLOW * 6 * 3600) - 1.0), 1e-14)
        self.assertGreater(zmax, 0.01)  # the wind did set up the water

    def test_full_dam_3d(self):
        closed, (w0, e0), (w1, e1), umax, dyeast, _ = run([[(20e3, -1e3), (20e3, 21e3)]], nz=5)
        self.assertEqual(umax, 0.0)
        self.assertLess(abs(e1 / e0 - 1.0), 1e-14)
        self.assertLess(abs(w1 / (w0 + FLOW * 6 * 3600) - 1.0), 1e-14)
        self.assertEqual(dyeast, 0.0)

    def test_gap(self):
        # a 4 km gap in the middle: water crosses
        closed, (w0, e0), (w1, e1), umax, _, _ = run([[(20e3, -1e3), (20e3, 8e3)], [(20e3, 12e3), (20e3, 21e3)]])
        self.assertEqual(closed, 16)
        self.assertGreater(umax, 0.0)
        self.assertGreater(abs(e1 / e0 - 1.0), 1e-6)
        self.assertLess(abs((w1 + e1) / (w0 + e0 + FLOW * 6 * 3600) - 1.0), 1e-14)

    def test_staircase(self):
        # a diagonal dam on a regular grid closes a staircase of U and V faces
        x, y = np.meshgrid(np.arange(21) / 2.0, np.arange(21) / 2.0)  # 10 x 10 cells of 1
        u, v, _ = pygetm.dams.faces(x, y, [(0.0, 0.1), (10.0, 10.1)])
        self.assertEqual((u.sum(), v.sum()), (9, 9))  # cell (j, i) below the line: j <= i; it touches the rest at 18 faces
        # watertight: no open face between a cell below the line and one above
        jj, ii = np.mgrid[:10, :10]
        below = jj <= ii
        self.assertFalse((below[:, :-1] != below[:, 1:])[~u[:, 1:-1]].any())
        self.assertFalse((below[:-1, :] != below[1:, :])[~v[1:-1, :]].any())


class _Domain:
    """The parts of a domain that check_dams reads (rank 0's global supergrid mask)."""

    class comm:
        rank = 0

    def __init__(self, shape):
        self._mask = np.ones((2 * shape[0] + 1, 2 * shape[1] + 1), int)
        self.mask = self._mask


class TestDamsInANest(unittest.TestCase):
    """pygetm.nesting.masks.check_dams: a dam on a level's interface is refused; in a
    child's band (cells set from its parent) it is reopened."""

    def child(self):
        from pygetm.nesting.geometry import load_levels

        cfg = {"root": dict(nx=20, ny=20, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
               "child": dict(parent="root", ratio=2, time_ratio=2, istart=5, istop=15, jstart=5, jstop=15)}
        levels = load_levels(cfg, halo=2, root_band=0, active=False)
        return next(l for l in levels if l.name == "child")

    def dam(self, child, i_face, rows):
        """Close the U faces west of child domain column ``i_face`` over ``rows``."""
        d = _Domain(child.domain_shape)
        d._mask[2 * rows[0] + 1 : 2 * rows[1] : 2, 2 * i_face] = 0
        return d

    def test_inside(self):
        from pygetm.nesting import masks

        child = self.child()
        h = child.band
        d = self.dam(child, h + 10, (h + 2, h + 18))  # well inside
        masks.check_dams(child, d)
        self.assertEqual(int((d._mask[1::2, ::2] == 0).sum()), 16)

    def test_on_interface_refused(self):
        from pygetm.nesting import masks

        child = self.child()
        h = child.band
        d = self.dam(child, h, (h + 2, h + 6))  # the child's west edge: band | window
        with self.assertRaisesRegex(ValueError, "interface"):
            masks.check_dams(child, d)

    def test_in_band_reopened(self):
        from pygetm.nesting import masks

        child = self.child()
        h = child.band
        d = self.dam(child, 1, (h + 2, h + 6))  # between two band cells
        masks.check_dams(child, d)
        self.assertEqual(int((d._mask[1::2, ::2] == 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
