import unittest
import datetime

import numpy as np

import pygetm
from pygetm import parallel
from pygetm.simulation import Coupler
from pygetm.nesting import split_world
from pygetm.util.interpolate import interp_columns


class TestInterpColumns(unittest.TestCase):
    def test_identity(self):
        rng = np.random.default_rng(1)
        xp = np.cumsum(rng.random((5, 4, 10)), axis=-1)
        fp = rng.random(xp.shape)
        np.testing.assert_array_equal(interp_columns(xp, xp, fp, axis=-1), fp)

    def test_linear_profiles_are_reproduced(self):
        # Source and target columns with different depths and layer counts
        rng = np.random.default_rng(2)
        xp = -np.sort(rng.random((7, 12)) * 50.0, axis=0)[::-1]  # (nz_src, np)
        x = -np.sort(rng.random((5, 12)) * 40.0, axis=0)[::-1]  # (nz_tgt, np)
        a, b = rng.random(12), rng.random(12)
        fp = a * xp + b
        f = interp_columns(x, xp, fp, axis=0)
        inside = (x >= xp.min(axis=0)) & (x <= xp.max(axis=0))
        np.testing.assert_allclose(f[inside], (a * x + b)[inside], rtol=1e-12)

    def test_nearest_value_outside_range(self):
        xp = np.array([[-30.0, -20.0, -10.0]])
        fp = np.array([[3.0, 2.0, 1.0]])
        f = interp_columns(np.array([[-50.0, -15.0, 0.0]]), xp, fp, axis=-1)
        np.testing.assert_allclose(f, [[3.0, 1.5, 1.0]])

    def test_invalid_points_skipped(self):
        xp = np.array([[-30.0, -20.0, -10.0], [-3.0, -2.0, -1.0]])
        fp = np.array([[np.nan, 2.0, 1.0], [np.nan, np.nan, np.nan]])
        f = interp_columns(np.array([[-30.0, -15.0], [-2.0, -1.0]]), xp, fp, axis=-1)
        np.testing.assert_allclose(f[0], [2.0, 1.5])
        self.assertTrue(np.isnan(f[1]).all())

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            interp_columns(np.zeros((3, 4)), np.zeros((3, 5)), np.zeros((3, 5)))


class TestSplitWorld(unittest.TestCase):
    def test_split(self):
        world = parallel.MPI.COMM_WORLD
        if world.size == 1:
            ctx = split_world({"root": 1})
            self.assertEqual((ctx.name, ctx.comm.size), ("root", 1))
            return
        ranks = {"root": world.size - 1, "child": 1}
        ctx = split_world(ranks)
        expected = "root" if world.rank < world.size - 1 else "child"
        self.assertEqual(ctx.name, expected)
        self.assertEqual(ctx.comm.size, ranks[expected])
        self.assertEqual(ctx.nesting_comm.size, world.size)
        self.assertEqual(ctx.offsets, {"root": 0, "child": world.size - 1})
        names = ctx.nesting_comm.allgather(ctx.name)
        self.assertEqual(names, ["root"] * (world.size - 1) + ["child"])

    def test_wrong_total(self):
        world = parallel.MPI.COMM_WORLD
        with self.assertRaises(ValueError):
            split_world({"root": world.size + 1})


class RecordingCoupler(Coupler):
    def __init__(self):
        self.calls = []

    def on_start(self, sim):
        self.calls.append(("start", sim.istep))

    def before_state_update(self, sim, macro_active):
        self.calls.append(("before", sim.istep, macro_active))

    def after_state_update(self, sim, macro_active):
        self.calls.append(("after", sim.istep, macro_active))

    def on_finish(self, sim):
        self.calls.append(("finish", sim.istep))


class TestCoupler(unittest.TestCase):
    def test_hook_order(self):
        domain = pygetm.domain.create_cartesian(
            500.0 * np.arange(20),
            500.0 * np.arange(10),
            f=0,
            H=20,
            logger=parallel.get_logger(level="ERROR"),
        )
        airsea = pygetm.airsea.Fluxes()
        sim = pygetm.Simulation(
            domain, runtype=pygetm.RunType.BAROTROPIC_2D, airsea=airsea
        )
        sim.airsea.taux.set(0.0)
        sim.airsea.tauy.set(0.0)
        sim.airsea.sp.set(0.0)
        coupler = RecordingCoupler()
        sim.add_coupler(coupler)
        sim.start(datetime.datetime(2000, 1, 1), timestep=10.0, split_factor=2)
        for _ in range(4):
            sim.advance()
        sim.finish()
        expected = [("start", 0)]
        for istep in range(1, 5):
            macro = istep % 2 == 0
            expected += [("before", istep, macro), ("after", istep, macro)]
        expected.append(("finish", 4))
        self.assertEqual(coupler.calls, expected)


if __name__ == "__main__":
    unittest.main()
