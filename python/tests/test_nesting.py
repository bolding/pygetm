import unittest
import datetime

import numpy as np

import pygetm
from pygetm import parallel
from pygetm.simulation import Coupler, COUPLING_STAGES
from pygetm.nesting import split_world, CellSet, outer_halo, hole_rim
from pygetm.nesting import variables
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

    def on_stage(self, sim, stage, macro_active):
        self.calls.append((stage, sim.istep, macro_active))

    def on_finish(self, sim):
        self.calls.append(("finish", sim.istep))


def barotropic_simulation(nx=20, ny=10):
    domain = pygetm.domain.create_cartesian(
        500.0 * np.arange(nx),
        500.0 * np.arange(ny),
        f=0,
        H=20,
        logger=parallel.get_logger(level="ERROR"),
    )
    sim = pygetm.Simulation(
        domain, runtype=pygetm.RunType.BAROTROPIC_2D, airsea=pygetm.airsea.Fluxes()
    )
    sim.airsea.taux.set(0.0)
    sim.airsea.tauy.set(0.0)
    sim.airsea.sp.set(0.0)
    return sim


def allsum(value):
    return parallel.MPI.COMM_WORLD.allreduce(value)


class TestCellSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nx, cls.ny = 20, 10
        cls.sim = barotropic_simulation(cls.nx, cls.ny)

    def test_outer_halo_t(self):
        T = self.sim.T
        ring = outer_halo(T)
        h = T.halox
        self.assertEqual(T.haloy, h)
        expected = (self.nx + 2 * h) * (self.ny + 2 * h) - self.nx * self.ny
        # Neighbouring subdomains may both hold a ring cell (in their halos),
        # so count unique cells over all ranks
        jg, ig = ring.global_indices()
        cells = set(zip(jg.tolist(), ig.tolist()))
        allcells = set().union(*parallel.MPI.COMM_WORLD.allgather(cells))
        self.assertEqual(len(allcells), expected)
        for j, i in cells:
            self.assertTrue(j < 0 or j >= self.ny or i < 0 or i >= self.nx)
            self.assertTrue(-h <= j < self.ny + h and -h <= i < self.nx + h)

    def test_outer_halo_x_grid_has_overlap(self):
        # X grid (corners) has one more point in each direction
        X = self.sim.X
        h = X.halox
        jg, ig = outer_halo(X).global_indices()
        cells = set(zip(jg.tolist(), ig.tolist()))
        allcells = set().union(*parallel.MPI.COMM_WORLD.allgather(cells))
        nx, ny = self.nx + 1, self.ny + 1
        self.assertEqual(len(allcells), (nx + 2 * h) * (ny + 2 * h) - nx * ny)

    def test_gather_scatter_2d(self):
        T = self.sim.T
        a = T.array(fill=0.0)
        rng = np.random.default_rng(parallel.MPI.COMM_WORLD.rank)
        a.all_values[...] = rng.random(a.all_values.shape)
        cells = outer_halo(T)
        values = cells.gather(a)
        self.assertEqual(values.shape, (cells.size,))
        self.assertEqual(cells.count(a), cells.size)
        b = T.array(fill=-1.0)
        cells.scatter(b, values)
        np.testing.assert_array_equal(b.all_values[cells.j, cells.i], values)
        untouched = np.ones(b.all_values.shape, dtype=bool)
        untouched[cells.j, cells.i] = False
        self.assertTrue((b.all_values[untouched] == -1.0).all())

    def test_gather_scatter_3d(self):
        # 3D layout: all cells of level 0 first, then level 1, ...
        T = self.sim.T
        nz = 3
        data = np.arange(nz * T.ny_ * T.nx_, dtype=float).reshape(nz, T.ny_, T.nx_)
        cells = CellSet(T, [0, 1, 2], [3, 4, 5])
        a = T.array(fill=0.0)
        a3 = _FakeArray(T, data)
        values = cells.gather(a3)
        self.assertEqual(cells.count(a3), nz * 3)
        np.testing.assert_array_equal(values[:3], data[0, [0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(values[3:6], data[1, [0, 1, 2], [3, 4, 5]])
        b3 = _FakeArray(T, np.zeros_like(data))
        cells.scatter(b3, values)
        np.testing.assert_array_equal(
            b3.all_values[:, [0, 1, 2], [3, 4, 5]], data[:, [0, 1, 2], [3, 4, 5]]
        )
        # pack a 2D and a 3D array into one buffer, then unpack
        a.all_values[...] = 7.0
        buf = cells.pack([a, a3])
        self.assertEqual(buf.size, 3 + nz * 3)
        a.all_values[...] = 0.0
        b3.all_values[...] = 0.0
        cells.unpack([a, b3], buf)
        np.testing.assert_array_equal(a.all_values[[0, 1, 2], [3, 4, 5]], 7.0)
        np.testing.assert_array_equal(
            b3.all_values[:, [0, 1, 2], [3, 4, 5]], data[:, [0, 1, 2], [3, 4, 5]]
        )

    def test_wrong_grid(self):
        cells = CellSet(self.sim.T, [0], [0])
        with self.assertRaises(ValueError):
            cells.gather(self.sim.U.array(fill=0.0))

    def test_hole_rim(self):
        T = self.sim.T
        hole = np.zeros((self.ny, self.nx), dtype=bool)
        hole[2:8, 5:15] = True
        width = 2
        rim = hole_rim(T, hole, width)
        jg, ig = rim.global_indices()
        self.assertTrue(hole[jg, ig].all())
        # Each rim cell is within `width` of a cell outside the hole
        for j, i in zip(jg, ig):
            window = hole[
                max(j - width, 0) : j + width + 1, max(i - width, 0) : i + width + 1
            ]
            self.assertFalse(window.all())
        cells = set(zip(jg.tolist(), ig.tolist()))
        allcells = set().union(*parallel.MPI.COMM_WORLD.allgather(cells))
        # hole 6 x 10 with a rim of depth 2 leaves an inner 2 x 6 block
        self.assertEqual(len(allcells), 6 * 10 - 2 * 6)


class _FakeArray:
    # Minimal stand-in for a 3D core.Array (2D runs have no vertical dimension)
    def __init__(self, grid, values):
        self.grid = grid
        self.all_values = values
        self.name = "fake"


class TestVariables(unittest.TestCase):
    def test_stages_are_known(self):
        for group in variables.EXCHANGE_GROUPS:
            self.assertIn(group.stage, COUPLING_STAGES)
            self.assertIn(group.grid, ("T", "U", "V"))

    def test_resolve_barotropic(self):
        sim = barotropic_simulation()
        found = {
            g.name: [a.name for a in variables.resolve(sim, g)]
            for g in variables.EXCHANGE_GROUPS
        }
        self.assertEqual(found["U"], ["U"])
        self.assertEqual(found["V"], ["V"])
        self.assertEqual(found["z"], ["zt"])
        self.assertEqual(found["ts"], [])
        for g in variables.EXCHANGE_GROUPS:
            for a in variables.resolve(sim, g):
                self.assertIs(a.grid, getattr(sim, g.grid))


class TestAdvectionFlux(unittest.TestCase):
    def test_face_fluxes_close_cell_budgets(self):
        # The amounts returned in flux_u/flux_v must explain every cell's change
        # in tracer content, for every scheme (needed for flux matching)
        domain = pygetm.domain.create_cartesian(
            np.arange(31) * 1e3,
            np.arange(21) * 1e3,
            interfaces=True,
            H=1.0,
            f=0.0,
            logger=parallel.get_logger(level="ERROR"),
        )
        T = domain.create_grids(1, halox=2, haloy=2, velocity_grids=1)
        if T is None:
            return
        rng = np.random.default_rng(1)
        h = T.halox
        for scheme in pygetm.AdvectionScheme:
            if scheme.name == "DEFAULT":
                continue
            with self.subTest(scheme=scheme.name):
                f = T.array(fill=0.0)
                f.all_values[...] = rng.random(f.all_values.shape)
                f.update_halos()
                u = T.ugrid.array(fill=300.0)
                v = T.vgrid.array(fill=200.0)
                u.all_values[T.ugrid.mask.all_values == 0] = 0.0
                v.all_values[T.vgrid.mask.all_values == 0] = 0.0
                fu = T.ugrid.array(fill=0.0)
                fv = T.vgrid.array(fill=0.0)
                adv = pygetm.operators.Advection(T, scheme=scheme)
                c0 = f.all_values * T.D.all_values * T.area.all_values
                adv(u, v, 1.0, f, flux_u=fu, flux_v=fv)
                c1 = f.all_values * adv.D * T.area.all_values
                net = (
                    fu.all_values[h:-h, h - 1 : -h - 1]
                    - fu.all_values[h:-h, h:-h]
                    + fv.all_values[h - 1 : -h - 1, h:-h]
                    - fv.all_values[h:-h, h:-h]
                )
                water = T.mask.all_values[h:-h, h:-h] == 1
                err = np.abs((c1 - c0)[h:-h, h:-h] - net)[water].max()
                self.assertLess(err, 1e-12 * np.abs(c0).max())
                self.assertGreater(np.abs(fu.all_values).max(), 0.0)


class TestCoupler(unittest.TestCase):
    def test_hook_order(self):
        sim = barotropic_simulation()
        coupler = RecordingCoupler()
        sim.add_coupler(coupler)
        sim.start(datetime.datetime(2000, 1, 1), timestep=10.0, split_factor=2)
        for _ in range(4):
            sim.advance()
        sim.finish()
        expected = [("start", 0)]
        for istep in range(1, 5):
            macro = istep % 2 == 0
            expected += [
                ("before", istep, macro),
                ("transports_2d", istep, macro),
                ("elevation", istep, macro),
                ("after", istep, macro),
            ]
        expected.append(("finish", 4))
        self.assertEqual(coupler.calls, expected)


if __name__ == "__main__":
    unittest.main()
