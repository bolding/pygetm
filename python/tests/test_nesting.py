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
        # kernel test: serial on every process (no decomposition involved)
        domain = pygetm.domain.create_cartesian(
            np.arange(61) * 1e3,
            np.arange(41) * 1e3,
            interfaces=True,
            H=1.0,
            f=0.0,
            comm=parallel.MPI.COMM_SELF,
            logger=parallel.get_logger(level="ERROR", comm=parallel.MPI.COMM_SELF),
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


class TestExternalCells(unittest.TestCase):
    def test_external_cells(self):
        # Cells marked external are wet with valid depth and open faces, are
        # not updated by advection, and are excluded from ACTIVE-only sums
        nx, ny = 40, 24
        domain = pygetm.domain.create_cartesian(
            np.arange(nx + 1) * 1e3,
            np.arange(ny + 1) * 1e3,
            interfaces=True,
            H=10.0,
            f=0.0,
            logger=parallel.get_logger(level="ERROR"),
        )
        external = np.zeros((ny, nx), dtype=bool)
        external[:, :2] = True  # a band on the west side
        domain.external = external
        T = domain.create_grids(1, halox=2, haloy=2, velocity_grids=1)
        if T is None:
            return
        from pygetm.constants import CellType

        jg, ig = np.meshgrid(
            np.arange(T.ny_) - T.haloy + T.tiling.yoffset,
            np.arange(T.nx_) - T.halox + T.tiling.xoffset,
            indexing="ij",
        )
        inside = (jg >= 0) & (jg < ny) & (ig >= 0) & (ig < nx)
        own = np.zeros_like(inside)  # this subdomain's cells (no halos)
        own[T.haloy : -T.haloy, T.halox : -T.halox] = True
        inside &= own
        ext = inside & (ig < 2)
        self.assertTrue((T.mask.all_values[ext] == CellType.EXTERNAL).all())
        self.assertTrue((T.mask.all_values[inside & ~ext] == CellType.ACTIVE).all())
        # depth valid (not fill) in external cells
        self.assertTrue(np.allclose(T.H.all_values[ext], 10.0))
        self.assertTrue(np.allclose(T.D.all_values[ext], 10.0))
        # faces between external and active cells are open and computed
        uopen = T.ugrid.mask.all_values[inside & (ig == 1)]
        self.assertTrue((uopen == CellType.ACTIVE).all())
        # faces with external cells on both sides are external
        uext = T.ugrid.mask.all_values[inside & (ig == 0)]
        self.assertTrue((uext == CellType.EXTERNAL).all())
        self.assertTrue(np.allclose(T.ugrid.D.all_values[inside & (ig == 0)], 10.0))
        # advection: external cells keep their values, active ones change
        f = T.array(fill=0.0)
        f.all_values[ext] = 1.0
        f.update_halos()
        u = T.ugrid.array(fill=0.0)
        u.all_values[T.ugrid.mask.all_values != 0] = 300.0
        v = T.vgrid.array(fill=0.0)
        before = f.all_values.copy()
        pygetm.operators.Advection(T, scheme=pygetm.AdvectionScheme.UPSTREAM)(u, v, 1.0, f)
        np.testing.assert_array_equal(f.all_values[ext], before[ext])
        first_active = inside & (ig == 2)
        if first_active.any():
            self.assertTrue((f.all_values[first_active] > 0.0).all())


def small_hierarchy(halo=2):
    from pygetm.nesting.geometry import load_levels

    return load_levels(
        {
            "root": dict(nx=40, ny=30, lon0=0.0, lat0=50.0, dlon=0.5, dlat=0.5),
            "child": dict(parent="root", istart=10, istop=26, jstart=8, jstop=20, ratio=3),
            # touches the child's south edge, as belt_sea in skagerrak
            "gc": dict(parent="child", istart=20, istop=40, jstart=0, jstop=15, ratio=3),
        },
        halo=halo,
    )


def tiling_info(nrow, ncol, ny_ext, nx_ext, xoff=0, yoff=0, removed=(), halo=2):
    # pyGETM-like tiling description: equal subdomains, optional protrusion
    # (negative offsets) and removed tiles (map -1, ranks renumbered)
    nx_sub = -(-(nx_ext - xoff) // ncol)
    ny_sub = -(-(ny_ext - yoff) // nrow)
    rank_map = -np.ones((nrow, ncol), dtype=int)
    rank = 0
    for r in range(nrow):
        for c in range(ncol):
            if (r, c) not in removed:
                rank_map[r, c] = rank
                rank += 1
    return dict(map=rank_map.tolist(), nx_sub=nx_sub, ny_sub=ny_sub,
                xoffset_global=xoff, yoffset_global=yoff, halo=halo)


class TestNestingGeometry(unittest.TestCase):
    def test_levels(self):
        levels = small_hierarchy()
        root, child, gc = levels
        self.assertEqual((child.nx, child.ny), (48, 36))
        self.assertEqual(gc.steps_per_root, 9)
        self.assertEqual(root.external().shape, root.extended_shape)
        # band and hole are external, the rest is computed
        self.assertTrue(root.external()[:2].all())
        self.assertFalse(root.external()[2 + 8 - 3, 2 + 10 - 3])
        self.assertTrue(root.external()[2 + 8, 2 + 10])
        # rim of the child in the root lies in the hole, next to computed cells
        rim = root.rims["child"]
        self.assertTrue((~root.computed[rim[:, 0], rim[:, 1]]).all())
        # child touching the parent's edge: rim reaches HALO * ratio from the edge
        self.assertEqual(child.rims["gc"][:, 0].min(), 0)
        self.assertGreaterEqual(child.rims["gc"][:, 0].max(), 2 * 3 - 1)
        bj, bi = child.ring_cells()
        self.assertEqual(bj.size, child.extended_shape[0] * child.extended_shape[1] - child.nx * child.ny)

    def test_root_without_band(self):
        from pygetm.nesting.geometry import load_levels

        levels = load_levels(
            {
                "root": dict(nx=40, ny=30, lon0=0.0, lat0=50.0, dlon=0.5, dlat=0.5),
                "child": dict(parent="root", istart=10, istop=26, jstart=8, jstop=20, ratio=3),
            },
            root_band=0,
        )
        root, child = levels
        self.assertEqual(root.extended_shape, (30, 40))
        self.assertEqual(child.band, 2)
        ext = root.external()
        self.assertTrue(ext[8:20, 10:26].all())
        self.assertEqual(int(ext.sum()), 12 * 16)


class TestNestingOperators(unittest.TestCase):
    def test_prolong_restrict(self):
        from pygetm.nesting.operators import prolong, restrict, limited_slopes

        rng = np.random.default_rng(0)
        r = 3
        c, sx, sy = rng.random(20), rng.normal(size=20), rng.normal(size=20)
        a = np.arange(r)[None, :, None]
        b = np.arange(r)[None, None, :]
        blk = prolong(c[:, None, None], sx[:, None, None], sy[:, None, None], a, b, r)
        np.testing.assert_allclose(blk.mean(axis=(1, 2)), c, atol=1e-14)
        np.testing.assert_allclose(restrict(blk, np.ones_like(blk)), c, atol=1e-14)
        w = np.ones_like(blk)
        w[:, 0, 0] = 0.0
        blk[:, 0, 0] = np.nan  # dry cell: ignored
        self.assertTrue(np.isfinite(restrict(blk, w)).all())
        wet = np.ones((3, 6), bool)
        wet[1, 3] = False
        sx2, _ = limited_slopes(np.tile(np.arange(6.0), (3, 1)), wet)
        self.assertEqual(sx2[1, 2], 0.0)
        self.assertEqual(sx2[0, 2], 1.0)


class TestNestingMasks(unittest.TestCase):
    def test_child_led_consistency(self):
        from pygetm.nesting.masks import make_consistent, check_consistent

        levels = small_hierarchy()
        rng = np.random.default_rng(1)
        wet = {l.name: rng.random(l.extended_shape) > 0.3 for l in levels}
        H = {l.name: 5.0 + 10.0 * rng.random(l.extended_shape) for l in levels}
        self.assertTrue(any(v for d in check_consistent(levels, wet).values() for v in d.values()))
        wet2, H2, _ = make_consistent(levels, wet, H)
        self.assertTrue(all(v == 0 for d in check_consistent(levels, wet2).values() for v in d.values()))
        wet3, _, changes = make_consistent(levels, wet2, H2)
        self.assertTrue(all(v == 0 for d in changes.values() for v in d.values()))
        for l in levels:
            np.testing.assert_array_equal(wet3[l.name], wet2[l.name])


class TestNestingPlan(unittest.TestCase):
    def _check(self, levels, layouts):
        from pygetm.nesting.plan import LevelTiling, Plan

        tilings, off = {}, 0
        for lev in levels:
            info = tiling_info(*layouts[lev.name][:2], *lev.extended_shape, **layouts[lev.name][2])
            tilings[lev.name] = LevelTiling(info, lev.halo, off)
            off += tilings[lev.name].ntiles
        sends, recvs = {}, {}
        for k, lev in enumerate(levels[1:]):
            plan = Plan(lev, tilings[lev.parent.name], tilings[lev.name], k)
            chk = plan.check()
            self.assertEqual(chk["ring_duplicates"], 0)
            self.assertEqual(chk["block_duplicates"], 0)
            self.assertEqual(chk["face_duplicates"], 0)
            for role, til in (("parent", plan.ptil), ("child", plan.ctil)):
                for t in range(til.ntiles):
                    me = til.offset + t
                    for line in plan.describe(me, role, t):
                        parts = line.split()
                        tag = parts[-2]
                        n = int(parts[-1].split("=")[1])
                        if "→" in line:
                            key = (me, int(parts[parts.index("→") + 2]), tag, n)
                            sends[key] = sends.get(key, 0) + 1
                        else:
                            key = (int(parts[parts.index("from") + 2]), me, tag, n)
                            recvs[key] = recvs.get(key, 0) + 1
            # every ring cell whose owners exist is covered exactly once
            bj, bi = lev.ring_cells()
            pj, pi, _, _ = lev.ring_source(bj, bi)
            owned = (plan.ctil.owner(bj, bi) >= 0) & (plan.ptil.owner(pj, pi) >= 0)
            self.assertEqual(sum(v[2].size for v in plan.pro.values()), int(owned.sum()))
        self.assertEqual(sends, recvs)
        self.assertTrue(all(v == 1 for v in sends.values()))

    def test_plans_simple(self):
        levels = small_hierarchy()
        self._check(levels, {"root": (1, 1, {}), "child": (1, 1, {}), "gc": (1, 1, {})})

    def test_plans_unequal_protruding_removed(self):
        levels = small_hierarchy()
        self._check(levels, {
            "root": (2, 3, dict(xoff=-4)),
            "child": (3, 2, dict(yoff=-5, removed=[(0, 1)])),
            "gc": (2, 4, dict(xoff=-7, removed=[(1, 3), (0, 0)])),
        })


class TestNestingRuntime(unittest.TestCase):
    """Two-level run with NestingRun/NestedTracer: mass conserved to round-off
    (all levels + outflow through the root boundary). Needs >= 2 processes."""

    def test_two_levels_conserve_mass(self):
        from pygetm.nesting import load_levels
        from pygetm.nesting.runtime import NestingRun, NestedTracer

        world = parallel.MPI.COMM_WORLD
        if world.size < 2:
            self.skipTest("needs at least 2 MPI processes")
        levels = load_levels(
            {
                "root": dict(nx=40, ny=30, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
                "child": dict(parent="root", istart=10, istop=22, jstart=8, jstop=18, ratio=3),
            }
        )
        nroot = (world.size + 1) // 2
        nest = NestingRun(levels, {"root": nroot, "child": world.size - nroot})
        lev = nest.level

        def make_domain(lev, comm):
            lon, lat = lev.edges()
            return pygetm.domain.create_cartesian(
                lon * 1e4, lat * 1e4, interfaces=True, H=1.0, f=0.0, comm=comm,
                logger=parallel.get_logger(level="ERROR", comm=comm),
            )

        T = nest.build(make_domain)
        content = []
        outflow = 0.0
        if T is not None:
            gj, gi = nest.global_indices()
            f = T.array(fill=0.0)
            x, y = gi + 0.5, gj + 0.5
            lon = lev.lon0 + x * lev.dlon
            lat = lev.lat0 + y * lev.dlat
            f.all_values[...] = np.exp(-((lon[None, :] - 12.0) ** 2 + (lat[:, None] - 10.0) ** 2) / 8.0)
            if lev.parent is None:
                band = (gj[:, None] < 0) | (gj[:, None] >= lev.ny) | (gi[None, :] < 0) | (gi[None, :] >= lev.nx)
                f.all_values[band] = 0.0
            f.update_halos()
            dt = 1.0 / lev.steps_per_root
            dx = lev.dlon * 1e4
            u = T.ugrid.array(fill=0.4 * dx / dt)
            v = T.vgrid.array(fill=0.2 * dx / dt)
            u.all_values[T.ugrid.mask.all_values == 0] = 0.0
            v.all_values[T.vgrid.mask.all_values == 0] = 0.0
            advect = pygetm.operators.Advection(T, scheme=pygetm.AdvectionScheme.SUPERBEE)
            tracer = NestedTracer(nest, f)
            tracer.start()
            content.append(tracer.level_content())
            for _ in range(20 * lev.steps_per_root):
                tracer.step(lambda fu, fv: advect(u, v, dt, f, flux_u=fu, flux_v=fv))
            content.append(tracer.level_content())
            outflow = tracer.outflow
        total = nest.world.allreduce(np.array(content) if content else np.zeros(2))
        total[1] += nest.world.allreduce(outflow)
        self.assertGreater(total[0], 0.0)
        self.assertLess(abs(total[1] - total[0]) / total[0], 1e-13)


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
