import unittest
import datetime

import numpy as np

import pygetm
from pygetm import parallel
from pygetm.simulation import Coupler, COUPLING_STAGES
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

    def test_mpmd_blocks(self):
        from pygetm.nesting import mpmd_blocks

        e = [("root", 0), ("root", 0), ("child", 1), ("gc", 2), ("gc", 2)]
        self.assertEqual(mpmd_blocks(e), {"root": (0, 2), "child": (2, 1), "gc": (3, 2)})
        self.assertEqual(mpmd_blocks([("root", -1)] * 3), {"root": (0, 3)})
        with self.assertRaises(ValueError):  # two domains in one component
            mpmd_blocks([("root", 0), ("child", 0), ("gc", 1)])
        # one application (srun --multi-prog): the names define the components
        e = [("root", 0), ("root", 0), ("child", 0), ("gc", 0)]
        self.assertEqual(mpmd_blocks(e), {"root": (0, 2), "child": (2, 1), "gc": (3, 1)})
        self.assertEqual(mpmd_blocks([("root", -1), ("child", -1)]), {"root": (0, 1), "child": (1, 1)})
        with self.assertRaises(ValueError):  # not contiguous
            mpmd_blocks([("root", 0), ("child", 0), ("root", 0)])
        with self.assertRaises(ValueError):  # one domain in two components
            mpmd_blocks([("root", 0), ("child", 1), ("root", 2)])

    def test_split_mpmd(self):
        # a plain (non-MPMD) launch: every process names the same domain
        from pygetm.nesting import split_mpmd

        world = parallel.MPI.COMM_WORLD
        ctx = split_mpmd("root")
        self.assertEqual((ctx.name, ctx.comm.size, ctx.sizes), ("root", world.size, {"root": world.size}))


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


def active_hierarchy(halo=2):
    """small_hierarchy with an L-shaped active region for the child (its
    north-east quarter stays with the root) and a grandchild on the child's
    south edge (registers passed upwards)."""
    from pygetm.nesting.geometry import load_levels

    act = np.ones((12, 16), dtype=bool)
    act[6:, 8:] = False
    return load_levels(
        {
            "root": dict(nx=40, ny=30, lon0=0.0, lat0=50.0, dlon=0.5, dlat=0.5),
            "child": dict(parent="root", istart=10, istop=26, jstart=8, jstop=20, ratio=3, active=act.tolist()),
            "gc": dict(parent="child", istart=3, istop=15, jstart=0, jstop=9, ratio=3),
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

    def test_tiling_mask(self):
        # only hole cells nothing reads (the hole minus the rim) need no subdomain
        root, child, gc = small_hierarchy()
        tm = root.tiling_mask()
        self.assertEqual(tm.shape, root.extended_shape)
        self.assertTrue(tm[:2].all())
        free = ~tm[2:-2, 2:-2]
        self.assertTrue((root.hole[free]).all())
        self.assertEqual(int(free.sum()), (12 - 2 * 2) * (16 - 2 * 2))
        rim = root.rims["child"]
        self.assertTrue(tm[2 + rim[:, 0], 2 + rim[:, 1]].all())
        # a grandchild on the child's edge: the faces passed upwards keep their hole cells
        tm = child.tiling_mask()
        wf = child.wfaces["gc"]
        hj, hi = wf["hole"]
        up = ~wf["inside"]
        self.assertTrue(up.any())
        self.assertTrue(tm[2 + hj[up], 2 + hi[up]].all())
        self.assertTrue(child.hole[~tm[2:-2, 2:-2]].all())

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

    def test_active_region(self):
        root, child, gc = active_hierarchy()
        # hole = the child's active region; the root computes the rest of the window
        self.assertTrue(root.hole[8:14, 10:26].all())
        self.assertFalse(root.hole[14:20, 18:26].any())
        self.assertTrue(root.computed[14:20, 18:26].all())
        np.testing.assert_array_equal(root.external(), ~np.pad(root.computed, 2, constant_values=False))
        # ring: non-active cells within the band of the active region, also inside the window
        b = child.band
        bj, bi = child.ring_cells()
        self.assertTrue(((bj >= 0) & (bj < child.ny) & (bi >= 0) & (bi < child.nx)).any())
        self.assertFalse(child.cactive[np.clip(bj, 0, child.ny - 1), np.clip(bi, 0, child.nx - 1)][
            (bj >= 0) & (bj < child.ny) & (bi >= 0) & (bi < child.nx)].any())
        self.assertTrue(child.unused()[b + child.ny - 1, b + child.nx - 1])
        # interface faces of the child in the root: the perimeter of the L
        wf = root.wfaces["child"]
        self.assertEqual(wf["is_x"].size, 2 * 6 + 2 * 16 + 2 * 12 - 2 * 6)
        self.assertTrue(wf["inside"].all())
        self.assertEqual(child.edge["is_x"].size, wf["is_x"].size * 3)
        # the child's edge faces lie between an active and a non-active cell
        cj, ci = child.edge["inner"]
        self.assertTrue(child.cactive[cj, ci].all())
        # grandchild on the child's south edge: those faces are passed upwards
        g = child.wfaces["gc"]
        up = g["up"][~g["inside"]]
        self.assertTrue(up.size > 0 and (up >= 0).all())
        e = child.edge
        for m in np.nonzero(~g["inside"])[0]:
            k = g["up"][m]
            self.assertEqual((e["face"][0][k], e["face"][1][k]), (g["face"][0][m], g["face"][1][m]))
            self.assertEqual((e["inner"][0][k], e["inner"][1][k]), (g["hole"][0][m], g["hole"][1][m]))
        # full window: unchanged behaviour
        levels = small_hierarchy()
        bj, bi = levels[1].ring_cells()
        self.assertEqual(bj.size, levels[1].extended_shape[0] * levels[1].extended_shape[1] - levels[1].nx * levels[1].ny)

    def test_active_region_from_seed(self):
        from pygetm.nesting.geometry import Level, active_region

        root = Level("root", dict(nx=20, ny=10, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0))
        child = Level("child", dict(istart=2, istop=18, jstart=1, jstop=9, ratio=2), root)
        wet = np.ones((child.ny, child.nx), dtype=bool)
        wet[:, 15] = False  # land wall at lon 9.5..10 (child column 15)
        act = active_region(child, dict(seed=[3.0, 4.0], barriers=[{"lat_gt": 7.0, "lon_lt": 6.0}]), wet)
        self.assertEqual(act.shape, (8, 16))
        self.assertFalse(act[:, 8:].any())  # behind the land wall
        self.assertFalse(act[6:, :4].any())  # behind the barrier
        self.assertTrue(act[:6, :7].all())
        with self.assertRaises(ValueError):
            active_region(child, dict(seed=[15.0, 4.0], barriers=[]), wet & False)
        # a list of seeds: the water behind the wall becomes active too
        act2 = active_region(child, dict(seed=[[3.0, 4.0], [15.0, 4.0]], barriers=[{"lat_gt": 7.0, "lon_lt": 6.0}]), wet)
        self.assertTrue(act2[:, 8:].all())
        self.assertTrue((act2[:, :8] == act[:, :8]).all())


    def test_outer_sides(self):
        """A child whose window reaches the root's edge owns that part of the
        open boundary: no ring and no interface faces beyond it, and its band
        there is dry after the mask consistency."""
        from pygetm.nesting.geometry import load_levels
        from pygetm.nesting import masks

        cfg = dict(
            root=dict(nx=10, ny=8, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
            east=dict(parent="root", istart=6, istop=10, jstart=0, jstop=5, ratio=2),
            inner=dict(parent="root", istart=2, istop=5, jstart=2, jstop=6, ratio=2),  # 2 cells from the edges
        )
        levels = {l.name: l for l in load_levels(cfg, halo=2, root_band=0)}
        east, inner = levels["east"], levels["inner"]
        self.assertEqual(east.outer_sides, "ES")
        self.assertEqual(inner.outer_sides, "")
        h = east.band
        ring = east._ring_mask()
        self.assertFalse(ring[:, h + east.nx :].any())  # nothing beyond the east edge
        self.assertFalse(ring[:h, :].any())  # nor beyond the south edge
        self.assertTrue(ring[h:, :h].any())  # the west and north bands are ring
        self.assertTrue(ring[h + east.ny :, h:].any())
        self.assertTrue(east.outer_mask()[:, h + east.nx :].all())
        # the ring's parent cells all lie inside the root
        pj, pi, _, _ = east.ring_source(*east.ring_cells())
        self.assertTrue(((pj >= 0) & (pj < 8) & (pi >= 0) & (pi < 10)).all())
        # interface faces: only the west and north sides of the window
        wf = levels["root"].wfaces["east"]
        oj, oi = wf["out"]
        self.assertTrue(((oj >= 0) & (oj < 8) & (oi >= 0) & (oi < 10)).all())
        self.assertEqual(wf["is_x"].size, 5 + 4)
        self.assertEqual(east.edge["is_x"].size, 2 * (5 + 4))
        # an all-wet setup: the band beyond the outer sides becomes dry
        wet = {n: np.ones(l.extended_shape, dtype=bool) for n, l in levels.items()}
        H = {n: np.full(l.extended_shape, 10.0) for n, l in levels.items()}
        wet2, _, changes = masks.make_consistent(levels.values(), wet, H)
        self.assertFalse(wet2["east"][east.outer_mask()].any())
        self.assertTrue(wet2["east"][h : h + east.ny, h : h + east.nx].all())
        self.assertFalse(any(masks.check_consistent(levels.values(), wet2)["east"].values()))

    def test_outer_sides_grandchild(self):
        """A grandchild at an outer side of its parent owns that part of the
        open boundary too; on the parent's other sides it has interface faces
        on the parent's interface."""
        from pygetm.nesting.geometry import load_levels
        from pygetm.nesting import masks

        cfg = dict(
            root=dict(nx=10, ny=8, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
            east=dict(parent="root", istart=5, istop=10, jstart=2, jstop=6, ratio=2),
            # reaches east's east edge (outer) and its south edge (not outer)
            gc=dict(parent="east", istart=4, istop=10, jstart=0, jstop=6, ratio=3),
        )
        levels = {l.name: l for l in load_levels(cfg, halo=2, root_band=0)}
        east, gc = levels["east"], levels["gc"]
        self.assertEqual(east.outer_sides, "E")
        self.assertEqual(gc.outer_sides, "E")
        h = gc.band
        ring = gc._ring_mask()
        self.assertFalse(ring[:, h + gc.nx :].any())
        self.assertTrue(ring[:h, h:].any())  # south: ring from east's band (over the root)
        wf = east.wfaces["gc"]
        oj, oi = wf["out"]
        self.assertTrue((oi < east.nx).all())  # no interface faces beyond the outer side
        self.assertEqual(wf["is_x"].size, 6 + 6 + 6)  # west, south, north
        south = oj < 0
        self.assertEqual(int(south.sum()), 6)
        self.assertTrue((wf["up"][south] >= 0).all())  # on east's interface with the root
        self.assertTrue((wf["up"][~south] < 0).all())
        wet = {n: np.ones(l.extended_shape, dtype=bool) for n, l in levels.items()}
        H = {n: np.full(l.extended_shape, 10.0) for n, l in levels.items()}
        wet2, _, _ = masks.make_consistent(levels.values(), wet, H)
        self.assertFalse(wet2["gc"][gc.outer_mask()].any())
        self.assertFalse(any(masks.check_consistent(levels.values(), wet2)["gc"].values()))


    def test_parent_rim_in_hole(self):
        """A grandchild within halo parent cells of its parent's interface (not touching it): the root's rim
        blocks lie partly in the child's hole, so the child must send up after its own children."""
        from pygetm.nesting.geometry import load_levels

        def child_flag(gi0):
            spec = {"root": dict(nx=40, ny=30, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
                    "child": dict(parent="root", istart=10, istop=30, jstart=5, jstop=25, ratio=3),
                    "grand": dict(parent="child", istart=gi0, istop=gi0 + 20, jstart=15, jstop=40, ratio=2)}
            return {lev.name: lev.parent_rim_in_hole for lev in load_levels(spec, root_band=0)}

        near = child_flag(3)  # one root cell inside the child's west edge
        self.assertTrue(near["child"])
        self.assertFalse(near["root"])
        self.assertFalse(near["grand"])
        self.assertFalse(child_flag(12)["child"])  # four root cells inside: clear of the root's rim

    def test_layout_refusals(self):
        """A window 1 cell from the root's outer edge, and a layout with two roots, are refused."""
        from pygetm.nesting.geometry import load_levels

        root = dict(nx=10, ny=8, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0)
        with self.assertRaisesRegex(ValueError, "from the root's outer edge"):
            load_levels(dict(root=root, c=dict(parent="root", istart=1, istop=4, jstart=2, jstop=6, ratio=2)), root_band=0)
        with self.assertRaisesRegex(ValueError, "exactly one root"):
            load_levels(dict(a=root, b=dict(root)), root_band=0)

class TestNestingSponge(unittest.TestCase):
    def test_water_distance_and_weight(self):
        from pygetm.nesting.sponge import sponge_weight, water_distance

        wet = np.ones((5, 7), dtype=bool)
        wet[:4, 3] = False  # a wall with a gap in the top row
        source = np.zeros_like(wet)
        source[:, 0] = True
        d = water_distance(source, wet)
        np.testing.assert_array_equal(d[:, 0], 0.0)
        np.testing.assert_array_equal(d[0, :3], [0.0, 1.0, 2.0])
        self.assertTrue(np.isinf(d[:4, 3]).all())
        self.assertEqual(d[4, 3], 3.0)  # through the gap only
        self.assertEqual(d[0, 4], 7.0)  # around the wall (3 + 4), not across it (4)
        w = sponge_weight(d, 2.0)
        np.testing.assert_allclose(w[0, :4], [0.0, 1.0, 0.5, 0.0])
        self.assertEqual(w[0, 4], 0.0)
        self.assertTrue((w[~wet] == 0.0).all())


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

    def test_limited_slopes_at(self):
        # the point version equals the full field everywhere (edges, land)
        from pygetm.nesting.operators import limited_slopes, limited_slopes_at

        rng = np.random.default_rng(1)
        c = rng.normal(size=(9, 11))
        wet = rng.random((9, 11)) > 0.25
        j, i = (x.ravel() for x in np.meshgrid(np.arange(9), np.arange(11), indexing="ij"))
        for w in (wet, None):
            sx, sy = limited_slopes(c, w)
            px, py = limited_slopes_at(c, w, j, i)
            self.assertTrue(np.array_equal(px, sx[j, i]))
            self.assertTrue(np.array_equal(py, sy[j, i]))


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

    def test_child_led_consistency_active(self):
        from pygetm.nesting.masks import make_consistent, check_consistent

        levels = active_hierarchy()
        rng = np.random.default_rng(2)
        wet = {l.name: rng.random(l.extended_shape) > 0.3 for l in levels}
        H = {l.name: 5.0 + 10.0 * rng.random(l.extended_shape) for l in levels}
        wet2, H2, changes = make_consistent(levels, wet, H)
        self.assertGreater(changes["child"]["unused_dried"], 0)
        self.assertTrue(all(v == 0 for d in check_consistent(levels, wet2).values() for v in d.values()))
        # root cells of the window outside the active region are left alone
        np.testing.assert_array_equal(wet2["root"][2 + 14 : 2 + 20, 2 + 18 : 2 + 26], wet["root"][2 + 14 : 2 + 20, 2 + 18 : 2 + 26])


    def test_faces_closed_over_closed_subfaces(self):
        # rule 3: a parent interface face between two wet cells whose child
        # sub-faces are all closed is closed in the parent, and in its domain mask
        from pygetm.nesting.masks import make_consistent, check_consistent, close_faces

        levels = small_hierarchy()
        root, child = levels[0], levels[1]
        wet = {l.name: np.ones(l.extended_shape, bool) for l in levels}
        H = {l.name: np.full(l.extended_shape, 10.0) for l in levels}
        f = root.wfaces[child.name]
        n = int(np.nonzero(f["inside"] & f["is_x"])[0][0])  # one x interface face
        e = child.edge
        r, h = child.ratio, child.band
        cj, ci = e["inner"]
        k = slice(n * r, (n + 1) * r)
        wet[child.name][cj[k] + h, ci[k] + h] = False  # the child's cells along it dry
        wet2, H2, _ = make_consistent(levels, wet, H)
        cf = root.closed_faces
        self.assertEqual(cf["is_x"].size, 1)
        self.assertTrue(cf["is_x"][0])
        self.assertEqual((cf["face"][0][0], cf["face"][1][0]), (f["face"][0][n], f["face"][1][n]))
        self.assertTrue(all(v == 0 for d in check_consistent(levels, wet2).values() for v in d.values()))
        # in the domain mask: the U point closed, its T neighbours still wet
        import types
        hb = root.band
        m = np.ones((2 * root.extended_shape[0] + 1, 2 * root.extended_shape[1] + 1), int)
        dom = types.SimpleNamespace(mask=m, _mask=m)
        close_faces(root, dom)
        fj, fi = f["face"][0][n] + hb, f["face"][1][n] + hb
        self.assertEqual(dom.mask[2 * fj + 1, 2 * fi + 2], 0)
        self.assertEqual(dom.mask[2 * fj + 1, 2 * fi + 1], 1)
        self.assertEqual(dom.mask[2 * fj + 1, 2 * fi + 3], 1)
        self.assertEqual(int((dom.mask == 0).sum()), 1)

        # a process other than the domain's rank 0: no global mask, and pyGETM's
        # Domain.mask raises there; nothing to close
        class _Other:
            _mask = None

            @property
            def mask(self):
                raise AttributeError("'NoneType' object has no attribute 'flags'")

        close_faces(root, _Other())


    def test_interface_depths(self):
        # the layout keeps interfaces out of drying water: shallow ring cells or
        # parent source cells are counted against interface_min_depth
        from pygetm.nesting.masks import interface_depths, interface_min_depth

        levels = small_hierarchy()
        root, child = levels[0], levels[1]
        wet = {l.name: np.ones(l.extended_shape, bool) for l in levels}
        H = {l.name: np.full(l.extended_shape, 10.0) for l in levels}
        lim = {child.name: 3.0}
        d = interface_depths(levels, wet, H, lim)[child.name]
        self.assertEqual(d["interface_shallow"], 0)
        self.assertEqual(d["min_depth_ring"], 10.0)
        bj, bi = child.ring_cells()
        H[child.name][bj[0] + child.band, bi[0] + child.band] = 1.0  # one shallow ring cell
        pj, pi, _, _ = child.ring_source(bj, bi)
        H[root.name][pj[-1] + root.band, pi[-1] + root.band] = 2.0  # one shallow parent source cell
        d = interface_depths(levels, wet, H, lim)[child.name]
        self.assertEqual(d["interface_shallow"], 2)
        self.assertEqual((d["min_depth_ring"], d["min_depth_parent"]), (1.0, 2.0))
        self.assertEqual(interface_depths(levels, wet, H, {})[child.name]["interface_shallow"], 0)  # not set
        cfg = {"config": {"interface_min_depth": 2.5}, child.name: {"interface_min_depth": 4}}
        self.assertEqual((interface_min_depth(cfg, child.name), interface_min_depth(cfg, root.name)), (4.0, 2.5))
        self.assertEqual(interface_min_depth({}, child.name), 0.0)


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
            plan = Plan(lev, tilings[lev.parent.name], tilings[lev.name])
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

    def test_plans_active(self):
        levels = active_hierarchy()
        self._check(levels, {"root": (1, 1, {}), "child": (1, 1, {}), "gc": (1, 1, {})})
        self._check(levels, {
            "root": (2, 3, dict(xoff=-4)),
            "child": (3, 2, dict(yoff=-5, removed=[(2, 1)])),
            "gc": (2, 2, dict(xoff=-3)),
        })


def run_barotropic(ratio, nroot, nsteps, amplitude=0.5, coriolis=1e-4, slope=True, children=True, active=None,
                   time_ratio=None, dt_root=20.0, **coupler_kwargs):
    """Closed basin, root (+ child), 2D barotropic; returns (volumes per
    output, gathered z of this level or None, level name, world)."""
    from pygetm.nesting import load_levels
    from pygetm.nesting.runtime import NestingRun
    from pygetm.nesting.nested2d import Nested2D

    cfg = {"root": dict(nx=40, ny=30, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0)}
    if children:
        cfg["child"] = dict(parent="root", istart=12, istop=24, jstart=8, jstop=20, ratio=ratio)
        if active is not None:
            cfg["child"]["active"] = active.tolist()
        if time_ratio is not None:
            cfg["child"]["time_ratio"] = time_ratio
    levels = load_levels(cfg, root_band=0)
    world = parallel.MPI.COMM_WORLD
    ranks = {"root": nroot, "child": world.size - nroot} if children else {"root": world.size}
    nest = NestingRun(levels, ranks)

    def make_domain(lev, comm):
        lon, lat = lev.edges()
        xc = 0.5 * (lon[1:] + lon[:-1])
        H = (20.0 + (10.0 * xc / 40.0 if slope else 0.0))[None, :] * np.ones((lat.size - 1, 1))
        if lev.parent is not None:
            H = np.where(lev.unused(), np.nan, H)  # window cells the child does not use
        return pygetm.domain.create_cartesian(
            lon * 2e3, lat * 2e3, interfaces=True, H=H, f=coriolis, comm=comm,
            logger=parallel.get_logger(level="ERROR", comm=comm),
        )

    def make_sim(lev, domain):
        sim = pygetm.Simulation(
            domain, runtype=pygetm.RunType.BAROTROPIC_2D, airsea=pygetm.airsea.Fluxes(),
            logger=parallel.get_logger(level="ERROR", comm=domain.comm),
        )
        sim.airsea.taux.set(0.0)
        sim.airsea.tauy.set(0.0)
        sim.airsea.sp.set(0.0)
        return sim

    sim = nest.build_simulation(make_domain, make_sim)
    cpl = Nested2D(nest, sim, **coupler_kwargs)
    sim.add_coupler(cpl)
    lev = nest.level
    gj, gi = nest.global_indices()
    x = lev.lon0 + (gi + 0.5) * lev.dlon
    y = lev.lat0 + (gj + 0.5) * lev.dlat
    sim.T.z.all_values[...] = amplitude * np.exp(-((x[None, :] - 18.0) ** 2 + (y[:, None] - 14.0) ** 2) / 8.0)
    sim.T.z.all_values[sim.T.mask.all_values == 0] = 0.0
    import datetime

    sim.start(datetime.datetime(2000, 1, 1), timestep=dt_root / lev.steps_per_root, split_factor=1, report=0)
    vols = [cpl.level_volume()]
    for _ in range(nsteps * lev.steps_per_root):
        sim.advance()
    vols.append(cpl.level_volume())
    sim.finish()
    g = sim.T.z.gather()
    return nest.world.allreduce(np.array(vols)), (None if g is None else np.array(g)), lev.name, nest.world


class TestNestingBarotropic(unittest.TestCase):
    """2D barotropic nesting (Nested2D). Needs >= 2 processes."""

    def setUp(self):
        if parallel.MPI.COMM_WORLD.size < 2:
            self.skipTest("needs at least 2 MPI processes")

    def test_volume_conserved(self):
        nroot = (parallel.MPI.COMM_WORLD.size + 1) // 2
        vols, _, _, _ = run_barotropic(3, nroot, 60)
        self.assertLess(abs(vols[1] - vols[0]) / vols[0], 1e-13)

    def test_volume_conserved_active(self):
        # L-shaped active region (south-west corner of the window stays with the root)
        act = np.ones((12, 12), dtype=bool)
        act[:5, :5] = False
        nroot = (parallel.MPI.COMM_WORLD.size + 1) // 2
        vols, _, _, _ = run_barotropic(3, nroot, 60, active=act)
        self.assertLess(abs(vols[1] - vols[0]) / vols[0], 1e-13)

    def test_volume_conserved_time_ratio_one(self):
        # child with the parent's time step (ratio 3 in space, 1 in time)
        nroot = (parallel.MPI.COMM_WORLD.size + 1) // 2
        vols, z, _, _ = run_barotropic(3, nroot, 120, time_ratio=1, dt_root=6.0)
        self.assertLess(abs(vols[1] - vols[0]) / vols[0], 1e-13)
        if z is not None:
            self.assertTrue(np.isfinite(z[np.isfinite(z)]).all())

    def test_volume_conserved_sponge(self):
        # the sponge damps momentum only: volume stays conserved
        nroot = (parallel.MPI.COMM_WORLD.size + 1) // 2
        vols, z, _, _ = run_barotropic(3, nroot, 60, sponge_width=1.0, sponge_alpha=0.01)
        self.assertLess(abs(vols[1] - vols[0]) / vols[0], 1e-13)
        if z is not None:
            self.assertTrue(np.isfinite(z[np.isfinite(z)]).all())

    def test_tracer_time_ratio(self):
        from pygetm.nesting import load_levels

        levels = load_levels({
            "root": dict(nx=40, ny=30, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
            "child": dict(parent="root", istart=10, istop=22, jstart=8, jstop=18, ratio=3, time_ratio=1),
            "gc": dict(parent="child", istart=6, istop=18, jstart=6, jstop=18, ratio=2, time_ratio=4),
        })
        self.assertEqual([l.steps_per_root for l in levels], [1, 1, 4])
        self.assertEqual(levels[2].time_ratio, 4)

    def test_lake_at_rest(self):
        nroot = (parallel.MPI.COMM_WORLD.size + 1) // 2
        _, z, _, world = run_barotropic(3, nroot, 30, amplitude=0.0)
        if z is not None:
            self.assertEqual(float(np.abs(z[np.isfinite(z)]).max()), 0.0)

    def test_ratio_one_twin(self):
        world = parallel.MPI.COMM_WORLD
        nroot = (world.size + 1) // 2
        _, z_nested, name, nworld = run_barotropic(1, nroot, 100)
        # single domain on the root's processes only
        color = 0 if world.rank < nroot else parallel.MPI.UNDEFINED
        sub = world.Split(color, world.rank)
        single = None
        if sub != parallel.MPI.COMM_NULL:
            from pygetm.nesting import load_levels

            lev = load_levels({"root": dict(nx=40, ny=30, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0)}, root_band=0)[0]
            lon, lat = lev.edges()
            xc = 0.5 * (lon[1:] + lon[:-1])
            H = (20.0 + 10.0 * xc / 40.0)[None, :] * np.ones((lat.size - 1, 1))
            domain = pygetm.domain.create_cartesian(
                lon * 2e3, lat * 2e3, interfaces=True, H=H, f=1e-4, comm=sub,
                logger=parallel.get_logger(level="ERROR", comm=sub),
            )
            sim = pygetm.Simulation(domain, runtype=pygetm.RunType.BAROTROPIC_2D,
                                    airsea=pygetm.airsea.Fluxes(),
                                    logger=parallel.get_logger(level="ERROR", comm=sub))
            sim.airsea.taux.set(0.0)
            sim.airsea.tauy.set(0.0)
            sim.airsea.sp.set(0.0)
            gj = np.arange(sim.T.ny_) - sim.T.haloy + sim.T.tiling.yoffset
            gi = np.arange(sim.T.nx_) - sim.T.halox + sim.T.tiling.xoffset
            sim.T.z.all_values[...] = 0.5 * np.exp(-(((gi + 0.5)[None, :] - 18.0) ** 2 + ((gj + 0.5)[:, None] - 14.0) ** 2) / 8.0)
            sim.T.z.all_values[sim.T.mask.all_values == 0] = 0.0
            import datetime

            sim.start(datetime.datetime(2000, 1, 1), timestep=20.0, split_factor=1, report=0)
            for _ in range(100):
                sim.advance()
            sim.finish()
            g = sim.T.z.gather()
            single = None if g is None else np.array(g)
        # compare on world rank 0 (root's rank 0, holds both root fields)
        child = nworld.bcast(z_nested if (name == "child" and z_nested is not None) else None,
                             root=nroot)
        if world.rank == 0:
            hole = np.zeros_like(single, dtype=bool)
            hole[8:20, 12:24] = True
            scale = np.abs(single).max()
            self.assertLess(np.abs(z_nested - single)[~hole].max() / scale, 1e-3)
            self.assertLess(np.abs(child[2:-2, 2:-2] - single[8:20, 12:24]).max() / scale, 1e-3)


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


class TestNestingMPMD(unittest.TestCase):
    """MPMD launch (one mpiexec component per level, NestedModel) against the
    single-program NestingRun: identical z on both levels. Starts mpiexec
    itself, so it runs only in a serial test session."""

    def test_mpmd_equals_single_program(self):
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        if parallel.MPI.COMM_WORLD.size > 1:
            self.skipTest("starts its own mpiexec; run serially")
        mpiexec = shutil.which("mpiexec")
        if mpiexec is None:
            self.skipTest("no mpiexec")
        case = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nesting_mpmd_case.py")
        env = dict(os.environ)
        for k in list(env):  # a fresh MPI launch, not this process's singleton
            if k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")):
                env.pop(k)
        with tempfile.TemporaryDirectory() as out:
            py = sys.executable
            cmds = (
                [mpiexec, "-n", "2", py, case, "mpmd", "root", out, ":", "-n", "1", py, case, "mpmd", "child", out],
                [mpiexec, "-n", "3", py, case, "single", "-", out],
            )
            for cmd in cmds:
                r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for name in ("root", "child"):
                a = np.load(os.path.join(out, f"mpmd_{name}.npy"))
                b = np.load(os.path.join(out, f"single_{name}.npy"))
                self.assertTrue(np.isfinite(a).any())
                self.assertTrue(np.array_equal(a, b, equal_nan=True), name)



class TestNestingDecomposition(unittest.TestCase):
    """The same nest with the root on 1 and on 4 processes (2 x 2) is bit-identical; the child's east and
    north edges lie on the root's subdomain boundaries there (nesting_mpmd_case.LAYOUT_EDGE), where the
    transports at the interface faces belong to another tile than the cells they correct (Plan.uv_of).
    Starts mpiexec itself."""

    def test_root_tiling(self):
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        if parallel.MPI.COMM_WORLD.size > 1:
            self.skipTest("starts its own mpiexec; run serially")
        mpiexec = shutil.which("mpiexec")
        if mpiexec is None:
            self.skipTest("no mpiexec")
        case = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nesting_mpmd_case.py")
        env = dict(os.environ)
        for k in list(env):  # a fresh MPI launch, not this process's singleton
            if k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")):
                env.pop(k)
        with tempfile.TemporaryDirectory() as out:
            for n in (2, 5):
                r = subprocess.run([mpiexec, "-n", str(n), sys.executable, case, "single", "-", out, "edge"], env=env,
                                   capture_output=True, text=True, timeout=600)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for name in ("root", "child"):
                a = np.load(os.path.join(out, f"single2_{name}.npy"))
                b = np.load(os.path.join(out, f"single5_{name}.npy"))
                self.assertTrue(np.isfinite(a).any())
                self.assertTrue(np.array_equal(a, b, equal_nan=True), name)

    def test_plan_roles(self):
        """Plan: with the root split at the child's east edge, the faces there are corrected by the eastern tile
        and their transports set by the western one."""
        from pygetm.nesting.plan import LevelTiling, Plan
        from pygetm.nesting.geometry import load_levels

        spec = {"root": dict(nx=40, ny=30, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
                "child": dict(parent="root", istart=10, istop=20, jstart=6, jstop=15, ratio=3)}
        levels = load_levels(spec, root_band=0)
        root, child = levels
        ptil = LevelTiling(dict(map=[[0, 1]], nx_sub=20, ny_sub=30, xoffset_global=0, yoffset_global=0, halo=2), 0, 0)
        ctil = LevelTiling(dict(map=[[0]], nx_sub=child.nx + 2 * child.band, ny_sub=child.ny + 2 * child.band,
                                xoffset_global=0, yoffset_global=0, halo=2), child.band, 2)
        plan = Plan(child, ptil, ctil)
        wf = root.wfaces["child"]
        east = wf["inside"] & wf["is_x"] & (wf["out"][1] == 20)
        n_east = int(east.sum())
        self.assertGreater(n_east, 0)
        for p, other in ((0, 1), (1, 0)):
            f = plan.faces_of[p]
            e = east[f]
            # tile 1 owns the cells east of the hole, tile 0 the U points of those faces (i = 19)
            self.assertEqual(int((e & plan.corr_of[p]).sum()), n_east if p == 1 else 0)
            self.assertEqual(int((e & plan.uv_of[p]).sum()), n_east if p == 0 else 0)
        # every inside face is corrected once and has its transport set once
        for role in (plan.corr_of, plan.uv_of):
            seen = np.concatenate([plan.faces_of[p][role[p]] for p in (0, 1)])
            inside = np.nonzero(wf["inside"])[0]
            self.assertTrue(np.array_equal(np.sort(seen[np.isin(seen, inside)]), inside))


@unittest.skipUnless(__import__("os").environ.get("PYGETM_TEST_FABM"), "FABM tests on demand: PYGETM_TEST_FABM=1")
class TestNestingFABM(unittest.TestCase):
    """Two levels in 3D with FABM (nesting_fabm_case.py: npzd + benthic predator, a bottom state variable):
    Nested3D exchanges FABM's pelagic state by default, nitrogen crosses the interface, and the total over
    both levels is conserved. On demand only (PYGETM_TEST_FABM=1). Starts mpiexec itself."""

    def test_fabm_conserved(self):
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        if parallel.MPI.COMM_WORLD.size > 1:
            self.skipTest("starts its own mpiexec; run serially")
        mpiexec = shutil.which("mpiexec")
        if mpiexec is None:
            self.skipTest("no mpiexec")
        here = os.path.dirname(os.path.abspath(__file__))
        case = os.path.join(here, "nesting_fabm_case.py")
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")):
                env.pop(k)
        env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
        py = sys.executable
        with tempfile.TemporaryDirectory() as out:
            cmd = [mpiexec, "-n", "2", py, case, "root", out, ":", "-n", "1", py, case, "child", out]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            a, b = (np.load(os.path.join(out, f"{n}.npz")) for n in ("root", "child"))
            for name in ("npzd_nut", "npzd_phy", "npzd_zoo", "npzd_det"):
                self.assertIn(name, list(b["tracers"]))
            total = a["nitrogen"] + b["nitrogen"]
            child = b["nitrogen"]
            self.assertGreater(abs(child[-1] - child[0]) / child[0], 1e-6)  # nitrogen crossed the interface
            self.assertLess(np.abs(total - total[0]).max() / total[0], 1e-7)


def _launch_3d(mode, out, *opts):
    """nesting_3d_case.py: root on 2 processes, child on 1 (one mpiexec component per level)."""
    import os
    import shutil
    import subprocess
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    case = os.path.join(here, "nesting_3d_case.py")
    env = dict(os.environ)
    for k in list(env):  # a fresh MPI launch, not this process's singleton
        if k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")):
            env.pop(k)
    env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
    py = sys.executable
    cmd = [shutil.which("mpiexec"), "-n", "2", py, case, mode, "root", out, *opts,
           ":", "-n", "1", py, case, mode, "child", out, *opts]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)


def _skip_unless_mpiexec(test):
    import shutil

    if parallel.MPI.COMM_WORLD.size > 1:
        test.skipTest("starts its own mpiexec; run serially")
    if shutil.which("mpiexec") is None:
        test.skipTest("no mpiexec")


class TestNesting3D(unittest.TestCase):
    """Two levels in 3D (nesting_3d_case.py: sigma layers 5 and 8, a temperature front across the child's west
    interface, uniform salinity, a closed basin): heat crosses the interface and the heat over both levels is
    conserved (with the limiter's deferred amounts; to the coupling's 1e-7, the sync follows the tracer step),
    salinity stays uniform, with the PPM and PCM remaps, and with a child on its own 3D step (3D sub-stepping)
    whose ring is extrapolated in time. Starts mpiexec itself."""

    def check(self, *opts, ratio_3d=1):
        import os
        import tempfile

        _skip_unless_mpiexec(self)
        with tempfile.TemporaryDirectory() as out:
            r = _launch_3d("run", out, *opts)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            a, b = (np.load(os.path.join(out, f"run_{n}.npz")) for n in ("root", "child"))
            heat = a["heat"] + b["heat"]
            self.assertGreater(abs(b["heat"][-1] - b["heat"][0]) / abs(b["heat"][0]), 1e-6)  # heat crossed
            self.assertLess(np.abs(heat - heat[0]).max() / abs(heat[0]), 1e-7)
            for lev in (a, b):
                self.assertLess(max(abs(lev["salt_min"].min() - 35.0), abs(lev["salt_max"].max() - 35.0)), 1e-10)
            self.assertEqual(int(b["ratio_3d"]), ratio_3d)
            if ratio_3d > 1:
                self.assertGreater(int(b["n_extrapolated"]), 0)

    def test_ppm(self):
        self.check("scheme=ppm")

    def test_pcm(self):
        self.check("scheme=pcm")

    def test_substeps_extrapolate(self):
        self.check("scheme=ppm", "sub", ratio_3d=3)


class TestNestingRestart3D(unittest.TestCase):
    """The 3D nest of nesting_3d_case.py with the child on its own 3D step and its ring extrapolated in time,
    restarted halfway, continues bit-identically on both levels (every state field, and the heat series): with a
    smooth front (the flux limiter does not clip), and with a step in temperature at the interface (it clips, and
    the restart file holds deferred amounts). Starts mpiexec itself."""

    def test_restart(self):
        self.check("sub")

    def test_restart_with_deferred_corrections(self):
        import os

        import netCDF4

        out = self.check("sub", "sharp", keep=True)
        try:
            self.assertGreater(int(np.load(os.path.join(out, "first_root.npz"))["n_clipped"]), 0)
            with netCDF4.Dataset(os.path.join(out, "first_root.nc")) as nc:
                d = [np.ma.filled(nc[v][:], 0.0) for v in nc.variables if v.startswith("nest_flux_deferred")]
            self.assertTrue(d and max(float(np.abs(x).max()) for x in d) > 0.0)
        finally:
            import shutil

            shutil.rmtree(out, ignore_errors=True)

    def check(self, *opts, keep=False):
        import os
        import tempfile

        import xarray as xr

        _skip_unless_mpiexec(self)
        out = tempfile.mkdtemp()
        try:
            for mode in ("full", "first", "second"):
                r = _launch_3d(mode, out, *opts)
                self.assertEqual(r.returncode, 0, mode + ": " + r.stdout[-3000:] + r.stderr[-3000:])
            for name in ("root", "child"):
                with xr.open_dataset(os.path.join(out, f"full_{name}.nc")) as a, \
                        xr.open_dataset(os.path.join(out, f"second_{name}.nc")) as b:
                    for v in ("temp", "salt", "nest_layout_id"):
                        self.assertIn(v, a)
                    for v in a.data_vars:
                        x, y = a[v].values, b[v].values
                        # pyGETM writes NaN where nothing is computed after a restart, 0 in a continuous run
                        same = (x == y) | (np.isnan(x) & np.isnan(y)) | (np.isnan(y) & (x == 0.0))
                        self.assertTrue(same.all(), f"{name} {v}")
                f = np.load(os.path.join(out, f"full_{name}.npz"))
                s = np.load(os.path.join(out, f"second_{name}.npz"))
                self.assertTrue(np.array_equal(f["heat"][-len(s["heat"]):], s["heat"]), name)
        finally:
            if not keep:
                import shutil

                shutil.rmtree(out, ignore_errors=True)
        return out


class TestNestingThreeLevels(unittest.TestCase):
    """Root, child and grandchild in 2D (nesting_levels_case.py; 2 + 2 + 1 processes): the water volume over all
    three levels is conserved to round-off and water reaches the grandchild, with the grandchild on the child's
    interface with the root (its fluxes passed up through the child) and one root cell inside it (the root's rim
    reaches into the child's hole); in both the child sends up after its grandchild. Starts mpiexec itself."""

    def check(self, where):
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        _skip_unless_mpiexec(self)
        here = os.path.dirname(os.path.abspath(__file__))
        case = os.path.join(here, "nesting_levels_case.py")
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")):
                env.pop(k)
        env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
        py = sys.executable
        with tempfile.TemporaryDirectory() as out:
            cmd = [shutil.which("mpiexec"), "-n", "2", py, case, "root", out, where, ":", "-n", "2", py, case, "child",
                   out, where, ":", "-n", "1", py, case, "grand", out, where]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            v = {n: np.load(os.path.join(out, f"{n}.npz")) for n in ("root", "child", "grand")}
            total = sum(x["volume"] for x in v.values())
            scale = max(np.abs(x["volume"]).max() for x in v.values())
            self.assertLess(np.abs(total - total[0]).max() / scale, 1e-12)
            self.assertGreater(np.abs(v["grand"]["volume"] - v["grand"]["volume"][0]).max() / scale, 0.1)
            self.assertTrue(bool(v["child"]["up_after_children"]))
            self.assertFalse(bool(v["root"]["up_after_children"]))

    def test_grandchild_on_interface(self):
        self.check("on")

    def test_grandchild_near_interface(self):
        self.check("near")

class TestNestingErrors(unittest.TestCase):
    """A failing level stops the whole nested job instead of leaving the others waiting (nesting_error_case.py:
    the child raises, asks for a level not in the layout, calls sys.exit(3) in the run, or runs 3D without
    Nested3D); each within the time-out, with its message. Starts mpiexec itself."""

    def check(self, what, message, status=None):
        import os
        import shutil
        import subprocess
        import sys

        _skip_unless_mpiexec(self)
        here = os.path.dirname(os.path.abspath(__file__))
        case = os.path.join(here, "nesting_error_case.py")
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")):
                env.pop(k)
        env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
        py = sys.executable
        cmd = [shutil.which("mpiexec"), "-n", "2", py, case, what, "root", ":", "-n", "1", py, case, what, "child"]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            self.fail(f"{what}: the other levels kept waiting (no abort within 60 s)")
        self.assertNotEqual(r.returncode, 0)
        if status is not None:
            self.assertEqual(r.returncode, status)
        log = r.stdout + r.stderr
        self.assertIn("aborting all levels", log)
        self.assertIn(message, log)

    def test_exception(self):
        self.check("raise", "the child fails before the run")

    def test_unknown_level(self):
        self.check("badname", "is not in the layout")

    def test_sys_exit(self):
        self.check("exit", "exit with status 3", status=3)

    def test_couplers_differ(self):
        self.check("no3d", "differ in run type or nesting couplers")

class TestNestingRemovedTiles(unittest.TestCase):
    """A root whose tiling removes a subdomain (nesting_removed_case.py: 3 x 3 on 8 processes, the centre all in the
    hole's interior) runs bit-identically to the root on 1 process (both levels; the root outside the removed
    subdomain, which the gather fills), and conserves the volume over both levels. Starts mpiexec itself."""

    def test_removed_subdomain(self):
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        _skip_unless_mpiexec(self)
        here = os.path.dirname(os.path.abspath(__file__))
        case = os.path.join(here, "nesting_removed_case.py")
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")):
                env.pop(k)
        env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
        py = sys.executable
        with tempfile.TemporaryDirectory() as out:
            for n in (1, 8):
                cmd = [shutil.which("mpiexec"), "-n", str(n), py, case, "root", out, ":", "-n", "1", py, case, "child", out]
                r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
                self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            load = lambda lev, n: np.load(os.path.join(out, f"{lev}_{n}.npz"))
            self.assertEqual(int(load("root", 8)["removed"]), 1)
            self.assertEqual(int(load("root", 1)["removed"]), 0)
            a, b = load("root", 1)["z"], load("root", 8)["z"]
            kept = np.abs(b) < 1e10  # the removed subdomain: the gather's fill value
            self.assertEqual(int((~kept).sum()), 14 * 10)
            self.assertTrue(np.array_equal(a[kept], b[kept], equal_nan=True))
            self.assertTrue(np.array_equal(load("child", 1)["z"], load("child", 8)["z"], equal_nan=True))
            v = load("root", 8)["volume"] + load("child", 8)["volume"]
            self.assertLess(np.abs(v - v[0]).max() / np.abs(load("child", 8)["volume"]).max(), 1e-12)

class TestNestingFeatures(unittest.TestCase):
    """Two levels in 2D with the features that meet the nesting (nesting_features_case.py), the volume over both
    levels conserved to round-off (or grown by exactly the rivers' inflow): a child whose window reaches the root's
    edge (an outer side), rivers given to every level (each kept by one), a dam in each level, a sluice between the
    levels, a bank in the child that falls dry. Starts mpiexec itself."""

    def run_case(self, what):
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        _skip_unless_mpiexec(self)
        here = os.path.dirname(os.path.abspath(__file__))
        case = os.path.join(here, "nesting_features_case.py")
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")):
                env.pop(k)
        env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
        py = sys.executable
        with tempfile.TemporaryDirectory() as out:
            cmd = [shutil.which("mpiexec"), "-n", "2", py, case, what, "root", out, ":", "-n", "1", py, case, what,
                   "child", out]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            a, b = (dict(np.load(os.path.join(out, f"{n}.npz"))) for n in ("root", "child"))
        V = a["volume"] + b["volume"]
        scale = max(np.abs(a["volume"]).max(), np.abs(b["volume"]).max())
        return a, b, V, scale

    def assertConserved(self, V, scale, inflow=None):
        expected = V[0] + (0.0 if inflow is None else inflow)
        self.assertLess(np.abs(V - expected).max() / scale, 1e-12)

    def test_outer_side(self):
        a, b, V, scale = self.run_case("outer")
        self.assertEqual(str(b["outer_sides"]), "E")
        self.assertConserved(V, scale)

    def test_rivers(self):
        a, b, V, scale = self.run_case("rivers")
        self.assertEqual((int(a["rivers"]), int(b["rivers"])), (1, 1))  # each kept by one level
        self.assertConserved(V, scale, inflow=100.0 * a["time"])  # 2 x 50 m3 s-1

    def test_dams(self):
        a, b, V, scale = self.run_case("dam")
        self.assertGreater(int(a["dam_faces"]), 0)
        self.assertGreater(int(b["dam_faces"]), 0)
        self.assertConserved(V, scale)

    def test_sluice(self):
        a, b, V, scale = self.run_case("sluice")
        self.assertGreater(float(b["sluice_volume"]), 1e4)  # water went from the root to the child
        self.assertConserved(V, scale)

    def test_drying(self):
        a, b, V, scale = self.run_case("drying")
        self.assertGreater(int(b["dry_seen"]), 0)
        self.assertConserved(V, scale)

class TestNestingRestart(unittest.TestCase):
    """A nested 2D run (root on 2 processes, child with time ratio 3) restarted
    halfway continues bit-identically (nesting_restart_case.py); a restart file
    written in the middle of a root step is refused. Starts mpiexec itself."""

    def test_restart(self):
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        import xarray as xr

        if parallel.MPI.COMM_WORLD.size > 1:
            self.skipTest("starts its own mpiexec; run serially")
        mpiexec = shutil.which("mpiexec")
        if mpiexec is None:
            self.skipTest("no mpiexec")
        here = os.path.dirname(os.path.abspath(__file__))
        case = os.path.join(here, "nesting_restart_case.py")
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")):
                env.pop(k)
        env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
        py = sys.executable
        with tempfile.TemporaryDirectory() as out:
            def launch(mode):
                cmd = [mpiexec, "-n", "2", py, case, mode, "root", out, ":", "-n", "1", py, case, mode, "child", out]
                return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

            for mode in ("full", "first", "second", "mid"):
                r = launch(mode)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for name in ("root", "child"):
                with xr.open_dataset(os.path.join(out, f"full_{name}.nc")) as a, \
                        xr.open_dataset(os.path.join(out, f"second_{name}.nc")) as b:
                    self.assertIn("nest_root_phase", a)
                    for v in a.data_vars:
                        x, y = a[v].values, b[v].values
                        # pyGETM writes NaN on land after a restart, 0 in a continuous run
                        same = (x == y) | (np.isnan(x) & np.isnan(y)) | (np.isnan(y) & (x == 0.0))
                        self.assertTrue(same.all(), f"{name} {v}")
            r = launch("from_mid")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("middle of a root 3D step (child: 2 of 3 steps)", r.stdout + r.stderr)
            r = subprocess.run([mpiexec, "-n", "2", py, case, "alone", "root", out], env=env, capture_output=True, text=True,
                               timeout=600)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("restart file(s) of root written with another set of levels", r.stdout + r.stderr)
            with xr.open_dataset(os.path.join(out, "full_root.nc")) as a:
                self.assertIn("nest_layout_id", a)


if __name__ == "__main__":
    unittest.main()
