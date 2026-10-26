"""Running a nesting hierarchy: processes, domains, exchanges.

:class:`NestingRun` places every level on its own group of MPI processes
(:func:`pygetm.nesting.split_world`), builds the level's pyGETM grids with the
ring band and the hole marked external (:attr:`pygetm.domain.Domain.external`),
shares the tilings of all levels and builds the exchange plans
(:class:`pygetm.nesting.plan.Plan`).

:class:`NestedTracer` exchanges one tracer between the levels:

* the child's ring from the parent (limited linear reconstruction of the
  parent's (c, sx, sy) snapshot, interpolated in time over the child's
  sub-steps), at the start and the end of every child step;
* the parent's rim from the child (volume-weighted mean over wet child cells),
  once per parent step;
* flux matching: every level keeps a register of the transports through its
  outer edge faces over its sub-steps; after the child's sub-steps the parent
  corrects its cells next to the window by (parent - child) transport, and
  registers of children on the parent's edge are passed upwards.

The transports are the tracer amounts the advection actually moved through
every face (``Advection(..., flux_u=, flux_v=)``). The time loop belongs to
the caller: every level calls :meth:`NestedTracer.step` once per own step, in
the same order on all ranks; MPI messages enforce the order between levels.
"""

from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
from mpi4py import MPI

from .geometry import SIDES, Level
from .hierarchy import split_world
from .plan import LevelTiling, Plan, gather_tilings
from . import operators as ops


class NestingRun:
    """Processes, grids and exchange plans of a nesting hierarchy (one level
    per group of processes)."""

    def __init__(self, levels: List[Level], ranks: Mapping[str, int], comm: Optional[MPI.Comm] = None):
        self.levels = levels
        ctx = split_world({l.name: int(ranks[l.name]) for l in levels}, comm=comm)
        self.world = ctx.nesting_comm
        self.comm = ctx.comm
        self.offsets = ctx.offsets
        self.level = next(l for l in levels if l.name == ctx.name)
        self.grid = None
        self.tilings: Dict[str, LevelTiling] = {}
        self.up: Optional[Plan] = None
        self.down: List[Plan] = []

    def build(self, make_domain: Callable, nz: int = 1, **grid_kwargs):
        """Build this level's grids. ``make_domain(level, comm)`` returns the
        pyGETM domain of the level including its band (window extended by
        ``level.band`` cells on every side). Returns the T grid, or None if
        pyGETM's tiling leaves this process unused."""
        lev = self.level
        domain = make_domain(lev, self.comm)
        domain.external = lev.external()
        grid_kwargs.setdefault("velocity_grids", 1)
        self.grid = domain.create_grids(nz, halox=lev.halo, haloy=lev.halo, **grid_kwargs)
        self.tilings = gather_tilings(self.levels, self.world, lev, self.grid, self.offsets)
        links = {c.name: k for k, c in enumerate(self.levels[1:])}
        til = self.tilings[lev.name]
        if lev.parent is not None:
            self.up = Plan(lev, self.tilings[lev.parent.name], til, links[lev.name])
        self.down = [Plan(c, til, self.tilings[c.name], links[c.name]) for c in lev.children]
        return self.grid

    @property
    def tile(self) -> Optional[int]:
        return None if self.grid is None else self.grid.tiling.rank

    def describe(self) -> List[str]:
        """This process's nesting messages (format of check_print_plan.py)."""
        me, lev = self.world.rank, self.level
        if self.grid is None:
            return [f"  [{lev.name} world {me}]  unused by pyGETM's tiling"]
        til, t = self.tilings[lev.name], self.tile
        lines = [f"  [{lev.name} tile {t} / world {me}]  pyGETM tiling {til.map.shape}, nx_sub={til.nx_sub}"
                 f" ny_sub={til.ny_sub} offsets x={til.xoff} y={til.yoff}, removed land tiles {til.removed}"]
        if self.up is not None:
            lines += self.up.describe(me, "child", t)
        for p in self.down:
            lines += p.describe(me, "parent", t)
        return lines

    def print_plan(self):
        """Print the messages of all processes in rank order (collective)."""
        lines = self.describe()
        for rank in range(self.world.size):
            if rank == self.world.rank:
                print("\n".join(lines), flush=True)
            self.world.Barrier()

    def global_indices(self):
        """Global (j, i) of every row/column of this process's local arrays."""
        T, lev = self.grid, self.level
        y0, x0 = self.tilings[lev.name].origin(self.tile)
        h = T.halox
        return np.arange(T.ny_) - h + y0, np.arange(T.nx_) - h + x0


class NestedTracer:
    """Exchange of one tracer (a T-grid :class:`pygetm.core.Array`) between
    this level, its parent and its children."""

    def __init__(self, run: NestingRun, f, break_ring=False, break_rim=False, break_flux=False):
        self.run = run
        self.f = f
        T = run.grid
        lev = run.level
        self.lev = lev
        self.t = run.tile
        self.til = run.tilings[lev.name]
        self.world = run.world
        self.up, self.down = run.up, run.down
        self.break_ring, self.break_rim, self.break_flux = break_ring, break_rim, break_flux
        self.wetmask = T.mask.all_values != 0
        self.content = T.D.all_values * T.area.all_values  # per unit concentration
        h = T.halox
        self.computed = T.mask.all_values[h:-h, h:-h] == 1
        self.halo = h
        # tracer amount through every face during a step (from the advection)
        self.flux = {"x": T.ugrid.array(fill=0.0), "y": T.vgrid.array(fill=0.0)}
        # this level's flux register (outer edge faces), kept by the owner of
        # the inner edge cell
        self.reg = {sd: np.zeros(len(lev.edge[sd]["inner"][0])) for sd in SIDES}
        self.own_edge = {sd: self.owned(*lev.edge[sd]["inner"]) for sd in SIDES}
        self.outflow = 0.0  # root: tracer amount that left through the boundary
        if self.up is not None:
            ring = [(p, v) for (p, tt), v in self.up.pro.items() if tt == self.t]
            cat = lambda k: np.concatenate([v[k] for _, v in ring]) if ring else np.zeros(0, int)
            self._ring = ring
            self._ring_a, self._ring_b = cat(4), cat(5)
            self._rj, self._ri = self.loc(cat(2), cat(3))
        self._fps = [None] * len(self.down)
        self._k = 0

    # --- helpers -----------------------------------------------------------

    def loc(self, j, i):
        return self.til.local(self.t, j, i)

    def owned(self, j, i):
        return self.til.owner(j, i) == self.t

    def _face_values(self, axis, j, i):
        return self.flux[axis].all_values[self.loc(j, i)]

    # --- ring (child side) -------------------------------------------------

    def _recv_snapshot(self):
        parts = []
        for p, v in self._ring:
            buf = np.empty((3, v[2].size))
            self.world.Recv(buf, source=self.up.ptil.offset + p, tag=self.up.tag_p)
            parts.append(buf)
        return np.concatenate(parts, axis=1) if parts else np.zeros((3, 0))

    def _set_ring(self, snap):
        if not self.break_ring and self._rj.size:
            vals = ops.prolong(snap[0], snap[1], snap[2], self._ring_a, self._ring_b, self.lev.ratio)
            keep = self.wetmask[self._rj, self._ri]
            self.f.all_values[self._rj[keep], self._ri[keep]] = vals[keep]
        self.f.update_halos()

    # --- parent side ----------------------------------------------------------

    def _send_snapshot(self, plan):
        f = self.f.all_values
        sx, sy = ops.limited_slopes(f, self.wetmask)
        reqs = []
        for (p, tt), (pj, pi, *_rest) in plan.pro.items():
            if p == self.t:
                jj, ii = self.loc(pj, pi)
                buf = np.ascontiguousarray(np.stack([f[jj, ii], sx[jj, ii], sy[jj, ii]]))
                reqs.append(self.world.Isend(buf, dest=plan.ctil.offset + tt, tag=plan.tag_p))
        MPI.Request.Waitall(reqs)

    def _recv_restriction(self, plan):
        mine = plan.rim_of[self.t]
        if not len(mine):
            return
        r = plan.child.ratio
        # child cells in removed (all-land) tiles are never sent: weight 0
        blk = np.zeros((len(mine), r, r))
        wgt = np.zeros((len(mine), r, r))
        for (tt, p), (k, a, b, _, _) in plan.res.items():
            if p == self.t:
                buf = np.empty((2, k.size))
                self.world.Recv(buf, source=plan.ctil.offset + tt, tag=plan.tag_r)
                blk[k, a, b] = buf[0]
                wgt[k, a, b] = buf[1]
        if self.break_rim:
            return
        # volume-weighted over wet child cells (ratio 1: the value itself)
        vals = blk[:, 0, 0].copy() if r == 1 else ops.restrict(blk, wgt)
        if r == 1:
            vals[wgt[:, 0, 0] <= 0] = np.nan
        jj, ii = self.loc(mine[:, 0], mine[:, 1])
        ok = np.isfinite(vals)
        self.f.all_values[jj[ok], ii[ok]] = vals[ok]

    def _window_fluxes(self, plan):
        wf = self.lev.wfaces[plan.child.name]
        return np.array([self._face_values(wf[sd]["axis"], wf[sd]["face"][0][m], wf[sd]["face"][1][m])
                         for sd, m in plan.faces_of[self.t]])

    def _match_fluxes(self, plan, fp):
        faces = plan.faces_of[self.t]
        if not faces:
            return
        r = plan.child.ratio
        sub = np.zeros((len(faces), r))  # faces in removed tiles carry no flux
        for (tt, p), (fpos, fa, _, _) in sorted(plan.flx.items()):
            if p == self.t:
                buf = np.empty(fpos.size)
                self.world.Recv(buf, source=plan.ctil.offset + tt, tag=plan.tag_f)
                sub[fpos, fa] = buf
        fc = ops.face_sum(sub)
        wf = self.lev.wfaces[plan.child.name]
        for n, (sd, m) in enumerate(faces):
            w = wf[sd]
            if w["inside"]:
                if not self.break_flux:
                    oj, oi = self.loc(w["out"][0][m], w["out"][1][m])
                    self.f.all_values[oj, oi] += w["sign"] * (fp[n] - fc[n]) / self.content[oj, oi]
            elif self.lev.parent is not None:
                self.reg[sd][w["edge_index"][m]] += fc[n]

    def _couple_down(self, fps):
        for plan in self.down:
            self._send_snapshot(plan)
        for plan, fp in zip(self.down, fps):
            self._recv_restriction(plan)
            if fp is not None:
                self._match_fluxes(plan, fp)
        self.f.update_halos()

    # --- child side: restriction and registers upwards --------------------------

    def _send_up(self, fluxes=True):
        reqs = []
        f = self.f.all_values
        for (tt, p), (_, _, _, cj, ci) in self.up.res.items():
            if tt == self.t:
                jj, ii = self.loc(cj, ci)
                weight = np.where(self.wetmask[jj, ii], self.content[jj, ii], 0.0)
                buf = np.ascontiguousarray(np.stack([np.where(weight > 0, f[jj, ii], 0.0), weight]))
                reqs.append(self.world.Isend(buf, dest=self.up.ptil.offset + p, tag=self.up.tag_r))
        if fluxes:
            for (tt, p), (_, _, fside, fk) in self.up.flx.items():
                if tt == self.t:
                    vals = np.array([self.reg[sd][kk] for sd, kk in zip(fside, fk)])
                    reqs.append(self.world.Isend(vals, dest=self.up.ptil.offset + p, tag=self.up.tag_f))
            for sd in SIDES:
                self.reg[sd][:] = 0.0
        MPI.Request.Waitall(reqs)

    def _own_edge_fluxes(self):
        for sd in SIDES:
            e = self.lev.edge[sd]
            keep = self.own_edge[sd] & self.lev.computed[e["inner"]]
            if keep.any():
                self.reg[sd][keep] += self._face_values(e["axis"], e["face"][0][keep], e["face"][1][keep])

    def _root_outflow(self):
        total = 0.0
        for sd in SIDES:
            e = self.lev.edge[sd]
            keep = self.owned(*e["inner"])
            if keep.any():
                amount = self._face_values(e["axis"], e["face"][0][keep], e["face"][1][keep]).sum()
                total += amount if sd in "EN" else -amount
        return total

    # --- public ---------------------------------------------------------------

    def start(self):
        """Initial exchanges (no transport yet). Call once on every process."""
        if self.up is not None:
            self._s_prev = self._recv_snapshot()
            self._set_ring(self._s_prev)
        self._couple_down([None] * len(self.down))
        if self.up is not None:
            self._send_up(fluxes=False)

    def step(self, advect: Callable):
        """One step of this level. ``advect(flux_u, flux_v)`` must advance the
        tracer by one step and add the transported amounts to flux_u/flux_v
        (e.g. ``Advection(u, v, dt, f, flux_u=flux_u, flux_v=flux_v)``)."""
        up, r = self.up, self.lev.ratio
        k = self._k % r
        if up is not None:
            if k == 0:
                self._s_new = self._recv_snapshot()
            self._set_ring(ops_interp(self._s_prev, self._s_new, k / r))
        self.flux["x"].all_values[...] = 0.0
        self.flux["y"].all_values[...] = 0.0
        advect(self.flux["x"], self.flux["y"])
        self.f.update_halos()
        fps = [self._window_fluxes(plan) for plan in self.down]
        if up is not None:
            self._own_edge_fluxes()
            self._set_ring(ops_interp(self._s_prev, self._s_new, (k + 1) / r))
        else:
            self.outflow += self._root_outflow()
        self._couple_down(fps)
        if up is not None and k == r - 1:
            self._s_prev = self._s_new
            self._send_up()
        self._k += 1

    def level_content(self) -> float:
        """Tracer content of the cells this level computes (this process)."""
        h = self.halo
        return float((self.f.all_values * self.content)[h:-h, h:-h][self.computed].sum())


def ops_interp(s0, s1, a):
    return (1.0 - a) * s0 + a * s1
