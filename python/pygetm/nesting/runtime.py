"""Running a nesting hierarchy: processes, domains, exchanges.

:class:`NestingRun` places every level on its own group of MPI processes
(:func:`pygetm.nesting.split_world`), builds the level's pyGETM grids with the
ring band and the hole marked external (:attr:`pygetm.domain.Domain.external`),
shares the tilings of all levels and builds the exchange plans
(:class:`pygetm.nesting.plan.Plan`).

:class:`NestedTracer` exchanges one tracer between the levels. Schedule
"sync" (default): all levels step at the same time and exchange at the end
of every parent step (the ring is held at the parent's state of the last
sync). The old "sequential" schedule, kept for comparison:

* the child's ring from the parent (limited linear reconstruction of the
  parent's (c, sx, sy) snapshot, interpolated in time over the child's
  sub-steps), at the start and the end of every child step;
* the parent's rim from the child (volume-weighted mean over wet child cells),
  once per parent step;
* flux matching: every level keeps a register of the transports through its
  edge faces (its interface with the parent) over its sub-steps; after the
  child's sub-steps the parent corrects its cells next to the hole by
  (parent - child) transport, and registers of children on the parent's own
  interface are passed upwards.

The transports are the tracer amounts the advection actually moved through
every face (``Advection(..., flux_u=, flux_v=)``). The time loop belongs to
the caller: every level calls :meth:`NestedTracer.step` once per own step, in
the same order on all ranks; MPI messages enforce the order between levels.
"""

from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
from mpi4py import MPI

from .geometry import Level
from .hierarchy import NestingContext, split_world
from .plan import LevelTiling, Plan, gather_tilings
from . import operators as ops


def install_abort_hook(comm: MPI.Comm, label: str):
    """Make an unhandled exception on any process abort the whole run.

    The levels of a nesting run wait for each other in MPI: if one level
    stops with an exception, the others would wait forever. The hook prints
    the traceback with the level and rank, then calls ``comm.Abort``."""
    import sys
    import traceback

    previous = sys.excepthook

    def hook(etype, value, tb):
        traceback.print_exception(etype, value, tb)
        sys.stderr.write(f"[nesting: {label}, rank {comm.rank}] unhandled exception - aborting all levels\n")
        sys.stderr.flush()
        try:
            comm.Abort(1)
        finally:
            previous(etype, value, tb)

    sys.excepthook = hook


class NestingRun:
    """Processes, grids and exchange plans of a nesting hierarchy (one level
    per group of processes). With ``abort_on_error`` (default), an unhandled
    exception on any process aborts all levels (:func:`install_abort_hook`)."""

    def __init__(self, levels: List[Level], ranks: Optional[Mapping[str, int]] = None, comm: Optional[MPI.Comm] = None,
                 abort_on_error: bool = True, interface_comms: bool = True,
                 context: Optional[NestingContext] = None):
        self.levels = levels
        if context is None:
            ctx = split_world({l.name: int(ranks[l.name]) for l in levels}, comm=comm)
        else:  # e.g. from split_mpmd (one mpiexec component per level)
            ctx = context
            missing = [l.name for l in levels if l.name not in ctx.sizes]
            if missing:
                raise ValueError(f"no processes for level(s) {missing}: {dict(ctx.sizes)}")
        self.world = ctx.nesting_comm
        self.comm = ctx.comm
        self.offsets = ctx.offsets
        self.sizes = ctx.sizes
        #: exchanges between a parent and a child on a communicator of their
        #: own (the parent's processes, then the child's); False: on the
        #: shared nesting communicator (as before, for comparison)
        self.interface_comms = interface_comms
        self.level = next(l for l in levels if l.name == ctx.name)
        if abort_on_error:
            install_abort_hook(self.world, self.level.name)
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
        with self._tiling_hint():
            grid = domain.create_grids(nz, halox=lev.halo, haloy=lev.halo, **grid_kwargs)
        return self._setup(grid)

    def build_simulation(self, make_domain: Callable, make_simulation: Callable):
        """Build this level's :class:`pygetm.Simulation`. ``make_domain(level,
        comm)`` returns the level's domain including its band;
        ``make_simulation(level, domain)`` the simulation (its halo must equal
        ``level.halo``). The band and the hole are marked external before the
        simulation builds its grids. Returns the simulation."""
        lev = self.level
        domain = make_domain(lev, self.comm)
        domain.external = lev.external()
        with self._tiling_hint():
            sim = make_simulation(lev, domain)
        if sim.T is not None and sim.T.halox != lev.halo:
            raise ValueError(f"simulation halo {sim.T.halox} differs from nesting halo {lev.halo}")
        self._setup(sim.T)
        return sim

    def _tiling_hint(self):
        """Context: a failing subdomain division names the level and its
        number of processes (then abort_on_error stops all levels)."""
        import contextlib

        run = self

        @contextlib.contextmanager
        def ctx():
            try:
                yield
            except Exception as e:
                if "subdomain" in str(e).lower() or "decompos" in str(e).lower():
                    raise RuntimeError(
                        f"level {run.level.name}: no subdomain division for {run.comm.size} processes"
                        f" ({run.level.nx} x {run.level.ny} cells) - use fewer processes for this level"
                        " (see nesting-balance in pygetm-nesting)") from e
                raise

        return ctx()

    def _setup(self, grid):
        lev = self.level
        self.grid = grid
        self.tilings = gather_tilings(self.levels, self.world, lev, self.grid, self.offsets)
        links = {c.name: k for k, c in enumerate(self.levels[1:])}
        til = self.tilings[lev.name]
        if lev.parent is not None:
            self.up = Plan(lev, self.tilings[lev.parent.name], til, links[lev.name])
        self.down = [Plan(c, til, self.tilings[c.name], links[c.name]) for c in lev.children]
        self._interface_comms()
        return self.grid

    def _interface_comms(self):
        """Give every plan of this process the communicator of its interface
        (``plan.comm``) and the ranks of the parent's and the child's tile 0
        in it (``plan.poff``, ``plan.coff``). Interfaces are created in the
        same (layout) order on all processes; each creation is collective
        over that interface's processes only."""
        lev = self.level
        if not self.interface_comms:
            for plan in ([self.up] if self.up is not None else []) + self.down:
                plan.comm, plan.poff, plan.coff = self.world, plan.ptil.offset, plan.ctil.offset
            return
        group = self.world.Get_group()
        block = lambda name: list(range(self.offsets[name], self.offsets[name] + self.sizes[name]))
        for k, child in enumerate(self.levels[1:]):
            par = child.parent
            if lev is not par and lev is not child:
                continue
            g = group.Incl(block(par.name) + block(child.name))
            comm = self.world.Create_group(g, tag=k)
            g.Free()
            plan = self.up if lev is child else self.down[lev.children.index(child)]
            plan.comm, plan.poff, plan.coff = comm, 0, self.sizes[par.name]
        group.Free()

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

    def __init__(self, run: NestingRun, f, break_ring=False, break_rim=False, break_flux=False,
                 schedule: str = "sync"):
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
        #: "sync" (default): all levels step at the same time, exchange at the
        #: end of every parent step, the ring is held at the parent's state of
        #: the last sync; "sequential": the old order (child after parent, ring
        #: interpolated in time), kept for comparison
        if schedule not in ("sync", "sequential"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.schedule = schedule
        self.wetmask = T.mask.all_values != 0
        self.content = T.D.all_values * T.area.all_values  # per unit concentration
        h = T.halox
        self.computed = T.mask.all_values[h:-h, h:-h] == 1
        self.halo = h
        # tracer amount through every face during a step (from the advection)
        self.flux = {"x": T.ugrid.array(fill=0.0), "y": T.vgrid.array(fill=0.0)}
        # this level's flux register (edge faces), kept by the owner of the
        # inner edge cell
        self.reg = np.zeros(lev.edge["is_x"].size)
        self.own_edge = self.owned(*lev.edge["inner"])
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

    def _face_values(self, is_x, j, i):
        jj, ii = self.loc(j, i)
        out = np.empty(np.shape(is_x))
        out[is_x] = self.flux["x"].all_values[jj[is_x], ii[is_x]]
        out[~is_x] = self.flux["y"].all_values[jj[~is_x], ii[~is_x]]
        return out

    # --- ring (child side) -------------------------------------------------

    def _recv_snapshot(self):
        parts = []
        for p, v in self._ring:
            buf = np.empty((3, v[2].size))
            self.up.comm.Recv(buf, source=self.up.poff + p, tag=self.up.tag_p)
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
                reqs.append(plan.comm.Isend(buf, dest=plan.coff + tt, tag=plan.tag_p))
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
                plan.comm.Recv(buf, source=plan.coff + tt, tag=plan.tag_r)
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
        m = plan.faces_of[self.t]
        return self._face_values(wf["is_x"][m], wf["face"][0][m], wf["face"][1][m])

    def _match_fluxes(self, plan, fp):
        faces = plan.faces_of[self.t]
        if not faces.size:
            return
        r = plan.child.ratio
        sub = np.zeros((faces.size, r))  # faces in removed tiles carry no flux
        for (tt, p), (fpos, fa, _) in sorted(plan.flx.items()):
            if p == self.t:
                buf = np.empty(fpos.size)
                plan.comm.Recv(buf, source=plan.coff + tt, tag=plan.tag_f)
                sub[fpos, fa] = buf
        fc = ops.face_sum(sub)
        wf = self.lev.wfaces[plan.child.name]
        # all faces at once; np.add.at adds in face order, as a loop would
        ins = wf["inside"][faces]
        m_in = faces[ins]
        if m_in.size and not self.break_flux:
            oj, oi = self.loc(wf["out"][0][m_in], wf["out"][1][m_in])
            np.add.at(self.f.all_values, (oj, oi), wf["sign"][m_in] * (fp[ins] - fc[ins]) / self.content[oj, oi])
        up = ~ins & (wf["up"][faces] >= 0)
        if up.any():
            np.add.at(self.reg, wf["up"][faces][up], fc[up])

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
                reqs.append(self.up.comm.Isend(buf, dest=self.up.poff + p, tag=self.up.tag_r))
        if fluxes:
            for (tt, p), (_, _, fk) in self.up.flx.items():
                if tt == self.t:
                    vals = np.ascontiguousarray(self.reg[fk])
                    reqs.append(self.up.comm.Isend(vals, dest=self.up.poff + p, tag=self.up.tag_f))
            self.reg = np.zeros_like(self.reg)
        MPI.Request.Waitall(reqs)

    def _own_edge_fluxes(self):
        e = self.lev.edge
        keep = self.own_edge & self.lev.computed[e["inner"]]
        if keep.any():
            self.reg[keep] += self._face_values(e["is_x"][keep], e["face"][0][keep], e["face"][1][keep])

    def _root_outflow(self):
        e = self.lev.edge
        keep = self.own_edge
        if not keep.any():
            return 0.0
        amount = self._face_values(e["is_x"][keep], e["face"][0][keep], e["face"][1][keep])
        return float((e["sign"][keep] * amount).sum())

    # --- public ---------------------------------------------------------------

    def _sync(self, fps, with_parent: bool, fluxes: bool):
        """One sync point: (a) restriction and flux matching from the
        children, (b) restriction and registers to the parent, (c) the
        parent's corrected state for the ring, (d) this level's state to the
        children (an order without cycles over the hierarchy)."""
        for plan, fp in zip(self.down, fps):
            self._recv_restriction(plan)
            if fp is not None:
                self._match_fluxes(plan, fp)
        if self.down:
            self.f.update_halos()
        if with_parent:
            self._send_up(fluxes=fluxes)
            self._s_hold = self._recv_snapshot()
            self._set_ring(self._s_hold)
        for plan in self.down:
            self._send_snapshot(plan)

    def start(self):
        """Initial exchanges (no transport yet). Call once on every process."""
        if self.schedule == "sync":
            self._sync([None] * len(self.down), self.up is not None, fluxes=False)
            return
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
        up, r = self.up, self.lev.time_ratio  # sub-steps per parent step
        k = self._k % r
        if self.schedule == "sync":
            if up is not None:
                self._set_ring(self._s_hold)
            self.flux["x"].all_values[...] = 0.0
            self.flux["y"].all_values[...] = 0.0
            advect(self.flux["x"], self.flux["y"])
            self.f.update_halos()
            fps = [self._window_fluxes(plan) for plan in self.down]
            if up is not None:
                self._own_edge_fluxes()
            else:
                self.outflow += self._root_outflow()
            self._sync(fps, up is not None and k == r - 1, fluxes=True)
            self._k += 1
            return
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
