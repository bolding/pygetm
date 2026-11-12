"""Two-way nesting of 2D barotropic pyGETM simulations (z, U, V).

All models step at the same time; at the end of every parent step they
meet at a sync point: children send the halo inside the hole and their flux
registers, the parent matches the fluxes of the same step and sends its
corrected state; the child continues its ring from the parent's last two
syncs (linear in time) over its next ``time_ratio`` sub-steps. Models only
wait in MPI at the syncs.

Every parent and child is a :class:`pygetm.Simulation` with
``runtype=BAROTROPIC_2D``; :class:`Nested2D` is a
:class:`pygetm.simulation.Coupler` attached to each of them. Per parent step n
(t_n -> t_n+1):

* parent, after its state update: receive the children's restriction
  (z: area-weighted mean over wet child cells; U, V: mean over the child
  faces along each parent face) for the rim, and their volume flux
  registers; correct z of the parent cells next to the window by
  (parent - child) volume through each shared face, and continue from the
  child's mean transport at those faces (without this, the correction loop
  is unstable in deep water); then send its state to the children: z with
  limited slopes at the parent cells covering the child's ring, and the
  transports U, V at the parent faces around the child's ring faces. All
  this happens before pyGETM recomputes depths and pressure gradients from z.
* child, sub-step k of r (r = time_ratio, steps per parent step):
  - after its momentum update: ring U, V at its half step, extrapolated
    linearly from the parent's last two syncs; in space linear across and
    constant along parent faces;
  - after its elevation update: ring z at its time, the same way; add the
    volume through its edge faces (U dy dt) to its flux register;
  - after the last sub-step: send restriction and register to the parent.

Faces with ring (or hole) cells on both sides are EXTERNAL
(:attr:`pygetm.domain.Domain.external`), so the model keeps the transports
set there; faces between ring and computed cells are computed by the level
itself and are the interface faces matched by the flux registers.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI

from ..simulation import Coupler
from .geometry import Level
from .plan import LevelTiling
from .runtime import NestingRun, ops_interp
from . import operators as ops



class HaloWaitTimer:
    """Time blocked in MPI_Waitall of pyGETM's halo updates (all of them: the
    model's own and those the nesting calls), on this process.

    pyGETM's :class:`pygetm.parallel.HaloUpdater` calls the module-level
    ``Waitall`` of :mod:`pygetm.parallel`; :meth:`install` replaces it by a
    timed version (two MPI.Wtime calls per halo update). Installed by
    :class:`Nested2D`, once per process."""

    total = 0.0
    _installed = False

    @classmethod
    def install(cls):
        if cls._installed:
            return
        from .. import parallel

        waitall = parallel.Waitall

        def timed_waitall(reqs, *args, **kwargs):
            t0 = MPI.Wtime()
            try:
                return waitall(reqs, *args, **kwargs)
            finally:
                cls.total += MPI.Wtime() - t0

        parallel.Waitall = timed_waitall
        cls._installed = True


class ExchangeClock:
    """Wall-clock time of the nesting exchanges of one process, split into
    waiting (blocked until the partner's message is there: MPI_Probe before
    every receive, and the completion of sends) and transfer (the receive
    itself); and the halo updates the nesting calls (``halo``: all of it,
    ``halo_wait``: the part blocked in MPI_Waitall, i.e. waiting for the
    neighbouring tiles of the same level)."""

    def __init__(self):
        HaloWaitTimer.install()
        self.wait = 0.0
        self.transfer = 0.0
        self.halo = 0.0
        self.halo_wait = 0.0
        self.halo_wait0 = HaloWaitTimer.total

    def halos(self, *fields):
        """Halo update of *fields* (pyGETM arrays), timed."""
        t0, w0 = MPI.Wtime(), HaloWaitTimer.total
        for f in fields:
            f.update_halos()
        self.halo += MPI.Wtime() - t0
        self.halo_wait += HaloWaitTimer.total - w0

    def model_halo_wait(self) -> float:
        """Time blocked in the model's own halo updates (not the nesting's)
        since the last reset."""
        return max(HaloWaitTimer.total - self.halo_wait0 - self.halo_wait, 0.0)

    def recv(self, comm, buf, source: int, tag: int):
        t0 = MPI.Wtime()
        comm.Probe(source=source, tag=tag)
        t1 = MPI.Wtime()
        comm.Recv(buf, source=source, tag=tag)
        self.wait += t1 - t0
        self.transfer += MPI.Wtime() - t1

    def waitall(self, reqs):
        t0 = MPI.Wtime()
        MPI.Request.Waitall(reqs)
        self.wait += MPI.Wtime() - t0

    def reset(self):
        self.wait = self.transfer = self.halo = self.halo_wait = 0.0
        self.halo_wait0 = HaloWaitTimer.total


class PointTransfer:
    """Weighted transfer from points of one level to points of another, on
    their tilings: value(dst n) = sum_k w[n, k] * value(src[n, k]). Points
    are (j, i) indices of T cells or of the U/V faces east/north of them;
    the owner of a point is the owner of its T cell. Sums are formed in a
    fixed order (k), so results do not depend on the decomposition."""

    def __init__(self, src_til: LevelTiling, dst_til: LevelTiling, dst, src, weights, tag: int, label: str,
                 src_off: Optional[int] = None, dst_off: Optional[int] = None):
        dj, di = (np.asarray(x) for x in dst)
        #: ranks of tile 0 of source and destination on the exchange
        #: communicator (default: the tilings' offsets on the shared one)
        self.src_off = src_til.offset if src_off is None else src_off
        self.dst_off = dst_til.offset if dst_off is None else dst_off
        sj, si = (np.asarray(x) for x in src)
        w = np.asarray(weights, dtype=float)
        self.src_til, self.dst_til, self.tag, self.label = src_til, dst_til, tag, label
        self.K = w.shape[1]
        dst_own = dst_til.owner(dj, di)
        src_own = src_til.owner(sj, si)
        self.dst_of: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.pairs: Dict[Tuple[int, int], tuple] = {}
        for t in range(dst_til.ntiles):
            idx = np.nonzero(dst_own == t)[0]
            self.dst_of[t] = (dj[idx], di[idx], w[idx])
            if not idx.size:
                continue
            pos, k = np.meshgrid(np.arange(idx.size), np.arange(self.K), indexing="ij")
            pos, k = pos.ravel(), k.ravel()
            own = src_own[idx][pos, k]
            use = (own >= 0) & (w[idx][pos, k] != 0)
            for p in np.unique(own[use]):
                sel = use & (own == p)
                self.pairs[(int(p), t)] = (pos[sel], k[sel], sj[idx][pos[sel], k[sel]], si[idx][pos[sel], k[sel]])

    def send(self, world, me: int, values: np.ndarray, loc):
        """Source side: send this tile's values (an ``all_values`` array)."""
        reqs = []
        for (p, t), (_, _, sj, si) in self.pairs.items():
            if p == me:
                buf = np.ascontiguousarray(values[loc(sj, si)])
                reqs.append(world.Isend(buf, dest=self.dst_off + t, tag=self.tag))
        return reqs

    def recv(self, world, me: int, clock: Optional[ExchangeClock] = None) -> np.ndarray:
        """Destination side: weighted values at this tile's points."""
        dj, di, w = self.dst_of.get(me, (np.zeros(0, int),) * 2 + (np.zeros((0, self.K)),))
        vals = np.zeros((dj.size, self.K))
        for (p, t), (pos, k, sj, _) in sorted(self.pairs.items()):
            if t == me:
                buf = np.empty(sj.size)
                if clock is None:
                    world.Recv(buf, source=self.src_off + p, tag=self.tag)
                else:
                    clock.recv(world, buf, self.src_off + p, self.tag)
                vals[pos, k] = buf
        out = np.zeros(dj.size)
        for kk in range(self.K):
            out = out + w[:, kk] * vals[:, kk]
        return out

    def points(self, me: int):
        dj, di, _ = self.dst_of.get(me, (np.zeros(0, int), np.zeros(0, int), None))
        return dj, di

    def describe(self, me: int, role: str, tile: int) -> List[str]:
        out = []
        for (p, t), v in self.pairs.items():
            if role == "src" and p == tile:
                out.append(f"[rank {me}]  {self.label}  local[points]  → rank {self.dst_til.offset + t}  tag={self.tag}  n={v[0].size}")
            if role == "dst" and t == tile:
                out.append(f"[rank {me}]  {self.label}  from rank {self.src_til.offset + p}  tag={self.tag}  n_parent={v[0].size}")
        return out


def _outside_masks(lev: Level):
    """Lookups (global j, i; band included) of the non-active cells and of
    the ring cells of a child."""
    b = lev.band
    act = np.zeros(lev.extended_shape, dtype=bool)
    act[b : b + lev.ny, b : b + lev.nx] = lev.cactive
    ring = lev._ring_mask()
    return (lambda j, i: ~act[j + b, i + b]), (lambda j, i: ring[j + b, i + b])


def ring_faces(child: Level, axis: str):
    """Child U (axis x) or V (axis y) points with non-active cells on both
    sides, at least one of them a ring cell, their parent source faces (2 per
    point) and weights: linear across, constant along the parent faces."""
    b, r = child.band, child.ratio
    i0, _, j0, _ = child.window
    outside, ring = _outside_masks(child)
    if axis == "x":
        j, i = np.meshgrid(np.arange(-b, child.ny + b), np.arange(-b, child.nx + b - 1), indexing="ij")
        keep = outside(j, i) & outside(j, i + 1) & (ring(j, i) | ring(j, i + 1))
        j, i = j[keep], i[keep]
        fl, rem = np.floor_divide(i + 1, r), np.mod(i + 1, r)
        fw = i0 + fl - 1
        pj = j0 + np.floor_divide(j, r)
        src_j = np.stack([pj, pj], axis=1)
        src_i = np.stack([fw, fw + 1], axis=1)
    else:
        j, i = np.meshgrid(np.arange(-b, child.ny + b - 1), np.arange(-b, child.nx + b), indexing="ij")
        keep = outside(j, i) & outside(j + 1, i) & (ring(j, i) | ring(j + 1, i))
        j, i = j[keep], i[keep]
        fl, rem = np.floor_divide(j + 1, r), np.mod(j + 1, r)
        fs = j0 + fl - 1
        pi = i0 + np.floor_divide(i, r)
        src_j = np.stack([fs, fs + 1], axis=1)
        src_i = np.stack([pi, pi], axis=1)
    w1 = rem / r
    weights = np.stack([1.0 - w1, w1], axis=1)
    return (j, i), (src_j, src_i), weights


def rim_faces(child: Level, axis: str):
    """Parent U/V points with hole cells on both sides and at least one rim
    cell, their child source faces (r per point, mean)."""
    par, r = child.parent, child.ratio
    i0, i1, j0, j1 = child.window
    rim = np.zeros((par.ny, par.nx), dtype=bool)
    rim[par.rims[child.name][:, 0], par.rims[child.name][:, 1]] = True
    hole = par.hole_of(child)
    a = np.arange(r)
    if axis == "x":
        pj, pf = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1 - 1), indexing="ij")
        keep = hole[pj, pf] & hole[pj, pf + 1] & (rim[pj, pf] | rim[pj, pf + 1])
        pj, pf = pj[keep], pf[keep]
        src_j = (pj - j0)[:, None] * r + a[None, :]
        src_i = np.repeat(((pf - i0 + 1) * r - 1)[:, None], r, axis=1)
    else:
        pf, pi = np.meshgrid(np.arange(j0, j1 - 1), np.arange(i0, i1), indexing="ij")
        keep = hole[pf, pi] & hole[pf + 1, pi] & (rim[pf, pi] | rim[pf + 1, pi])
        pf, pi = pf[keep], pi[keep]
        src_j = np.repeat(((pf - j0 + 1) * r - 1)[:, None], r, axis=1)
        src_i = (pi - i0)[:, None] * r + a[None, :]
        pj, pf = pf, pi
    weights = np.full(src_j.shape, 1.0 / r)
    return (pj, pf), (src_j, src_i), weights


def face_transfers(child: Level, plan, link: int, tag_base: int = 600):
    """U/V ring (parent -> child) and rim (child -> parent) transfers of the
    interface of *plan* (on ``plan.comm``)."""
    ptil, ctil, po, co = plan.ptil, plan.ctil, plan.poff, plan.coff
    out = {}
    for n, axis in enumerate("xy"):
        dst, src, w = ring_faces(child, axis)
        out[f"ring_{axis}"] = PointTransfer(ptil, ctil, dst, src, w, tag_base + 4 * link + n, "UV"[n] + "r", po, co)
        dst, src, w = rim_faces(child, axis)
        out[f"rim_{axis}"] = PointTransfer(ctil, ptil, dst, src, w, tag_base + 4 * link + 2 + n, "UV"[n] + "R", co, po)
    return out


class Nested2D(Coupler):
    """Coupler exchanging z, U, V of a 2D barotropic simulation with its
    parent and children (see module documentation)."""

    def __init__(self, nest: NestingRun, sim, sponge_width=0.0, sponge_alpha=0.1):
        self.nest, self.sim = nest, sim
        self.lev = lev = nest.level
        self.t = nest.tile
        self.til = nest.tilings[lev.name]
        self.world = nest.world
        self.up, self.down = nest.up, nest.down
        #: sponge (child levels; see :mod:`pygetm.nesting.sponge`): damping of
        #: the transports over ``sponge_width`` parent cells next to the
        #: interface (counted through water), with An dt / dx^2 =
        #: ``sponge_alpha`` at the interface, decaying linearly inward. Added
        #: to any An already set. 0 (default): no sponge. Needed for stability
        #: with time_ratio > 1 and flux matching.
        self.sponge_width, self.sponge_alpha = float(sponge_width), float(sponge_alpha)
        if self.sponge_width < 0.0 or self.sponge_alpha < 0.0:
            raise ValueError("sponge_width and sponge_alpha must be >= 0")
        self._snap_old = None
        T = sim.T
        self.z, self.U, self.V = T.z, sim.momentum.U, sim.momentum.V
        self.wetmask = T.mask.all_values != 0
        self.area = T.area.all_values
        self.H = T.H.all_values
        self.dyu, self.dxv = T.ugrid.dy.all_values, T.vgrid.dx.all_values
        links = {c.name: k for k, c in enumerate(nest.levels[1:])}
        self.up_faces = None
        if lev.parent is not None:
            self.up_faces = face_transfers(lev, self.up, links[lev.name])
        self.down_faces = [face_transfers(c, plan, links[c.name]) for c, plan in zip(lev.children, self.down)]
        self.reg = np.zeros(lev.edge["is_x"].size)
        self.own_edge = self.til.owner(*lev.edge["inner"]) == self.t
        if self.up is not None:
            ring = [(p, v) for (p, tt), v in self.up.pro.items() if tt == self.t]
            cat = lambda k: np.concatenate([v[k] for _, v in ring]) if ring else np.zeros(0, int)
            self._ring = ring
            self._ring_a, self._ring_b = cat(4), cat(5)
            self._rj, self._ri = self.loc(cat(2), cat(3))
            self._ru = self.loc(*self.up_faces["ring_x"].points(self.t))
            self._rv = self.loc(*self.up_faces["ring_y"].points(self.t))
            #: lowest ring elevation: the bed + Dmin (drying)
            self._ring_zmin = -T.H.all_values[self._rj, self._ri] + float(sim.Dmin)
            self._dmin = float(sim.Dmin)
        self._k = 0
        self._fps = []
        self._restart_arrays()
        self._phase_array()
        #: wall-clock monitoring (seconds, this process): time in the syncs
        #: with parent and children (waiting in MPI + the exchange itself)
        #: and the elapsed time since the end of on_start
        self.t_sync = 0.0
        self._t_start = None
        #: waiting and transfer within the syncs (MPI calls of the exchanges)
        self.clock = ExchangeClock()
        #: time in the coupler's own work during the model step (ring values,
        #: window volumes, edge registers)
        self.t_stage = 0.0
        #: largest elevation correction (m) by flux matching so far (diagnostic)
        self.max_correction = 0.0
        self.max_correction_info = None
        #: drying (diagnostics, parent side): the least water depth (m) of a
        #: parent cell after the flux-matching correction, and of a rim cell
        #: after the restriction, so far
        self.min_depth_corrected = np.inf
        self.min_depth_corrected_info = None
        self.min_depth_rim = np.inf
        #: least water depth (m) the ring has had (diagnostic; ring cells at the
        #: child's bed + Dmin are dry: the interface has fallen dry)
        self.min_depth_ring = np.inf

    # --- restart ----------------------------------------------------------------

    #: the parent's ring state at the sync before the last one (``_snap_old``,
    #: for the extrapolation in time): z with its slopes on the T grid at the
    #: ring cells, U/V at the ring faces. Everything else
    #: the coupler needs is rebuilt by the initial sync of :meth:`on_start` from
    #: the restarted model state (restart files are written after the sync).
    _RESTART = (("T", "z"), ("T", "zsx"), ("T", "zsy"), ("U", "u"), ("V", "v"))

    def _restart_arrays(self):
        """Arrays (part of the model state, so in pyGETM's restart files) with
        the previous ring snapshot, and ``nest_ring_old_set`` (1 where it is
        stored, 0 before the first sync with fluxes: a cold start)."""
        from ..constants import CellType

        self._rs = None
        if self.up is None:
            return
        T = self.sim.T
        grids = {"T": T, "U": T.ugrid, "V": T.vgrid}
        attrs = dict(_part_of_state=True, _valid_at=tuple(CellType))  # nothing masked: exact values back
        self._rs = [grids[g].array(name=f"nest_ring_{n}_old", fill=0.0,
                                   long_name=f"nesting: parent ring state {n} at the previous sync", attrs=dict(attrs))
                    for g, n in self._RESTART]
        self._rs_set = T.array(name="nest_ring_old_set", fill=0.0,
                               long_name="nesting: previous ring state stored (1) or not (0)", attrs=dict(attrs))

    def _phase_array(self):
        """``nest_root_phase``: this level's steps since the last root 3D step
        (in a 2D run: the last root step), part of the model state. The nesting
        restarts its sub-step counters, sums and registers at a root 3D step, so
        a restart file written in the middle of one is refused (:meth:`on_start`).
        Optional: a file without it (older, or from upstream pyGETM) loads
        unchecked."""
        from ..constants import CellType

        self._phase = self.sim.T.array(
            name="nest_root_phase", fill=0.0,
            long_name="nesting: steps of this level since the last root 3D step",
            attrs=dict(_part_of_state=True, _restart_optional=True, _valid_at=tuple(CellType)))
        self._period = 1
        #: ``nest_layout_id``: :meth:`layout_signature` of this level, part of the model state, so that a restart file
        #: written with another set of levels around this one is refused (:meth:`on_start`). Optional: a file without
        #: it loads unchecked (the array keeps this run's own value).
        self._layout_id = self.sim.T.array(
            name="nest_layout_id", fill=self.layout_signature(),
            long_name="nesting: signature of this level, its parent and its children (geometry, active regions)",
            attrs=dict(_part_of_state=True, _restart_optional=True, _valid_at=tuple(CellType)))

    def layout_signature(self) -> float:
        """CRC32 (exact in a double) of what this level's state depends on: its own grid and window, its parent's grid,
        and its children (names, windows, ratios, time ratios, active regions). A parent's former hole holds no valid
        state once a child is left out, and a child's ring cells move with its window."""
        import zlib

        def one(lev, with_active):
            geo = (lev.name, lev.parent.name if lev.parent is not None else "", lev.window, lev.ratio, lev.time_ratio,
                   lev.nx, lev.ny, getattr(lev, "lon0", None), getattr(lev, "lat0", None),
                   getattr(lev, "dlon", None), getattr(lev, "dlat", None))
            b = repr(geo).encode()
            act = getattr(lev, "active", None) if with_active else None
            if act is not None:
                b += np.ascontiguousarray(np.asarray(act, dtype=np.uint8)).tobytes()
            return b

        lev = self.lev
        data = one(lev, True) + (one(lev.parent, False) if lev.parent is not None else b"")
        data += b"".join(one(c, True) for c in lev.children)
        return float(zlib.crc32(data))

    def _clean_external_after_restart(self, sim):
        """After a restart: the restart file holds NaN where the writer masks the
        field, which includes this level's EXTERNAL cells inside the domain (a
        parent's hole, a child's band); a continuous run has finite values there.
        Nothing computed uses them (the rim and ring are set by the syncs), but
        diagnostics such as Coriolis are evaluated there and pyGETM's start-up
        check would stop at the NaN. Set them to 0."""
        from ..constants import CellType

        if not getattr(sim, "_restarted", False):
            return
        for field in sim.output_manager.fields.values():
            if not field.attrs.get("_part_of_state", False):
                continue
            a = getattr(field, "array", field)
            if not hasattr(a, "grid") or not hasattr(a, "all_values") or a.ndim == 0 or getattr(a, "on_boundary", False):
                continue
            ext = a.grid.mask.all_values == CellType.EXTERNAL
            v = a.all_values
            bad = ~np.isfinite(v) & np.broadcast_to(ext, v.shape)
            v[bad] = 0.0

    def _check_restart_phase(self, sim):
        """Steps of this level per root 3D step (the longest 3D step of all
        levels: the root's); after a restart, all levels stop if any file was
        written in the middle of a root 3D step."""
        from ..constants import RunType

        own = sim.timestep if sim.runtype == RunType.BAROTROPIC_2D else sim.macrotimestep
        root_macro = self.world.allreduce(float(own), op=MPI.MAX)
        self._period = max(1, int(round(root_macro / sim.timestep)))
        phase = 0
        if getattr(sim, "_restarted", False):
            v = self._phase.all_values
            v = v[np.isfinite(v)]  # a restart file holds NaN on land
            phase = int(round(float(v.max()))) if v.size else 0
        bad = sorted({(n, p, P) for n, p, P in self.world.allgather((self.lev.name, phase, self._period)) if p})
        if bad:
            raise ValueError(
                "nesting: restart file(s) written in the middle of a root 3D step ("
                + ", ".join(f"{n}: {p} of {P} steps" for n, p, P in bad)
                + "); write restarts at whole root 3D steps")
        # a restart file written with another set of levels around a level (a child left out or added, a window moved)
        own_id = self.layout_signature()
        changed = False
        if getattr(sim, "_restarted", False):
            v = self._layout_id.all_values
            v = v[np.isfinite(v)]
            changed = bool(v.size) and float(v.max()) != own_id
        self._layout_id.fill(own_id)
        moved = sorted({n for n, c in self.world.allgather((self.lev.name, changed)) if c})
        if moved:
            raise ValueError(
                "nesting: restart file(s) of " + ", ".join(moved) + " written with another set of levels around them (a "
                "child left out or added, or a window, ratio or active region changed): a parent's former hole holds no "
                "valid state. Start these levels from initial conditions, or restart with the levels the files were "
                "written with")

    def _restart_points(self):
        return (self._rj, self._ri), (self._rj, self._ri), (self._rj, self._ri), self._ru, self._rv

    def _snap_to_restart(self):
        """Keep the restart arrays equal to ``_snap_old``."""
        if self._rs is None:
            return
        if self._snap_old is None:
            self._rs_set.all_values[...] = 0.0
            return
        z, u, v = self._snap_old
        for a, pts, vals in zip(self._rs, self._restart_points(), (z[0], z[1], z[2], u, v)):
            a.all_values[pts] = vals
        self._rs_set.all_values[...] = 1.0

    def _snap_from_restart(self):
        """``_snap_old`` from the restart arrays if a restart stored one (else
        None, as at a cold start)."""
        if self._rs is None or not (self._rs_set.all_values > 0.5).any():
            return
        vals = [a.all_values[pts].copy() for a, pts in zip(self._rs, self._restart_points())]
        self._snap_old = (np.stack(vals[:3]), vals[3], vals[4])

    # --- helpers --------------------------------------------------------------

    def loc(self, j, i):
        return self.til.local(self.t, j, i)

    def _dt(self):
        return self.sim.timestep

    def _face_volume(self, is_x, j, i):
        jj, ii = self.loc(j, i)
        out = np.empty(np.shape(is_x))
        x, y = is_x, ~is_x
        out[x] = self.U.all_values[jj[x], ii[x]] * self.dyu[jj[x], ii[x]] * self._dt()
        out[y] = self.V.all_values[jj[y], ii[y]] * self.dxv[jj[y], ii[y]] * self._dt()
        return out

    # --- child side -------------------------------------------------------------

    def _recv_snapshot(self):
        parts = []
        for p, v in self._ring:
            buf = np.empty((3, v[2].size))
            self.clock.recv(self.up.comm, buf, self.up.poff + p, self.up.tag_p)
            parts.append(buf)
        z = np.concatenate(parts, axis=1) if parts else np.zeros((3, 0))
        u = self.up_faces["ring_x"].recv(self.up.comm, self.t, self.clock)
        v = self.up_faces["ring_y"].recv(self.up.comm, self.t, self.clock)
        return z, u, v

    def _set_ring_z(self, snap, halos=True):
        if self._rj.size:
            vals = ops.prolong(snap[0], snap[1], snap[2], self._ring_a, self._ring_b, self.lev.ratio)
            # drying: the parent's (block mean) elevation can lie below a ring cell's
            # own bed; there the ring cell is dry, as the child's own dry cells
            # (pyGETM keeps those at -H + Dmin). Ring cells are not part of the
            # child's volume, so conservation is unchanged.
            np.maximum(vals, self._ring_zmin, out=vals)
            keep = self.wetmask[self._rj, self._ri]
            if keep.any():  # diagnostic: the layout should keep this well above Dcrit (masks.interface_depths)
                d = vals[keep] - self._ring_zmin[keep] + self._dmin
                self.min_depth_ring = min(self.min_depth_ring, float(d.min()))
            self.z.all_values[self._rj[keep], self._ri[keep]] = vals[keep]
        if halos:
            self.clock.halos(self.z)

    def _alpha(self, grid, pts):
        """pyGETM's drying factor at faces ``pts`` of ``grid`` (U or V) from the
        child's own water depth there: 0 at D <= 2 Dmin, 1 at D >= Dcrit
        (as pygetm.Simulation.update_depth, _pygetm.alpha). pyGETM computes it
        only at faces it updates; the ring faces are set from the parent."""
        sim = self.sim
        D = grid.D.all_values[pts]
        dmin2 = 2.0 * sim.Dmin
        return np.clip((D - dmin2) / (sim.Dcrit - dmin2), 0.0, 1.0)

    def _set_ring_uv(self, u, v, halos=True):
        # drying: the parent's transport through a (coarser) face that is wet in
        # the parent, scaled by the child's own drying factor at its ring face, so
        # no water moves over the child's dry ground (alpha = 1 in deep water)
        Ug, Vg = self.sim.U, self.sim.V
        self.U.all_values[self._ru] = u * self._alpha(Ug, self._ru)
        self.V.all_values[self._rv] = v * self._alpha(Vg, self._rv)
        if halos:
            self.clock.halos(self.U, self.V)

    def _send_up(self, fluxes=True, wait=True):
        reqs = []
        z = self.z.all_values
        for (tt, p), (_, _, _, cj, ci) in self.up.res.items():
            if tt == self.t:
                jj, ii = self.loc(cj, ci)
                weight = np.where(self.wetmask[jj, ii], self.area[jj, ii], 0.0)
                buf = np.ascontiguousarray(np.stack([np.where(weight > 0, z[jj, ii], 0.0), weight]))
                reqs.append(self.up.comm.Isend(buf, dest=self.up.poff + p, tag=self.up.tag_r))
        reqs += self.up_faces["rim_x"].send(self.up.comm, self.t, self.U.all_values, self.loc)
        reqs += self.up_faces["rim_y"].send(self.up.comm, self.t, self.V.all_values, self.loc)
        if fluxes:
            for (tt, p), (_, _, fk) in self.up.flx.items():
                if tt == self.t:
                    vals = np.ascontiguousarray(self.reg[fk])
                    reqs.append(self.up.comm.Isend(vals, dest=self.up.poff + p, tag=self.up.tag_f))
            self.reg = np.zeros_like(self.reg)
        if not wait:
            return reqs
        self.clock.waitall(reqs)

    def _own_edge_fluxes(self):
        e = self.lev.edge
        keep = self.own_edge & self.lev.computed[e["inner"]]
        if keep.any():
            self.reg[keep] += self._face_volume(e["is_x"][keep], e["face"][0][keep], e["face"][1][keep])

    # --- parent side ------------------------------------------------------------

    def _send_snapshot(self, plan, faces):
        z = self.z.all_values
        reqs = []
        for (p, tt), (pj, pi, *_rest) in plan.pro.items():
            if p == self.t:
                jj, ii = self.loc(pj, pi)
                sx, sy = ops.limited_slopes_at(z, self.wetmask, jj, ii)
                buf = np.ascontiguousarray(np.stack([z[jj, ii], sx, sy]))
                reqs.append(plan.comm.Isend(buf, dest=plan.coff + tt, tag=plan.tag_p))
        reqs += faces["ring_x"].send(plan.comm, self.t, self.U.all_values, self.loc)
        reqs += faces["ring_y"].send(plan.comm, self.t, self.V.all_values, self.loc)
        return reqs

    def _window_volumes(self, plan):
        wf = self.lev.wfaces[plan.child.name]
        m = plan.faces_of[self.t]
        return self._face_volume(wf["is_x"][m], wf["face"][0][m], wf["face"][1][m])

    def _recv_child(self, plan, faces, fp):
        # rim z (area-weighted mean over wet child cells)
        mine = plan.rim_of[self.t]
        if len(mine):
            r = plan.child.ratio
            blk = np.zeros((len(mine), r, r))
            wgt = np.zeros((len(mine), r, r))
            for (tt, p), (k, a, b, _, _) in plan.res.items():
                if p == self.t:
                    buf = np.empty((2, k.size))
                    self.clock.recv(plan.comm, buf, plan.coff + tt, plan.tag_r)
                    blk[k, a, b] = buf[0]
                    wgt[k, a, b] = buf[1]
            vals = blk[:, 0, 0].copy() if r == 1 else ops.restrict(blk, wgt)
            if r == 1:
                vals[wgt[:, 0, 0] <= 0] = np.nan
            jj, ii = self.loc(mine[:, 0], mine[:, 1])
            ok = np.isfinite(vals)
            self.z.all_values[jj[ok], ii[ok]] = vals[ok]
            if ok.any():  # diagnostic
                d = self.H[jj[ok], ii[ok]] + vals[ok]
                self.min_depth_rim = min(self.min_depth_rim, float(d.min()))
        # rim U, V (mean over child faces)
        u = faces["rim_x"].recv(plan.comm, self.t, self.clock)
        v = faces["rim_y"].recv(plan.comm, self.t, self.clock)
        # only at faces open in this level: a land face keeps no transport (beyond a
        # curvilinear child's conforming band the child's faces lie elsewhere)
        for arr, f, val in ((self.U, faces["rim_x"], u), (self.V, faces["rim_y"], v)):
            fj, fi = self.loc(*f.points(self.t))
            op = arr.grid.mask.all_values[fj, fi] != 0
            arr.all_values[fj[op], fi[op]] = val[op]
        # volume flux matching
        if fp is None:
            return
        flist = plan.faces_of[self.t]
        if not flist.size:
            return
        r = plan.child.ratio
        sub = np.zeros((flist.size, r))
        for (tt, p), (fpos, fa, _) in sorted(plan.flx.items()):
            if p == self.t:
                buf = np.empty(fpos.size)
                self.clock.recv(plan.comm, buf, plan.coff + tt, plan.tag_f)
                sub[fpos, fa] = buf
        fc = ops.face_sum(sub)
        wf = self.lev.wfaces[plan.child.name]
        # all faces at once; np.add.at adds in face order, as a loop would
        ins = wf["inside"][flist]
        m_in = flist[ins]
        if m_in.size:
            oj, oi = self.loc(wf["out"][0][m_in], wf["out"][1][m_in])
            fpi, fci = fp[ins], fc[ins]
            dz = wf["sign"][m_in] * (fpi - fci) / self.area[oj, oi]
            np.add.at(self.z.all_values, (oj, oi), dz)
            d = np.where(self.wetmask[oj, oi], self.H[oj, oi] + self.z.all_values[oj, oi], np.inf)  # diagnostic
            k = int(np.argmin(d))
            if d[k] < self.min_depth_corrected:
                self.min_depth_corrected = float(d[k])
                self.min_depth_corrected_info = (self._k, int(oj[k]), int(oi[k]), float(dz[k]), float(d[k] - dz[k]))
            # the parent continues from the child's mean transport
            fj, fi = self.loc(wf["face"][0][m_in], wf["face"][1][m_in])
            x = wf["is_x"][m_in]
            newU = fci[x] / (self.dyu[fj[x], fi[x]] * self._dt())
            newV = fci[~x] / (self.dxv[fj[~x], fi[~x]] * self._dt())
            # 3D runs: the transports summed over the macro step for the
            # 3D momentum (Ui, Vi) hold this step's own U, V; the volume
            # that moved is the child's (the z correction above)
            mom = self.sim.momentum
            if getattr(mom, "_U_cum", None) is not None and self.sim.runtype > 1:
                mom._U_cum[fj[x], fi[x]] += newU - self.U.all_values[fj[x], fi[x]]
                mom._V_cum[fj[~x], fi[~x]] += newV - self.V.all_values[fj[~x], fi[~x]]
            self.U.all_values[fj[x], fi[x]] = newU
            self.V.all_values[fj[~x], fi[~x]] = newV
            k = int(np.argmax(np.abs(dz)))
            if abs(dz[k]) > self.max_correction:  # diagnostic
                m = m_in[k]
                self.max_correction = abs(dz[k])
                face = (wf["face"][0][m], wf["face"][1][m])
                self.max_correction_info = (self._k, "U" if wf["is_x"][m] else "V", face, float(fpi[k]), float(fci[k]), float(dz[k]))
        up = ~ins & (wf["up"][flist] >= 0)
        if up.any():
            np.add.at(self.reg, wf["up"][flist][up], fc[up])

    def _sync(self, fps, with_parent: bool, fluxes: bool):
        """One sync point, in an order without cycles over the hierarchy: (a) to
        the parent: restriction and registers (what goes up does not depend on
        the children); (b) from the children: halo in the hole and flux
        registers, then flux matching; (c) to the children: this level's
        corrected state for their rings; (d) from the parent: its state for the
        ring (what goes down does not depend on the parent). The sweeps up and
        down do not wait for each other. A level whose children pass fluxes on
        up (a child touching its edge) sends up after its children."""
        t0 = MPI.Wtime()
        if not hasattr(self, "_up_after_children"):
            self._up_after_children = any(
                bool((np.asarray(self.lev.wfaces[plan.child.name]["up"]) >= 0).any()) for plan in self.down)
        reqs = []
        if with_parent and not self._up_after_children:
            reqs += self._send_up(fluxes=fluxes, wait=False)
        for plan, faces, fp in zip(self.down, self.down_faces, fps):
            self._recv_child(plan, faces, fp)
        if with_parent and self._up_after_children:
            reqs += self._send_up(fluxes=fluxes, wait=False)
        if self.down:
            self.clock.halos(self.z, self.U, self.V)
            for plan, faces in zip(self.down, self.down_faces):
                reqs += self._send_snapshot(plan, faces)
        if with_parent:
            self._snap_old = getattr(self, "_snap", None) if fluxes else None
            self._snap = self._recv_snapshot()
            self._set_ring_z(self._snap[0])
            self._set_ring_uv(self._snap[1], self._snap[2])
        self.clock.waitall(reqs)
        if with_parent and fluxes:
            self._snap_to_restart()
        if self._t_start is not None:
            self.t_sync += MPI.Wtime() - t0

    # --- Coupler hooks -------------------------------------------------------------

    def _set_sponge(self, sim):
        """Add the sponge to An and redo pyGETM's interpolation of An to the
        U and V grids (momentum.start has done it before the couplers start;
        the time step is known only now)."""
        from .sponge import sponge_weight, water_distance

        lev, T, mom = self.lev, sim.T, sim.momentum
        # the level's wet cells including its band, on every process
        gmask = T.mask.gather()
        wet_dom = self.nest.comm.bcast(None if gmask is None else np.asarray(gmask) != 0)
        if wet_dom.shape != lev.domain_shape:
            raise ValueError(f"{lev.name}: gathered mask has shape {wet_dom.shape}, expected {lev.domain_shape}")
        wet = lev.to_view(wet_dom)
        dist = water_distance(lev._ring_mask() & wet, wet)
        w_glob = lev.to_domain(sponge_weight(dist, self.sponge_width * lev.ratio), 0.0)
        gj, gi = self.nest.global_indices()
        oj, oi = lev.view_offset
        jj = np.clip(gj + lev.band + oj, 0, w_glob.shape[0] - 1)
        ii = np.clip(gi + lev.band + oi, 0, w_glob.shape[1] - 1)
        w = w_glob[np.ix_(jj, ii)] * self.wetmask
        dx2 = np.minimum(T.dx.all_values, T.dy.all_values) ** 2
        mom.An.all_values[...] += np.where(w > 0.0, self.sponge_alpha * dx2 / sim.timestep * w, 0.0)
        mom.An_uu = mom.An.interp(mom.U.grid.ugrid.array(fill=np.nan))
        mom.An_uv = mom.An.interp(mom.U.grid.vgrid.array(fill=np.nan))
        mom.An_vu = mom.An.interp(mom.V.grid.ugrid.array(fill=np.nan))
        mom.An_vv = mom.An.interp(mom.V.grid.vgrid.array(fill=np.nan))
        # no damping where interpolated from masked cells (fill values)
        for a in (mom.An_uu, mom.An_uv, mom.An_vu, mom.An_vv):
            v = a.all_values
            v[~np.isfinite(v) | (v < 0.0)] = 0.0
        #: largest sponge An (m2 s-1) of this process (diagnostic)
        self.sponge_max = float(mom.An.all_values[w > 0].max(initial=0.0))

    def on_start(self, sim):
        self._check_restart_phase(sim)
        self._clean_external_after_restart(sim)
        if self.sponge_width > 0.0 and self.up is not None:
            self._set_sponge(sim)
        self._sync([None] * len(self.down), self.up is not None, fluxes=False)
        if self.up is not None:
            self._snap_from_restart()  # after a restart: the snapshot before the last sync
        self.clock.reset()
        self._t_start = MPI.Wtime()

    def on_stage(self, sim, stage, macro_active):
        t0 = MPI.Wtime()
        self._on_stage(sim, stage, macro_active)
        if self._t_start is not None:
            self.t_stage += MPI.Wtime() - t0

    def _on_stage(self, sim, stage, macro_active):
        r = self.lev.time_ratio  # sub-steps per parent step
        k = self._k % r
        # the ring continued linearly in time from the parent's last two syncs
        # (until there are two: held at the last one)
        ex = self._snap_old is not None
        if stage == "transports_2d" and self.up is not None:
            if ex:  # parent U at n-1/2 and n-3/2 -> child half step n + (k+1/2)/r
                a = 1.0 + (k + 0.5) / r
                self._set_ring_uv(*(ops_interp(self._snap_old[m], self._snap[m], a) for m in (1, 2)))
            else:
                self._set_ring_uv(*self._snap[1:3])
        elif stage == "elevation":
            if self.up is not None:
                if ex:  # parent z at n and n-1 -> child time n + (k+1)/r
                    self._set_ring_z(ops_interp(self._snap_old[0], self._snap[0], 1.0 + (k + 1) / r))
                else:
                    self._set_ring_z(self._snap[0])
                self._own_edge_fluxes()
            self._fps = [self._window_volumes(plan) for plan in self.down]

    def after_state_update(self, sim, macro_active):
        self._phase.all_values[...] = sim.istep % self._period  # restart state (see _phase_array)
        r = self.lev.time_ratio  # sub-steps per parent step
        k = self._k % r
        self._sync(self._fps, self.up is not None and k == r - 1, fluxes=True)
        self._k += 1

    def timing(self) -> dict:
        """Wall-clock monitoring of this process since the start (seconds):

        * ``elapsed``;
        * ``compute``: the model's own step (elapsed - sync - stage),
          including the model's own halo updates;
        * ``model_halo_wait``: of ``compute``, the time blocked in the
          model's own halo updates (waiting for the other tiles of the same
          level; ``compute - model_halo_wait`` is the computation itself);
        * ``work``: the nesting's own computations (restriction, flux
          matching, ring values; sync + stage - wait - transfer - halo);
        * ``halo``: the halo updates the nesting calls (``halo_wait`` of it
          blocked in MPI, i.e. waiting for the other tiles of the level);
        * ``transfer``: receiving the exchanged data;
        * ``wait``: blocked until a partner's data is there (idle);
        * ``sync``: all time in the syncs (wait + transfer + part of work
          and halo);
        * ``idle``: wait / elapsed.

        compute + work + halo + transfer + wait = elapsed. Schedule "sync";
        everything excludes the initial sync.

        For cost models: ``steps`` (model steps), ``cells`` (interior cells
        of this subdomain, land included), ``wet`` (of them unmasked) and
        ``open_boundary`` (open boundary points of this subdomain)."""
        elapsed = MPI.Wtime() - self._t_start if self._t_start is not None else 0.0
        c = self.clock
        T = self.sim.T
        mask = T.mask.all_values[T.haloy:T.haloy + T.ny, T.halox:T.halox + T.nx]
        work = max(self.t_sync + self.t_stage - c.wait - c.transfer - c.halo, 0.0)
        return dict(level=self.lev.name, tile=self.t, elapsed=elapsed, sync=self.t_sync,
                    compute=max(elapsed - self.t_sync - self.t_stage, 0.0), model_halo_wait=c.model_halo_wait(),
                    work=work, halo=c.halo, halo_wait=c.halo_wait,
                    transfer=c.transfer, wait=c.wait, idle=c.wait / elapsed if elapsed > 0 else 0.0,
                    steps=self._k, cells=int(mask.size), wet=int((mask != 0).sum()),
                    open_boundary=int(getattr(self.sim.open_boundaries, "np", 0)),
                    min_depth_ring=self.min_depth_ring)

    def describe(self) -> List[str]:
        """Messages of the U/V transfers of this process."""
        me, t = self.world.rank, self.t
        out = []
        if self.up_faces is not None:
            out += self.up_faces["ring_x"].describe(me, "dst", t) + self.up_faces["ring_y"].describe(me, "dst", t)
            out += self.up_faces["rim_x"].describe(me, "src", t) + self.up_faces["rim_y"].describe(me, "src", t)
        for faces in self.down_faces:
            out += faces["ring_x"].describe(me, "src", t) + faces["ring_y"].describe(me, "src", t)
            out += faces["rim_x"].describe(me, "dst", t) + faces["rim_y"].describe(me, "dst", t)
        return out

    def level_volume(self) -> float:
        """Water volume above rest (sum z * area) of the cells this level
        computes (this process)."""
        h = self.sim.T.halox
        active = self.sim.T.mask.all_values[h:-h, h:-h] == 1
        return float((self.z.all_values * self.area)[h:-h, h:-h][active].sum())
