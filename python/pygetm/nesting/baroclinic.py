"""Two-way nesting of 3D tracers (temperature, salinity, passive tracers)
between levels whose vertical grids differ (number of layers, coordinate,
column depth): the first step of the 3D coupling, pygetm-nesting
docs/plan_21_3d_exchange.md.

:class:`NestedTracers3D` is a :class:`pygetm.simulation.Coupler` attached to
every level next to :class:`pygetm.nesting.barotropic.NestedBarotropic`
(which keeps coupling z, U, V at every 2D step). It acts at the ``tracers``
stage of every macro (3D) time step. All levels must have the same macro
time step: a child's split factor is its parent's times its time ratio, so
that the macro steps of all levels end together and every macro step is a
sync point for the tracers.

At a sync, in an order without cycles over the hierarchy (as the 2D sync,
"up_first"):

* **rim (child -> parent):** the child sends, for each parent cell of the
  rim, its r x r columns (layer thicknesses, concentrations, area; 0 for
  land). The parent remaps every child column onto its own layers with
  :func:`pygetm.nesting.vremap.rim_columns_prepare` (the upper ``keep`` of
  min(D_c, D_p) unstretched, the part below stretched to the parent's depth;
  concentrations by the child volume each parent layer receives). A uniform
  field stays uniform. The rim then holds the child's content per child
  volume; the volumes differ by the elevation the parent's rim gets at the
  next 2D sync (the 2D sync follows the tracer stage), so the content is
  exact only up to that.
* **ring (parent -> child):** the parent sends the column (thicknesses,
  concentrations) of the parent cell covering each child ring cell; the
  child remaps it onto its own layers in that cell (the same stretch, by
  volume: a uniform field stays uniform). Horizontally piecewise constant
  (first version), held over the child's next macro step (the parent's
  state of the last sync).

Not yet: 3D flux matching (the parent's tracer fluxes through the
interface faces are not corrected to the child's, so the composite content
is not conserved exactly), the ring's 3D velocities (the child computes
its own at the interface faces; pyGETM adjusts them to the 2D transports),
limited slopes in the ring, time interpolation of the ring.
"""

from typing import List, Optional, Sequence

import numpy as np
from mpi4py import MPI

from ..simulation import Coupler
from .runtime import NestingRun
from . import vremap

#: tag offset (added to the 2D plan's tags) of the 3D messages
TAG_3D = 1000

# pyGETM's layers run from the bottom (0) to the surface (nz - 1); vremap's
# columns from the surface down: the messages hold columns surface first.


class NestedTracers3D(Coupler):
    """Exchange of 3D tracers (T-grid arrays ``(nz, ny, nx)``) with the parent
    and the children, at every macro time step.

    Args:
        nest: the level's :class:`NestingRun`
        sim: the level's simulation
        tracers: names of the tracers (attributes of ``sim``, e.g. "temp",
            "salt") or arrays; default: temp and salt where present
        keep: fraction of min(D_c, D_p) below the surface that is not
            stretched in the vertical remap (B''; 0: linear stretch, B)
        break_ring, break_rim: diagnostics, leave out one direction
    """

    def __init__(self, nest: NestingRun, sim, tracers: Optional[Sequence] = None, keep: float = 0.8,
                 break_ring: bool = False, break_rim: bool = False):
        self.nest, self.sim = nest, sim
        self.lev = nest.level
        self.t = nest.tile
        self.til = nest.tilings[self.lev.name]
        self.up, self.down = nest.up, nest.down
        self.keep = keep
        self.break_ring, self.break_rim = break_ring, break_rim
        self._tracer_spec = tracers
        self.t_sync = 0.0
        self.t_wait = 0.0
        self.n_sync = 0
        self.max_ring_change = 0.0

    # --- helpers -----------------------------------------------------------

    def loc(self, j, i):
        return self.til.local(self.t, j, i)

    def _tracers(self, sim):
        if self._tracer_spec is None:
            names = [n for n in ("temp", "salt") if getattr(sim, n, None) is not None]
            return [getattr(sim, n) for n in names]
        return [getattr(sim, n) if isinstance(n, str) else n for n in self._tracer_spec]

    def _recv(self, comm, buf, source, tag):
        t0 = MPI.Wtime()
        comm.Recv(buf, source=source, tag=tag)
        self.t_wait += MPI.Wtime() - t0

    # --- child side ------------------------------------------------------------

    def _send_up(self) -> List[MPI.Request]:
        """Rim: the child columns of every parent rim cell (h, c per tracer,
        area or 0 for land)."""
        reqs = []
        T = self.sim.T
        h = T.hn.all_values
        a = T.area.all_values
        for (tt, p), (_, _, _, cj, ci) in self.up.res.items():
            if tt != self.t:
                continue
            jj, ii = self.loc(cj, ci)
            w = np.where(self.wet[jj, ii], a[jj, ii], 0.0)
            parts = [h[::-1, jj, ii]] + [np.where(w > 0, f.all_values[::-1, jj, ii], 0.0) for f in self.tracers]
            buf = np.ascontiguousarray(np.concatenate(parts + [w[None, :]], axis=0))
            reqs.append(self.up.comm.Isend(buf, dest=self.up.poff + p, tag=self.up.tag_r + TAG_3D))
        return reqs

    def _recv_ring(self):
        """Ring: the parent columns for this tile's ring cells, remapped onto
        the child's layers there."""
        if not self._ring:
            return
        nzp = self.parent_nz
        nt = len(self.tracers)
        parts = []
        for p, v in self._ring:
            buf = np.empty(((1 + nt) * nzp, v[2].size))
            self._recv(self.up.comm, buf, self.up.poff + p, self.up.tag_p + TAG_3D)
            parts.append(buf)
        buf = np.concatenate(parts, axis=1)
        if self.break_ring:
            return
        hs = buf[:nzp].T  # (n, nzp)
        cs = buf[nzp:].reshape(nt, nzp, -1).transpose(0, 2, 1)  # (nt, n, nzp)
        rj, ri = self._rj, self._ri
        hd = self.sim.T.hn.all_values[::-1, rj, ri].T  # (n, nzc), surface first
        ok = self.wet[rj, ri] & (hs.sum(axis=1) > 0.0) & (hd.sum(axis=1) > 0.0)
        if not ok.any():
            return
        hs, cs, hd = hs[ok], cs[:, ok], hd[ok]
        shared = vremap.rim_columns_prepare(hs[:, None, :], np.ones((hs.shape[0], 1)), hd, keep=self.keep)
        c = vremap.rim_columns_apply(shared, cs[:, :, None, :], hs[:, None, :], hd, by_volume=True)
        for f, cf in zip(self.tracers, c):
            new = cf.T[::-1]  # pyGETM: layer 0 at the bottom
            old = f.all_values[:, rj[ok], ri[ok]]
            self.max_ring_change = max(self.max_ring_change, float(np.abs(new - old).max(initial=0.0)))
            f.all_values[:, rj[ok], ri[ok]] = new

    # --- parent side -------------------------------------------------------------

    def _send_down(self, plan) -> List[MPI.Request]:
        reqs = []
        h = self.sim.T.hn.all_values
        for (p, tt), (pj, pi, *_rest) in plan.pro.items():
            if p != self.t:
                continue
            jj, ii = self.loc(pj, pi)
            parts = [h[::-1, jj, ii]] + [f.all_values[::-1, jj, ii] for f in self.tracers]
            buf = np.ascontiguousarray(np.concatenate(parts, axis=0))
            reqs.append(plan.comm.Isend(buf, dest=plan.coff + tt, tag=plan.tag_p + TAG_3D))
        return reqs

    def _recv_rim(self, plan, nzc: int):
        mine = plan.rim_of[self.t]
        if not len(mine):
            return
        r = plan.child.ratio
        nt = len(self.tracers)
        n = len(mine)
        H = np.zeros((n, r * r, nzc))
        C = np.zeros((nt, n, r * r, nzc))
        W = np.zeros((n, r * r))
        for (tt, p), (k, a, b, _, _) in plan.res.items():
            if p != self.t:
                continue
            buf = np.empty(((1 + nt) * nzc + 1, k.size))
            self._recv(plan.comm, buf, plan.coff + tt, plan.tag_r + TAG_3D)
            m = a * r + b
            H[k, m] = buf[:nzc].T
            C[:, k, m] = buf[nzc:(1 + nt) * nzc].reshape(nt, nzc, -1).transpose(0, 2, 1)
            W[k, m] = buf[-1]
        if self.break_rim:
            return
        jj, ii = self.loc(mine[:, 0], mine[:, 1])
        hp = self.sim.T.hn.all_values[::-1, jj, ii].T  # (n, nzp), surface first
        ok = (W.sum(axis=1) > 0.0) & self.wet[jj, ii] & (hp.sum(axis=1) > 0.0)
        if not ok.any():
            return
        H, C, W, hp = H[ok], C[:, ok], W[ok], hp[ok]
        # land child columns (weight 0): any positive thicknesses, they add nothing
        dry = (W <= 0.0) | (H.sum(axis=-1) <= 0.0)
        W = np.where(dry, 0.0, W)
        H = np.where(dry[..., None], hp.sum(axis=-1)[:, None, None] / nzc, H)
        shared = vremap.rim_columns_prepare(H, W, hp, keep=self.keep)
        # concentrations by the child volume each parent layer receives: the
        # child columns' content per child volume. Their total volume differs
        # from the parent cell's by the elevation the parent's rim gets only
        # at the next 2D sync, so restoring the content exactly in the
        # parent's volume ("fix") would make a uniform field non-uniform.
        c = vremap.rim_columns_apply(shared, C, H, hp, by_volume=True)
        for f, cf in zip(self.tracers, c):
            f.all_values[:, jj[ok], ii[ok]] = cf.T[::-1]

    # --- sync --------------------------------------------------------------------

    def _sync(self):
        t0 = MPI.Wtime()
        reqs = []
        if self.up is not None and not self._up_after_children:
            reqs += self._send_up()
        for plan, nzc in zip(self.down, self.child_nz):
            self._recv_rim(plan, nzc)
        if self.up is not None and self._up_after_children:
            reqs += self._send_up()
        if self.down:
            for f in self.tracers:
                f.update_halos()
            for plan in self.down:
                reqs += self._send_down(plan)
        if self.up is not None:
            self._recv_ring()
            for f in self.tracers:
                f.update_halos()
        MPI.Request.Waitall(reqs)
        self.t_sync += MPI.Wtime() - t0
        self.n_sync += 1

    # --- Coupler hooks -------------------------------------------------------------

    def on_start(self, sim):
        T = sim.T
        self.tracers = self._tracers(sim)
        self.wet = T.mask.all_values != 0
        h = T.halox
        self.computed = T.mask.all_values[h:-h, h:-h] == 1
        self.halo = h
        world = self.nest.world
        # all levels share the macro time step (every macro step is a sync)
        dts = world.allgather((self.lev.name, float(sim.macrotimestep), int(T.hn.all_values.shape[0]),
                               len(self.tracers)))
        steps = {d[1] for d in dts}
        if max(steps) - min(steps) > 1e-6 * max(steps):
            raise ValueError("NestedTracers3D: the levels need the same macro time step (split factor of a child ="
                             f" its parent's x its time ratio): {sorted(set((d[0], d[1]) for d in dts))}")
        if len({d[3] for d in dts}) != 1:
            raise ValueError(f"NestedTracers3D: the levels exchange different numbers of tracers: {dts}")
        nz = {d[0]: d[2] for d in dts}
        self.nz = nz[self.lev.name]
        self.parent_nz = nz[self.lev.parent.name] if self.lev.parent is not None else None
        self.child_nz = [nz[plan.child.name] for plan in self.down]
        self._up_after_children = any(
            bool((np.asarray(self.lev.wfaces[plan.child.name]["up"]) >= 0).any()) for plan in self.down)
        self._ring = []
        if self.up is not None:
            self._ring = [(p, v) for (p, tt), v in sorted(self.up.pro.items()) if tt == self.t]
            cat = lambda k: np.concatenate([v[k] for _, v in self._ring]) if self._ring else np.zeros(0, int)
            self._rj, self._ri = self.loc(cat(2), cat(3))
        self._sync()
        self.max_ring_change = 0.0
        self.t_sync = self.t_wait = 0.0
        self.n_sync = 0

    def on_stage(self, sim, stage, macro_active):
        if stage == "tracers" and macro_active:
            self._sync()

    # --- diagnostics -----------------------------------------------------------------

    def level_content(self) -> np.ndarray:
        """Content (sum of c h area) of every tracer over the cells this
        level computes (this process)."""
        T, h = self.sim.T, self.halo
        vol = (T.hn.all_values * T.area.all_values)[:, h:-h, h:-h]
        return np.array([float((f.all_values[:, h:-h, h:-h] * vol)[:, self.computed].sum()) for f in self.tracers])

    def timing(self) -> dict:
        return dict(sync=self.t_sync, wait=self.t_wait, n_sync=self.n_sync, max_ring_change=self.max_ring_change)
