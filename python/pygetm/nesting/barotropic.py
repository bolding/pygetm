"""Two-way nesting of 2D barotropic pyGETM simulations (z, U, V).

Schedule (``NestedBarotropic(schedule=...)``):

* "sync" (default): all models step at the same time; at the end of every
  parent step they meet at a sync point: children send the halo inside the
  hole and their flux registers, the parent matches the fluxes of the same
  step and sends its corrected state, which the child holds as its ring for
  its next ``time_ratio`` sub-steps. Models only wait in MPI at the syncs.
* "sequential" (old, kept for comparison; described below): the child
  steps after the parent and interpolates its ring in time.

Every parent and child is a :class:`pygetm.Simulation` with
``runtype=BAROTROPIC_2D``; :class:`NestedBarotropic` is a
:class:`pygetm.simulation.Coupler` attached to each of them. Per parent step n
(t_n -> t_n+1):

* parent, after its state update: send a snapshot to the children - z(n+1)
  with limited slopes at the parent cells covering the child's ring, and the
  transports U, V(n+1/2) at the parent faces around the child's ring faces;
  then wait for the children's sub-steps, receive their restriction
  (z: area-weighted mean over wet child cells; U, V: mean over the child
  faces along each parent face) for the rim, and their volume flux
  registers; correct z of the parent cells next to the window by
  (parent - child) volume through each shared face, and continue from the
  child's mean transport at those faces (without this, the correction loop
  is unstable in deep water); send the corrected z to the children (the
  start of their next parent step). All this happens before pyGETM
  recomputes depths and pressure gradients from z.
* child, sub-step k of r (r = time_ratio, steps per parent step):
  - after its momentum update: ring U, V at t_n + (k + 1/2)/r, interpolated
    (and beyond t_n+1/2 extrapolated) between the parent's U(n-1/2) and
    U(n+1/2); in space linear across and constant along parent faces;
  - after its elevation update: ring z at t_n + (k + 1)/r, interpolated
    between the parent's z(n) and z(n+1); add the volume through its edge
    faces (U dy dt) to its flux register;
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

#: tag offset of the parent's corrected z snapshot
TAG_CORRECTED = 50


class PointTransfer:
    """Weighted transfer from points of one level to points of another, on
    their tilings: value(dst n) = sum_k w[n, k] * value(src[n, k]). Points
    are (j, i) indices of T cells or of the U/V faces east/north of them;
    the owner of a point is the owner of its T cell. Sums are formed in a
    fixed order (k), so results do not depend on the decomposition."""

    def __init__(self, src_til: LevelTiling, dst_til: LevelTiling, dst, src, weights, tag: int, label: str):
        dj, di = (np.asarray(x) for x in dst)
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
                reqs.append(world.Isend(buf, dest=self.dst_til.offset + t, tag=self.tag))
        return reqs

    def recv(self, world, me: int) -> np.ndarray:
        """Destination side: weighted values at this tile's points."""
        dj, di, w = self.dst_of.get(me, (np.zeros(0, int),) * 2 + (np.zeros((0, self.K)),))
        vals = np.zeros((dj.size, self.K))
        for (p, t), (pos, k, sj, _) in sorted(self.pairs.items()):
            if t == me:
                buf = np.empty(sj.size)
                world.Recv(buf, source=self.src_til.offset + p, tag=self.tag)
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


def face_transfers(child: Level, ptil: LevelTiling, ctil: LevelTiling, link: int, tag_base: int = 600):
    """U/V ring (parent -> child) and rim (child -> parent) transfers."""
    out = {}
    for n, axis in enumerate("xy"):
        dst, src, w = ring_faces(child, axis)
        out[f"ring_{axis}"] = PointTransfer(ptil, ctil, dst, src, w, tag_base + 4 * link + n, "UV"[n] + "r")
        dst, src, w = rim_faces(child, axis)
        out[f"rim_{axis}"] = PointTransfer(ctil, ptil, dst, src, w, tag_base + 4 * link + 2 + n, "UV"[n] + "R")
    return out


class NestedBarotropic(Coupler):
    """Coupler exchanging z, U, V of a 2D barotropic simulation with its
    parent and children (see module documentation)."""

    def __init__(self, nest: NestingRun, sim, break_ring=False, break_rim=False, break_flux=False,
                 replace_transport=True, schedule="sync", ring_time="hold"):
        self.nest, self.sim = nest, sim
        self.lev = lev = nest.level
        self.t = nest.tile
        self.til = nest.tilings[lev.name]
        self.world = nest.world
        self.up, self.down = nest.up, nest.down
        self.break_ring, self.break_rim, self.break_flux = break_ring, break_rim, break_flux
        #: after flux matching, set the parent's transport at each window face
        #: to the child's mean transport (stabilises the coupling in deep water)
        self.replace_transport = replace_transport
        #: "sync" (default): all models step at the same time and exchange at
        #: the end of every parent step; the child holds its ring from the
        #: parent's state at the last sync. "sequential" (old, for comparison):
        #: the child steps after the parent and interpolates its ring in time
        #: between the parent's n and n+1.
        if schedule not in ("sync", "sequential"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.schedule = schedule
        #: schedule "sync": the ring between syncs is "hold" (the parent's state
        #: at the last sync) or "extrapolate" (linear in time from the parent's
        #: last two syncs; z to the child's time, U/V to its half steps)
        if ring_time not in ("hold", "extrapolate"):
            raise ValueError(f"unknown ring_time {ring_time!r}")
        self.ring_time = ring_time
        self._snap_old = None
        T = sim.T
        self.z, self.U, self.V = T.z, sim.momentum.U, sim.momentum.V
        self.wetmask = T.mask.all_values != 0
        self.area = T.area.all_values
        self.dyu, self.dxv = T.ugrid.dy.all_values, T.vgrid.dx.all_values
        links = {c.name: k for k, c in enumerate(nest.levels[1:])}
        self.up_faces = None
        if lev.parent is not None:
            self.up_faces = face_transfers(lev, nest.tilings[lev.parent.name], self.til, links[lev.name])
        self.down_faces = [face_transfers(c, self.til, nest.tilings[c.name], links[c.name]) for c in lev.children]
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
        self._k = 0
        self._fps = []
        #: largest elevation correction (m) by flux matching so far (diagnostic)
        self.max_correction = 0.0
        self.max_correction_info = None

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
            self.world.Recv(buf, source=self.up.ptil.offset + p, tag=self.up.tag_p)
            parts.append(buf)
        z = np.concatenate(parts, axis=1) if parts else np.zeros((3, 0))
        u = self.up_faces["ring_x"].recv(self.world, self.t)
        v = self.up_faces["ring_y"].recv(self.world, self.t)
        return z, u, v

    def _set_ring_z(self, snap):
        if not self.break_ring and self._rj.size:
            vals = ops.prolong(snap[0], snap[1], snap[2], self._ring_a, self._ring_b, self.lev.ratio)
            keep = self.wetmask[self._rj, self._ri]
            self.z.all_values[self._rj[keep], self._ri[keep]] = vals[keep]
        self.z.update_halos()

    def _set_ring_uv(self, u, v):
        if not self.break_ring:
            self.U.all_values[self._ru] = u
            self.V.all_values[self._rv] = v
        self.U.update_halos()
        self.V.update_halos()

    def _send_up(self, fluxes=True):
        reqs = []
        z = self.z.all_values
        for (tt, p), (_, _, _, cj, ci) in self.up.res.items():
            if tt == self.t:
                jj, ii = self.loc(cj, ci)
                weight = np.where(self.wetmask[jj, ii], self.area[jj, ii], 0.0)
                buf = np.ascontiguousarray(np.stack([np.where(weight > 0, z[jj, ii], 0.0), weight]))
                reqs.append(self.world.Isend(buf, dest=self.up.ptil.offset + p, tag=self.up.tag_r))
        reqs += self.up_faces["rim_x"].send(self.world, self.t, self.U.all_values, self.loc)
        reqs += self.up_faces["rim_y"].send(self.world, self.t, self.V.all_values, self.loc)
        if fluxes:
            for (tt, p), (_, _, fk) in self.up.flx.items():
                if tt == self.t:
                    vals = np.ascontiguousarray(self.reg[fk])
                    reqs.append(self.world.Isend(vals, dest=self.up.ptil.offset + p, tag=self.up.tag_f))
            self.reg = np.zeros_like(self.reg)
        MPI.Request.Waitall(reqs)

    def _own_edge_fluxes(self):
        e = self.lev.edge
        keep = self.own_edge & self.lev.computed[e["inner"]]
        if keep.any():
            self.reg[keep] += self._face_volume(e["is_x"][keep], e["face"][0][keep], e["face"][1][keep])

    # --- parent side ------------------------------------------------------------

    def _send_snapshot(self, plan, faces):
        z = self.z.all_values
        sx, sy = ops.limited_slopes(z, self.wetmask)
        reqs = []
        for (p, tt), (pj, pi, *_rest) in plan.pro.items():
            if p == self.t:
                jj, ii = self.loc(pj, pi)
                buf = np.ascontiguousarray(np.stack([z[jj, ii], sx[jj, ii], sy[jj, ii]]))
                reqs.append(self.world.Isend(buf, dest=plan.ctil.offset + tt, tag=plan.tag_p))
        reqs += faces["ring_x"].send(self.world, self.t, self.U.all_values, self.loc)
        reqs += faces["ring_y"].send(self.world, self.t, self.V.all_values, self.loc)
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
                    self.world.Recv(buf, source=plan.ctil.offset + tt, tag=plan.tag_r)
                    blk[k, a, b] = buf[0]
                    wgt[k, a, b] = buf[1]
            if not self.break_rim:
                vals = blk[:, 0, 0].copy() if r == 1 else ops.restrict(blk, wgt)
                if r == 1:
                    vals[wgt[:, 0, 0] <= 0] = np.nan
                jj, ii = self.loc(mine[:, 0], mine[:, 1])
                ok = np.isfinite(vals)
                self.z.all_values[jj[ok], ii[ok]] = vals[ok]
        # rim U, V (mean over child faces)
        u = faces["rim_x"].recv(self.world, self.t)
        v = faces["rim_y"].recv(self.world, self.t)
        if not self.break_rim:
            self.U.all_values[self.loc(*faces["rim_x"].points(self.t))] = u
            self.V.all_values[self.loc(*faces["rim_y"].points(self.t))] = v
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
                self.world.Recv(buf, source=plan.ctil.offset + tt, tag=plan.tag_f)
                sub[fpos, fa] = buf
        fc = ops.face_sum(sub)
        wf = self.lev.wfaces[plan.child.name]
        for n, m in enumerate(flist):
            if wf["inside"][m]:
                oj, oi = self.loc(wf["out"][0][m], wf["out"][1][m])
                dz = wf["sign"][m] * (fp[n] - fc[n]) / self.area[oj, oi]
                if not self.break_flux:
                    self.z.all_values[oj, oi] += dz
                    if self.replace_transport:
                        # the parent continues from the child's mean transport
                        fj, fi = self.loc(wf["face"][0][m], wf["face"][1][m])
                        if wf["is_x"][m]:
                            self.U.all_values[fj, fi] = fc[n] / (self.dyu[fj, fi] * self._dt())
                        else:
                            self.V.all_values[fj, fi] = fc[n] / (self.dxv[fj, fi] * self._dt())
                if abs(dz) > self.max_correction:  # diagnostic
                    self.max_correction = abs(dz)
                    face = (wf["face"][0][m], wf["face"][1][m])
                    self.max_correction_info = (self._k, "U" if wf["is_x"][m] else "V", face, float(fp[n]), float(fc[n]), float(dz))
            elif wf["up"][m] >= 0:
                self.reg[wf["up"][m]] += fc[n]

    def _send_corrected_z(self, plan):
        """After the volume correction: the corrected z (with slopes) at the
        cells the child's ring is filled from. The child uses it as the start
        of its next parent step, so ring and parent stay consistent."""
        z = self.z.all_values
        sx, sy = ops.limited_slopes(z, self.wetmask)
        reqs = []
        for (p, tt), (pj, pi, *_rest) in plan.pro.items():
            if p == self.t:
                jj, ii = self.loc(pj, pi)
                buf = np.ascontiguousarray(np.stack([z[jj, ii], sx[jj, ii], sy[jj, ii]]))
                reqs.append(self.world.Isend(buf, dest=plan.ctil.offset + tt, tag=plan.tag_p + TAG_CORRECTED))
        return reqs

    def _couple_down(self, fps):
        reqs = []
        for plan, faces in zip(self.down, self.down_faces):
            reqs += self._send_snapshot(plan, faces)
        MPI.Request.Waitall(reqs)
        for plan, faces, fp in zip(self.down, self.down_faces, fps):
            self._recv_child(plan, faces, fp)
        self.z.update_halos()
        self.U.update_halos()
        self.V.update_halos()
        if any(fp is not None for fp in fps):
            reqs = []
            for plan in self.down:
                reqs += self._send_corrected_z(plan)
            MPI.Request.Waitall(reqs)

    def _sync(self, fps, with_parent: bool, fluxes: bool):
        """One sync point (schedule "sync"), in an order without cycles over
        the hierarchy: (a) from the children: halo in the hole and flux
        registers, then flux matching; (b) to the parent: restriction and
        registers; (c) from the parent: its (corrected) state for the ring;
        (d) to the children: this level's state for their rings."""
        for plan, faces, fp in zip(self.down, self.down_faces, fps):
            self._recv_child(plan, faces, fp)
        if self.down:
            self.z.update_halos()
            self.U.update_halos()
            self.V.update_halos()
        if with_parent:
            self._send_up(fluxes=fluxes)
            self._snap_old = getattr(self, "_snap", None) if fluxes else None
            self._snap = self._recv_snapshot()
            self._set_ring_z(self._snap[0])
            self._set_ring_uv(self._snap[1], self._snap[2])
        reqs = []
        for plan, faces in zip(self.down, self.down_faces):
            reqs += self._send_snapshot(plan, faces)
        MPI.Request.Waitall(reqs)

    # --- Coupler hooks -------------------------------------------------------------

    def on_start(self, sim):
        if self.schedule == "sync":
            self._sync([None] * len(self.down), self.up is not None, fluxes=False)
            return
        if self.up is not None:
            snap = self._recv_snapshot()
            self._prev = self._new = snap
            self._set_ring_z(snap[0])
            self._set_ring_uv(snap[1], snap[2])
        self._couple_down([None] * len(self.down))
        if self.up is not None:
            self._send_up(fluxes=False)

    def before_state_update(self, sim, macro_active):
        if self.schedule == "sequential" and self.up is not None and self._k % self.lev.time_ratio == 0:
            self._new = self._recv_snapshot()

    def on_stage(self, sim, stage, macro_active):
        r = self.lev.time_ratio  # sub-steps per parent step
        k = self._k % r
        if self.schedule == "sync":
            # ring held at the parent's state of the last sync (external
            # cells/faces: pyGETM keeps them, set again for safety)
            ex = self.ring_time == "extrapolate" and self._snap_old is not None
            if stage == "transports_2d" and self.up is not None:
                if ex:  # parent U at n-1/2 and n-3/2 -> child half step n + (k+1/2)/r
                    a = 1.0 + (k + 0.5) / r
                    self._set_ring_uv(ops_interp(self._snap_old[1], self._snap[1], a),
                                      ops_interp(self._snap_old[2], self._snap[2], a))
                else:
                    self._set_ring_uv(self._snap[1], self._snap[2])
            elif stage == "elevation":
                if self.up is not None:
                    if ex:  # parent z at n and n-1 -> child time n + (k+1)/r
                        self._set_ring_z(ops_interp(self._snap_old[0], self._snap[0], 1.0 + (k + 1) / r))
                    else:
                        self._set_ring_z(self._snap[0])
                    self._own_edge_fluxes()
                self._fps = [self._window_volumes(plan) for plan in self.down]
            return
        if stage == "transports_2d" and self.up is not None:
            # parent U lives at n-1/2 and n+1/2: interpolate up to n+1/2,
            # hold beyond (extrapolation is unstable in deep water)
            a = min((k + 0.5) / r + 0.5, 1.0)
            self._set_ring_uv(ops_interp(self._prev[1], self._new[1], a), ops_interp(self._prev[2], self._new[2], a))
        elif stage == "elevation":
            if self.up is not None:
                self._set_ring_z(ops_interp(self._prev[0], self._new[0], (k + 1) / r))
                self._own_edge_fluxes()
            self._fps = [self._window_volumes(plan) for plan in self.down]

    def _recv_corrected_z(self):
        parts = []
        for p, v in self._ring:
            buf = np.empty((3, v[2].size))
            self.world.Recv(buf, source=self.up.ptil.offset + p, tag=self.up.tag_p + TAG_CORRECTED)
            parts.append(buf)
        return np.concatenate(parts, axis=1) if parts else np.zeros((3, 0))

    def after_state_update(self, sim, macro_active):
        r = self.lev.time_ratio  # sub-steps per parent step
        k = self._k % r
        if self.schedule == "sync":
            self._sync(self._fps, self.up is not None and k == r - 1, fluxes=True)
            self._k += 1
            return
        self._couple_down(self._fps)
        if self.up is not None and k == r - 1:
            self._send_up()
            # the parent's z after its volume correction is the start of the
            # next parent step (and the ring's value now)
            zc = self._recv_corrected_z()
            self._new = (zc,) + tuple(self._new[1:])
            self._set_ring_z(zc)
            self._prev = self._new
        self._k += 1

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
