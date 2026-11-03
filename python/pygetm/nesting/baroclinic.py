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

* **flux matching (child -> parent, ``flux_matching=True``):** pyGETM's
  tracer advection adds the amount through every face and layer to arrays
  of this coupler (:attr:`pygetm.tracer.TracerCollection.face_fluxes`).
  The child sends, with the rim, the profiles through its edge faces (r
  sub-faces per parent interface face) over the 3D step, with its face
  layer thicknesses. The parent remaps every sub-face profile onto the
  layers of its own face (the same stretch; the amounts are kept exactly),
  adds the r sub-faces, and corrects the parent cell next to the face layer
  by layer by (parent - child) amount. Faces of a child that lie on the
  parent's own interface are passed upwards, in the parent's layers, as in
  2D.

* **3D transports (child -> parent, ``replace_transport=True``):** at the
  ``momentum_3d`` stage the child sends its layer transports through its
  edge faces; the parent remaps them onto its face layers, keeps its own
  column total (which its elevation used) and sets pk, qk there, then
  recomputes ww, uk, vk. Parent and child then move the same water through
  every layer of the interface, and the flux correction conserves the
  content exactly while keeping uniform fields uniform.

* **ring velocities (parent -> child, ``ring_velocities=True``):** at every
  sync the parent sends its layer transports and face thicknesses at the
  faces around the child's ring faces (2 per face, linear across and
  constant along the parent faces, as the 2D ring). At its next
  ``momentum_3d`` stage the child remaps them onto its own face layers,
  shifts them to its own column total (the ring's 2D transport, which its
  elevation used) and sets pk, qk there, then recomputes ww, uk, vk. Held
  over the 3D step.

Not yet: limited slopes in the ring, time interpolation of the ring.
"""

from typing import List, Optional, Sequence

import numpy as np
from mpi4py import MPI

from ..constants import CENTERS
from ..simulation import Coupler
from .runtime import NestingRun
from . import vremap

#: tag offset (added to the 2D plan's tags) of the 3D messages
TAG_3D = 1000

# pyGETM's layers run from the bottom (0) to the surface (nz - 1); vremap's
# columns from the surface down: the messages hold columns surface first.


def remap_amounts(F, h_src, h_dst, keep=0.0):
    """Amounts per layer (``F [..., ns]``, e.g. the tracer amount through a
    face) of a column with thicknesses ``h_src`` onto the layers ``h_dst``
    (surface first), the source column stretched to the target's depth below
    ``keep`` x min(depths). The sum over the layers is kept exactly. Columns
    with no thickness carry nothing."""
    Ds, Dd = h_src.sum(axis=-1), h_dst.sum(axis=-1)
    dry = (Ds <= 0.0) | (Dd <= 0.0)
    nd = h_dst.shape[-1]
    h_dst = np.where(dry[..., None], 1.0 / nd, h_dst)
    h_src = np.where(dry[..., None], 1.0 / h_src.shape[-1], h_src)
    F = np.where(dry[..., None], 0.0, F)
    zs = vremap.stretch(vremap.interfaces(h_src), h_src.sum(axis=-1), h_dst.sum(axis=-1), keep)
    prep = vremap.prepare(zs, vremap.interfaces(h_dst))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(h_src > 0.0, F / h_src, 0.0)
    return vremap.apply(prep, c, h_src)


def remap_conc_z(c, h_src, h_dst, w=None, scheme="ppm"):
    """Concentrations ``c [nt, ..., m, ns]`` of source columns (thicknesses ``h_src
    [..., m, ns]``, surface first) onto target layers ``h_dst [..., nd]``, with every
    source interface kept at its depth below the surface: only the bottom source
    layer is cut or extended to the target's depth. Concentrations by the volume each
    target layer receives, the m source columns weighted by ``w [..., m]`` (default
    1): a horizontally uniform field (flat isopycnals) stays exactly so, whatever the
    depths, and a uniform field uniform."""
    zs = vremap.interfaces(h_src)
    Dd = h_dst.sum(axis=-1)[..., None, None]
    zs = np.minimum(zs, Dd)
    zs[..., -1] = Dd[..., 0]
    h_eff = np.diff(zs, axis=-1)
    zd = np.broadcast_to(vremap.interfaces(h_dst)[..., None, :], zs.shape[:-1] + (h_dst.shape[-1] + 1,))
    prep = vremap.prepare(zs, zd)
    w = np.ones(h_src.shape[:-1]) if w is None else w
    C = (w[..., None] * vremap.apply(prep, c, h_eff, scheme)).sum(axis=-2)
    V = (w[..., None] * vremap.apply(prep, np.ones_like(h_eff), h_eff)).sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(V > 0.0, C / V, 0.0)


def remap_column(c, h_src, h_dst, keep=0.0, z=False, scheme="ppm"):
    if z:
        return remap_conc_z(c[:, :, None, :], h_src[:, None, :], h_dst, scheme=scheme)
    """Concentrations ``c [nt, n, ns]`` of columns with thicknesses ``h_src`` onto
    the layers ``h_dst [n, nd]`` (surface first), the source stretched to the
    target's depth below ``keep`` x min(depths), by volume (uniform stays uniform);
    ``z``: interfaces kept at their depths (:func:`remap_conc_z`)."""
    shared = vremap.rim_columns_prepare(h_src[:, None, :], np.ones((h_src.shape[0], 1)), h_dst, keep=keep)
    return vremap.rim_columns_apply(shared, c[:, :, None, :], h_src[:, None, :], h_dst, by_volume=True)


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
                 break_ring: bool = False, break_rim: bool = False, flux_matching: bool = True,
                 flux_form: str = "consistent", replace_transport: bool = True, ring_velocities: bool = True,
                 ring_detail: bool = True, vertical_map: str = "z", remap_scheme: str = "ppm"):
        self.nest, self.sim = nest, sim
        self.lev = nest.level
        self.t = nest.tile
        self.til = nest.tilings[self.lev.name]
        self.up, self.down = nest.up, nest.down
        self.keep = keep
        self.break_ring, self.break_rim = break_ring, break_rim
        self.break_flux = False
        self.flux_matching = flux_matching
        #: correction of the parent cell next to an interface face, layer k
        #: (amounts F of tracer and V of water through the face, parent p and
        #: child c over the 3D step; the 2D coupling adds the column's volume
        #: difference dV = sum_k (V_p - V_c)_k to the parent cell at its own
        #: concentrations):
        #: "content": dc_k = sign ((F_p - F_c)_k - c_k dV h_k / D) / vol_k -
        #: the composite content is conserved, a uniform field does not stay
        #: uniform (the layers' volume fluxes differ); "consistent" (default):
        #: dc_k = sign ((F_p - F_c)_k - c_k (V_p - V_c)_k) / vol_k - a uniform
        #: field stays uniform, the content is conserved up to
        #: sum_k c_k ((V_p - V_c)_k - dV h_k / D)
        if flux_form not in ("content", "consistent"):
            raise ValueError(f"flux_form {flux_form!r}")
        self.flux_form = flux_form
        #: at the momentum_3d stage, the parent's layer transports at the window
        #: faces become the child's (remapped onto the parent's face layers, the
        #: parent's column total kept), and its vertical velocity is
        #: recomputed: parent and child then move the same water through every
        #: layer of the interface, and the tracer correction is exact and keeps
        #: uniform fields uniform
        self.replace_transport = replace_transport
        #: the child's 3D transports at its ring faces from the parent's (sent
        #: at every sync, set at the child's next momentum_3d stage)
        self.ring_velocities = ring_velocities
        #: the ring's profile = the parent's, remapped, + the child's own vertical
        #: detail: (child's adjacent interior column remapped onto the ring cell's
        #: layers) - (the same column through the parent's layers and back). The
        #: parent's layer means cannot carry the child's finer vertical structure;
        #: without this the ring's stepped profile next to the child's smooth one
        #: is a horizontal density gradient at the interface (IDEAL V2: spurious
        #: currents of 1-2 cm/s in a lake at rest, 10x the single-model error).
        #: Limited to the range of the parent's value and the child's adjacent value
        self.ring_detail = ring_detail
        #: remap of concentrations (ring, rim): "z" (default) keeps every interface at
        #: its depth below the surface and cuts or extends only the bottom layer, so
        #: flat isopycnals stay flat over a sloping bottom; "stretch": the upper
        #: ``keep`` of the column at its depth, the rest stretched (the amounts of the
        #: flux register and the transports are always remapped with the stretch,
        #: which keeps their sums exactly). IDEAL V2 (lake at rest over a slope):
        #: "stretch" displaces the pycnocline where the column is shallow
        if vertical_map not in ("z", "stretch"):
            raise ValueError(f"vertical_map {vertical_map!r}")
        self.zmap = vertical_map == "z"
        #: reconstruction inside each source layer when concentrations are remapped
        #: (ring, rim; with vertical_map "z"): "ppm" (default, limited parabola,
        #: Colella and Woodward 1984) or "pcm" (constant). PPM carries a thin
        #: pycnocline from coarse layers onto finer ones 5-20x more accurately;
        #: both keep the content exactly and add no new extrema
        if remap_scheme not in ("ppm", "pcm"):
            raise ValueError(f"remap_scheme {remap_scheme!r}")
        self.scheme = remap_scheme
        self._ring_uv = None
        self.t_transport = 0.0
        self.max_correction = 0.0
        self._tracer_spec = tracers
        # wall-clock time per part of the exchange (inclusive of the MPI waits inside it)
        self.parts = {}
        for name in ("_own_edge_fluxes", "_send_up", "_recv_ring", "_send_down", "_recv_rim", "_match_fluxes",
                     "_send_ring_uv", "_recv_ring_uv", "_set_ring_uv", "_send_transports_up", "_recv_transports"):
            self._timed(name)
        self.t_sync = 0.0
        self._fluxes_ready = False
        self.t_wait = 0.0
        self.n_sync = 0
        self.max_ring_change = 0.0

    # --- helpers -----------------------------------------------------------

    def _timed(self, name):
        f = getattr(self, name)

        def wrapper(*a, **kw):
            t0, w0 = MPI.Wtime(), self.t_wait
            try:
                return f(*a, **kw)
            finally:
                p = self.parts.setdefault(name, [0.0, 0.0])
                p[0] += MPI.Wtime() - t0
                p[1] += self.t_wait - w0  # of which waiting for the partner (MPI receives)

        setattr(self, name, wrapper)

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

    def _face_profiles(self, arrays, is_x, j, i):
        """(n, nz) profiles, surface first, of a U array and a V array at faces (j, i)."""
        jj, ii = self.loc(j, i)
        out = np.empty((np.size(is_x), self.nz))
        out[is_x] = arrays[0].all_values[::-1, jj[is_x], ii[is_x]].T
        out[~is_x] = arrays[1].all_values[::-1, jj[~is_x], ii[~is_x]].T
        return out

    def _own_edge_fluxes(self):
        e = self.lev.edge
        keep = self._own_edge & self.lev.computed[e["inner"]]
        if keep.any():
            args = (e["is_x"][keep], e["face"][0][keep], e["face"][1][keep])
            for n, fl in enumerate(self.fluxes):
                self.reg[n, keep] += self._face_profiles(fl, *args)
            self.reg[-1, keep] += self._volume_profiles(*args)

    def _volume_profiles(self, is_x, j, i):
        """Water through faces (j, i) over the 3D step per layer, surface first
        (layer transports pk, qk at the half step, as the tracer advection)."""
        m = self.sim.momentum
        vol = self._face_profiles((m.pk, m.qk), is_x, j, i)
        jj, ii = self.loc(j, i)
        w = np.where(is_x, self.sim.U.dy.all_values[jj, ii], self.sim.V.dx.all_values[jj, ii])
        return np.where(np.isfinite(vol), vol, 0.0) * (w * self.sim.macrotimestep)[:, None]

    def _send_up(self) -> List[MPI.Request]:
        """Rim: the child columns of every parent rim cell (h, c per tracer,
        area or 0 for land); with flux matching, the register of every edge
        face (its layer thicknesses and the amounts per tracer and layer)."""
        reqs = []
        if self.flux_matching and self._fluxes_ready:
            e = self.lev.edge
            for (tt, p), (_, _, fk) in self.up.flx.items():
                if tt != self.t:
                    continue
                hface = self._face_profiles((self.sim.U.hn, self.sim.V.hn), e["is_x"][fk], e["face"][0][fk], e["face"][1][fk])
                hface = np.where(np.isfinite(hface), hface, 0.0)
                buf = np.ascontiguousarray(np.concatenate([hface.T] + [r[fk].T for r in self.reg], axis=0))
                reqs.append(self.up.comm.Isend(buf, dest=self.up.poff + p, tag=self.up.tag_f + TAG_3D))
            self.reg[...] = 0.0
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
        c = remap_column(cs, hs, hd, self.keep, self.zmap, self.scheme)
        if self.ring_detail:
            ij, ii_ = self._rin_j[ok], self._rin_i[ok]
            m = self._rin_ok[ok].copy()
            if m.any():
                hin = self.sim.T.hn.all_values[::-1, ij, ii_].T
                cin = np.stack([f.all_values[::-1, ij, ii_].T for f in self.tracers])
                m &= np.isfinite(hin).all(axis=1) & (hin.sum(axis=1) > 0.0)
                z = self.zmap
                sc = self.scheme
                direct = remap_column(cin[:, m], hin[m], hd[m], self.keep, z, sc)
                back = remap_column(remap_column(cin[:, m], hin[m], hs[m], self.keep, z, sc), hs[m], hd[m], self.keep, z, sc)
                # limited to the range of the parent's value and the child's adjacent one
                # (layer by layer): the correction adds no new extrema (IDEAL V6)
                c0 = c[:, m]
                c[:, m] = np.clip(c0 + direct - back, np.minimum(c0, direct), np.maximum(c0, direct))
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
        if self.zmap:
            c = remap_conc_z(C, H, hp, W, self.scheme)
            for f, cf in zip(self.tracers, c):
                f.all_values[:, jj[ok], ii[ok]] = cf.T[::-1]
            return
        shared = vremap.rim_columns_prepare(H, W, hp, keep=self.keep)
        # concentrations by the child volume each parent layer receives: the
        # child columns' content per child volume. Their total volume differs
        # from the parent cell's by the elevation the parent's rim gets only
        # at the next 2D sync, so restoring the content exactly in the
        # parent's volume ("fix") would make a uniform field non-uniform.
        c = vremap.rim_columns_apply(shared, C, H, hp, by_volume=True)
        for f, cf in zip(self.tracers, c):
            f.all_values[:, jj[ok], ii[ok]] = cf.T[::-1]

    def _match_fluxes(self, plan, nzc: int):
        flist = plan.faces_of[self.t]
        if not flist.size:
            return
        r = plan.child.ratio
        nt = len(self.tracers) + 1  # the tracers and the water
        H = np.zeros((flist.size, r, nzc))
        F = np.zeros((nt, flist.size, r, nzc))
        for (tt, p), (fpos, fa, _) in sorted(plan.flx.items()):
            if p != self.t:
                continue
            buf = np.empty(((1 + nt) * nzc, fpos.size))
            self._recv(plan.comm, buf, plan.coff + tt, plan.tag_f + TAG_3D)
            H[fpos, fa] = buf[:nzc].T
            F[:, fpos, fa] = buf[nzc:].reshape(nt, nzc, -1).transpose(0, 2, 1)
        wf = self.lev.wfaces[plan.child.name]
        args = (wf["is_x"][flist], wf["face"][0][flist], wf["face"][1][flist])
        hp = self._face_profiles((self.sim.U.hn, self.sim.V.hn), *args)  # (nf, nzp)
        hp = np.where(np.isfinite(hp), hp, 0.0)
        # child amounts on the parent face's layers, r sub-faces added
        fc = remap_amounts(F, H, np.broadcast_to(hp[:, None, :], H.shape[:2] + hp.shape[-1:]), self.keep).sum(axis=-2)
        ins = wf["inside"][flist]
        m_in = flist[ins]
        if m_in.size and not self.break_flux:
            oj, oi = self.loc(wf["out"][0][m_in], wf["out"][1][m_in])
            T = self.sim.T
            vol = (T.hn.all_values[::-1, oj, oi] * T.area.all_values[oj, oi]).T  # (n, nz), surface first
            sign = wf["sign"][m_in][:, None]
            kk = np.arange(self.nz)[::-1]
            dV = self._volume_profiles(*(a[ins] for a in args)) - fc[-1][ins]  # water, parent - child (n, nz)
            hT = T.hn.all_values[::-1, oj, oi].T
            frac = hT / np.maximum(hT.sum(axis=1, keepdims=True), 1e-30)
            for n, (f, fl) in enumerate(zip(self.tracers, self.fluxes)):
                fp = self._face_profiles(fl, *(a[ins] for a in args))
                c = f.all_values[::-1, oj, oi].T
                if self.flux_form == "consistent":
                    amount = fp - fc[n][ins] - c * dV
                else:
                    amount = fp - fc[n][ins] - c * dV.sum(axis=1, keepdims=True) * frac
                with np.errstate(invalid="ignore", divide="ignore"):
                    dc = np.where(vol > 0.0, sign * amount / vol, 0.0)
                self.max_correction = max(self.max_correction, float(np.abs(dc).max(initial=0.0)))
                # np.add.at in face order (several faces may share a cell)
                np.add.at(f.all_values, (kk[None, :], oj[:, None], oi[:, None]), dc)
        up = ~ins & (wf["up"][flist] >= 0)
        if up.any():
            for n in range(nt):  # tracers and water, in this level's layers
                np.add.at(self.reg[n], wf["up"][flist][up], fc[n][up])

    # --- 3D transports at the interface (momentum_3d stage) ------------------------

    def _send_transports_up(self) -> List[MPI.Request]:
        reqs = []
        e = self.lev.edge
        m = self.sim.momentum
        for (tt, p), (_, _, fk) in self.up.flx.items():
            if tt != self.t:
                continue
            args = (e["is_x"][fk], e["face"][0][fk], e["face"][1][fk])
            h = self._face_profiles((self.sim.U.hn, self.sim.V.hn), *args)
            h = np.where(np.isfinite(h), h, 0.0)
            q = self._volume_profiles(*args) / self.sim.macrotimestep  # m3 s-1 per layer
            buf = np.ascontiguousarray(np.concatenate([h.T, q.T], axis=0))
            reqs.append(self.up.comm.Isend(buf, dest=self.up.poff + p, tag=self.up.tag_f + TAG_3D + 300))
        return reqs

    def _recv_transports(self, plan, nzc: int) -> bool:
        flist = plan.faces_of[self.t]
        if not flist.size:
            return False
        r = plan.child.ratio
        H = np.zeros((flist.size, r, nzc))
        Q = np.zeros((flist.size, r, nzc))
        for (tt, p), (fpos, fa, _) in sorted(plan.flx.items()):
            if p != self.t:
                continue
            buf = np.empty((2 * nzc, fpos.size))
            self._recv(plan.comm, buf, plan.coff + tt, plan.tag_f + TAG_3D + 300)
            H[fpos, fa] = buf[:nzc].T
            Q[fpos, fa] = buf[nzc:].T
        wf = self.lev.wfaces[plan.child.name]
        ins = wf["inside"][flist]
        if not ins.any():
            return False
        m_in = flist[ins]
        is_x, fj, fi = wf["is_x"][m_in], wf["face"][0][m_in], wf["face"][1][m_in]
        hp = self._face_profiles((self.sim.U.hn, self.sim.V.hn), is_x, fj, fi)
        hp = np.where(np.isfinite(hp), hp, 0.0)
        qc = remap_amounts(Q[ins], H[ins], np.broadcast_to(hp[:, None, :], H[ins].shape[:2] + hp.shape[-1:]),
                           self.keep).sum(axis=-2)  # (n, nzp) m3 s-1
        mom = self.sim.momentum
        jj, ii = self.loc(fj, fi)
        width = np.where(is_x, self.sim.U.dy.all_values[jj, ii], self.sim.V.dx.all_values[jj, ii])
        own = self._face_profiles((mom.pk, mom.qk), is_x, fj, fi)  # m2 s-1
        own = np.where(np.isfinite(own), own, 0.0)
        D = hp.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            new = qc / width[:, None]
            # keep the parent's column total (its elevation used it)
            new = new + (own.sum(axis=1, keepdims=True) - new.sum(axis=1, keepdims=True)) * np.where(D > 0, hp / D, 0.0)
        new = np.where(D > 0, new, own)
        kk = np.arange(self.nz)[::-1]
        x = is_x
        mom.pk.all_values[kk[None, :], jj[x][:, None], ii[x][:, None]] = new[x]
        mom.qk.all_values[kk[None, :], jj[~x][:, None], ii[~x][:, None]] = new[~x]
        return True

    # --- 3D transports in the child's ring (parent -> child) ------------------------

    def _ring_transfers(self, plan, link_tag):
        from .barotropic import PointTransfer, ring_faces

        out = {}
        for n, axis in enumerate("xy"):
            dst, src, w = ring_faces(plan.child, axis)
            out[axis] = PointTransfer(plan.ptil, plan.ctil, dst, src, w, link_tag + 2 + n, "3D" + "UV"[n] + "r",
                                      plan.poff, plan.coff)
        return out

    def _send_ring_uv(self, plan, tx) -> List[MPI.Request]:
        reqs = []
        mom = self.sim.momentum
        for axis, grid, tp in (("x", self.sim.U, mom.pk), ("y", self.sim.V, mom.qk)):
            for (p, t), (_, _, sj, si) in tx[axis].pairs.items():
                if p != self.t:
                    continue
                jj, ii = self.loc(sj, si)
                buf = np.concatenate([grid.hn.all_values[::-1, jj, ii], tp.all_values[::-1, jj, ii]], axis=0)
                buf = np.ascontiguousarray(np.where(np.isfinite(buf), buf, 0.0))
                reqs.append(plan.comm.Isend(buf, dest=plan.coff + t, tag=tx[axis].tag))
        return reqs

    def _recv_ring_uv(self):
        nzp = self.parent_nz
        out = {}
        for axis in "xy":
            tx = self._ring_tx[axis]
            dj, di, w = tx.dst_of.get(self.t, (np.zeros(0, int), np.zeros(0, int), np.zeros((0, tx.K))))
            vals = np.zeros((dj.size, tx.K, 2 * nzp))
            for (p, t), (pos, k, sj, _) in sorted(tx.pairs.items()):
                if t != self.t:
                    continue
                buf = np.empty((2 * nzp, sj.size))
                self._recv(self.up.comm, buf, self.up.poff + p, tx.tag)
                vals[pos, k] = buf.T
            # weights of the source faces with water, normalised
            wet = vals[:, :, :nzp].sum(axis=2) > 0.0
            ww = np.where(wet, w, 0.0)
            tot = ww.sum(axis=1, keepdims=True)
            ww = np.where(tot > 0.0, ww / np.where(tot > 0.0, tot, 1.0), 0.0)
            hs = (ww[:, :, None] * vals[:, :, :nzp]).sum(axis=1)
            qs = (ww[:, :, None] * vals[:, :, nzp:]).sum(axis=1)
            out[axis] = (dj, di, hs, qs)
        self._ring_uv = out

    def _set_ring_uv(self, sim):
        mom = sim.momentum
        kk = np.arange(self.nz)[::-1]
        for axis, grid, tp in (("x", sim.U, mom.pk), ("y", sim.V, mom.qk)):
            dj, di, hs, qs = self._ring_uv[axis]
            if not dj.size:
                continue
            jj, ii = self.loc(dj, di)
            hd = grid.hn.all_values[::-1, jj, ii].T
            hd = np.where(np.isfinite(hd), hd, 0.0)
            own = tp.all_values[::-1, jj, ii].T
            own = np.where(np.isfinite(own), own, 0.0)
            ok = (hd.sum(axis=1) > 0.0) & (hs.sum(axis=1) > 0.0)
            if not ok.any():
                continue
            hd, hs_, qs_, own = hd[ok], hs[ok], qs[ok], own[ok]
            new = remap_amounts(qs_, hs_, hd, self.keep)
            D = hd.sum(axis=1, keepdims=True)
            new = new + (own.sum(axis=1, keepdims=True) - new.sum(axis=1, keepdims=True)) * hd / D
            tp.all_values[kk[None, :], jj[ok][:, None], ii[ok][:, None]] = new

    def _sync_transports(self, sim):
        t0 = MPI.Wtime()
        reqs = self._send_transports_up() if (self.up is not None and self.replace_transport) else []
        changed = False
        if self.replace_transport:
            for plan, nzc in zip(self.down, self.child_nz):
                changed |= self._recv_transports(plan, nzc)
        MPI.Request.Waitall(reqs)
        ring = self.ring_velocities and self.up is not None and self._ring_uv is not None
        if ring:
            self._set_ring_uv(sim)
        if (self.down and self.replace_transport) or ring:
            mom = sim.momentum
            mom.pk.update_halos()
            mom.qk.update_halos()
            # as Momentum.update_diagnostics: w from the transports, velocities
            from .. import _pygetm
            _pygetm.w_momentum_3d(mom.pk, mom.qk, sim.macrotimestep, mom.ww)
            np.divide(mom.pk.all_values, mom.pk.grid.hn.all_values, out=mom.uk.all_values)
            np.divide(mom.qk.all_values, mom.qk.grid.hn.all_values, out=mom.vk.all_values)
        self.t_transport += MPI.Wtime() - t0

    # --- sync --------------------------------------------------------------------

    def _sync(self):
        t0 = MPI.Wtime()
        reqs = []
        if self.flux_matching and self._fluxes_ready and self.up is not None:
            self._own_edge_fluxes()
        if self.up is not None and not self._up_after_children:
            reqs += self._send_up()
        for plan, nzc in zip(self.down, self.child_nz):
            self._recv_rim(plan, nzc)
            if self.flux_matching and self._fluxes_ready:
                self._match_fluxes(plan, nzc)
        if self.up is not None and self._up_after_children:
            reqs += self._send_up()
        if self.down:
            for f in self.tracers:
                f.update_halos()
            for plan, tx in zip(self.down, self._ring_tx_down):
                reqs += self._send_down(plan)
                if self.ring_velocities:
                    reqs += self._send_ring_uv(plan, tx)
        if self.up is not None:
            if self.ring_detail:
                for f in self.tracers:  # the interior cells next to the ring may lie in the halo
                    f.update_halos()
            self._recv_ring()
            if self.ring_velocities:
                self._recv_ring_uv()
            for f in self.tracers:
                f.update_halos()
        MPI.Request.Waitall(reqs)
        if self.flux_matching:
            for fl in self.fluxes:
                fl[0].all_values[...] = 0.0
                fl[1].all_values[...] = 0.0
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
            # the child's interior cell next to each ring cell (within the halo)
            gj = np.clip(cat(2), 0, self.lev.ny - 1)
            gi = np.clip(cat(3), 0, self.lev.nx - 1)
            self._rin_ok = self.lev.computed[gj, gi]
            self._rin_j, self._rin_i = self.loc(gj, gi)
        self.fluxes = []
        if self.flux_matching:
            U, V = sim.U, sim.V
            for f in self.tracers:
                fl = (U.array(z=CENTERS, fill=0.0), V.array(z=CENTERS, fill=0.0))
                sim.tracers.face_fluxes[f.name] = fl
                self.fluxes.append(fl)
            if self.up is not None:
                e = self.lev.edge
                self._own_edge = self.til.owner(*e["inner"]) == self.t
            # this level's register: amount per tracer, edge face and layer (surface first)
            nedge = self.lev.edge["is_x"].size if self.up is not None else 0
            self.reg = np.zeros((len(self.tracers) + 1, nedge, self.nz))  # tracers, then the water
        self._ring_tx_down = [self._ring_transfers(plan, plan.tag_p + TAG_3D + 500) for plan in self.down]
        self._ring_tx = self._ring_transfers(self.up, self.up.tag_p + TAG_3D + 500) if self.up is not None else None
        self._sync()  # initial: no fluxes yet
        self._fluxes_ready = True
        self.max_ring_change = 0.0
        self.t_sync = self.t_wait = 0.0
        self.parts = {}  # the start-up sync does not count
        self.n_sync = 0

    def on_stage(self, sim, stage, macro_active):
        if stage == "momentum_3d" and macro_active and (self.replace_transport or self.ring_velocities):
            self._sync_transports(sim)
        elif stage == "tracers" and macro_active:
            self._sync()

    # --- diagnostics -----------------------------------------------------------------

    def level_content(self) -> np.ndarray:
        """Content (sum of c h area) of every tracer over the cells this
        level computes (this process)."""
        T, h = self.sim.T, self.halo
        vol = (T.hn.all_values * T.area.all_values)[:, h:-h, h:-h]
        return np.array([float((f.all_values[:, h:-h, h:-h] * vol)[:, self.computed].sum()) for f in self.tracers])

    def timing(self) -> dict:
        return dict(sync=self.t_sync, wait=self.t_wait, n_sync=self.n_sync, max_ring_change=self.max_ring_change,
                    max_correction=self.max_correction, transport=self.t_transport,
                    parts={k.lstrip("_"): v[0] for k, v in self.parts.items()},
                    parts_wait={k.lstrip("_"): v[1] for k, v in self.parts.items()})
