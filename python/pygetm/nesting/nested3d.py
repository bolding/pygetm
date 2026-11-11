"""Two-way nesting of 3D tracers (temperature, salinity, passive tracers)
between levels whose vertical grids differ (number of layers, coordinate,
column depth): the first step of the 3D coupling, pygetm-nesting
docs/plan_21_3d_exchange.md.

:class:`Nested3D` is a :class:`pygetm.simulation.Coupler` attached to
every level next to :class:`pygetm.nesting.nested2d.Nested2D`
(which keeps coupling z, U, V at every 2D step). It acts at the ``tracers``
stage of every macro (3D) time step.

**3D sub-stepping:** a child's macro step may be its parent's divided by a whole
number R (:attr:`Nested3D.ratio_3d`), which follows from the split factors:
R = parent's split x child's time ratio / child's split (R = 1: all levels share
the macro step). Everything between the two levels happens once per parent macro
step, at the sync at the end of the child's R-th macro step. In between the child
holds the ring (concentrations and the parent's layer transports, the latter
shifted at every macro step to the column total of its own 2D ring), and adds up
its tracer amounts and its water through the edge faces. The parent therefore
sees the sum over all sub-steps in its correction and its transports.

At a sync, in an order without cycles over the hierarchy (as the 2D sync,
"up_first"):

* **rim (child -> parent):** the child sends, for each parent cell of the
  rim, its r x r columns (layer thicknesses, concentrations, area; 0 for
  land). The parent remaps them onto its own layers with every interface
  kept at its depth below the surface (only the bottom layer cut or
  extended; PPM or PCM inside the layers, the child's ``remap_scheme``),
  concentrations by the child volume each parent layer receives. A uniform
  field stays uniform, and flat isopycnals stay flat over a sloping bottom.
  The rim then holds the child's content per child volume; the volumes
  differ by the elevation the parent's rim gets at the next 2D sync (the 2D
  sync follows the tracer stage), so the content is exact only up to that.
* **ring (parent -> child):** the parent sends the column (thicknesses,
  concentrations) of the parent cell covering each child ring cell and of its
  four neighbours; the child remaps them onto its own layers in that cell (as
  the rim), adds its own vertical detail and limited slopes within the parent
  cell (``ring_slopes``), and holds or extrapolates it over its next R macro
  steps (``ring_time``).

* **flux matching (child -> parent):** pyGETM's
  tracer advection adds the amount through every face and layer to arrays
  of this coupler (:attr:`pygetm.tracer.TracerCollection.face_fluxes`).
  The child sends, with the rim, the profiles through its edge faces (r
  sub-faces per parent interface face) over the parent's 3D step, with its face
  layer thicknesses. The parent remaps every sub-face profile onto the
  layers of its own face (the same stretch; the amounts are kept exactly),
  adds the r sub-faces, and corrects the parent cell next to the face layer
  by layer by (parent - child) amount. Faces of a child that lie on the
  parent's own interface are passed upwards, in the parent's layers, as in
  2D.

* **3D transports (child -> parent):** at the
  ``momentum_3d`` stage the child sends its layer transports through its
  edge faces (the mean over its R macro steps); the parent remaps them onto its face layers, keeps its own
  column total (which its elevation used) and sets pk, qk there, then
  recomputes ww, uk, vk. Parent and child then move the same water through
  every layer of the interface, and the flux correction conserves the
  content exactly while keeping uniform fields uniform.

* **ring velocities (parent -> child):** at every
  sync the parent sends its layer transports and face thicknesses at the
  faces around the child's ring faces (2 per face, linear across and
  constant along the parent faces, as the 2D ring). At its next
  ``momentum_3d`` stage the child remaps them onto its own face layers,
  shifts them to its own column total (the ring's 2D transport, which its
  elevation used) and sets pk, qk there, then recomputes ww, uk, vk. Held
  over the parent's 3D step.

* **ring in time (``ring_time``):** "hold" (default) keeps the ring at the parent's
  state of the last sync over the child's R macro steps; "extrapolate" (for R > 1)
  continues it linearly from the parent's last two syncs, c = c_n + a (c_n - c_{n-1})
  with a = m / R at the child's m-th macro step after the sync (m = 0: the sync's
  values, so R = 1 is unchanged), limited layer by layer to the range of the parent's
  cell and its four neighbours at the last sync, remapped onto the ring cell's layers
  (over one parent macro step the water moves less than a parent cell, so the true
  value lies in that range: no new extrema). The previous ring is part of the restart
  state. The vertical structure of the ring velocities is held (their column totals
  follow the 2D ring, which is extrapolated).

The ring is linear within each parent cell, with limited slopes (``ring_slopes``, default; off: constant per parent cell).
"""

from typing import List, Optional, Sequence

import numpy as np
from mpi4py import MPI

from ..constants import CENTERS, CellType
from ..simulation import Coupler
from .runtime import NestingRun
from . import vremap
from .operators import minmod, subcell_offset

#: tag offset (added to the 2D plan's tags) of the 3D messages
TAG_3D = 1000

# pyGETM's layers run from the bottom (0) to the surface (nz - 1); vremap's
# columns from the surface down: the messages hold columns surface first.


#: fraction of min(D_c, D_p) below the surface kept at its depth when the amounts of the flux
#: register and the transports are remapped between columns of different depth (the rest stretched)
KEEP = 0.8


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
    zd = np.broadcast_to(vremap.interfaces(h_dst), zs.shape[:-1] + (h_dst.shape[-1] + 1,))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(h_src > 0.0, F / h_src, 0.0)
    return vremap.contents(zs, zd, c, h_src)


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
    w = np.ones(h_src.shape[:-1]) if w is None else w
    C = (w[..., None] * vremap.contents(zs, zd, c, h_eff, scheme)).sum(axis=-2)
    V = (w[..., None] * vremap.contents(zs, zd, np.ones(h_eff.shape), h_eff)).sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(V > 0.0, C / V, 0.0)


def remap_column(c, h_src, h_dst, scheme="ppm"):
    """Concentrations ``c [nt, n, ns]`` of columns with thicknesses ``h_src`` onto
    the layers ``h_dst [n, nd]`` (surface first), every source interface kept at
    its depth below the surface (:func:`remap_conc_z`), compiled when available."""
    k = vremap._kernel("remap_z")
    if k is None:
        return remap_conc_z(c[:, :, None, :], h_src[:, None, :], h_dst, scheme=scheme)
    hs = np.ascontiguousarray(h_src, dtype=float)
    hd = np.ascontiguousarray(h_dst, dtype=float)
    cc = np.ascontiguousarray(c, dtype=float)
    out = np.empty(cc.shape[:2] + (hd.shape[-1],))
    k(hs, hd, np.ascontiguousarray(hd.sum(axis=-1)), cc, out, 1 if scheme == "ppm" else 0)  # D as numpy sums it
    return out


def macro_ratios(levels) -> dict:
    """Macro steps of every child per macro step of its parent. ``levels``: name ->
    (parent name or None, macro time step, split factor, time ratio). A child's macro
    step must be its parent's divided by a whole number."""
    ratios = {}
    for name, (parent, macro, split, time_ratio) in levels.items():
        if parent is None:
            continue
        pmacro, psplit = levels[parent][1], levels[parent][2]
        q, rem = divmod(psplit * time_ratio, split)
        if q < 1 or rem != 0 or abs(pmacro - q * macro) > 1e-6 * pmacro:
            raise ValueError(f"Nested3D: the 3D step of {name} ({macro} s, split {split}, time ratio {time_ratio}) is not"
                             f" the 3D step of its parent {parent} ({pmacro} s, split {psplit}) divided by a whole number")
        ratios[name] = q
    return ratios


class Nested3D(Coupler):
    """Exchange of 3D tracers (T-grid arrays ``(nz, ny, nx)``) with the parent
    and the children, at every macro time step.

    Args:
        nest: the level's :class:`NestingRun`
        sim: the level's simulation
        tracers: names of the tracers (attributes of ``sim``, e.g. "temp",
            "salt") or arrays; default: temp and salt where present
        remap_scheme: "ppm" or "pcm" (see below), per interface
        ring_slopes: the ring linear within each parent cell (see below)
        ring_time: "hold" or "extrapolate" (children with 3D sub-stepping)
    """

    def __init__(self, nest: NestingRun, sim, tracers: Optional[Sequence] = None, remap_scheme: str = "ppm",
                 ring_slopes: bool = True, ring_time: str = "hold"):
        self.nest, self.sim = nest, sim
        self.lev = nest.level
        self.t = nest.tile
        self.til = nest.tilings[self.lev.name]
        self.up, self.down = nest.up, nest.down
        #: the stretch of the amounts (flux register, transports) between columns of
        #: different depth: the upper KEEP of min(D_c, D_p) at its depth, the rest
        #: stretched; their sums over the layers are kept exactly
        self.keep = KEEP
        #: reconstruction inside each source layer when concentrations are remapped
        #: (ring, rim): "ppm" (default, limited parabola,
        #: Colella and Woodward 1984) or "pcm" (constant). PPM carries a thin
        #: pycnocline from coarse layers onto finer ones 5-20x more accurately;
        #: both keep the content exactly and add no new extrema. Per interface: a child
        #: level's setting is used for its ring and for its rim in the parent (the
        #: root's own setting is not used). IDEAL V5/V7: PCM makes the exchange's work
        #: 25-35 % cheaper; the pycnocline next to the interfaces 4-8 % less accurate
        if remap_scheme not in ("ppm", "pcm"):
            raise ValueError(f"remap_scheme {remap_scheme!r}")
        self.scheme = remap_scheme
        #: the ring horizontally linear within each parent cell (as the 2D ring's
        #: elevation; default): the parent also sends the columns of the four
        #: neighbours of every ring source cell; the child remaps all five onto the
        #: ring cell's own layers and takes minmod-limited slopes layer by layer of
        #: their anomalies against the child's adjacent column seen through each
        #: column's layers (see :meth:`_recv_ring`), within the five columns' range.
        #: Without it the ring is piecewise constant per parent cell, with steps at the
        #: parent cells' edges. IDEAL V6 (a dye front) against one model at the shelf's
        #: 1 km: the depth-mean dye error 45 % smaller on the shelf and in the strait;
        #: V2 (at rest) about as without. The same on all levels.
        self.ring_slopes = bool(ring_slopes)
        #: the ring between two syncs of a child with 3D sub-stepping: "hold" or
        #: "extrapolate" (see the module's description). The same on all levels
        #: (with "extrapolate" the parent sends the neighbour columns for the limiter).
        if ring_time not in ("hold", "extrapolate"):
            raise ValueError(f"ring_time {ring_time!r}")
        self.ring_time = ring_time
        self._extrap = ring_time == "extrapolate"
        self._rc = None  # ring on the child's layers at the last sync: (new, previous, lo, hi), NaN where unset
        self._ring_uv = None
        #: this level's macro steps per macro step of its parent (from the split
        #: factors, set in :meth:`on_start`); the exchange with the parent happens
        #: at every ``ratio_3d``-th macro step
        self.ratio_3d = 1
        self._water_sum = {}
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
        #: The flux correction (parent side) is limited, without new extrema: the corrections of all
        #: children at a sync are added per cell and layer, the corrected value is clipped
        #: to the range of the cell, its four neighbours and the layers above and below
        #: (before the correction); the column's net cut-off amount is spread over its
        #: layers that have room left within their own range, and only what does not fit
        #: is kept per cell and layer (``nest_flux_deferred_<k>``, an amount; restart
        #: state) and added at the next sync. The content is conserved over time (with the deferred amounts, see
        #: :meth:`deferred_content`). Off: the correction as it comes (new extrema of
        #: 1e-4 in IDEAL V6 without it).
        self.n_clipped = 0
        self.n_seen_reused = 0  # ring slopes: neighbours' round trips not remapped (the same layers)
        self.max_deferred = 0.0
        self._restart_ring_arrays()  # needs the tracers
        self._deferred_arrays()

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

    def _up_active(self) -> bool:
        """Does this macro step end one of the parent's (a sync with the parent)?"""
        return self.up is not None and self.sim.istep % (self.sim.split_factor * self.ratio_3d) == 0

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
        if self._fluxes_ready:
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
            # column by column, surface first: (1 + nt, k, nz) then the k weights
            k, nt, nz = jj.size, len(self.tracers), h.shape[0]
            buf = np.empty((1 + nt) * k * nz + k)
            B = buf[: (1 + nt) * k * nz].reshape(1 + nt, k, nz)
            vremap.gather_columns(h, jj, ii, out=B[0])
            for n, f in enumerate(self.tracers):
                vremap.gather_columns(f.all_values, jj, ii, out=B[1 + n])
            B[1:, w <= 0.0] = 0.0
            buf[-k:] = w
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
        nnb = 5 if (self.ring_slopes or self._extrap) else 1
        for p, v in self._ring:
            buf = np.empty((nnb, 1 + nt, v[2].size, nzp))  # columns surface first
            self._recv(self.up.comm, buf, self.up.poff + p, self.up.tag_p + TAG_3D)
            parts.append(buf)
        buf = np.concatenate(parts, axis=2)
        hs = buf[0, 0]  # (n, nzp)
        cs = buf[0, 1:]  # (nt, n, nzp)
        rj, ri = self._rj, self._ri
        hd = vremap.gather_columns(self.sim.T.hn.all_values, rj, ri)  # (n, nzc), surface first
        ok = self.wet[rj, ri] & (hs.sum(axis=1) > 0.0) & (hd.sum(axis=1) > 0.0)
        if not ok.any():
            return
        hs, cs, hd = hs[ok], cs[:, ok], hd[ok]
        c = remap_column(cs, hs, hd, self.scheme)
        sc = self.scheme
        # the child's own column next to each ring cell (one cell inwards): its vertical
        # detail for the ring (ring_detail) and the reference for the slopes
        m = np.zeros(c.shape[1], bool)
        ij, ii_ = self._rin_j[ok], self._rin_i[ok]
        m = self._rin_ok[ok].copy()
        if m.any():
            hin = vremap.gather_columns(self.sim.T.hn.all_values, ij, ii_)
            cin = np.empty((len(self.tracers),) + hin.shape)
            for n, f in enumerate(self.tracers):
                vremap.gather_columns(f.all_values, ij, ii_, out=cin[n])
            m &= np.isfinite(hin).all(axis=1) & (hin.sum(axis=1) > 0.0)

        def seen_through(h_src):
            """the child's adjacent column as the ring sees it through a parent column with
            layers h_src: remapped onto them and back onto the ring cell's layers"""
            return remap_column(remap_column(cin[:, m], hin[m], h_src[m], sc), h_src[m], hd[m], sc)

        lo = hi = None
        back = None
        if self.ring_slopes and not self._extrap:
            # minmod slopes per layer of the anomalies, taken on each parent column's own layers: the
            # parent column (the ring cell's and its W, E, S, N neighbours) minus the child's adjacent
            # column remapped onto those layers, then remapped once onto the ring cell's layers. The
            # remap's errors (coarse layers of other depths; a horizontally uniform field) are in both
            # parts and cancel; a one-sided difference counts only in layers above the bottom of the
            # neighbour, the ring cell and the child's column, and is zero towards land or where the
            # child has no adjacent column. The range of the five columns on the ring cell's layers is
            # the ring value plus each difference (12 remaps per ring cell and tracer; the neighbours
            # themselves are not remapped). With the child detail, the ring cell's own anomaly is
            # what the parent adds to the child's column.
            hks = [buf[k, 0][ok] for k in range(1, 5)]
            wks = [hk.sum(axis=1) > 0.0 for hk in hks]
            if m.any():
                nm = int(m.sum())
                need = [np.ones(nm, bool)] + [w[m] for w in wks]  # land neighbours: nothing to remap
                N = np.concatenate(need)
                H = np.concatenate([hs[m]] + [h[m] for h in hks])[N]
                Cp = np.concatenate([cs[:, m]] + [buf[k, 1:][:, ok][:, m] for k in range(1, 5)], axis=1)[:, N]
                idx = np.tile(np.arange(nm), 5)[N]
                p_ = remap_column(cin[:, m][:, idx], hin[m][idx], H, sc)  # the child's column on the parent's layers
                A = remap_column(Cp - p_, H, hd[m][idx], sc)  # the anomalies on the ring cell's layers
                st = np.zeros((A.shape[0], 5 * nm, A.shape[2]))
                st[:, N] = A
                a0 = st[:, :nm]
                zb = np.cumsum(hd[m], axis=1)
                dmax = np.minimum(hd[m].sum(axis=1), hin[m].sum(axis=1))
                nb = []  # each neighbour minus the ring cell, on the ring cell's layers
                for k in range(4):
                    lim = np.minimum(dmax, hks[k][m].sum(axis=1))
                    valid = (zb <= lim[:, None] + 1e-9)[None] & wks[k][m][None, :, None]
                    nb.append(np.where(valid, st[:, (k + 1) * nm:(k + 2) * nm] - a0, 0.0))
                sx = minmod(-nb[0], nb[1])  # W: backward, E: forward
                sy = minmod(-nb[2], nb[3])  # S: backward, N: forward
                cu = c[:, m]
                zero = np.zeros_like(cu)
                lo_m = cu + np.minimum.reduce([zero] + nb)
                hi_m = cu + np.maximum.reduce([zero] + nb)
                c[:, m] = np.clip(cu + sx * self._ring_ox[ok][m][None, :, None] + sy * self._ring_oy[ok][m][None, :, None], lo_m, hi_m)
                back = cu - a0  # the child's column as the ring sees it through the parent's (for the child detail)
        elif self.ring_slopes or self._extrap:
            # the neighbours on the ring cell's layers (land: the cell itself)
            # all four in one remap (the columns are independent)
            hks = [buf[k, 0][ok] for k in range(1, 5)]
            wks = [hk.sum(axis=1) > 0.0 for hk in hks]
            W = np.concatenate(wks)
            nbs = [c.copy() for _ in range(4)]
            if W.any():
                r = remap_column(np.concatenate([buf[k, 1:][:, ok] for k in range(1, 5)], axis=1)[:, W],
                                 np.concatenate(hks)[W], np.tile(hd, (4, 1))[W], self.scheme)
                pos = np.cumsum(W) - 1
                n0 = c.shape[1]
                for k in range(4):
                    sel = W[k * n0:(k + 1) * n0]
                    nbs[k][:, sel] = r[:, pos[k * n0:(k + 1) * n0][sel]]
            lo = np.minimum.reduce([c] + nbs)  # the range of the five columns (land: the cell itself)
            hi = np.maximum.reduce([c] + nbs)
            if self.ring_slopes:
                # minmod slopes per layer of the anomalies: each parent column minus the child's
                # adjacent column seen through that parent column's layers. The remap's errors
                # (coarse layers of other depths; a horizontally uniform field) are the same in
                # both and cancel; a one-sided difference counts only in layers above the bottom
                # of the neighbour, the ring cell and the child's column, and is zero towards land
                # or where the child has no adjacent column. The result stays within the range of
                # the five columns. (Without the anomalies: slopes of 0.04-0.08 C per parent cell
                # in IDEAL V2 at rest, 8-11x larger.)
                sx = np.zeros_like(c)
                sy = np.zeros_like(c)
                if m.any():
                    # the child's column seen through the ring cell's parent column and its four neighbours,
                    # in two remaps of the stacked columns; a neighbour with exactly the ring cell's parent
                    # layers sees it the same way (the same numbers), so only the others are remapped
                    hk5 = [hs] + [np.where(wks[k][:, None], hks[k], hs) for k in range(4)]
                    nm = int(m.sum())
                    hsm = hs[m]
                    need = [np.ones(nm, bool)] + [~(h[m] == hsm).all(axis=1) for h in hk5[1:]]
                    N = np.concatenate(need)
                    H = np.concatenate([h[m] for h in hk5])[N]
                    idx = np.tile(np.arange(nm), 5)[N]
                    r = remap_column(remap_column(cin[:, m][:, idx], hin[m][idx], H, sc),
                                     H, hd[m][idx], sc)
                    st = np.empty((r.shape[0], 5 * nm, r.shape[2]))
                    st[:, N] = r
                    back = st[:, :nm]
                    for k in range(1, 5):
                        same = ~need[k]
                        st[:, k * nm:(k + 1) * nm][:, same] = back[:, same]
                    self.n_seen_reused += int((~N).sum())
                    a0 = c[:, m] - back
                    zb = np.cumsum(hd[m], axis=1)
                    dmax = np.minimum(hd[m].sum(axis=1), hin[m].sum(axis=1))
                    d = []
                    for k in range(4):
                        hk = hk5[k + 1]
                        ak = nbs[k][:, m] - st[:, (k + 1) * nm:(k + 2) * nm]
                        dk = (ak - a0) if k in (1, 3) else (a0 - ak)  # E, N: forward; W, S: backward
                        lim = np.minimum(dmax, hk[m].sum(axis=1))
                        dk = np.where((zb <= lim[:, None] + 1e-9)[None] & wks[k][m][None, :, None], dk, 0.0)
                        d.append(dk)
                    sx[:, m] = minmod(d[0], d[1])
                    sy[:, m] = minmod(d[2], d[3])
                c = np.clip(c + sx * self._ring_ox[ok][None, :, None] + sy * self._ring_oy[ok][None, :, None], lo, hi)
        if m.any():
            direct = remap_column(cin[:, m], hin[m], hd[m], sc)
            if back is None:
                back = seen_through(hs)
            # limited to the range of the parent's value and the child's adjacent one
            # (layer by layer): the correction adds no new extrema (IDEAL V6)
            c0 = c[:, m]
            c[:, m] = np.clip(c0 + direct - back, np.minimum(c0, direct), np.maximum(c0, direct))
        jo, io = rj[ok], ri[ok]
        for f, cf in zip(self.tracers, c):
            old = vremap.gather_columns(f.all_values, jo, io)
            self.max_ring_change = max(self.max_ring_change, float(np.abs(cf - old).max(initial=0.0)))
            vremap.scatter_columns(f.all_values, jo, io, cf)
        if self._extrap:
            self._store_ring(ok, c, np.minimum(lo, c), np.maximum(hi, c))

    # --- the ring in time (ring_time "extrapolate") ---------------------------------

    def _store_ring(self, ok, c, lo, hi):
        """Keep the ring of this sync (all ring cells, NaN where not set) and move
        the previous one back; the previous one is also written to the restart arrays."""
        shape = (len(self.tracers), self._rj.size, self.nz)
        new = np.full(shape, np.nan)
        new[:, ok] = c
        lo_, hi_ = np.full(shape, np.nan), np.full(shape, np.nan)
        lo_[:, ok], hi_[:, ok] = lo, hi
        prev = self._rc[0] if self._rc is not None else np.full(shape, np.nan)
        self._rc = (new, prev, lo_, hi_)
        self._prev_to_restart()

    def _restart_ring_arrays(self):
        """Arrays (part of the model state) with the previous ring, per tracer, at the
        ring cells, and where it is set."""
        self._rs3 = None
        if self.up is None or not self._extrap:
            return
        T = self.sim.T
        attrs = dict(_part_of_state=True, _valid_at=tuple(CellType))  # nothing masked: exact values back
        n = len(self._tracers(self.sim))
        self._rs3 = [T.array(name=f"nest_ring3d_old_{k}", z=CENTERS, fill=0.0,
                             long_name=f"nesting: 3D ring of tracer {k} at the previous sync", attrs=dict(attrs))
                     for k in range(n)]
        self._rs3_set = T.array(name="nest_ring3d_old_set", fill=0.0,
                                long_name="nesting: previous 3D ring stored (1) or not (0)", attrs=dict(attrs))

    def _deferred_arrays(self):
        """Arrays (part of the model state) with the amounts of the flux correction
        deferred by the limiter, per tracer (levels with children)."""
        self._deferred = None
        self._acc = None
        self._corr_cells = []
        if not self.down:
            return
        T = self.sim.T
        attrs = dict(_part_of_state=True, _restart_optional=True, _valid_at=tuple(CellType))
        n = len(self._tracers(self.sim))
        self._deferred = [T.array(name=f"nest_flux_deferred_{k}", z=CENTERS, fill=0.0,
                                  long_name=f"nesting: flux correction of tracer {k} deferred by the limiter (amount)",
                                  attrs=dict(attrs))
                          for k in range(n)]

    def deferred_content(self) -> np.ndarray:
        """The deferred amounts of every tracer in the cells this level computes
        (this process); add them to :meth:`level_content` for conservation."""
        n = len(self.tracers)
        if self._deferred is None:
            return np.zeros(n)
        h = self.halo
        return np.array([float(a.all_values[:, h:-h, h:-h][:, self.computed].sum()) for a in self._deferred])

    def _apply_corrections(self):
        """The limited flux correction of this sync (see the constructor)."""
        if self._deferred is None:
            return
        cells = list(self._corr_cells)
        self._corr_cells = []
        pend = np.zeros(self.wet.shape, bool)
        for a in self._deferred:
            pend |= (a.all_values != 0.0).any(axis=0)
        if pend.any():
            cells.append(np.nonzero(pend))
        if not cells:
            return
        J = np.concatenate([c[0] for c in cells])
        I = np.concatenate([c[1] for c in cells])
        key = np.unique(J * self.wet.shape[1] + I)
        J, I = key // self.wet.shape[1], key % self.wet.shape[1]
        T = self.sim.T
        ny, nx = self.wet.shape
        vol = T.hn.all_values[:, J, I] * T.area.all_values[J, I]  # (nz, n)
        ok = vol > 0.0
        for n, f in enumerate(self.tracers):
            a = f.all_values
            c = a[:, J, I]
            lo, hi = c.copy(), c.copy()
            for dj, di in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                j2, i2 = np.clip(J + dj, 0, ny - 1), np.clip(I + di, 0, nx - 1)
                w = self.wet[j2, i2] & (T.hn.all_values[:, j2, i2] > 0.0)
                v = np.where(w, a[:, j2, i2], c)
                lo, hi = np.minimum(lo, v), np.maximum(hi, v)
            for dk in (-1, 1):  # the layers below and above (layer 0 at the bottom)
                v = np.roll(c, dk, axis=0)
                wk = np.roll(ok, dk, axis=0)
                if dk == -1:
                    v[-1], wk[-1] = c[-1], True
                else:
                    v[0], wk[0] = c[0], True
                v = np.where(wk, v, c)
                lo, hi = np.minimum(lo, v), np.maximum(hi, v)
            acc = self._acc[n][:, J, I]
            dfr = self._deferred[n].all_values[:, J, I]
            with np.errstate(invalid="ignore", divide="ignore"):
                new = np.where(ok, c + acc + dfr / vol, c)
            lim = np.clip(new, lo, hi)
            rest = np.where(ok, (new - lim) * vol, 0.0)
            clipped = ok & (lim != new)
            # spread the column's net excess over its layers with room left within their range; defer what does not fit
            E = rest.sum(axis=0)
            up = np.where(ok, (hi - lim) * vol, 0.0)
            down = np.where(ok, (lim - lo) * vol, 0.0)
            room = np.where(E > 0.0, up, down)
            R = room.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                put = np.where(R > 0.0, np.minimum(np.abs(E), R) / R, 0.0)  # fraction of the room used
                lim = np.where(ok, lim + np.sign(E) * put * room / vol, lim)
                left = E - np.sign(E) * put * R  # same sign as E
                same = np.where(rest * E > 0.0, rest, 0.0)
                S = same.sum(axis=0)
                rest = np.where(S != 0.0, same * (left / S), 0.0)
            rest = np.where(ok, rest, dfr)  # a dry cell keeps its deferred amount
            a[:, J, I] = np.where(ok, lim, c)
            self._deferred[n].all_values[:, J, I] = rest
            self._acc[n][:, J, I] = 0.0
            self.n_clipped += int(clipped.sum())
            d = ok & (rest != 0.0)
            if d.any():
                self.max_deferred = max(self.max_deferred, float(np.abs(rest[d] / vol[d]).max()))

    def _prev_to_restart(self):
        if self._rs3 is None or not self._rj.size:
            return
        prev = self._rc[1]
        set_ = np.isfinite(prev).all(axis=(0, 2))
        for a, pv in zip(self._rs3, prev):
            vremap.scatter_columns(a.all_values, self._rj, self._ri, np.where(set_[:, None], pv, 0.0))
        self._rs3_set.all_values[self._rj, self._ri] = set_.astype(float)

    def _prev_from_restart(self):
        """The previous ring from the restart arrays (as they were before the initial sync)."""
        if self._rs3 is None or not self._rj.size:
            return None
        set_ = self._rs3_set.all_values[self._rj, self._ri] > 0.5
        if not set_.any():
            return None
        prev = np.stack([vremap.gather_columns(a.all_values, self._rj, self._ri) for a in self._rs3])
        prev[:, ~set_] = np.nan
        return prev

    def _extrapolate_ring(self, sim):
        """At the child's m-th macro step after a sync: the ring continued in time."""
        R = self.ratio_3d
        m = (sim.istep // sim.split_factor - 1) % R  # macro steps since the last sync, before this one
        if m == 0 or self._rc is None:
            return
        new, prev, lo, hi = self._rc
        a = m / R
        ok = np.isfinite(new).all(axis=(0, 2)) & np.isfinite(prev).all(axis=(0, 2)) & self.wet[self._rj, self._ri]
        if not ok.any():
            return
        c = np.clip(new[:, ok] + a * (new[:, ok] - prev[:, ok]), lo[:, ok], hi[:, ok])
        jo, io = self._rj[ok], self._ri[ok]
        for f, cf in zip(self.tracers, c):
            vremap.scatter_columns(f.all_values, jo, io, cf)
        self.n_extrapolated += 1

    # --- parent side -------------------------------------------------------------

    def _send_down(self, plan) -> List[MPI.Request]:
        reqs = []
        h = self.sim.T.hn.all_values
        for (p, tt), (pj, pi, *_rest) in plan.pro.items():
            if p != self.t:
                continue
            jj, ii = self.loc(pj, pi)
            # the source column, and with ring_slopes its W, E, S, N neighbours (land: thickness 0)
            nb = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)) if (self.ring_slopes or self._extrap) else ((0, 0),)
            buf = np.empty((len(nb), 1 + len(self.tracers), jj.size, h.shape[0]))  # columns surface first
            ny, nx = self.wet.shape
            for k, (dj, di) in enumerate(nb):
                j2 = np.clip(jj + dj, 0, ny - 1)
                i2 = np.clip(ii + di, 0, nx - 1)
                vremap.gather_columns(h, j2, i2, out=buf[k, 0])
                for n, f in enumerate(self.tracers):
                    vremap.gather_columns(f.all_values, j2, i2, out=buf[k, 1 + n])
                if k:
                    buf[k, 0][~self.wet[j2, i2]] = 0.0
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
            nk = k.size
            buf = np.empty((1 + nt) * nk * nzc + nk)
            self._recv(plan.comm, buf, plan.coff + tt, plan.tag_r + TAG_3D)
            m = a * r + b
            B = buf[: (1 + nt) * nk * nzc].reshape(1 + nt, nk, nzc)
            H[k, m] = B[0]
            C[:, k, m] = B[1:]
            W[k, m] = buf[-nk:]
        jj, ii = self.loc(mine[:, 0], mine[:, 1])
        hp = vremap.gather_columns(self.sim.T.hn.all_values, jj, ii)  # (n, nzp), surface first
        ok = (W.sum(axis=1) > 0.0) & self.wet[jj, ii] & (hp.sum(axis=1) > 0.0)
        if not ok.any():
            return
        H, C, W, hp = H[ok], C[:, ok], W[ok], hp[ok]
        # land child columns (weight 0): any positive thicknesses, they add nothing
        dry = (W <= 0.0) | (H.sum(axis=-1) <= 0.0)
        W = np.where(dry, 0.0, W)
        H = np.where(dry[..., None], hp.sum(axis=-1)[:, None, None] / nzc, H)
        c = remap_conc_z(C, H, hp, W, self.child_scheme[self.down.index(plan)])
        jo, io = jj[ok], ii[ok]
        for f, cf in zip(self.tracers, c):
            vremap.scatter_columns(f.all_values, jo, io, cf)

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
        if m_in.size:
            oj, oi = self.loc(wf["out"][0][m_in], wf["out"][1][m_in])
            T = self.sim.T
            vol = (T.hn.all_values[::-1, oj, oi] * T.area.all_values[oj, oi]).T  # (n, nz), surface first
            sign = wf["sign"][m_in][:, None]
            kk = np.arange(self.nz)[::-1]
            dV = self._volume_profiles(*(a[ins] for a in args)) - fc[-1][ins]  # water, parent - child (n, nz)
            for n, (f, fl) in enumerate(zip(self.tracers, self.fluxes)):
                fp = self._face_profiles(fl, *(a[ins] for a in args))
                c = f.all_values[::-1, oj, oi].T
                amount = fp - fc[n][ins] - c * dV
                with np.errstate(invalid="ignore", divide="ignore"):
                    dc = np.where(vol > 0.0, sign * amount / vol, 0.0)
                self.max_correction = max(self.max_correction, float(np.abs(dc).max(initial=0.0)))
                # np.add.at in face order (several faces may share a cell)
                if self._deferred is not None:  # limited: collected, applied after all children
                    if self._acc is None:
                        self._acc = [np.zeros_like(g.all_values) for g in self.tracers]
                    np.add.at(self._acc[n], (kk[None, :], oj[:, None], oi[:, None]), dc)
                else:
                    np.add.at(f.all_values, (kk[None, :], oj[:, None], oi[:, None]), dc)
            if self._deferred is not None:
                self._corr_cells.append((oj, oi))
        up = ~ins & (wf["up"][flist] >= 0)
        if up.any():
            for n in range(nt):  # tracers and water, in this level's layers
                np.add.at(self.reg[n], wf["up"][flist][up], fc[n][up])

    # --- 3D transports at the interface (momentum_3d stage) ------------------------

    def _add_water_up(self):
        """Add this macro step's water through the edge faces (every macro step)."""
        e = self.lev.edge
        for key, (_, _, fk) in self.up.flx.items():
            if key[0] != self.t:
                continue
            vol = self._volume_profiles(e["is_x"][fk], e["face"][0][fk], e["face"][1][fk])
            if key in self._water_sum:
                self._water_sum[key] += vol
            else:
                self._water_sum[key] = vol

    def _send_transports_up(self) -> List[MPI.Request]:
        reqs = []
        e = self.lev.edge
        for key, (_, _, fk) in self.up.flx.items():
            tt, p = key
            if tt != self.t:
                continue
            args = (e["is_x"][fk], e["face"][0][fk], e["face"][1][fk])
            h = self._face_profiles((self.sim.U.hn, self.sim.V.hn), *args)
            h = np.where(np.isfinite(h), h, 0.0)
            # m3 s-1 per layer, the mean over the parent's macro step
            q = self._water_sum.pop(key) / (self.sim.macrotimestep * self.ratio_3d)
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
        from .nested2d import PointTransfer, ring_faces

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
        reqs = []
        if self.up is not None:
            self._add_water_up()
            if self._up_active():
                reqs = self._send_transports_up()
        changed = False
        for plan, nzc in zip(self.down, self.child_nz):
            changed |= self._recv_transports(plan, nzc)
        MPI.Request.Waitall(reqs)
        ring = self.up is not None and self._ring_uv is not None
        if ring:
            self._set_ring_uv(sim)
        if self.down or ring:
            # the simulation then updates the momentum diagnostics (w, velocities,
            # friction, advection, Coriolis) from the replaced transports
            mom = sim.momentum
            mom.pk.update_halos()
            mom.qk.update_halos()
        self.t_transport += MPI.Wtime() - t0

    # --- sync --------------------------------------------------------------------

    def _sync(self):
        t0 = MPI.Wtime()
        reqs = []
        up = self._up_active()
        if self._fluxes_ready and self.up is not None:
            self._own_edge_fluxes()
        if up and not self._up_after_children:
            reqs += self._send_up()
        for plan, nzc in zip(self.down, self.child_nz):
            self._recv_rim(plan, nzc)
            if self._fluxes_ready:
                self._match_fluxes(plan, nzc)
        if self.down and self._fluxes_ready:
            self._apply_corrections()
        if up and self._up_after_children:
            reqs += self._send_up()
        if self.down:
            for f in self.tracers:
                f.update_halos()
            for plan, tx in zip(self.down, self._ring_tx_down):
                reqs += self._send_down(plan)
                reqs += self._send_ring_uv(plan, tx)
        if up:
            for f in self.tracers:  # the interior cells next to the ring may lie in the halo
                f.update_halos()
            self._recv_ring()
            self._recv_ring_uv()
            for f in self.tracers:
                f.update_halos()
        MPI.Request.Waitall(reqs)
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
        parent = self.lev.parent.name if self.lev.parent is not None else None
        dts = world.allgather((self.lev.name, float(sim.macrotimestep), int(T.hn.all_values.shape[0]),
                               len(self.tracers), (self.ring_slopes, self.ring_time), parent, int(sim.split_factor), self.lev.time_ratio,
                               self.scheme))
        # checked for all levels on all processes, so that all stop
        self.ratio_3d = macro_ratios({d[0]: (d[5], d[1], d[6], d[7]) for d in dts}).get(self.lev.name, 1)
        if len({d[3] for d in dts}) != 1:
            raise ValueError(f"Nested3D: the levels exchange different numbers of tracers: {dts}")
        if len({d[4] for d in dts}) != 1:
            raise ValueError(f"Nested3D: ring_slopes and ring_time must be the same on all levels: {dts}")
        nz = {d[0]: d[2] for d in dts}
        self.nz = nz[self.lev.name]
        self.parent_nz = nz[self.lev.parent.name] if self.lev.parent is not None else None
        self.child_nz = [nz[plan.child.name] for plan in self.down]
        # the remap scheme belongs to the interface: the child's decides its ring and its rim in the parent
        self.child_scheme = [{d[0]: d[8] for d in dts}[plan.child.name] for plan in self.down]
        self._up_after_children = any(
            bool((np.asarray(self.lev.wfaces[plan.child.name]["up"]) >= 0).any()) for plan in self.down)
        self._ring = []
        if self.up is not None:
            self._ring = [(p, v) for (p, tt), v in sorted(self.up.pro.items()) if tt == self.t]
            cat = lambda k: np.concatenate([v[k] for _, v in self._ring]) if self._ring else np.zeros(0, int)
            self._rj, self._ri = self.loc(cat(2), cat(3))
            r = self.lev.ratio
            self._ring_ox, self._ring_oy = subcell_offset(cat(5), r), subcell_offset(cat(4), r)  # x: b, y: a
            # the child's interior cell next to each ring cell (within the halo)
            gj = np.clip(cat(2), 0, self.lev.ny - 1)
            gi = np.clip(cat(3), 0, self.lev.nx - 1)
            self._rin_ok = self.lev.computed[gj, gi]
            self._rin_j, self._rin_i = self.loc(gj, gi)
        self.fluxes = []
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
        prev = self._prev_from_restart()  # before the initial sync overwrites the arrays
        self.n_extrapolated = 0
        self._sync()  # initial: no fluxes yet
        if prev is not None and self._rc is not None:  # after a restart: the ring before the last sync
            self._rc = (self._rc[0], prev, self._rc[2], self._rc[3])
            self._prev_to_restart()
        self._fluxes_ready = True
        self.max_ring_change = 0.0
        self.t_sync = self.t_wait = 0.0
        self.parts = {}  # the start-up sync does not count
        self.n_sync = 0

    def on_stage(self, sim, stage, macro_active):
        if stage == "momentum_3d_transports" and macro_active:
            self._sync_transports(sim)
        elif stage == "momentum_3d" and macro_active and self._extrap and self.up is not None and self.ratio_3d > 1:
            self._extrapolate_ring(sim)
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
        return dict(ratio_3d=self.ratio_3d, ring_time=self.ring_time, n_extrapolated=getattr(self, "n_extrapolated", 0), sync=self.t_sync, wait=self.t_wait, n_sync=self.n_sync, max_ring_change=self.max_ring_change,
                    max_correction=self.max_correction, n_clipped=self.n_clipped, max_deferred=self.max_deferred, n_seen_reused=self.n_seen_reused, transport=self.t_transport,
                    parts={k.lstrip("_"): v[0] for k, v in self.parts.items()},
                    parts_wait={k.lstrip("_"): v[1] for k, v in self.parts.items()})
