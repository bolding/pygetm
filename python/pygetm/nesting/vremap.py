"""
Conservative vertical remapping between water columns whose layers differ
(pygetm-nesting docs/plan_21_3d_exchange.md). Numpy only, vectorised over columns.

Columns are given by their layer thicknesses ``h[..., nz]`` from the surface
down; interfaces are depths below the surface, ``z[..., nz + 1]`` with
``z[..., 0] = 0``.

Piecewise-constant (PCM) remap through the cumulative content Q(z):
the content between two target interfaces is Q(z_b) - Q(z_t), with Q linear
inside each source layer. Exactly conservative (the sum telescopes) and
monotone (no new extrema).

The work is split so that it can be shared between scalars:

- :func:`prepare` (once per pair of layer distributions): where every target
  interface lies in the source column, as (layer index, fraction). It depends
  only on the layer thicknesses, so T, S and all FABM tracers share it.
- :func:`apply` (per scalar): cumulative content at the source interfaces,
  gathered and interpolated at the prepared positions, differenced.

The rim (child -> parent) in the two forms of the plan:

- :func:`rim_average` (A): average layer heights and contents of the r x r
  child columns by layer index, then one remap onto the parent's layers;
- :func:`rim_columns` (B): every child column stretched to the parent's depth
  and remapped onto the parent's layers, contents added area-weighted.
  ``keep`` > 0 gives B': the upper ``keep`` x min(D_c, D_p) of the column is
  not stretched, only the part below.

Both conserve the children's content exactly. With sigma layers in the
child (fixed fractions of the depth) A and B are the same: the stretched
child layer k lands exactly on layer k of the mean column.
"""

import numpy as np


def interfaces(h):
    """Depths of the layer interfaces (0 at the surface), shape [..., nz + 1]."""
    h = np.asarray(h, dtype=float)
    z = np.zeros(h.shape[:-1] + (h.shape[-1] + 1,))
    np.cumsum(h, axis=-1, out=z[..., 1:])
    return z


def stretch(z, D_from, D_to, keep=0.0):
    """Map interface depths of a column of depth ``D_from`` onto one of depth
    ``D_to``: linear (keep 0), or identity down to ``keep`` x min(D_from, D_to)
    and linear below (B')."""
    z = np.asarray(z, dtype=float)
    D_from = np.asarray(D_from, dtype=float)[..., None]
    D_to = np.asarray(D_to, dtype=float)[..., None]
    if keep <= 0.0:
        return z * (D_to / D_from)
    z0 = keep * np.minimum(D_from, D_to)
    below = z0 + (z - z0) * (D_to - z0) / (D_from - z0)
    return np.where(z <= z0, z, below)


def prepare(z_src, z_dst):
    """Positions of the target interfaces in the source column:
    (layer index j, fraction f) with z_dst = z_src[j] + f h_src[j].

    z_src [..., ns + 1], z_dst [..., nd + 1] (same leading shape; both start at
    0 and end at the same depth up to round-off)."""
    z_src = np.asarray(z_src, dtype=float)
    z_dst = np.asarray(z_dst, dtype=float)
    ns = z_src.shape[-1] - 1
    z_dst = np.broadcast_to(z_dst, z_src.shape[:-1] + z_dst.shape[-1:])
    # source layer holding each target interface: one searchsorted over all columns,
    # each column shifted by its index x (deepest interface + 1) so that the
    # concatenation is sorted
    lead = z_src.shape[:-1]
    n = int(np.prod(lead)) if lead else 1
    zs, zd = z_src.reshape(n, -1), z_dst.reshape(n, -1)
    span = max(float(np.max(zs)), float(np.max(zd))) + 1.0
    off = np.arange(n)[:, None] * span
    k = np.searchsorted((zs[:, 1:-1] + off).ravel(), (zd + off).ravel(), side="right")
    j = (k.reshape(n, -1) - np.arange(n)[:, None] * (ns - 1)).reshape(z_dst.shape)
    j = np.clip(j, 0, ns - 1)
    zt = np.take_along_axis(z_src, j, axis=-1)
    h = np.take_along_axis(np.diff(z_src, axis=-1), j, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(h > 0.0, (z_dst - zt) / h, 0.0)
    return j, np.clip(f, 0.0, 1.0)


def ppm_edges(c, h):
    """Left (upper) and right (lower) edge values of the limited piecewise-parabolic
    reconstruction (Colella and Woodward 1984, eqs. 1.6-1.10, for layers of unequal
    thickness) of layer means ``c [..., n]`` with thicknesses ``h [..., n]``. The
    outermost two layers and layers next to empty ones are piecewise constant. The
    parabola in each layer keeps its mean, and the limiter adds no new extrema."""
    c = np.asarray(c, dtype=float)
    h = np.broadcast_to(np.asarray(h, dtype=float), c.shape)
    n = c.shape[-1]
    aL = c.copy()
    aR = c.copy()
    if n < 4:
        return aL, aR
    with np.errstate(invalid="ignore", divide="ignore"):
        hm, h0, hp = h[..., :-2], h[..., 1:-1], h[..., 2:]
        cm, c0, cp = c[..., :-2], c[..., 1:-1], c[..., 2:]
        # slopes of the inner layers (1.7), monotonised (1.8)
        da = h0 / (hm + h0 + hp) * ((2 * hm + h0) / (hp + h0) * (cp - c0) + (h0 + 2 * hp) / (hm + h0) * (c0 - cm))
        ok = (cp - c0) * (c0 - cm) > 0.0
        dm = np.where(ok, np.sign(da) * np.minimum(np.abs(da), 2 * np.minimum(np.abs(c0 - cm), np.abs(cp - c0))), 0.0)
        dm = np.where(np.isfinite(dm), dm, 0.0)
        d = np.zeros_like(c)
        d[..., 1:-1] = dm
        # edge between layer j and j+1 (1.6), for j = 1 .. n-3 (needs j-1 .. j+2)
        hj_1, hj, hj1, hj2 = h[..., :-3], h[..., 1:-2], h[..., 2:-1], h[..., 3:]
        cj, cj1 = c[..., 1:-2], c[..., 2:-1]
        dj, dj1 = d[..., 1:-2], d[..., 2:-1]
        z1 = (hj_1 + hj) / (2 * hj + hj1)
        z2 = (hj2 + hj1) / (2 * hj1 + hj)
        e = (cj + hj / (hj + hj1) * (cj1 - cj)
             + 1.0 / (hj_1 + hj + hj1 + hj2) * (2 * hj1 * hj / (hj + hj1) * (z1 - z2) * (cj1 - cj)
                                                 - hj * z1 * dj1 + hj1 * z2 * dj))
        good = np.isfinite(e) & (hj_1 > 0) & (hj > 0) & (hj1 > 0) & (hj2 > 0)
        e = np.where(good, e, 0.5 * (cj + cj1))
    # inner layers 2 .. n-3 get both edges from e; the outer two layers stay constant
    aR[..., 2:-2] = e[..., 1:]       # lower edge of layer j = edge j+1/2
    aL[..., 2:-2] = e[..., :-1]      # upper edge of layer j = edge j-1/2
    inner = np.zeros(n, dtype=bool)
    inner[2:-2] = True
    # limiter (1.10)
    a = c
    flat = ((aR - a) * (a - aL) <= 0.0) | ~inner
    aL = np.where(flat, a, aL)
    aR = np.where(flat, a, aR)
    dA = aR - aL
    a6 = 6.0 * (a - 0.5 * (aL + aR))
    aL = np.where(dA * a6 > dA * dA, 3.0 * a - 2.0 * aR, aL)
    aR = np.where(-dA * dA > dA * a6, 3.0 * a - 2.0 * aL, aR)
    empty = ~(h > 0.0)
    return np.where(empty, a, aL), np.where(empty, a, aR)


def apply(prep, c_src, h_src, scheme="pcm"):
    """Contents of the target layers (per unit area) from source concentrations
    ``c_src [..., ns]`` with thicknesses ``h_src``; ``c_src`` may carry extra
    leading tracer axes (broadcast against the prepared positions).

    ``scheme``: the reconstruction inside each source layer, "pcm" (constant) or
    "ppm" (limited parabola, :func:`ppm_edges`). Both keep the column content
    exactly and add no new extrema."""
    j, f = prep
    c_src = np.asarray(c_src, dtype=float)
    q = c_src * h_src
    Q = np.concatenate([np.zeros(q.shape[:-1] + (1,)), np.cumsum(q, axis=-1)], axis=-1)
    jj = np.broadcast_to(j, q.shape[:-1] + j.shape[-1:])
    ff = np.broadcast_to(f, jj.shape)
    if scheme == "ppm":
        aL, aR = ppm_edges(c_src, h_src)
        dA = aR - aL
        a6 = 6.0 * (c_src - 0.5 * (aL + aR))
        hh = np.broadcast_to(h_src, q.shape)
        g = lambda x: np.take_along_axis(x, jj, axis=-1)
        part = g(aL) * ff + g(dA) * ff ** 2 / 2 + g(a6) * (ff ** 2 / 2 - ff ** 3 / 3)
        Qz = np.take_along_axis(Q, jj, axis=-1) + g(hh) * part
    elif scheme == "pcm":
        Qz = np.take_along_axis(Q, jj, axis=-1) + ff * np.take_along_axis(q, jj, axis=-1)
    else:
        raise ValueError(f"unknown remap scheme {scheme!r}")
    # the outer interfaces exactly: no content lost to round-off at the bottom
    Qz[..., 0] = 0.0
    Qz[..., -1] = Q[..., -1]
    return np.diff(Qz, axis=-1)


def remap(h_src, c_src, h_dst, scheme="pcm"):
    """Concentrations on the target layers (one call, no sharing)."""
    prep = prepare(interfaces(h_src), interfaces(h_dst))
    with np.errstate(invalid="ignore", divide="ignore"):
        return apply(prep, c_src, h_src, scheme) / h_dst


# --- the rim (child -> parent) --------------------------------------------------------------
#
# Child columns of one parent cell are along axis -2 (r*r, land columns weight 0):
#   h_c [..., m, nzc], c_c [(tracers,) ..., m, nzc], a_c [..., m]; parent layers h_p [..., nzp].


def rim_average_prepare(h_c, a_c, h_p):
    """A, the part shared by all scalars: mean layer heights and their
    positions in the parent's column."""
    w = a_c / a_c.sum(axis=-1, keepdims=True)
    hbar = (w[..., None] * h_c).sum(axis=-2)
    # the mean column has the parent's depth when the depths are consistent; scale
    # out the round-off so that both columns end at the same depth
    hbar = hbar * (h_p.sum(axis=-1, keepdims=True) / hbar.sum(axis=-1, keepdims=True))
    return w, hbar, prepare(interfaces(hbar), interfaces(h_p))


def rim_average_apply(shared, c_c, h_c, h_p):
    """A, per scalar: contents averaged by layer index, then remapped."""
    w, hbar, prep = shared
    Cbar = (w[..., None] * c_c * h_c).sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cbar = Cbar / hbar
    return apply(prep, cbar, hbar) / h_p


def rim_columns_prepare(h_c, a_c, h_p, keep=0.0):
    """B, the part shared by all scalars: every child column stretched to the
    parent's depth, and the parent's interfaces located in each stretched
    child column. Also the child water volume each parent layer receives
    (per unit area), ``V``: equal to the parent's layer thickness for the
    linear stretch with consistent depths, not for B'."""
    w = a_c / a_c.sum(axis=-1, keepdims=True)
    Dc = h_c.sum(axis=-1)
    Dp = h_p.sum(axis=-1)
    zc = stretch(interfaces(h_c), Dc, Dp[..., None], keep)
    zp = np.broadcast_to(interfaces(h_p)[..., None, :], zc.shape[:-1] + (h_p.shape[-1] + 1,))
    prep = prepare(zc, zp)
    V = (w[..., None] * apply(prep, np.ones_like(h_c), h_c)).sum(axis=-2)
    return w, prep, V


def rim_columns_apply(shared, c_c, h_c, h_p, by_volume=False, fix=None):
    """B, per scalar: the content of every child layer (c h_c) spread evenly
    over its stretched layer, the parts inside each parent layer added
    area-weighted.

    Default: concentration = content / parent layer thickness. The sum over
    the parent's layers is the children's content exactly.
    ``by_volume``: content / child volume received (``V``): a uniform field
    stays uniform exactly, but the content is conserved only when V = h_p
    (linear stretch, consistent depths); for B' it is not. ``fix`` ("add" or
    "scale") then restores the content exactly (B'')."""
    w, prep, V = shared
    C = (w[..., None] * apply(prep, c_c, h_c)).sum(axis=-2)
    if not by_volume:
        return C / h_p
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(V > 0.0, C / V, 0.0)
    if fix is None:
        return c
    # B'': by volume, then the column's content restored exactly (uniform fields stay
    # uniform, since then nothing is missing): "scale" multiplies (keeps the sign,
    # for positive tracers), "add" shifts by the missing content / D_p
    total = C.sum(axis=-1, keepdims=True)
    have = (c * h_p).sum(axis=-1, keepdims=True)
    if fix == "scale":
        with np.errstate(invalid="ignore", divide="ignore"):
            return c * np.where(have != 0.0, total / have, 1.0)
    return c + (total - have) / h_p.sum(axis=-1, keepdims=True)


def contents(zs, zd, c, h, scheme="pcm"):
    """Contents of the target layers (per unit area) for source interfaces ``zs [..., ns+1]`` and
    target interfaces ``zd [..., nd+1]`` (depths below the surface), concentrations ``c [(tracers,)
    ..., ns]`` and source thicknesses ``h [..., ns]``: the same as ``apply(prepare(zs, zd), c, h,
    scheme)``, with the compiled kernel (``_pygetm.vertical_remap``) when it is available."""
    try:
        from .. import _pygetm

        kernel = _pygetm.vertical_remap
    except (ImportError, AttributeError):
        kernel = None
    zs = np.asarray(zs, dtype=float)
    zd = np.asarray(zd, dtype=float)
    h = np.asarray(h, dtype=float)
    c = np.asarray(c, dtype=float)
    if kernel is None:
        return apply(prepare(zs, zd), c, h, scheme)
    lead = np.broadcast_shapes(zs.shape[:-1], zd.shape[:-1], h.shape[:-1])
    ns, nd = zs.shape[-1] - 1, zd.shape[-1] - 1
    extra = c.shape[: c.ndim - len(lead) - 1]
    n = int(np.prod(lead)) if lead else 1
    nt = int(np.prod(extra)) if extra else 1
    zs2 = np.ascontiguousarray(np.broadcast_to(zs, lead + (ns + 1,)).reshape(n, ns + 1))
    zd2 = np.ascontiguousarray(np.broadcast_to(zd, lead + (nd + 1,)).reshape(n, nd + 1))
    h2 = np.ascontiguousarray(np.broadcast_to(h, lead + (ns,)).reshape(n, ns))
    c2 = np.ascontiguousarray(np.broadcast_to(c, extra + lead + (ns,)).reshape(nt, n, ns))
    out = np.empty((nt, n, nd))
    kernel(zs2, zd2, h2, c2, out, 1 if scheme == "ppm" else 0)
    return out.reshape(extra + lead + (nd,))
