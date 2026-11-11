"""Land/sea consistency between nesting levels (child-led).

Every level has a wet mask and depth on its extended shape (window plus the
halo band; array index = global index + halo). In the zones where parent and
child overlap they are reconciled, the child's finer bathymetry leading:

1. bottom-up (deepest level first): a cell of the parent's hole (the child's
   active region) is wet if any of its ratio x ratio child cells is wet, with
   the mean depth of the wet ones. The parent does not compute there; these
   cells define the parent's faces at the interface and the rim the child
   fills.
2. top-down: a cell of the child's ring can only be wet where the parent
   cell it is filled from is wet. Otherwise the child would take water or
   tracer through a face for which the parent has no value - the one
   mismatch flux matching cannot repair. Child cells that are neither active
   nor ring (parts of the window the parent keeps) are made dry.
3. faces: a parent interface face between two wet cells whose r child
   sub-faces are all closed (the child's cells along the face dry, or its
   ring cells) is closed in the parent too. Flux matching would set its
   transport to zero anyway, but the parent's momentum equation would keep a
   transport there and the parent's advection would use it. The faces are
   stored per level (:attr:`Level.closed_faces`) and closed in the level's
   domain mask before its grids are built
   (:meth:`pygetm.nesting.runtime.NestingRun.build`).

Interfaces and drying: the layout is assumed to keep every interface out of
water that floods and dries - drying happens inside a level, where pyGETM
handles it (the coupling has a safety net for it in 2D, see
:mod:`pygetm.nesting.nested2d`, but not in 3D). :func:`interface_depths`
checks this against ``interface_min_depth`` (m below the depth datum) of the
layout: per level, or for all levels under ``config:``. Choose it as the
largest fall of the water level below the datum (tidal amplitude plus
set-down by wind) plus a margin above pyGETM's Dcrit.
"""

from typing import Dict, Iterable

import numpy as np

from .geometry import Level


def make_consistent(levels: Iterable[Level], wet: Dict[str, np.ndarray], H: Dict[str, np.ndarray]):
    """Child-led consistency of wet masks and depths (levels parents first).
    Returns new (wet, H) and, per level, the number of cells changed."""
    levels = list(levels)
    wet = {k: v.copy() for k, v in wet.items()}
    H = {k: np.where(wet[k], v, np.nan) for k, v in H.items()}
    changes = {l.name: {"hole_wetted": 0, "hole_dried": 0, "band_dried": 0, "unused_dried": 0} for l in levels}

    for lev in reversed(levels):
        if lev.parent is None:
            continue
        h, p, r = lev.band, lev.parent, lev.ratio
        i0, i1, j0, j1 = lev.window
        cw = wet[lev.name][h:-h, h:-h].reshape(j1 - j0, r, i1 - i0, r)
        cH = H[lev.name][h:-h, h:-h].reshape(j1 - j0, r, i1 - i0, r)
        anywet = cw.any(axis=(1, 3))
        n = cw.sum(axis=(1, 3))
        Hmean = np.where(cw, cH, 0.0).sum(axis=(1, 3)) / np.where(n > 0, n, 1)
        hp = p.band
        sl = (slice(j0 + hp, j1 + hp), slice(i0 + hp, i1 + hp))
        act = lev.active
        if lev.grid is not None:
            # curvilinear: beyond its conforming band the child's cells lie elsewhere;
            # there the parent keeps its own land (it must be land: check_conforming)
            act = act & conforming_parent_cells(lev, wet[lev.name])
            if (wet[p.name][sl] & lev.active & ~act).any():
                raise ValueError(f"{lev.name}: {p.name} has water in the window beyond the conforming band "
                                 f"({int((wet[p.name][sl] & lev.active & ~act).sum())} cells): it must be land there")
        before = wet[p.name][sl].copy()
        changes[p.name]["hole_wetted"] += int((act & anywet & ~before).sum())
        changes[p.name]["hole_dried"] += int((act & ~anywet & before).sum())
        wet[p.name][sl] = np.where(act, anywet, before)
        H[p.name][sl] = np.where(act, np.where(anywet, Hmean, np.nan), H[p.name][sl])

    for lev in levels:
        if lev.parent is None:
            continue
        h, p = lev.band, lev.parent
        bj, bi = lev.ring_cells()
        pj, pi, _, _ = lev.ring_source(bj, bi)
        src_wet = wet[p.name][pj + p.band, pi + p.band]
        dry = wet[lev.name][bj + h, bi + h] & ~src_wet
        changes[lev.name]["band_dried"] += int(dry.sum())
        wet[lev.name][bj[dry] + h, bi[dry] + h] = False
        H[lev.name][bj[dry] + h, bi[dry] + h] = np.nan
        unused = lev.unused() & wet[lev.name]
        changes[lev.name]["unused_dried"] += int(unused.sum())
        wet[lev.name][unused] = False
        H[lev.name][unused] = np.nan

    for lev in levels:
        lev.closed_faces = closed_faces(lev, wet)  # rule 3 (masks are unchanged)
    return wet, H, changes


def _open_without_subfaces(lev: Level, c: Level, wet: Dict[str, np.ndarray]) -> np.ndarray:
    """Per interface face of child ``c`` in ``lev`` (``lev.wfaces[c.name]``):
    open in ``lev`` (both cells wet, the outside one computed by ``lev``)
    while all r sub-faces of the child are closed."""
    f = lev.wfaces[c.name]
    hp, h, r = lev.band, c.band, c.ratio
    (hj, hi), (oj, oi) = f["hole"], f["out"]
    ins = f["inside"]
    oj, oi = np.clip(oj, -hp, lev.ny - 1 + hp), np.clip(oi, -hp, lev.nx - 1 + hp)
    pw = wet[lev.name]
    popen = ins & pw[hj + hp, hi + hp] & pw[oj + hp, oi + hp]
    e = c.edge
    (cj, ci), s = e["inner"], np.asarray(e["sign"]).astype(int)
    rj = np.where(e["is_x"], cj, cj + s)
    ri = np.where(e["is_x"], ci + s, ci)
    cw = wet[c.name]
    sub = cw[cj + h, ci + h] & cw[rj + h, ri + h]
    return popen & ~sub.reshape(-1, r).any(axis=1)


def closed_faces(lev: Level, wet: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Interface faces of ``lev``'s children to close in ``lev`` (rule 3):
    ``is_x`` (U face) and ``face`` (j, i) as in
    :func:`pygetm.nesting.geometry.boundary_faces` (U point (j, i) between
    cells (j, i) and (j, i + 1), V point between (j, i) and (j + 1, i))."""
    is_x, fj, fi = [np.zeros(0, bool)], [np.zeros(0, int)], [np.zeros(0, int)]
    for c in lev.children:
        bad = _open_without_subfaces(lev, c, wet)
        f = lev.wfaces[c.name]
        is_x.append(f["is_x"][bad])
        fj.append(f["face"][0][bad])
        fi.append(f["face"][1][bad])
    return dict(is_x=np.concatenate(is_x), face=(np.concatenate(fj), np.concatenate(fi)))


def close_faces(lev: Level, domain) -> None:
    """Close :attr:`Level.closed_faces` in ``domain`` (the level's domain
    including its band; its supergrid mask, before the grids are built)."""
    cf = getattr(lev, "closed_faces", None)
    # domain.mask raises on processes other than the domain's rank 0 (no global mask there)
    if cf is None or cf["is_x"].size == 0 or getattr(domain, "_mask", None) is None:
        return
    h = lev.band
    fj, fi = cf["face"][0] + h, cf["face"][1] + h
    x = cf["is_x"]
    # supergrid: T (j, i) at (2j + 1, 2i + 1); U east of it at (2j + 1, 2i + 2), V north at (2j + 2, 2i + 1)
    domain.mask[2 * fj[x] + 1, 2 * fi[x] + 2] = 0
    domain.mask[2 * fj[~x] + 2, 2 * fi[~x] + 1] = 0


def check_dams(lev: Level, domain) -> None:
    """Dams (:mod:`pygetm.dams`: faces closed between two wet cells) must stay off the
    level's interface: a closed face between a computed and an external cell raises
    (the parent's and the child's fluxes there would differ). Faces between two
    external cells (the child's band, the parent's hole) are reopened, and the
    corners among them: the level that computes those cells has the dam itself.
    Call before :func:`close_faces` (which closes interface faces by rule 3).
    The global mask lives on the domain's rank 0 only (an error there aborts all levels)."""
    if domain.comm.rank != 0 or domain._mask is None:
        return
    m = domain.mask
    ext = np.asarray(lev.external(), dtype=bool)
    tw = m[1::2, 1::2] != 0
    for name, um, a, b, ea, eb in (
        ("U", m[1::2, 2:-1:2], tw[:, :-1], tw[:, 1:], ext[:, :-1], ext[:, 1:]),
        ("V", m[2:-1:2, 1::2], tw[:-1, :], tw[1:, :], ext[:-1, :], ext[1:, :]),
    ):
        dam = (um == 0) & a & b
        bad = dam & (ea != eb)
        if bad.any():
            j, i = np.argwhere(bad)[0]
            raise ValueError(f"{lev.name}: a dam closes {int(bad.sum())} {name} face(s) on the level's interface"
                             f" (first between domain cells (j {j}, i {i}) and the next); move it off the interface")
        um[dam & ea & eb] = 1
    xm = m[2:-1:2, 2:-1:2]  # inner corners
    t4 = tw[:-1, :-1] & tw[:-1, 1:] & tw[1:, :-1] & tw[1:, 1:]
    e4 = ext[:-1, :-1] & ext[:-1, 1:] & ext[1:, :-1] & ext[1:, 1:]
    xm[(xm == 0) & t4 & e4] = 1


def check_consistent(levels: Iterable[Level], wet: Dict[str, np.ndarray]):
    """Violations of the two rules per child level (all zero if consistent)."""
    out = {}
    for lev in levels:
        if lev.parent is None:
            continue
        h, p, r = lev.band, lev.parent, lev.ratio
        i0, i1, j0, j1 = lev.window
        cw = wet[lev.name][h:-h, h:-h].reshape(j1 - j0, r, i1 - i0, r)
        hole = wet[p.name][j0 + p.band : j1 + p.band, i0 + p.band : i1 + p.band]
        bj, bi = lev.ring_cells()
        pj, pi, _, _ = lev.ring_source(bj, bi)
        out[lev.name] = {
            "hole_mismatch": int((lev.active & (hole != cw.any(axis=(1, 3)))).sum()),
            "unused_wet": int((lev.unused() & wet[lev.name]).sum()),
            "band_wet_over_dry_parent": int((wet[lev.name][bj + h, bi + h] & ~wet[p.name][pj + p.band, pi + p.band]).sum()),
            # open in the parent with all sub-faces closed and not closed by rule 3
            "face_open_over_closed": int(_open_without_subfaces(p, lev, wet).sum()) - _n_closed(p, lev, wet),
        }
    return out


def _n_closed(p: Level, c: Level, wet: Dict[str, np.ndarray]) -> int:
    """How many of the faces rule 3 finds for child ``c`` are in ``p.closed_faces``."""
    cf = getattr(p, "closed_faces", None)
    if cf is None or cf["is_x"].size == 0:
        return 0
    bad = _open_without_subfaces(p, c, wet)
    f = p.wfaces[c.name]
    have = set(zip(cf["is_x"].tolist(), cf["face"][0].tolist(), cf["face"][1].tolist()))
    want = zip(f["is_x"][bad].tolist(), f["face"][0][bad].tolist(), f["face"][1][bad].tolist())
    return sum(k in have for k in want)


def interface_min_depth(config: dict, name: str) -> float:
    """``interface_min_depth`` of level ``name`` in a layout (its own key, else
    ``config:``; 0: not checked)."""
    v = (config.get(name) or {}).get("interface_min_depth", (config.get("config") or {}).get("interface_min_depth", 0.0))
    return float(v or 0.0)


def interface_depths(levels: Iterable[Level], wet: Dict[str, np.ndarray], H: Dict[str, np.ndarray],
                     min_depth: Dict[str, float]) -> Dict[str, dict]:
    """The water at every child's interface: per child level, the least depth
    (``H``, positive down, on the extended shapes) of its wet ring cells and
    of the wet parent cells they are filled from, and how many of each are
    shallower than ``min_depth[child]`` (``interface_shallow``; 0 if no minimum
    is set)."""
    out = {}
    for lev in levels:
        if lev.parent is None:
            continue
        h, p = lev.band, lev.parent
        bj, bi = lev.ring_cells()
        pj, pi, _, _ = lev.ring_source(bj, bi)
        cw = wet[lev.name][bj + h, bi + h]
        cd = np.asarray(H[lev.name], dtype=float)[bj + h, bi + h][cw]
        pk = np.unique(np.stack([pj, pi]), axis=1) if pj.size else np.zeros((2, 0), int)
        pw = wet[p.name][pk[0] + p.band, pk[1] + p.band]
        pd = np.asarray(H[p.name], dtype=float)[pk[0] + p.band, pk[1] + p.band][pw]
        lim = float(min_depth.get(lev.name, 0.0))
        out[lev.name] = {
            "min_depth_ring": float(cd.min(initial=np.inf)),
            "min_depth_parent": float(pd.min(initial=np.inf)),
            "interface_min_depth": lim,
            "interface_shallow": int((cd < lim).sum() + (pd < lim).sum()) if lim > 0.0 else 0,
        }
    return out


def conforming_band(lev: Level, wet_level: np.ndarray) -> np.ndarray:
    """The level's cells (extended shape) the exchanges use where water crosses its
    interface: within ``2 r + band`` cells of a wet ring cell (the parent's rim of 2
    parent cells, the ring and halo)."""
    from .geometry import dilate

    ring_wet = lev._ring_mask() & wet_level
    return dilate(ring_wet, 2 * lev.ratio + lev.band)


def conforming_parent_cells(lev: Level, wet_level: np.ndarray) -> np.ndarray:
    """Parent cells of the window (window shape) that hold cells of the conforming band."""
    h, r = lev.band, lev.ratio
    i0, i1, j0, j1 = lev.window
    band = conforming_band(lev, wet_level)[h:-h, h:-h]
    return band.reshape(j1 - j0, r, i1 - i0, r).any(axis=(1, 3))


def check_conforming(levels: Iterable[Level], wet: Dict[str, np.ndarray], tol: float = 1e-6) -> None:
    """Curvilinear levels (``grid``): raise unless

    1. the grid coincides with the parent's grid refined by the ratio (to ``tol``
       of a cell) over the band the exchanges use where water crosses the
       interface: the parent's rim (2 parent cells, ``2 r`` child cells) and the
       ring and halo (``band`` cells) around every wet ring cell;
    2. no water is counted twice: wherever a wet cell of the level lies
       physically outside that band, the parent is land or the level's own hole.

    Then every index relation the nesting uses is also a geometric one, and the
    exchanges are those between regular grids (docs: methods.html#curvi)."""
    from .geometry import dilate

    for lev in levels:
        if lev.grid is None:
            continue
        p, r, h = lev.parent, lev.ratio, lev.band
        if p.grid is not None:
            raise ValueError(f"{lev.name}: a curvilinear level in a curvilinear parent is not supported")
        w = wet[lev.name]
        if not (lev._ring_mask() & w).any():
            raise ValueError(f"{lev.name}: no water crosses the interface")
        band = conforming_band(lev, w)
        # the supergrid points of those cells (corners, faces and centre)
        sb = np.zeros((2 * band.shape[0] + 1, 2 * band.shape[1] + 1), dtype=bool)
        for a in range(3):
            for b in range(3):
                sb[a : a + 2 * band.shape[0] : 2, b : b + 2 * band.shape[1] : 2] |= band
        x, y = lev.supergrid()
        xr, yr = lev.regular_supergrid()
        dev = np.maximum(np.abs(x - xr), np.abs(y - yr))[sb]
        cell = min(abs(lev.dlon), abs(lev.dlat))
        if np.nanmax(dev) > tol * cell:
            raise ValueError(f"{lev.name}: the grid differs from the refined parent grid by up to {np.nanmax(dev):.3g} "
                             f"(a cell is {cell:.3g}) in the band where water crosses the interface "
                             f"({2 * r + h} cells around the wet ring cells): it must coincide there, or the parent "
                             f"must be land there so that no water crosses the interface")
        # 2. wet cells outside the band: the parent cell under each must be land or the hole
        xc, yc = x[1::2, 1::2], y[1::2, 1::2]
        out = w & ~band
        if out.any():
            pi = np.floor((xc[out] - p.lon0) / p.dlon).astype(int)
            pj = np.floor((yc[out] - p.lat0) / p.dlat).astype(int)
            inside = (pi >= 0) & (pi < p.nx) & (pj >= 0) & (pj < p.ny)
            pw = wet[p.name][pj[inside] + p.band, pi[inside] + p.band]
            hole = p.hole[pj[inside], pi[inside]] if getattr(p, "hole", None) is not None else np.zeros(pw.shape, bool)
            twice = int((pw & ~hole).sum())
            if twice:
                raise ValueError(f"{lev.name}: {twice} wet cells lie over water that {p.name} computes "
                                 "(the parent must be land under the level outside its conforming band)")
            # wet cells outside the parent's domain are allowed: nothing lies beneath them


def check_overhang(lev: Level, wet_domain: np.ndarray, wet: Dict[str, np.ndarray]) -> None:
    """A level's overhang (``overhang``): raise if any of its wet cells lies over
    water its parent computes (outside the parent's domain is allowed)."""
    if not lev.has_overhang:
        return
    p = lev.parent
    x, y = lev.domain_supergrid()
    xc, yc = x[1::2, 1::2], y[1::2, 1::2]
    over = np.ones(lev.domain_shape, dtype=bool)
    oj, oi = lev.view_offset
    ny, nx = lev.extended_shape
    over[oj : oj + ny, oi : oi + nx] = lev._overhang_band()
    sel = over & wet_domain
    if not sel.any():
        return
    pi = np.floor((xc[sel] - p.lon0) / p.dlon).astype(int)
    pj = np.floor((yc[sel] - p.lat0) / p.dlat).astype(int)
    inside = (pi >= 0) & (pi < p.nx) & (pj >= 0) & (pj < p.ny)
    pw = wet[p.name][pj[inside] + p.band, pi[inside] + p.band]
    hole = p.hole[pj[inside], pi[inside]]
    twice = int((pw & ~hole).sum())
    if twice:
        raise ValueError(f"{lev.name}: {twice} wet cells of its overhang lie over water that {p.name} computes")
