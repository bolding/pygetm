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
    if cf is None or cf["is_x"].size == 0 or domain.mask is None:
        return
    h = lev.band
    fj, fi = cf["face"][0] + h, cf["face"][1] + h
    x = cf["is_x"]
    # supergrid: T (j, i) at (2j + 1, 2i + 1); U east of it at (2j + 1, 2i + 2), V north at (2j + 2, 2i + 1)
    domain.mask[2 * fj[x] + 1, 2 * fi[x] + 2] = 0
    domain.mask[2 * fj[~x] + 2, 2 * fi[~x] + 1] = 0


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
