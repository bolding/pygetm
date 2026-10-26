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
    return wet, H, changes


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
        }
    return out
