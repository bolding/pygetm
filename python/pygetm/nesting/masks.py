"""Land/sea consistency between nesting levels (child-led).

Every level has a wet mask and depth on its extended shape (window plus the
halo band; array index = global index + halo). In the zones where parent and
child overlap they are reconciled, the child's finer bathymetry leading:

1. bottom-up (deepest level first): a cell of the parent's hole is wet if any
   of its ratio x ratio child cells is wet, with the mean depth of the wet
   ones. The parent does not compute there; these cells define the parent's
   faces at the interface and the rim the child fills.
2. top-down: a cell of the child's band (its ring) can only be wet where the
   parent cell it is filled from is wet. Otherwise the child would take water
   or tracer through a face for which the parent has no value - the one
   mismatch flux matching cannot repair.
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
    changes = {l.name: {"hole_wetted": 0, "hole_dried": 0, "band_dried": 0} for l in levels}

    for lev in reversed(levels):
        if lev.parent is None:
            continue
        h, p, r = lev.halo, lev.parent, lev.ratio
        i0, i1, j0, j1 = lev.window
        cw = wet[lev.name][h:-h, h:-h].reshape(j1 - j0, r, i1 - i0, r)
        cH = H[lev.name][h:-h, h:-h].reshape(j1 - j0, r, i1 - i0, r)
        anywet = cw.any(axis=(1, 3))
        n = cw.sum(axis=(1, 3))
        Hmean = np.where(cw, cH, 0.0).sum(axis=(1, 3)) / np.where(n > 0, n, 1)
        hp = p.halo
        sl = (slice(j0 + hp, j1 + hp), slice(i0 + hp, i1 + hp))
        before = wet[p.name][sl].copy()
        changes[p.name]["hole_wetted"] += int((anywet & ~before).sum())
        changes[p.name]["hole_dried"] += int((~anywet & before).sum())
        wet[p.name][sl] = anywet
        H[p.name][sl] = np.where(anywet, Hmean, np.nan)

    for lev in levels:
        if lev.parent is None:
            continue
        h, p = lev.halo, lev.parent
        bj, bi = lev.ring_cells()
        pj, pi, _, _ = lev.ring_source(bj, bi)
        src_wet = wet[p.name][pj + p.halo, pi + p.halo]
        dry = wet[lev.name][bj + h, bi + h] & ~src_wet
        changes[lev.name]["band_dried"] += int(dry.sum())
        wet[lev.name][bj[dry] + h, bi[dry] + h] = False
        H[lev.name][bj[dry] + h, bi[dry] + h] = np.nan
    return wet, H, changes


def check_consistent(levels: Iterable[Level], wet: Dict[str, np.ndarray]):
    """Violations of the two rules per child level (all zero if consistent)."""
    out = {}
    for lev in levels:
        if lev.parent is None:
            continue
        h, p, r = lev.halo, lev.parent, lev.ratio
        i0, i1, j0, j1 = lev.window
        cw = wet[lev.name][h:-h, h:-h].reshape(j1 - j0, r, i1 - i0, r)
        hole = wet[p.name][j0 + p.halo : j1 + p.halo, i0 + p.halo : i1 + p.halo]
        bj, bi = lev.ring_cells()
        pj, pi, _, _ = lev.ring_source(bj, bi)
        out[lev.name] = {
            "hole_mismatch": int((hole != cw.any(axis=(1, 3))).sum()),
            "band_wet_over_dry_parent": int((wet[lev.name][bj + h, bi + h] & ~wet[p.name][pj + p.halo, pi + p.halo]).sum()),
        }
    return out
