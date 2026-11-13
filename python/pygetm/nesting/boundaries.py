"""Open boundaries along the outer edges of a level: the root's edges, and a child's outer sides (a window reaching
the root's edge), along every wet stretch of the cells the level computes. One rule for every way a nested model is set
up (pygetm-config's generated scripts, pygetm-nesting's tools), computed from the depths the run uses (after the
nesting's consistency, :meth:`pygetm.nesting.NestedModel.share_depth`)."""

from typing import List, Tuple

import numpy as np

from .geometry import Level

#: side letter -> pygetm.open_boundaries.Side name
SIDES = {"W": "LEFT", "E": "RIGHT", "S": "BOTTOM", "N": "TOP"}


def edge_segments(wet, band=0, sides="WESN", exclude=None):
    """Open boundaries along every wet stretch of a level's outer edges.

    ``wet`` (ny, nx): T cells of the level's pyGETM domain (band included).
    The boundary lies ``band`` cells in from the domain edge: on the root
    (band 0) its outer cells, on a child with outer sides (a window reaching
    the root's edge) its last window row/column, the band beyond being dry.
    ``sides``: the edges that are open ("WESN" for the root, the child's
    ``outer_sides``). Corners are excluded where the perpendicular side is
    open too, so that boundaries do not cross. A cell is a boundary cell when
    it and its inner neighbour are wet and it is not in ``exclude`` (the
    root: its hole, the part of its edge a child owns; the inner neighbour
    may be a hole cell).

    Returns [(side, l, m0, m1)] in the order the run scripts add them (all
    west, east, south, north stretches), in domain indices: side "W"/"E" at
    column l with rows m0..m1-1, "S"/"N" at row l with columns m0..m1-1."""
    wet = np.asarray(wet, dtype=bool)
    ny, nx = wet.shape
    h = band
    own = wet if exclude is None else wet & ~np.asarray(exclude, dtype=bool)

    def runs(cells):
        idx = np.nonzero(cells)[0]
        if not idx.size:
            return []
        breaks = np.nonzero(np.diff(idx) > 1)[0]
        starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        stops = np.concatenate([idx[breaks] + 1, [idx[-1] + 1]])
        return list(zip(starts.tolist(), stops.tolist()))

    def along(n, low, high):
        """the window's cells along an edge, without the corners of open perpendicular sides"""
        m = np.zeros(n, bool)
        m[h : n - h] = True
        if low in sides:
            m[h] = False
        if high in sides:
            m[n - h - 1] = False
        return m

    segs = []
    lines = {"W": (h, h + 1), "E": (nx - 1 - h, nx - 2 - h), "S": (h, h + 1), "N": (ny - 1 - h, ny - 2 - h)}
    for side in "WESN":
        if side not in sides:
            continue
        l, inner = lines[side]
        if side in "WE":
            cells = own[:, l] & wet[:, inner] & along(ny, "S", "N")
        else:
            cells = own[l, :] & wet[inner, :] & along(nx, "W", "E")
        segs += [(side, l, a, b) for a, b in runs(cells)]
    return segs


def level_edge_segments(lev: Level, wet: np.ndarray, sides: str = None) -> List[Tuple[str, int, int, int]]:
    """The open boundary stretches of level ``lev`` (:func:`edge_segments`): the root along all its edges (its band is
    0: the boundary is its outer cells), a child along its outer sides (on its last window row/column, the band beyond
    is dry); without the cells the level does not compute (:meth:`Level.external`: the root's hole, the part of its
    edge a child owns). ``wet``: the level's wet cells on its domain (band included), e.g. ``np.isfinite(H)`` of the
    depth after :meth:`~pygetm.nesting.NestedModel.share_depth`. A child without outer sides: none. ``sides``: only
    these of them (e.g. "W"; default all)."""
    own = "WESN" if lev.parent is None else lev.outer_sides
    if sides is not None:
        own = "".join(c for c in own if c in sides)
    if not own:
        return []
    return edge_segments(np.asarray(wet, dtype=bool), lev.band, own, lev.external())


def add_edge_boundaries(domain, lev: Level, wet: np.ndarray, type_2d: int, type_3d: int,
                        prefix: str = "", sides: str = None) -> List[Tuple[str, int, int, int]]:
    """Add the open boundaries of :func:`level_edge_segments` to ``domain`` (before its grids are created), named
    ``<prefix><side letter><k>`` (k counting all of them, in the order added: W, E, S, N), with the given 2D and 3D
    boundary types (e.g. ``pygetm.FLATHER_TRANSPORT``, ``pygetm.ZERO_GRADIENT``); ``sides``: only these (default
    all of the level's). Returns the stretches."""
    from ..open_boundaries import Side

    segs = level_edge_segments(lev, wet, sides)
    for k, (side, l, m0, m1) in enumerate(segs):
        domain.open_boundaries.add_by_index(side=getattr(Side, SIDES[side]), l=int(l), mstart=int(m0), mstop=int(m1),
                                            type_2d=type_2d, type_3d=type_3d, name=f"{prefix}{side}{k}")
    return segs
