"""Geometry of a nesting hierarchy.

A hierarchy is a root plus children, each child a refinement by an integer
ratio of a rectangular window of its parent's T grid (0-based, exclusive
stop), as in a nesting YAML file (e.g. pygetm-nesting
config/nesting_atlantic.yaml)::

    root:      {nx: 100, ny: 46, lon0: -20.0, lat0: 48.0, dlon: 0.5, dlat: 0.5}
    north_sea: {parent: root, istart: 30, istop: 70, jstart: 4, jstop: 28, ratio: 3}

Global cell indices of a level run from 0 to nx-1, ny-1; its ring (the halo
band around it, filled from the parent) has indices -halo..-1 and
n..n+halo-1. A parent does not compute in its hole (the union of its
children's windows); the hole cells that something reads form the rim,
filled from the child.

Face convention: the U face with index (j, i) is the east face of T cell
(j, i), the V face (j, i) its north face.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

SIDES = ("W", "E", "S", "N")


class Level:
    """One level of the hierarchy: grid, window in the parent, hole, rims and
    the faces at its interfaces. Call :meth:`finish` once all children have
    been added (:func:`load_levels` does this)."""

    def __init__(self, name: str, cfg: Mapping[str, Any], parent: Optional["Level"] = None, halo: int = 2):
        self.name = name
        self.parent = parent
        self.children: List["Level"] = []
        self.halo = halo
        if parent is None:
            self.ratio, self.window = 1, None
            self.nx, self.ny = int(cfg["nx"]), int(cfg["ny"])
            self.lon0, self.lat0 = float(cfg["lon0"]), float(cfg["lat0"])
            self.dlon, self.dlat = float(cfg["dlon"]), float(cfg["dlat"])
        else:
            r = int(cfg["ratio"])
            i0, i1 = int(cfg["istart"]), int(cfg["istop"])
            j0, j1 = int(cfg["jstart"]), int(cfg["jstop"])
            if not (0 <= i0 < i1 <= parent.nx and 0 <= j0 < j1 <= parent.ny):
                raise ValueError(f"{name}: window {(i0, i1, j0, j1)} outside {parent.name}")
            self.ratio, self.window = r, (i0, i1, j0, j1)
            self.nx, self.ny = (i1 - i0) * r, (j1 - j0) * r
            self.dlon, self.dlat = parent.dlon / r, parent.dlat / r
            self.lon0 = parent.lon0 + i0 * parent.dlon
            self.lat0 = parent.lat0 + j0 * parent.dlat
            parent.children.append(self)
        self.index = -1
        self.steps_per_root = 1 if parent is None else parent.steps_per_root * self.ratio

    @property
    def extended_shape(self):
        """Shape of the level including its ring band."""
        return (self.ny + 2 * self.halo, self.nx + 2 * self.halo)

    def edges(self):
        """Cell edges (degrees) in x and y, including the band."""
        h = self.halo
        lon = self.lon0 + np.arange(-h, self.nx + h + 1) * self.dlon
        lat = self.lat0 + np.arange(-h, self.ny + h + 1) * self.dlat
        return lon, lat

    def finish(self):
        """Hole, rims, edge faces and window faces.

        The rim is every hole cell that something reads, so the child must
        fill it:

        * cells within ``halo`` of cells this level computes;
        * cells within ``halo * ratio`` of this level's domain edge, because
          its parent restricts those into its own rim (matters where a child
          touches this level's edge).
        """
        h = self.halo
        hole = np.zeros((self.ny, self.nx), dtype=bool)
        for c in self.children:
            i0, i1, j0, j1 = c.window
            if hole[j0:j1, i0:i1].any():
                raise ValueError(f"overlapping child windows in {self.name}")
            hole[j0:j1, i0:i1] = True
        self.computed = ~hole
        needed = dilate(self.computed, h)
        if self.parent is not None:
            w = h * self.ratio
            needed[:w, :] = needed[-w:, :] = True
            needed[:, :w] = needed[:, -w:] = True
        self.rims: Dict[str, np.ndarray] = {}
        for c in self.children:
            i0, i1, j0, j1 = c.window
            rim = np.zeros_like(hole)
            rim[j0:j1, i0:i1] = needed[j0:j1, i0:i1]
            self.rims[c.name] = np.argwhere(rim)  # (n, 2) global (j, i), row-major

        # outer edge faces per side: inner cell and the face (U or V index)
        jj, ii = np.arange(self.ny), np.arange(self.nx)
        z_j, z_i = np.zeros_like(jj), np.zeros_like(ii)
        self.edge = {
            "W": dict(inner=(jj, z_j), face=(jj, z_j - 1), axis="x"),
            "E": dict(inner=(jj, z_j + self.nx - 1), face=(jj, z_j + self.nx - 1), axis="x"),
            "S": dict(inner=(z_i, ii), face=(z_i - 1, ii), axis="y"),
            "N": dict(inner=(z_i + self.ny - 1, ii), face=(z_i + self.ny - 1, ii), axis="y"),
        }
        if self.parent is not None:
            for side, e in self.edge.items():
                covered = np.zeros(e["inner"][0].shape, dtype=bool)
                for c in self.children:
                    covered |= in_window(c, *e["inner"])
                if not (self.computed[e["inner"]] | covered).all():
                    raise ValueError(f"{self.name} {side} edge: cells neither computed nor covered")

        # window faces of each child, per side, in this level's indices
        self.wfaces: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for c in self.children:
            i0, i1, j0, j1 = c.window
            jw, iw = np.arange(j0, j1), np.arange(i0, i1)
            f = {
                # face index, cell outside the window, sign of the correction
                "W": dict(face=(jw, jw * 0 + i0 - 1), out=(jw, jw * 0 + i0 - 1), sign=1.0, axis="x", inside=i0 > 0),
                "E": dict(face=(jw, jw * 0 + i1 - 1), out=(jw, jw * 0 + i1), sign=-1.0, axis="x", inside=i1 < self.nx),
                "S": dict(face=(iw * 0 + j0 - 1, iw), out=(iw * 0 + j0 - 1, iw), sign=1.0, axis="y", inside=j0 > 0),
                "N": dict(face=(iw * 0 + j1 - 1, iw), out=(iw * 0 + j1, iw), sign=-1.0, axis="y", inside=j1 < self.ny),
            }
            for side, d in f.items():
                # window on this level's edge: the face is this level's own
                # edge face with this index along that edge
                d["edge_index"] = jw if side in "WE" else iw
                if d["inside"] and not self.computed[d["out"]].all():
                    raise ValueError(f"{c.name}: {side} neighbour cells are not computed by {self.name}")
            self.wfaces[c.name] = f

    def external(self) -> np.ndarray:
        """Cells (including the band) set from outside: the band (filled from
        the parent, or a zero-inflow open boundary for the root) and the hole
        (filled from the children). For :attr:`pygetm.domain.Domain.external`."""
        h = self.halo
        ext = np.ones(self.extended_shape, dtype=bool)
        ext[h:-h, h:-h] = ~self.computed
        return ext

    def ring_cells(self):
        """Global (j, i) of all band cells, row-major."""
        h = self.halo
        band = np.ones(self.extended_shape, dtype=bool)
        band[h:-h, h:-h] = False
        j, i = np.nonzero(band)
        return j - h, i - h

    def ring_source(self, j, i):
        """Parent cell (global) covering child cells (j, i) (also in the band),
        and the position (a, b) of the child cell within it."""
        i0, _, j0, _ = self.window
        r = self.ratio
        return j0 + np.floor_divide(j, r), i0 + np.floor_divide(i, r), np.mod(j, r), np.mod(i, r)


def in_window(c: Level, j, i):
    i0, i1, j0, j1 = c.window
    return (j >= j0) & (j < j1) & (i >= i0) & (i < i1)


def dilate(mask: np.ndarray, width: int) -> np.ndarray:
    """Cells within ``width`` (Chebyshev distance) of a true cell."""
    out = mask.copy()
    ny, nx = mask.shape
    for dj in range(-width, width + 1):
        for di in range(-width, width + 1):
            if dj or di:
                src = mask[max(0, -dj) : ny - max(0, dj), max(0, -di) : nx - max(0, di)]
                out[max(0, dj) : ny - max(0, -dj), max(0, di) : nx - max(0, -di)] |= src
    return out


def load_levels(config, halo: int = 2) -> List[Level]:
    """Levels from a nesting YAML file (path) or mapping, parents before
    children; keys other than levels (e.g. ``config``) are ignored."""
    if not isinstance(config, Mapping):
        import yaml

        with open(config) as f:
            config = yaml.safe_load(f)
    specs = {k: v for k, v in config.items() if k != "config"}
    levels: List[Level] = []
    done: Dict[str, Level] = {}
    while len(done) < len(specs):
        progress = False
        for name, spec in specs.items():
            if name in done:
                continue
            parent = spec.get("parent")
            if parent is None or parent in done:
                lev = Level(name, spec, done.get(parent), halo=halo)
                lev.index = len(levels)
                levels.append(lev)
                done[name] = lev
                progress = True
        if not progress:
            raise ValueError("cyclic or missing parent in nesting configuration")
    for lev in levels:
        lev.finish()
    return levels
