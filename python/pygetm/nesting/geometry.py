"""Geometry of a nesting hierarchy.

A hierarchy is a root plus children, each child a refinement by an integer
ratio of a rectangular window of its parent's T grid (0-based, exclusive
stop), as in a nesting YAML file (e.g. pygetm-nesting
config/nesting_atlantic.yaml)::

    root:      {nx: 100, ny: 46, lon0: -20.0, lat0: 48.0, dlon: 0.5, dlat: 0.5}
    north_sea: {parent: root, istart: 30, istop: 70, jstart: 4, jstop: 28, ratio: 3}

The window defines the index mapping. The child computes its active region:
by default the whole window; with ``active`` (see :func:`active_region`)
only the window's parent cells connected to a seed point, e.g. the North Sea
and the Baltic without the Norwegian Sea of a rectangle around them. The
rest of the window stays with the parent.

Global cell indices of a level run from 0 to nx-1, ny-1; its band (cells
around the window that belong to its pyGETM domain) has indices -band..-1
and n..n+band-1. The ring is the non-active cells within ``band`` of the
active region (for a full window: the band); it is filled from the parent.
Other non-active cells are not used (dry). A parent does not compute in its
hole (the union of its children's active regions); the hole cells that
something reads form the rim, filled from the child.

Face convention: the U face with index (j, i) is the east face of T cell
(j, i), the V face (j, i) its north face. Interface faces are kept as flat
arrays (see :func:`boundary_faces`): x faces first, then y faces, row-major.
"""

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

SIDES = ("W", "E", "S", "N")


def boundary_faces(mask: np.ndarray) -> Dict[str, Any]:
    """Faces between cells of ``mask`` (ny, nx) and cells outside it
    (including cells outside the array). Returns flat arrays:

    * ``is_x``: U face (True) or V face (False);
    * ``face``: (j, i) of the U/V point;
    * ``inner``: (j, i) of the cell in the mask;
    * ``other``: (j, i) of the cell outside the mask (may lie outside the array);
    * ``sign``: +1 if ``other`` lies east/north of ``inner`` (a positive
      transport leaves the mask), -1 otherwise.

    Order: x faces row-major, then y faces row-major."""
    ny, nx = mask.shape
    pad = np.zeros((ny + 2, nx + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    out = []
    # x faces: U point (j, i) between cells (j, i) and (j, i + 1), i = -1..nx-1
    left, right = pad[1:-1, :-1], pad[1:-1, 1:]
    j, i = np.nonzero(left != right)
    i = i - 1
    lin = left[j, i + 1]
    out.append((np.ones(j.size, bool), j, i, np.where(lin, i, i + 1), np.where(lin, i + 1, i), np.where(lin, 1.0, -1.0)))
    # y faces: V point (j, i) between cells (j, i) and (j + 1, i), j = -1..ny-1
    low, high = pad[:-1, 1:-1], pad[1:, 1:-1]
    j, i = np.nonzero(low != high)
    j = j - 1
    lin = low[j + 1, i]
    out.append((np.zeros(j.size, bool), j, i, np.where(lin, j, j + 1), np.where(lin, j + 1, j), np.where(lin, 1.0, -1.0)))
    xs, ys = out
    cat = lambda k: np.concatenate([xs[k], ys[k]])
    is_x = cat(0)
    fj, fi = cat(1), cat(2)
    # inner/other cell along the axis of the face
    a_in, a_out = cat(3), cat(4)
    inner = (np.where(is_x, fj, a_in), np.where(is_x, a_in, fi))
    other = (np.where(is_x, fj, a_out), np.where(is_x, a_out, fi))
    return dict(is_x=is_x, face=(fj, fi), inner=inner, other=other, sign=cat(5))


class Level:
    """One level of the hierarchy: grid, window in the parent, active region,
    hole, rims and the faces at its interfaces. Call :meth:`finish` once all
    children have been added (:func:`load_levels` does this)."""

    def __init__(self, name: str, cfg: Mapping[str, Any], parent: Optional["Level"] = None,
                 halo: int = 2, band: Optional[int] = None):
        self.name = name
        self.parent = parent
        self.cfg = cfg
        self.children: List["Level"] = []
        #: width of stencils (rims are this wide around computed cells)
        self.halo = halo
        #: width of the band around the window (part of the level's pyGETM
        #: domain). Children need band = halo; the root may have 0 (its
        #: edges then are ordinary pyGETM open boundaries)
        self.band = halo if band is None else band
        if parent is not None and self.band != halo:
            raise ValueError(f"{name}: a child needs a band of {halo} cells")
        if parent is None:
            self.ratio, self.window, self.time_ratio = 1, None, 1
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
            #: steps per parent step (YAML ``time_ratio``, default: ratio); 1 =
            #: the parent's time step
            self.time_ratio = int(cfg.get("time_ratio", r))
            if self.time_ratio < 1:
                raise ValueError(f"{name}: time_ratio must be >= 1")
            self.nx, self.ny = (i1 - i0) * r, (j1 - j0) * r
            self.dlon, self.dlat = parent.dlon / r, parent.dlat / r
            self.lon0 = parent.lon0 + i0 * parent.dlon
            self.lat0 = parent.lat0 + j0 * parent.dlat
            parent.children.append(self)
        #: active region in parent cells of the window (j1-j0, i1-i0); the
        #: root and a full window: all True
        self.active: Optional[np.ndarray] = None
        self.set_active(None)
        self.index = -1
        self.steps_per_root = 1 if parent is None else parent.steps_per_root * self.time_ratio

    def set_active(self, mask: Optional[np.ndarray]):
        """Set the active region: a boolean mask of the window's parent cells
        (shape (jstop - jstart, istop - istart)); None for the whole window.
        Must be called before :meth:`finish`."""
        if self.parent is None:
            if mask is not None:
                raise ValueError(f"{self.name}: the root has no active region")
            self.active = np.ones((self.ny, self.nx), dtype=bool)
            self.cactive = self.active
            return
        r = self.ratio
        shape = (self.ny // r, self.nx // r)
        mask = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"{self.name}: active region has shape {mask.shape}, expected {shape}")
        if not mask.any():
            raise ValueError(f"{self.name}: empty active region")
        self.active = mask
        #: active region in this level's cells (ny, nx)
        self.cactive = np.repeat(np.repeat(mask, r, axis=0), r, axis=1)

    @property
    def extended_shape(self):
        """Shape of the level including its band."""
        return (self.ny + 2 * self.band, self.nx + 2 * self.band)

    def edges(self):
        """Cell edges (degrees) in x and y, including the band."""
        h = self.band
        lon = self.lon0 + np.arange(-h, self.nx + h + 1) * self.dlon
        lat = self.lat0 + np.arange(-h, self.ny + h + 1) * self.dlat
        return lon, lat

    def hole_of(self, child: "Level") -> np.ndarray:
        """The child's active region in this level's cells (ny, nx)."""
        i0, i1, j0, j1 = child.window
        hole = np.zeros((self.ny, self.nx), dtype=bool)
        hole[j0:j1, i0:i1] = child.active
        return hole

    def finish(self):
        """Hole, rims, interface faces (parents must be finished first).

        The rim is every hole cell that something reads, so the child must
        fill it:

        * cells within ``halo`` of cells this level computes;
        * the cells of the blocks of the parent's rim in this level: the
          parent restricts those into its own rim (matters where a child
          touches this level's interface).
        """
        h = self.halo
        hole = np.zeros((self.ny, self.nx), dtype=bool)
        for c in self.children:
            i0, i1, j0, j1 = c.window
            if (j0 < 0) or hole[j0:j1, i0:i1].any() or any(
                    _overlap(c.window, o.window) for o in self.children if o is not c):
                raise ValueError(f"overlapping child windows in {self.name}")
            if (c.active & ~self.cactive[j0:j1, i0:i1]).any():
                raise ValueError(f"{c.name}: active region outside the active region of {self.name}")
            hole |= self.hole_of(c)
        self.hole = hole
        #: cells this level computes
        self.computed = self.cactive & ~hole
        needed = dilate(self.computed, h)
        if self.parent is not None:
            prim = self.parent.rims[self.name]
            r = self.ratio
            i0, _, j0, _ = self.window
            for a in range(r):
                for b in range(r):
                    needed[(prim[:, 0] - j0) * r + a, (prim[:, 1] - i0) * r + b] = True
        self.rims: Dict[str, np.ndarray] = {}
        for c in self.children:
            self.rims[c.name] = np.argwhere(self.hole_of(c) & needed)  # (n, 2) global (j, i), row-major

        if self.parent is None:
            # the root's edge: faces on the boundary of its domain
            f = boundary_faces(self.cactive)
            self.edge = dict(is_x=f["is_x"], face=f["face"], inner=f["inner"], sign=f["sign"])
        # else: self.edge was set by the parent's finish (faces of the parent's
        # interface, r sub-faces each, in this level's indices)
        lookup = {}
        if self.parent is not None:
            e = self.edge
            for k in range(e["is_x"].size):
                lookup[(bool(e["is_x"][k]), int(e["face"][0][k]), int(e["face"][1][k]),
                        int(e["inner"][0][k]), int(e["inner"][1][k]))] = k

        # interface faces of every child, in this level's indices
        self.wfaces: Dict[str, Dict[str, Any]] = {}
        for c in self.children:
            f = boundary_faces(self.hole_of(c))
            oj, oi = f["other"]
            in_dom = (oj >= 0) & (oj < self.ny) & (oi >= 0) & (oi < self.nx)
            ojc, oic = np.clip(oj, 0, self.ny - 1), np.clip(oi, 0, self.nx - 1)
            inside = in_dom & self.computed[ojc, oic]
            if (in_dom & ~inside & self.cactive[ojc, oic]).any():
                raise ValueError(f"{c.name}: its active region touches another child's in {self.name}")
            up = -np.ones(inside.size, dtype=int)
            for n in np.nonzero(~inside)[0]:
                if self.parent is None:
                    continue  # the root's open boundary: nothing to pass on
                key = (bool(f["is_x"][n]), int(f["face"][0][n]), int(f["face"][1][n]),
                       int(f["inner"][0][n]), int(f["inner"][1][n]))
                if key not in lookup:
                    raise ValueError(f"{c.name}: interface face {key} is neither inside {self.name} nor on its interface")
                up[n] = lookup[key]
            self.wfaces[c.name] = dict(
                is_x=f["is_x"], face=f["face"], hole=f["inner"], out=f["other"],
                # sign of the correction of the outside cell: +1 if it lies west/south
                sign=-f["sign"], inside=inside, up=up,
            )
            c.edge = _child_edge(c, f)

    def external(self) -> np.ndarray:
        """Cells (including the band) set from outside or not used: all but
        the cells this level computes. For :attr:`pygetm.domain.Domain.external`."""
        h = self.band
        ext = np.ones(self.extended_shape, dtype=bool)
        ext[h : h + self.ny, h : h + self.nx] = ~self.computed
        return ext

    def tiling_mask(self) -> np.ndarray:
        """Cells (including the band) that need a subdomain: all but the hole
        cells nothing reads (not in a rim, not the hole cell of an interface
        face passed upwards). Subdomains holding only those are removed like
        all-land ones. For :attr:`pygetm.domain.Domain.tiling_mask`."""
        h = self.band
        unread = self.hole.copy()
        for rim in self.rims.values():
            unread[rim[:, 0], rim[:, 1]] = False
        for wf in self.wfaces.values():
            hj, hi = wf["hole"]
            unread[hj[~wf["inside"]], hi[~wf["inside"]]] = False
        tm = np.ones(self.extended_shape, dtype=bool)
        tm[h : h + self.ny, h : h + self.nx] = ~unread
        return tm

    def _ring_mask(self) -> np.ndarray:
        h = self.band
        act = np.zeros(self.extended_shape, dtype=bool)
        act[h : h + self.ny, h : h + self.nx] = self.cactive
        return dilate(act, h) & ~act

    def ring_cells(self):
        """Global (j, i) of all ring cells, row-major (for a full window: the
        whole band)."""
        h = self.band
        j, i = np.nonzero(self._ring_mask())
        return j - h, i - h

    def unused(self) -> np.ndarray:
        """Cells of the extended shape that are neither active nor ring: they
        must be dry (land)."""
        h = self.band
        act = np.zeros(self.extended_shape, dtype=bool)
        act[h : h + self.ny, h : h + self.nx] = self.cactive
        return ~act & ~self._ring_mask()

    def ring_source(self, j, i):
        """Parent cell (global) covering child cells (j, i) (also in the band),
        and the position (a, b) of the child cell within it."""
        i0, _, j0, _ = self.window
        r = self.ratio
        return j0 + np.floor_divide(j, r), i0 + np.floor_divide(i, r), np.mod(j, r), np.mod(i, r)


def _overlap(w1, w2) -> bool:
    return w1[0] < w2[1] and w2[0] < w1[1] and w1[2] < w2[3] and w2[2] < w1[3]


def _child_edge(c: Level, f: Dict[str, Any]) -> Dict[str, Any]:
    """The child's edge: r sub-faces per parent interface face n (index
    k = n * r + a), in the child's indices."""
    r = c.ratio
    i0, _, j0, _ = c.window
    n = f["is_x"].size
    is_x = np.repeat(f["is_x"], r)
    sign = np.repeat(f["sign"], r)  # outward from the hole = outward from the child
    a = np.tile(np.arange(r), n)
    hj, hi = (np.repeat(x, r) for x in f["inner"])
    # x face: rows along the face, column of the hole cell's west/east block edge
    first_j, first_i = (hj - j0) * r, (hi - i0) * r
    cj = np.where(is_x, first_j + a, np.where(sign > 0, first_j + r - 1, first_j))
    ci = np.where(is_x, np.where(sign > 0, first_i + r - 1, first_i), first_i + a)
    fj = np.where(is_x, cj, np.where(sign > 0, cj, cj - 1))
    fi = np.where(is_x, np.where(sign > 0, ci, ci - 1), ci)
    return dict(is_x=is_x, face=(fj, fi), inner=(cj, ci), sign=sign)


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


def active_region(level: Level, spec: Mapping[str, Any], wet: np.ndarray) -> np.ndarray:
    """Active region of a child from its YAML ``active`` entry::

        active:
          seed: [3.0, 55.0]                 # lon, lat of a wet point
          barriers:                         # every key must hold
            - {lat_gt: 61.0, lon_lt: 8.0}   # also lon_gt, lat_lt

    The parent keeps the child's wet cells (``wet``: window shape (ny, nx) or
    extended shape) that are not connected to the seed without crossing a
    barrier (barrier lines should lie on parent cell edges). A parent cell is
    active if it holds water connected to the seed or no water the parent
    keeps: land always belongs to the child (it costs nothing, and the active
    regions of nested levels then fit into each other). Returns the mask for
    :meth:`Level.set_active`."""
    from scipy import ndimage

    wet = np.asarray(wet, dtype=bool)
    if wet.shape == level.extended_shape and level.band:
        h = level.band
        wet = wet[h:-h, h:-h]
    if wet.shape != (level.ny, level.nx):
        raise ValueError(f"{level.name}: wet mask has shape {wet.shape}")
    lon = level.lon0 + (np.arange(level.nx) + 0.5) * level.dlon
    lat = level.lat0 + (np.arange(level.ny) + 0.5) * level.dlat
    LO, LA = np.meshgrid(lon, lat)
    ok = wet.copy()
    barred = np.zeros_like(ok)
    tests = {"lon_lt": lambda v: LO < v, "lon_gt": lambda v: LO > v, "lat_lt": lambda v: LA < v, "lat_gt": lambda v: LA > v}
    for b in spec.get("barriers", []) or []:
        sel = np.ones_like(ok)
        for key, v in b.items():
            sel &= tests[key](float(v))
        barred |= sel
    ok &= ~barred
    lab, _ = ndimage.label(ok)
    sx, sy = spec["seed"]
    seed = lab[int(np.argmin(abs(lat - float(sy)))), int(np.argmin(abs(lon - float(sx))))]
    if seed == 0:
        raise ValueError(f"{level.name}: seed {spec['seed']} is on land or behind a barrier")
    r = level.ratio
    connected = lab == seed
    # water the parent keeps: not connected to the seed, and continuing into
    # the parent's area (touching the window edge) or behind a barrier; water
    # cut off at this resolution (lagoons, fjord ends) counts as land
    other, _ = ndimage.label(wet & ~connected)
    edge = np.unique(np.concatenate([other[0], other[-1], other[:, 0], other[:, -1], other[barred & wet]]))
    kept = np.isin(other, edge[edge > 0])
    blocks = lambda m: m.reshape(level.ny // r, r, level.nx // r, r).any(axis=(1, 3))
    # every parent cell without water the parent keeps is active: land always
    # belongs to the child (it costs nothing; nested active regions then fit)
    return blocks(connected) | ~blocks(kept)


def load_levels(config, halo: int = 2, root_band: Optional[int] = None,
                wet: Optional[Mapping[str, np.ndarray]] = None, active: bool = True) -> List[Level]:
    """Levels from a nesting YAML file (path) or mapping, parents before
    children; keys other than levels (e.g. ``config``) are ignored.
    ``root_band``: width of the root's band (default ``halo``; 0 when the
    root's edges are pyGETM open boundaries, e.g. tides). ``wet``: wet masks
    per level (window or extended shape), needed for levels with an
    ``active`` entry (:func:`active_region`); an ``active`` entry may also be
    an explicit mask of the window's parent cells. ``active=False``: the
    geometry only (windows, grids, bands), without active regions, e.g. to
    build a level's bathymetry before the wet masks are known."""
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
                lev = Level(name, spec, done.get(parent), halo=halo,
                            band=root_band if parent is None else None)
                lev.index = len(levels)
                levels.append(lev)
                done[name] = lev
                progress = True
        if not progress:
            raise ValueError("cyclic or missing parent in nesting configuration")
    for lev in levels:
        spec = specs[lev.name].get("active")
        if spec is None or lev.parent is None or not active:
            continue
        if isinstance(spec, Mapping):
            if wet is None or lev.name not in wet:
                raise ValueError(f"{lev.name}: an active region from a seed needs the level's wet mask")
            lev.set_active(active_region(lev, spec, wet[lev.name]))
        else:
            lev.set_active(np.asarray(spec, dtype=bool))
    for lev in levels:
        lev.finish()
    return levels
