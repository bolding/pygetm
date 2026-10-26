"""Generic access to the cells that parent and child exchange.

A :class:`CellSet` is a fixed set of horizontal cells of one grid, in local
indices of ``Array.all_values`` (that is, including halos). It gathers values
from any array on that grid and scatters values into it. That covers 2D
arrays ``(ny, nx)``, 3D arrays on layer centres ``(nz, ny, nx)`` and on
interfaces ``(nz+1, ny, nx)`` alike. Several arrays can be packed into one
buffer, so that variables needed at the same time travel in one message.

Two kinds of cell sets are needed for nesting:

* :func:`outer_halo`: the child's cells in the ring around its global
  domain (``halo`` cells deep). pyGETM's own halo exchange leaves these alone,
  as there is no neighbouring subdomain, and the parent fills them.
* :func:`hole_rim`: the parent's cells in the hole occupied by the child, within
  a given distance of cells the parent computes. The child fills these.

This module only reads the grid and its tiling. It never calls pyGETM's halo
exchange (see :mod:`pygetm.nesting`).
"""

from typing import Iterable, Optional, Sequence

import numpy as np
import numpy.typing as npt

from .. import core


class CellSet:
    """Fixed set of horizontal cells on one grid, in local ``all_values`` indices.

    Values are gathered with layout ``(*extra, n)`` flattened in C order, where
    ``extra`` holds any leading (vertical) dimensions of the array. For a 3D
    array this means all cells of level 0 first, then level 1, and so on.
    """

    def __init__(self, grid: core.Grid, j: npt.ArrayLike, i: npt.ArrayLike):
        j = np.asarray(j, dtype=np.intp).ravel()
        i = np.asarray(i, dtype=np.intp).ravel()
        if j.shape != i.shape:
            raise ValueError(f"j and i differ in length: {j.size} vs {i.size}")
        ny, nx = grid.ny_, grid.nx_
        if j.size and (j.min() < 0 or j.max() >= ny or i.min() < 0 or i.max() >= nx):
            raise ValueError(f"cell indices outside local extent ({ny}, {nx})")
        self.grid = grid
        self.j = j
        self.i = i

    @property
    def size(self) -> int:
        return self.j.size

    def __len__(self) -> int:
        return self.size

    def _check(self, array: core.Array):
        if array.grid is not self.grid:
            raise ValueError(f"{array.name} is not defined on the grid of this cell set")

    def count(self, array: core.Array) -> int:
        """Number of values :meth:`gather` returns for ``array``."""
        self._check(array)
        return int(np.prod(array.all_values.shape[:-2], dtype=int)) * self.size

    def gather(
        self, array: core.Array, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return the values of ``array`` at these cells as a flat array."""
        self._check(array)
        values = array.all_values[..., self.j, self.i]
        if out is None:
            return values.ravel()
        out[...] = values.ravel()
        return out

    def scatter(self, array: core.Array, values: npt.ArrayLike):
        """Write a flat array of values (layout as :meth:`gather`) into ``array``."""
        self._check(array)
        shape = array.all_values.shape[:-2] + (self.size,)
        array.all_values[..., self.j, self.i] = np.reshape(values, shape)

    def pack(
        self, arrays: Sequence[core.Array], out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Gather several arrays into one flat buffer, one after the other."""
        n = sum(self.count(a) for a in arrays)
        if out is None:
            out = np.empty(n)
        elif out.size != n:
            raise ValueError(f"buffer has {out.size} values, {n} needed")
        start = 0
        for array in arrays:
            stop = start + self.count(array)
            self.gather(array, out[start:stop])
            start = stop
        return out

    def unpack(self, arrays: Sequence[core.Array], buffer: np.ndarray):
        """Scatter a buffer filled by :meth:`pack` back into several arrays."""
        n = sum(self.count(a) for a in arrays)
        if buffer.size != n:
            raise ValueError(f"buffer has {buffer.size} values, {n} expected")
        start = 0
        for array in arrays:
            stop = start + self.count(array)
            self.scatter(array, buffer[start:stop])
            start = stop

    def global_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Global (j, i) indices of these cells (negative or beyond the global
        extent for cells outside the global domain)."""
        xoff, yoff = _offsets(self.grid)
        return self.j - self.grid.haloy + yoff, self.i - self.grid.halox + xoff

    def where(self, keep: npt.ArrayLike) -> "CellSet":
        """Subset of these cells for which ``keep`` (one value per cell) is true."""
        keep = np.asarray(keep, dtype=bool)
        return CellSet(self.grid, self.j[keep], self.i[keep])


def _offsets(grid: core.Grid) -> tuple[int, int]:
    if grid.tiling is None:
        return 0, 0
    return grid.tiling.xoffset, grid.tiling.yoffset


def _global_extent(grid: core.Grid) -> tuple[int, int]:
    """Global (ny, nx) of a grid, including its overlap (e.g. the X grid)."""
    if grid.tiling is None:
        return grid.ny, grid.nx
    return grid.tiling.ny_glob + grid.overlap, grid.tiling.nx_glob + grid.overlap


def _local_global_index(grid: core.Grid) -> tuple[np.ndarray, np.ndarray]:
    """Global j (column vector) and i (row vector) for every local all_values cell."""
    xoff, yoff = _offsets(grid)
    jglob = np.arange(grid.ny_)[:, np.newaxis] - grid.haloy + yoff
    iglob = np.arange(grid.nx_)[np.newaxis, :] - grid.halox + xoff
    return jglob, iglob


def outer_halo(grid: core.Grid) -> CellSet:
    """Cells of this subdomain in the ring around the global domain.

    The ring is the band of ``halox`` columns and ``haloy`` rows (corners
    included) just outside the global domain. For a child, the parent fills it.

    These cells are usually in the subdomain's halo, but pyGETM's tilings may
    let subdomains protrude beyond the global domain (negative
    ``xoffset_global``), and then part of the ring lies in the subdomain's own
    (masked) interior. Both are included. Cells further out are not. Subdomains
    that do not touch the global edge get an empty set. Cells are ordered row by
    row (j, then i).
    """
    ny_glob, nx_glob = _global_extent(grid)
    jglob, iglob = _local_global_index(grid)
    hx, hy = grid.halox, grid.haloy
    band = (jglob >= -hy) & (jglob < ny_glob + hy) & (iglob >= -hx) & (iglob < nx_glob + hx)
    inside = (jglob >= 0) & (jglob < ny_glob) & (iglob >= 0) & (iglob < nx_glob)
    ring = np.broadcast_to(band & ~inside, (grid.ny_, grid.nx_))
    j, i = np.nonzero(ring)
    return CellSet(grid, j, i)


def hole_rim(grid: core.Grid, hole: npt.ArrayLike, width: int) -> CellSet:
    """Cells of the hole within ``width`` cells (including diagonals) of cells
    outside the hole, restricted to this subdomain's local extent incl. halos.

    Args:
        grid: grid of the parent
        hole: boolean array with the global shape of ``grid`` (see
            :func:`_global_extent`), true where the child replaces the parent
        width: depth of the rim, normally the halo width of the parent grid

    Cells are ordered row by row (j, then i).
    """
    hole = np.asarray(hole, dtype=bool)
    ny_glob, nx_glob = _global_extent(grid)
    if hole.shape != (ny_glob, nx_glob):
        raise ValueError(f"hole has shape {hole.shape}, expected {(ny_glob, nx_glob)}")
    near = _dilate(~hole, width)
    rim = hole & near

    jglob, iglob = _local_global_index(grid)
    jglob = np.broadcast_to(jglob, (grid.ny_, grid.nx_))
    iglob = np.broadcast_to(iglob, (grid.ny_, grid.nx_))
    inside = (jglob >= 0) & (jglob < ny_glob) & (iglob >= 0) & (iglob < nx_glob)
    local = np.zeros((grid.ny_, grid.nx_), dtype=bool)
    local[inside] = rim[jglob[inside], iglob[inside]]
    j, i = np.nonzero(local)
    return CellSet(grid, j, i)


def _dilate(mask: np.ndarray, width: int) -> np.ndarray:
    """Cells within ``width`` (Chebyshev distance) of a true cell."""
    out = mask.copy()
    ny, nx = mask.shape
    for dj in range(-width, width + 1):
        for di in range(-width, width + 1):
            if dj == 0 and di == 0:
                continue
            src = mask[max(0, -dj) : ny - max(0, dj), max(0, -di) : nx - max(0, di)]
            out[max(0, dj) : ny - max(0, -dj), max(0, di) : nx - max(0, -di)] |= src
    return out


def combine(sets: Iterable[CellSet]) -> CellSet:
    """Concatenate cell sets on the same grid (order is preserved)."""
    sets = list(sets)
    if not sets:
        raise ValueError("no cell sets to combine")
    grid = sets[0].grid
    if any(s.grid is not grid for s in sets):
        raise ValueError("cell sets are on different grids")
    return CellSet(
        grid, np.concatenate([s.j for s in sets]), np.concatenate([s.i for s in sets])
    )
