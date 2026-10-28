"""Conservative transfer operators between parent and child.

All operators work on plain numpy arrays and use a fixed summation order, so
that a decomposed (MPI) run reproduces a serial one bit for bit.

Prolongation (parent -> child ring)
    Limited piecewise-linear reconstruction inside each parent cell:
    value(a, b) = c + sx * xi(b) + sy * xi(a), xi(k) = (k + 0.5) / r - 0.5.
    The r x r sub-cells average to the parent value; at ratio 1 the parent
    value is copied. Slopes are minmod-limited and stop at land.

Restriction (child -> parent rim)
    Weighted mean (weights: volume D x area, 0 for dry cells) of the r x r
    child cells.

Flux matching
    Over one parent step, the transport through a parent face is the sum over
    its r child faces and the child's sub-steps; the parent cell next to the
    window is corrected by (parent flux - child flux).
"""

import numpy as np


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod limiter: the smaller of a and b if they have the same sign, else 0."""
    return np.where(a * b > 0.0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def limited_slopes(c: np.ndarray, wet: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """Minmod-limited slopes (per cell, in units of the cell) along x and y.

    Cells on the outermost row/column of ``c`` get slope 0. With a ``wet``
    mask, a difference towards a dry neighbour counts as 0, so the slope of a
    cell next to land is 0 in that direction (and dry cells get 0).
    """
    sx = np.zeros_like(c)
    sy = np.zeros_like(c)
    if wet is None:
        sx[:, 1:-1] = minmod(c[:, 1:-1] - c[:, :-2], c[:, 2:] - c[:, 1:-1])
        sy[1:-1, :] = minmod(c[1:-1, :] - c[:-2, :], c[2:, :] - c[1:-1, :])
        return sx, sy
    wet = np.asarray(wet, dtype=bool)
    with np.errstate(invalid="ignore"):
        dw = np.where(wet[:, 1:-1] & wet[:, :-2], c[:, 1:-1] - c[:, :-2], 0.0)
        de = np.where(wet[:, 1:-1] & wet[:, 2:], c[:, 2:] - c[:, 1:-1], 0.0)
        ds = np.where(wet[1:-1, :] & wet[:-2, :], c[1:-1, :] - c[:-2, :], 0.0)
        dn = np.where(wet[1:-1, :] & wet[2:, :], c[2:, :] - c[1:-1, :], 0.0)
    sx[:, 1:-1] = minmod(dw, de)
    sy[1:-1, :] = minmod(ds, dn)
    sx[~wet] = 0.0
    sy[~wet] = 0.0
    return sx, sy


def limited_slopes_at(c: np.ndarray, wet: "np.ndarray | None", j, i) -> "tuple[np.ndarray, np.ndarray]":
    """:func:`limited_slopes` at the cells (j, i) only, with the same
    arithmetic (identical values), for the few cells a parent sends."""
    j, i = np.asarray(j), np.asarray(i)
    ny, nx = c.shape
    if wet is None:
        wet = np.ones(c.shape, dtype=bool)
    wet = np.asarray(wet, dtype=bool)
    sx = np.zeros(j.shape)
    sy = np.zeros(j.shape)
    x = (i > 0) & (i < nx - 1)
    y = (j > 0) & (j < ny - 1)
    jx, ix = j[x], i[x]
    jy, iy = j[y], i[y]
    with np.errstate(invalid="ignore"):
        dw = np.where(wet[jx, ix] & wet[jx, ix - 1], c[jx, ix] - c[jx, ix - 1], 0.0)
        de = np.where(wet[jx, ix] & wet[jx, ix + 1], c[jx, ix + 1] - c[jx, ix], 0.0)
        ds = np.where(wet[jy, iy] & wet[jy - 1, iy], c[jy, iy] - c[jy - 1, iy], 0.0)
        dn = np.where(wet[jy, iy] & wet[jy + 1, iy], c[jy + 1, iy] - c[jy, iy], 0.0)
    sx[x] = minmod(dw, de)
    sy[y] = minmod(ds, dn)
    dry = ~wet[j, i]
    sx[dry] = 0.0
    sy[dry] = 0.0
    return sx, sy


def subcell_offset(k, r: int):
    """Centre of sub-cell k (0..r-1) relative to the parent cell centre, in
    parent-cell units."""
    return (np.asarray(k) + 0.5) / r - 0.5


def prolong(c, sx, sy, a, b, r: int) -> np.ndarray:
    """Value in child sub-cell (row a, column b) of parent cells (c, sx, sy)."""
    return c + sx * subcell_offset(b, r) + sy * subcell_offset(a, r)


def restrict(blk: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Weighted mean of (n, r, r) child blocks, summed in fixed order.

    ``area`` holds the weights: cell areas, or volumes (depth x area) with 0
    for dry cells. Blocks without weight (all dry) give NaN.
    """
    r = blk.shape[1]
    mass = np.zeros(blk.shape[0])
    total = np.zeros(blk.shape[0])
    for a in range(r):
        for b in range(r):
            w = area[:, a, b]
            mass = mass + np.where(w > 0, blk[:, a, b] * w, 0.0)
            total = total + w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, mass / np.where(total > 0, total, 1.0), np.nan)


def face_sum(sub: np.ndarray) -> np.ndarray:
    """Sum of (n, r) child-face transports per parent face, in fixed order."""
    total = np.zeros(sub.shape[0])
    for a in range(sub.shape[1]):
        total = total + sub[:, a]
    return total


__all__ = [
    "minmod",
    "limited_slopes",
    "limited_slopes_at",
    "subcell_offset",
    "prolong",
    "restrict",
    "face_sum",
]
