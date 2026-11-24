"""Dams and thin walls: lines of zero width that stop the flow between wet cells.

A dam is a polyline. Every U or V face whose connection between the two cell
centres crosses the line is closed (its mask is set to land on the supergrid), as
are the corners (X points) between two closed faces on one line, so that the dam
acts as a coast (free slip) on both sides. The cells on both sides stay water. This
needs nothing of pyGETM beyond its supergrid mask: U and V masks are taken from
it, and closing them is kept by :meth:`pygetm.domain.Domain.get_final_mask`.

The test is a crossing of segments, so it works on regular, curvilinear and
spherical (longitude/latitude, taken as a plane over each segment) grids. A dam
along cell edges closes exactly those edges; a dam at an angle closes a staircase.
A dam through a cell centre is ambiguous there: keep it off the centres.

    dam = pygetm.dams.close(domain, [(8.0, 54.0), (8.0, 54.3)])   # before create_grids / Simulation

Returns the number of U and V faces closed. A gap (an open sluice or a breach) is
two dams with the gap between them.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from .constants import CoordinateType


def _crossing(ax, ay, bx, by, p, q) -> np.ndarray:
    """Segments a-b (arrays) cross p-q (strictly on both sides of p-q's line; the
    ends of p-q count)."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    o1 = dx * (ay - p[1]) - dy * (ax - p[0])
    o2 = dx * (by - p[1]) - dy * (bx - p[0])
    ex, ey = bx - ax, by - ay
    o3 = ex * (p[1] - ay) - ey * (p[0] - ax)
    o4 = ex * (q[1] - ay) - ey * (q[0] - ax)
    return (o1 * o2 < 0.0) & (o3 * o4 <= 0.0)


def faces(x: np.ndarray, y: np.ndarray, line: Sequence[Tuple[float, float]]):
    """The U and V faces (as masks on the supergrid's U and V points,
    ``[1::2, ::2]`` and ``[::2, 1::2]``) and X corners (``[::2, ::2]``) closed by
    the polyline ``line`` on a supergrid with coordinates ``x``, ``y``."""
    line = [tuple(map(float, p)) for p in line]
    if len(line) < 2:
        raise ValueError("a dam needs at least two points")
    tx, ty = x[1::2, 1::2], y[1::2, 1::2]
    ny, nx = tx.shape
    u = np.zeros((ny, nx + 1), bool)  # supergrid [1::2, ::2]: face between T[j, i-1] and T[j, i]
    v = np.zeros((ny + 1, nx), bool)  # supergrid [::2, 1::2]: face between T[j-1, i] and T[j, i]
    for p, q in zip(line[:-1], line[1:]):
        u[:, 1:-1] |= _crossing(tx[:, :-1], ty[:, :-1], tx[:, 1:], ty[:, 1:], p, q)
        v[1:-1, :] |= _crossing(tx[:-1, :], ty[:-1, :], tx[1:, :], ty[1:, :], p, q)
    xc = np.zeros((ny + 1, nx + 1), bool)
    xc[1:-1, :] |= u[:-1, :] & u[1:, :]  # two closed U faces above each other: a wall north-south
    xc[:, 1:-1] |= v[:, :-1] & v[:, 1:]  # two closed V faces side by side: a wall east-west
    return u, v, xc


def close(domain, line: Sequence[Tuple[float, float]], coordinate_type: Optional[CoordinateType] = None) -> int:
    """Close the faces of ``domain`` crossed by the polyline ``line`` (points in
    ``coordinate_type``, default the domain's). Collective: call it on all
    processes of the domain, before its grids are created. Returns the number of
    faces closed between two wet cells."""
    n = 0
    if domain.comm.rank == 0:
        ct = coordinate_type or domain.coordinate_type
        if ct == CoordinateType.LONLAT:
            x, y = domain._lon, domain._lat
        elif ct == CoordinateType.XY:
            x, y = domain._x, domain._y
        else:  # index space: T centre (j, i) at (i + 0.5, j + 0.5)
            x = np.arange(2 * domain.nx + 1) / 2.0
            y = np.arange(2 * domain.ny + 1) / 2.0
            x, y = np.broadcast_arrays(x[None, :], y[:, None])
        if x is None or y is None:
            raise ValueError(f"dam: the domain has no {ct.name} coordinates")
        u, v, xc = faces(np.asarray(x, float), np.asarray(y, float), line)
        m = domain.mask
        twet = m[1::2, 1::2] != 0
        n = int((u[:, 1:-1] & twet[:, :-1] & twet[:, 1:]).sum() + (v[1:-1, :] & twet[:-1, :] & twet[1:, :]).sum())
        m[1::2, ::2][u] = 0
        m[::2, 1::2][v] = 0
        m[::2, ::2][xc] = 0
    return domain.comm.bcast(n)
