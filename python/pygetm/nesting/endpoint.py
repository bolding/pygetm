"""This process's end of the interfaces of its level, shared by the couplers
(:class:`pygetm.nesting.nested2d.Nested2D`, :class:`pygetm.nesting.nested3d.Nested3D`):
the tile, its tiling, the plans with the parent and the children, local indices,
the ring this tile receives, its edge faces, and the receive of the children's
sub-face messages."""

from typing import TYPE_CHECKING, Callable, List, Tuple

import numpy as np
import numpy.typing as npt

from .plan import Plan, Tag

if TYPE_CHECKING:
    from .runtime import NestingRun


class Endpoint:

    __slots__ = (
        "down",
        "lev",
        "nest",
        "own_edge",
        "ri",
        "ring",
        "ring_cells",
        "rj",
        "t",
        "til",
        "up",
        "world",
    )

    def __init__(self, nest: "NestingRun"):
        self.nest = nest
        self.lev = nest.level
        self.t = nest.tile
        self.til = nest.tilings[self.lev.name]
        self.world = nest.world
        #: the plan with the parent (None for the root) and with every child
        self.up, self.down = nest.up, nest.down
        #: the edge faces of this level whose inner cell this tile owns
        self.own_edge = self.til.owner(*self.lev.edge["inner"]) == self.t
        #: the ring messages this tile receives from the parent: [(parent tile, cells)]
        self.ring: List[Tuple[int, tuple]] = []
        self.ring_cells = None
        self.rj = self.ri = np.zeros(0, dtype=int)
        if self.up is not None:
            self.ring = self.up.ring_to(self.t)
            #: the ring cells: parent cell (a, b: sub-cell position in it) and child
            #: cell (bj, bi)
            self.ring_cells = self.up.ring_cells(self.t)
            self.rj, self.ri = self.loc(self.ring_cells["bj"], self.ring_cells["bi"])

    def loc(self, j: npt.ArrayLike, i: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Local (with halos) indices of global cells of this level on this tile."""
        return self.til.local(self.t, j, i)

    def recv_subfaces(
        self, plan: Plan, rows: int, tag: Tag, recv: Callable
    ) -> np.ndarray:
        """The children's messages about this tile's interface faces with child ``plan``
        (:attr:`Plan.faces_of`): every child tile sends ``rows`` values per sub-face it
        computes; returns them as (rows, faces, r sub-faces), 0 where none came.
        ``recv(comm, buf, source, tag)``: the coupler's timed receive."""
        flist = plan.faces_of[self.t]
        out = np.zeros((rows, flist.size, plan.child.ratio))
        for tt, fpos, fa in plan.faces_to(self.t):
            buf = np.empty((rows, fpos.size))
            recv(plan.comm, buf, plan.coff + tt, tag)
            out[:, fpos, fa] = buf
        return out
