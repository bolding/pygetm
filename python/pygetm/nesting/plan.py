"""Bookkeeping of the M:N exchanges between a parent and a child.

Every level runs on its own pyGETM tiling. :class:`LevelTiling` describes such
a tiling as seen by every rank (map, subdomain size, offsets; exchanged once
with :meth:`LevelTiling.allgather`), in the level's global cell indices:
the pyGETM domain of a level is its window extended by the ring band, so
global index g corresponds to pyGETM index g + band. Tiles that pyGETM removed
(all land) have no owner.

:class:`Plan` holds, for one parent/child link, which tile sends what to
which. Both sides compute the same lists from the shared geometry and tilings,
so no handshake is needed:

* P (prolongation): parent tile -> child tile, (c, sx, sy) of the parent
  cell covering each child ring cell (ring order);
* R (restriction): child tile -> parent tile, value and weight of the
  child cells of each rim cell's ratio x ratio block (rim-major, a, b);
* F (flux register): child tile -> parent tile, the child's transports
  through the interface faces (r sub-faces each). The parent tile correcting
  a face is the owner of the cell outside the hole (flux matching), or,
  where that cell is not computed by the parent (the parent's own interface),
  of the hole cell (passed upwards); the owner of the face's U/V point sets
  the transport there, and gets the face too where it is another tile.

This is the nesting exchange; it shares no code with pyGETM's halo exchange.
"""

import enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import Level


class LevelTiling:
    """A level's pyGETM tiling in the level's global cell indices."""

    def __init__(self, info: dict, band: int, world_offset: int, view_offset=(0, 0)):
        self.map = np.asarray(info["map"])
        self.nx_sub, self.ny_sub = info["nx_sub"], info["ny_sub"]
        self.xoff, self.yoff = info["xoffset_global"], info["yoffset_global"]
        self.halo = info["halo"]
        self.band = band
        #: pyGETM index = global index + these (the band, and a level's overhang to the south/west)
        self.offj, self.offi = band + int(view_offset[0]), band + int(view_offset[1])
        self.offset = world_offset
        self.ntiles = int((self.map >= 0).sum())

    @staticmethod
    def info(grid) -> dict:
        """Description of the tiling of a pyGETM grid (to be shared)."""
        tl = grid.tiling
        return dict(map=tl.map.tolist(), nx_sub=tl.nx_sub, ny_sub=tl.ny_sub,
                    xoffset_global=tl.xoffset_global, yoffset_global=tl.yoffset_global,
                    halo=grid.halox)

    @property
    def removed(self) -> int:
        """Number of tiles pyGETM removed (all land)."""
        return int((self.map < 0).sum())

    def owner(self, j, i) -> np.ndarray:
        """Rank (within the level) owning global cells (j, i); -1 if none."""
        row = np.floor_divide(np.asarray(j) + self.offj - self.yoff, self.ny_sub)
        col = np.floor_divide(np.asarray(i) + self.offi - self.xoff, self.nx_sub)
        ok = (row >= 0) & (row < self.map.shape[0]) & (col >= 0) & (col < self.map.shape[1])
        out = np.full(np.shape(row), -1)
        out[ok] = self.map[row[ok], col[ok]]
        return out

    def origin(self, t: int) -> Tuple[int, int]:
        """Global (j, i) of the first interior cell of tile t."""
        row, col = (int(x[0]) for x in np.nonzero(self.map == t))
        return row * self.ny_sub + self.yoff - self.offj, col * self.nx_sub + self.xoff - self.offi

    def local(self, t: int, j, i):
        """Index into ``Array.all_values`` of tile t for global cells (j, i)."""
        y0, x0 = self.origin(t)
        return np.asarray(j) - y0 + self.halo, np.asarray(i) - x0 + self.halo


def gather_tilings(levels: List[Level], world, level: Level, grid, offsets: Dict[str, int]) -> Dict[str, LevelTiling]:
    """Share the tilings of all levels with every rank (one allgather)."""
    info = LevelTiling.info(grid) if grid.tiling.rank == 0 else None
    tilings = {}
    for name, inf in world.allgather((level.name, info)):
        if inf is not None:
            lev = next(l for l in levels if l.name == name)
            tilings[name] = LevelTiling(inf, lev.band, offsets[name], lev.view_offset)
    return tilings


class Tag(enum.IntEnum):
    """The kinds of nesting messages (the MPI tag). Every interface has its own communicator
    (:attr:`Plan.comm`), so the kind alone tells messages between two processes apart."""

    RING = 1  # P: parent cells (value and slopes) for the child's ring, 2D
    RIM = 2  # R: child cells of the parent's rim blocks, 2D
    FLUX = 3  # F: the child's flux registers through the interface faces, 2D
    RING_U = 4  # U, V transports for the child's ring faces
    RING_V = 5
    RIM_U = 6  # mean child U, V transports at the parent's rim faces
    RIM_V = 7
    RING_3D = 11  # 3D: the parent's columns for the child's ring
    RIM_3D = 12  # 3D: the child's columns of the rim blocks
    FLUX_3D = 13  # 3D: layer thicknesses and amounts through the interface faces
    TRANSPORT_3D = 14  # 3D: the child's layer transports through the interface faces (momentum_3d stage)
    RING_U_3D = 15  # 3D: layer transports for the child's ring faces
    RING_V_3D = 16


class Plan:
    """Exchange lists for one parent/child link."""

    def __init__(self, child: Level, ptil: LevelTiling, ctil: LevelTiling):
        self.child, self.ptil, self.ctil = child, ptil, ctil
        r = child.ratio
        i0, _, j0, _ = child.window
        par = child.parent

        # P: ring cells of the child and the parent cells covering them;
        # cells in removed (all-land) tiles are land after mask consistency
        bj, bi = child.ring_cells()
        pj, pi, a, b = child.ring_source(bj, bi)
        cown, pown = ctil.owner(bj, bi), ptil.owner(pj, pi)
        keep = (cown >= 0) & (pown >= 0)
        bj, bi, pj, pi, a, b, cown, pown = (x[keep] for x in (bj, bi, pj, pi, a, b, cown, pown))
        self.pro: Dict[Tuple[int, int], tuple] = {}
        for key in sorted(set(zip(pown.tolist(), cown.tolist()))):
            sel = (pown == key[0]) & (cown == key[1])
            self.pro[key] = (pj[sel], pi[sel], bj[sel], bi[sel], a[sel], b[sel])

        # R: rim cells of the parent and the child cells of their blocks
        rim = par.rims[child.name]
        rown = ptil.owner(rim[:, 0], rim[:, 1])
        self.res: Dict[Tuple[int, int], tuple] = {}
        self.rim_of: Dict[int, np.ndarray] = {}
        for p in range(ptil.ntiles):
            mine = rim[rown == p]
            self.rim_of[p] = mine
            if not len(mine):
                continue
            k = np.repeat(np.arange(len(mine)), r * r)
            aa = np.tile(np.repeat(np.arange(r), r), len(mine))
            bb = np.tile(np.arange(r), r * len(mine))
            cj = (mine[k, 0] - j0) * r + aa
            ci = (mine[k, 1] - i0) * r + bb
            own = ctil.owner(cj, ci)
            for t in range(ctil.ntiles):
                sel = own == t
                if sel.any():
                    self.res[(t, p)] = (k[sel], aa[sel], bb[sel], cj[sel], ci[sel])

        # F: interface faces per parent tile, and the child edge faces
        # (r per interface face, k = n * r + a) feeding them
        wf = par.wfaces[child.name]
        hj, hi = wf["hole"]
        oj, oi = wf["out"]
        # two roles per face: the correction of the cell outside the hole (elevation, tracers; for a face
        # passed up: the hole cell's tile) and the transport the parent continues from, at the face's own
        # U/V point. An x face has its western cell's index, a y face its southern cell's, so with the
        # hole west or south of the outside cell the point is the hole cell's; where that is another tile
        # (a subdomain boundary on the interface) both tiles get the face. A point in a removed tile
        # stays with the correcting tile.
        face_owner = np.where(wf["inside"], ptil.owner(oj, oi), ptil.owner(hj, hi))
        point_owner = np.where(wf["inside"], ptil.owner(*wf["face"]), -1)
        point_owner = np.where(wf["inside"] & (point_owner < 0), face_owner, point_owner)
        #: interface face indices n handled by each parent tile (sorted), and per entry whether the tile
        #: applies the correction (:attr:`corr_of`) and sets the transport (:attr:`uv_of`)
        self.faces_of: Dict[int, np.ndarray] = {}
        self.corr_of: Dict[int, np.ndarray] = {}
        self.uv_of: Dict[int, np.ndarray] = {}
        self.flx: Dict[Tuple[int, int], tuple] = {}
        for p in range(ptil.ntiles):
            mine = np.nonzero((face_owner == p) | (point_owner == p))[0]
            self.faces_of[p] = mine
            self.corr_of[p] = face_owner[mine] == p
            self.uv_of[p] = point_owner[mine] == p
            if not mine.size:
                continue
            fpos = np.repeat(np.arange(mine.size), r)
            fa = np.tile(np.arange(r), mine.size)
            fk = mine[fpos] * r + fa
            own = ctil.owner(child.edge["inner"][0][fk], child.edge["inner"][1][fk])
            for t in range(ctil.ntiles):
                sel = own == t
                if sel.any():
                    self.flx[(t, p)] = (fpos[sel], fa[sel], fk[sel])

    # --- the lists one tile works through (sorted by the other side's tile: the same order on both sides) ---

    def ring_to(self, t: int) -> List[Tuple[int, tuple]]:
        """P entries child tile ``t`` receives: [(parent tile, (pj, pi, bj, bi, a, b))]."""
        return [(p, v) for (p, tt), v in sorted(self.pro.items()) if tt == t]

    def ring_cells(self, t: int) -> Dict[str, np.ndarray]:
        """The ring cells of child tile ``t`` in the order of :meth:`ring_to`: parent cells ``pj, pi``, child cells
        ``bj, bi`` (global), sub-cell position ``a`` (j), ``b`` (i) within the parent cell."""
        ring = self.ring_to(t)
        return {k: (np.concatenate([v[n] for _, v in ring]) if ring else np.zeros(0, int))
                for n, k in enumerate(("pj", "pi", "bj", "bi", "a", "b"))}

    def faces_to(self, p: int) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """F messages parent tile ``p`` receives: [(child tile, position in :attr:`faces_of` [p], sub-face)]."""
        return [(t, fpos, fa) for (t, pp), (fpos, fa, _) in sorted(self.flx.items()) if pp == p]

    def faces_from(self, t: int) -> List[Tuple[Tuple[int, int], np.ndarray]]:
        """F messages child tile ``t`` sends: [((t, parent tile), child edge faces fk)]."""
        return [(key, fk) for key, (_, _, fk) in sorted(self.flx.items()) if key[0] == t]

    def describe(self, me: int, role: str, tile: int) -> List[str]:
        """Messages of this rank, in the format of check_print_plan.py
        (pygetm-nesting): every send must match exactly one receive."""
        out = []
        pw, cw = self.ptil.offset, self.ctil.offset
        for (p, t), v in self.pro.items():
            if role == "parent" and p == tile:
                out.append(f"[rank {me}]  P  local[ring of {self.child.name}]  → rank {cw + t}  tag={int(Tag.RING)}  n={3 * v[0].size}")
            if role == "child" and t == tile:
                out.append(f"[rank {me}]  P  from rank {pw + p}  tag={int(Tag.RING)}  n_parent={3 * v[0].size}")
        for (t, p), v in self.res.items():
            if role == "parent" and p == tile:
                out.append(f"[rank {me}]  R  from rank {cw + t}  tag={int(Tag.RIM)}  n_parent={2 * v[0].size}")
            if role == "child" and t == tile:
                out.append(f"[rank {me}]  R  local[rim block]  → rank {pw + p}  tag={int(Tag.RIM)}  n={2 * v[0].size}")
        for (t, p), v in self.flx.items():
            if role == "parent" and p == tile:
                out.append(f"[rank {me}]  F  from rank {cw + t}  tag={int(Tag.FLUX)}  n_parent={v[0].size}")
            if role == "child" and t == tile:
                out.append(f"[rank {me}]  F  local[flux register]  → rank {pw + p}  tag={int(Tag.FLUX)}  n={v[0].size}")
        return out

    def check(self) -> Dict[str, int]:
        """Coverage checks (all zero if consistent): ring cells sent twice,
        rim block cells sent twice, window faces x sub-faces not covered."""
        ring = {}
        for v in self.pro.values():
            for j, i in zip(v[2].tolist(), v[3].tolist()):
                ring[(j, i)] = ring.get((j, i), 0) + 1
        block = {}
        for (t, p), v in self.res.items():
            for k, a, b in zip(v[0].tolist(), v[1].tolist(), v[2].tolist()):
                block[(p, k, a, b)] = block.get((p, k, a, b), 0) + 1
        faces = {}
        for (t, p), v in self.flx.items():
            for n, a in zip(v[0].tolist(), v[1].tolist()):
                faces[(p, n, a)] = faces.get((p, n, a), 0) + 1
        nfaces = sum(f.size for f in self.faces_of.values()) * self.child.ratio
        return {
            "ring_duplicates": sum(c - 1 for c in ring.values()),
            "block_duplicates": sum(c - 1 for c in block.values()),
            "face_duplicates": sum(c - 1 for c in faces.values()),
            "faces_uncovered": nfaces - len(faces),
        }
