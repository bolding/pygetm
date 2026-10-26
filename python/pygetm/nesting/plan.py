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
  through the interface faces (r sub-faces each). The parent tile handling
  a face is the owner of the cell outside the hole (flux matching), or,
  where that cell is not computed by the parent (the parent's own interface),
  of the hole cell (passed upwards).

This is the nesting exchange; it shares no code with pyGETM's halo exchange.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import Level


class LevelTiling:
    """A level's pyGETM tiling in the level's global cell indices."""

    def __init__(self, info: dict, band: int, world_offset: int):
        self.map = np.asarray(info["map"])
        self.nx_sub, self.ny_sub = info["nx_sub"], info["ny_sub"]
        self.xoff, self.yoff = info["xoffset_global"], info["yoffset_global"]
        self.halo = info["halo"]
        self.band = band
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
        row = np.floor_divide(np.asarray(j) + self.band - self.yoff, self.ny_sub)
        col = np.floor_divide(np.asarray(i) + self.band - self.xoff, self.nx_sub)
        ok = (row >= 0) & (row < self.map.shape[0]) & (col >= 0) & (col < self.map.shape[1])
        out = np.full(np.shape(row), -1)
        out[ok] = self.map[row[ok], col[ok]]
        return out

    def origin(self, t: int) -> Tuple[int, int]:
        """Global (j, i) of the first interior cell of tile t."""
        row, col = (int(x[0]) for x in np.nonzero(self.map == t))
        return row * self.ny_sub + self.yoff - self.band, col * self.nx_sub + self.xoff - self.band

    def local(self, t: int, j, i):
        """Index into ``Array.all_values`` of tile t for global cells (j, i)."""
        y0, x0 = self.origin(t)
        return np.asarray(j) - y0 + self.halo, np.asarray(i) - x0 + self.halo


def gather_tilings(levels: List[Level], world, level: Level, grid, offsets: Dict[str, int]) -> Dict[str, LevelTiling]:
    """Share the tilings of all levels with every rank (one allgather)."""
    info = LevelTiling.info(grid) if grid is not None and grid.tiling.rank == 0 else None
    tilings = {}
    for name, inf in world.allgather((level.name, info)):
        if inf is not None:
            lev = next(l for l in levels if l.name == name)
            tilings[name] = LevelTiling(inf, lev.band, offsets[name])
    return tilings


class Plan:
    """Exchange lists for one parent/child link."""

    def __init__(self, child: Level, ptil: LevelTiling, ctil: LevelTiling, link: int, tag_base: int = 200):
        self.child, self.ptil, self.ctil = child, ptil, ctil
        self.tag_p, self.tag_r, self.tag_f = tag_base + 3 * link, tag_base + 3 * link + 1, tag_base + 3 * link + 2
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
        face_owner = np.where(wf["inside"], ptil.owner(oj, oi), ptil.owner(hj, hi))
        #: interface face indices n handled by each parent tile
        self.faces_of: Dict[int, np.ndarray] = {}
        self.flx: Dict[Tuple[int, int], tuple] = {}
        for p in range(ptil.ntiles):
            mine = np.nonzero(face_owner == p)[0]
            self.faces_of[p] = mine
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

    def describe(self, me: int, role: str, tile: int) -> List[str]:
        """Messages of this rank, in the format of check_print_plan.py
        (pygetm-nesting): every send must match exactly one receive."""
        out = []
        pw, cw = self.ptil.offset, self.ctil.offset
        for (p, t), v in self.pro.items():
            if role == "parent" and p == tile:
                out.append(f"[rank {me}]  P  local[ring of {self.child.name}]  → rank {cw + t}  tag={self.tag_p}  n={3 * v[0].size}")
            if role == "child" and t == tile:
                out.append(f"[rank {me}]  P  from rank {pw + p}  tag={self.tag_p}  n_parent={3 * v[0].size}")
        for (t, p), v in self.res.items():
            if role == "parent" and p == tile:
                out.append(f"[rank {me}]  R  from rank {cw + t}  tag={self.tag_r}  n_parent={2 * v[0].size}")
            if role == "child" and t == tile:
                out.append(f"[rank {me}]  R  local[rim block]  → rank {pw + p}  tag={self.tag_r}  n={2 * v[0].size}")
        for (t, p), v in self.flx.items():
            if role == "parent" and p == tile:
                out.append(f"[rank {me}]  F  from rank {cw + t}  tag={self.tag_f}  n_parent={v[0].size}")
            if role == "child" and t == tile:
                out.append(f"[rank {me}]  F  local[flux register]  → rank {pw + p}  tag={self.tag_f}  n={v[0].size}")
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
