"""
pygetm.nesting.exchange — MPI boundary-strip exchange for nested-grid simulations.

Moved here from the pygetm-nesting repository (lib/exchange.py), where it was
developed and tested independently of pyGETM; it depends only on numpy and
mpi4py. It is organised in two phases:

Arakawa C-grid constants and tag helper
────────────────────────────────────────
  TGRID = 0, UGRID = 1, VGRID = 2
      Grid-type identifiers for the NE-convention Arakawa C-grid.

  make_tags(base, grid=TGRID) → (tag_w, tag_e, tag_s, tag_n)
      Encode direction and grid type into four MPI tags.  Tags for
      different grids are offset by _DIR_STRIDE=4; domain pairs must
      use *base* values spaced ≥ 4×n_grids apart to avoid collisions.

Planning (call once, before the time loop)
──────────────────────────────────────────
  child_window(child_geo, parent_geo)
      Derive the child region in parent T-grid indices (ci0,ci1,cj0,cj1)
      and the refinement ratio from two geometry dicts.

  bc_widths(ci0,ci1,cj0,cj1, nx,ny, halo,ratio, child_nx,child_ny)
      Number of child cells to overwrite on each of the four sides.

  best_tiling(n_ranks, ny, nx)
      Closest-aspect-ratio (nrows, ncols) factorisation of n_ranks.

  domain_slice(local_rank, nrows, ncols, ny, nx)
      Row/column slice owned by a given local rank.

  build_cell_mask(mask_cells, ny, nx)
      Boolean mask array from a list of rectangular exclusion regions.

  StripExchangePlan.build_send(...)       — parent: single-variable plan
  StripExchangePlan.build_recv(...)       — child:  single-variable plan
  StripExchangePlan.build_send_multi(..., n_vars)  — parent: packed n-var plan
  StripExchangePlan.build_recv_multi(..., n_vars)  — child:  packed n-var plan

Execution (call every timestep)
────────────────────────────────
  StripExchangePlan.send(comm, local_data)
      Non-blocking Isend of innermost parent strips to every child rank
      whose BC region overlaps this rank's subdomain.

  StripExchangePlan.recv(comm) → (w_up, e_up, s_up, n_up)
      Non-blocking Irecv from all contributing parent ranks; nearest-
      neighbour upscale; assemble into pre-allocated BC output arrays.

  StripExchangePlan.send_multi(comm, data_list)
      Like send but packs n_vars arrays into one message per direction.

  StripExchangePlan.recv_multi(comm) → list of (w_up,e_up,s_up,n_up)
      Like recv but unpacks n_vars variables; returns one tuple per var.

  apply_bc_sub(local_data, w_up,e_up,s_up,n_up, ...)
      Write upscaled strips into a child rank's local subdomain.

  assemble_global(local_data, jsl,isl, ny,nx, local_comm)
      Reduce all subdomain tiles onto local rank 0.

Two exchange strategies are provided
──────────────────────────────────────
  Gather / broadcast  (simple, suitable for O(10) ranks per domain)
    gather_send_strips / recv_bcast_strips
    Parent: all ranks Reduce(SUM) the strip → rank 0 → blocking Send.
    Child:  rank 0 Recv → Bcast to all.

  P2P  (preferred, scales to large tile counts)
    StripExchangePlan.build_send / build_recv + send / recv
    Parent: each rank that owns the innermost strip column or row posts
            Isend directly to every child rank whose BC band overlaps.
    Child:  each rank posts Irecv from every parent rank expected to
            contribute, then nearest-neighbour upscales in-place.
"""

import numpy as np
from mpi4py import MPI


# ── Arakawa C-grid type constants (NE convention) ─────────────────────────────

TGRID = 0   # Tracer / scalar grid — cell centres       (i,   j  )
UGRID = 1   # U-velocity grid      — east  face (x+½)  (i+½, j  )
VGRID = 2   # V-velocity grid      — north face (y+½)  (i,   j+½)

_DIR_STRIDE = 4   # directions W=0 E=1 S=2 N=3; grid_id × stride separates grids


def make_tags(base: int, grid: int = TGRID) -> "tuple[int,int,int,int]":
    """Return (tag_w, tag_e, tag_s, tag_n) for one domain pair and one grid type.

    Tags are encoded as ``base + direction + grid * _DIR_STRIDE`` where
    direction W=0, E=1, S=2, N=3.  Domain pairs must use *base* values spaced
    at least ``_DIR_STRIDE × n_grids`` apart to avoid collisions between pairs.
    Within one pair, TGRID/UGRID/VGRID tags are already distinct by construction.

    Example — root→north_sea pair, base=10:
      TGRID: (10, 11, 12, 13)
      UGRID: (14, 15, 16, 17)
      VGRID: (18, 19, 20, 21)
    north_sea→skagerrak pair, base=30  (≥ 10 + 4×3 = 22):
      TGRID: (30, 31, 32, 33)  ...

    Parameters
    ----------
    base : int — base tag for the W direction of the T-grid exchange
    grid : int — TGRID (0), UGRID (1), or VGRID (2)
    """
    off = grid * _DIR_STRIDE
    return base + off, base + 1 + off, base + 2 + off, base + 3 + off


# ── Geometry helpers ──────────────────────────────────────────────────────────

def child_window(child_geo: dict, parent_geo: dict) -> "tuple[int,int,int,int,int]":
    """Return (ci0, ci1, cj0, cj1, ratio) — child region in parent T-grid.

    ci0/ci1 use the same convention as the YAML istart/istop keys:
    ci0 is the first column of the parent that the child covers; ci1 is
    one past the last (exclusive right).  ratio = parent_dlon / child_dlon.

    Works for both Cartesian (dx/dy/x0/y0) and spherical (dlon/dlat/lon0/lat0)
    geometry dicts.  The key names ``dlon``/``dlat``/``lon0``/``lat0`` are tried
    first; ``dx``/``dy``/``x0``/``y0`` are used as fallback.

    Parameters
    ----------
    child_geo  : dict  with keys nx, ny, and origin+spacing (lon0/lat0/dlon/dlat
                       or x0/y0/dx/dy)
    parent_geo : dict  same keys for the parent domain
    """
    # Support spherical (lon0/lat0/dlon/dlat) and Cartesian (x0/y0/dx/dy)
    if "dlon" in parent_geo:
        p_dx, p_dy = parent_geo["dlon"], parent_geo["dlat"]
        p_x0, p_y0 = parent_geo["lon0"], parent_geo["lat0"]
        c_dx, c_dy = child_geo["dlon"],  child_geo["dlat"]
        c_x0, c_y0 = child_geo["lon0"],  child_geo["lat0"]
    else:
        p_dx, p_dy = parent_geo["dx"], parent_geo["dy"]
        p_x0, p_y0 = parent_geo["x0"], parent_geo["y0"]
        c_dx, c_dy = child_geo["dx"],   child_geo["dy"]
        c_x0, c_y0 = child_geo["x0"],   child_geo["y0"]

    ci0   = round((c_x0 - p_x0) / p_dx)
    ci1   = round((c_x0 + child_geo["nx"] * c_dx - p_x0) / p_dx)
    cj0   = round((c_y0 - p_y0) / p_dy)
    cj1   = round((c_y0 + child_geo["ny"] * c_dy - p_y0) / p_dy)
    ratio = round(p_dx / c_dx)
    return ci0, ci1, cj0, cj1, ratio


def bc_widths(
    ci0: int, ci1: int, cj0: int, cj1: int,
    nx:  int, ny:  int,
    halo: int, ratio: int,
    child_nx: int, child_ny: int,
) -> "tuple[int,int,int,int]":
    """Number of child cells to overwrite on each of the four sides.

    Returns (bc_w, bc_e, bc_s, bc_n).  A side gets width 0 when there is
    no adjacent parent cell (e.g. child touching the parent domain edge).
    """
    halo_w = ci0 - max(ci0 - halo, 0)
    halo_e = min(ci1 + halo, nx) - ci1
    halo_s = cj0 - max(cj0 - halo, 0)
    halo_n = min(cj1 + halo, ny) - cj1
    return (min(halo_w * ratio, child_nx // 2),
            min(halo_e * ratio, child_nx // 2),
            min(halo_s * ratio, child_ny // 2),
            min(halo_n * ratio, child_ny // 2))


# ── Upscaling ─────────────────────────────────────────────────────────────────

def upscale(strip: np.ndarray, ratio: int,
            target_rows: int, target_cols: int) -> np.ndarray:
    """Nearest-neighbour upscale via np.repeat, trimmed to (target_rows, target_cols)."""
    out = np.repeat(strip, ratio, axis=0)[:target_rows]
    out = np.repeat(out,   ratio, axis=1)[:, :target_cols]
    return out


# ── Cell-mask helper ──────────────────────────────────────────────────────────

def build_cell_mask(mask_cells: list, ny: int, nx: int) -> np.ndarray:
    """Build a boolean (ny, nx) mask from a list of rectangular regions.

    Parameters
    ----------
    mask_cells : list of [i0, j0, i1, j1]
        Each entry is a masked rectangle in the domain's T-grid:
          i0, i1 : first and last column (0-based, inclusive)
          j0, j1 : first and last row    (0-based, inclusive)
    ny, nx : domain dimensions

    Returns
    -------
    ndarray bool (ny, nx) — True where a cell is permanently masked.
    """
    arr = np.zeros((ny, nx), dtype=bool)
    for rect in mask_cells:
        i0, j0, i1, j1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        i0 = max(0, min(i0, nx - 1))
        i1 = max(0, min(i1, nx - 1))
        j0 = max(0, min(j0, ny - 1))
        j1 = max(0, min(j1, ny - 1))
        arr[j0:j1 + 1, i0:i1 + 1] = True
    return arr


# ── BC application ────────────────────────────────────────────────────────────

def apply_bc_sub(
    local_data: np.ndarray,
    w_up: np.ndarray, e_up: np.ndarray,
    s_up: np.ndarray, n_up: np.ndarray,
    bc_w: int, bc_e: int, bc_s: int, bc_n: int,
    child_ny: int, child_nx: int,
    jsl: slice, isl: slice,
    mask: "np.ndarray | None" = None,
) -> None:
    """Overwrite BC border cells in a local subdomain from upscaled parent strips.

    *_up arrays span the full child domain:
        w_up / e_up — shape (child_ny, bc_*)
        s_up / n_up — shape (bc_*, child_nx)
    Only the intersection of [jsl, isl] with each BC band is written.

    Parameters
    ----------
    mask : ndarray bool (child_ny, child_nx), optional
        Cells where mask is True are permanently excluded: they are not
        overwritten by BC values and retain the value -1.0 throughout.
        Pass the full-domain mask (built with build_cell_mask); the function
        extracts the relevant sub-region internally.
    """
    # West: child columns 0 .. bc_w-1
    if isl.start < bc_w:
        c1  = min(bc_w, isl.stop)
        src = w_up[jsl.start:jsl.stop, isl.start:c1]
        if mask is None:
            local_data[:, :c1 - isl.start] = src
        else:
            np.copyto(local_data[:, :c1 - isl.start], src,
                      where=~mask[jsl.start:jsl.stop, isl.start:c1])
    # East: child columns child_nx-bc_e .. child_nx-1
    east = child_nx - bc_e
    if isl.stop > east:
        c0  = max(east, isl.start)
        src = e_up[jsl.start:jsl.stop, c0 - east:isl.stop - east]
        if mask is None:
            local_data[:, c0 - isl.start:] = src
        else:
            np.copyto(local_data[:, c0 - isl.start:], src,
                      where=~mask[jsl.start:jsl.stop, c0:isl.stop])
    # South: child rows 0 .. bc_s-1
    if jsl.start < bc_s:
        r1  = min(bc_s, jsl.stop)
        src = s_up[jsl.start:r1, isl.start:isl.stop]
        if mask is None:
            local_data[:r1 - jsl.start, :] = src
        else:
            np.copyto(local_data[:r1 - jsl.start, :], src,
                      where=~mask[jsl.start:r1, isl.start:isl.stop])
    # North: child rows child_ny-bc_n .. child_ny-1
    north = child_ny - bc_n
    if jsl.stop > north:
        r0  = max(north, jsl.start)
        src = n_up[r0 - north:jsl.stop - north, isl.start:isl.stop]
        if mask is None:
            local_data[r0 - jsl.start:, :] = src
        else:
            np.copyto(local_data[r0 - jsl.start:, :], src,
                      where=~mask[r0:jsl.stop, isl.start:isl.stop])


# ── Subdomain decomposition ───────────────────────────────────────────────────

def best_tiling(n_ranks: int, ny: int, nx: int) -> "tuple[int,int]":
    """Return (nrows, ncols) with nrows×ncols == n_ranks, aspect ratio ≈ ny/nx."""
    best, best_score = (1, n_ranks), float("inf")
    for nr in range(1, n_ranks + 1):
        if n_ranks % nr == 0:
            nc    = n_ranks // nr
            score = abs(nr / nc - ny / nx)
            if score < best_score:
                best_score, best = score, (nr, nc)
    return best


def domain_slice(
    local_rank: int, nrows: int, ncols: int, ny: int, nx: int,
) -> "tuple[slice,slice]":
    """Return (jslice, islice) — global index range owned by local rank *local_rank*."""
    irow, icol = local_rank // ncols, local_rank % ncols
    j0 = (irow     * ny) // nrows;  j1 = ((irow + 1) * ny) // nrows
    i0 = (icol     * nx) // ncols;  i1 = ((icol + 1) * nx) // ncols
    return slice(j0, j1), slice(i0, i1)


# ── MPI collective helpers ────────────────────────────────────────────────────

def assemble_global(
    local_data: np.ndarray,
    jsl: slice, isl: slice,
    ny: int, nx: int,
    local_comm: "MPI.Comm",
) -> "np.ndarray | None":
    """Reduce all subdomain tiles onto local rank 0.

    Returns the full (ny, nx) array on local rank 0; None on all other ranks.
    """
    buf  = np.zeros((ny, nx), dtype=np.float64)
    buf[jsl, isl] = local_data
    recv = np.empty((ny, nx), dtype=np.float64) if local_comm.rank == 0 else None
    local_comm.Reduce(buf, recv, op=MPI.SUM, root=0)
    return recv


def gather_strip(
    local_data: np.ndarray,
    jsl: slice, isl: slice,
    gj: slice, gi: slice,
    local_comm: "MPI.Comm",
) -> "np.ndarray | None":
    """Reduce each rank's contribution to global strip [gj, gi] onto local rank 0.

    Returns the assembled strip on local rank 0; None on all other ranks.
    """
    sny = gj.stop - gj.start
    snx = gi.stop  - gi.start
    buf = np.zeros((sny, snx), dtype=np.float64)
    r0 = max(gj.start, jsl.start);  r1 = min(gj.stop, jsl.stop)
    c0 = max(gi.start, isl.start);  c1 = min(gi.stop,  isl.stop)
    if r0 < r1 and c0 < c1:
        buf[r0 - gj.start:r1 - gj.start, c0 - gi.start:c1 - gi.start] = \
            local_data[r0 - jsl.start:r1 - jsl.start, c0 - isl.start:c1 - isl.start]
    recv = np.empty((sny, snx), dtype=np.float64) if local_comm.rank == 0 else None
    local_comm.Reduce(buf, recv, op=MPI.SUM, root=0)
    return recv


def gather_send_strips(
    comm: "MPI.Comm",
    local_data: np.ndarray,
    jsl: slice, isl: slice,
    local_comm: "MPI.Comm",
    ci0: int, ci1: int, cj0: int, cj1: int,
    nx: int, ny: int,
    halo: int,
    dest_global: int,
    tag_w: int, tag_e: int, tag_s: int, tag_n: int,
) -> None:
    """Gather all four boundary strips to local rank 0, which sends them to dest_global.

    Parameters
    ----------
    comm        — COMM_WORLD (for the cross-domain Send)
    local_comm  — per-domain communicator (for the intra-domain Reduce)
    ci0..cj1    — child region in parent cell indices
    dest_global — global rank of the receiving domain's local rank 0
    """
    lr = local_comm.rank
    wi = max(ci0 - halo, 0);  we = ci0
    ei = ci1;                  ee = min(ci1 + halo, nx)
    sj = max(cj0 - halo, 0);  se = cj0
    nj = cj1;                  ne = min(cj1 + halo, ny)
    for tag, gj, gi in [
        (tag_w, slice(cj0, cj1), slice(wi, we)),
        (tag_e, slice(cj0, cj1), slice(ei, ee)),
        (tag_s, slice(sj, se),   slice(ci0, ci1)),
        (tag_n, slice(nj, ne),   slice(ci0, ci1)),
    ]:
        full = gather_strip(local_data, jsl, isl, gj, gi, local_comm)
        if lr == 0:
            comm.Send(np.ascontiguousarray(full), dest=dest_global, tag=tag)


def recv_bcast_strips(
    comm: "MPI.Comm",
    local_comm: "MPI.Comm",
    source_global: int,
    rows: int, cols: int,
    halo_w: int, halo_e: int, halo_s: int, halo_n: int,
    tag_w: int, tag_e: int, tag_s: int, tag_n: int,
) -> "tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]":
    """Rank 0 receives four strips from source_global; all ranks get them via Bcast.

    Returns (w, e, s, n) strip arrays valid on all ranks.

    Parameters
    ----------
    comm          — COMM_WORLD (for the cross-domain Recv)
    local_comm    — per-domain communicator (for the Bcast)
    source_global — global rank of the sending domain's local rank 0
    """
    lr = local_comm.rank
    w  = np.empty((rows, halo_w), dtype=np.float64)
    e  = np.empty((rows, halo_e), dtype=np.float64)
    s  = np.empty((halo_s, cols), dtype=np.float64)
    n  = np.empty((halo_n, cols), dtype=np.float64)
    if lr == 0:
        comm.Recv(w, source=source_global, tag=tag_w)
        comm.Recv(e, source=source_global, tag=tag_e)
        comm.Recv(s, source=source_global, tag=tag_s)
        comm.Recv(n, source=source_global, tag=tag_n)
    local_comm.Bcast(w, root=0)
    local_comm.Bcast(e, root=0)
    local_comm.Bcast(s, root=0)
    local_comm.Bcast(n, root=0)
    return w, e, s, n


# ── P2P strip exchange ────────────────────────────────────────────────────────

class StripExchangePlan:
    """Precomputed P2P boundary-strip exchange schedule for one MPI rank.

    The geometry (which ranks to send to / receive from, what local slices to
    extract, where to place upscaled values in the BC arrays) is fixed for the
    lifetime of a simulation.  Build the plan once before the time loop with
    :meth:`build_send` or :meth:`build_recv`, then call :meth:`send` /
    :meth:`recv` every step.

    send-side fields
    ----------------
    _sends : list of (r0, r1, c0, c1, dest, tag)
        Local slice [r0:r1, c0:c1] of the parent's subdomain array to send.
        For W/E strips the column range is a single cell; for S/N it is the
        row range.  All slices are in *local* (0-based) coordinates.
    _send_bufs : pre-allocated contiguous send buffers (one per entry)

    recv-side fields
    ----------------
    _recv_srcs, _recv_tags : MPI source ranks and tags (one per entry)
    _recv_bufs  : pre-allocated raw receive buffers — MPI writes here
    _idx_arrs   : pre-computed repeat index arrays (nearest-neighbour weights);
                  idx[i] = i // ratio, cached once to avoid np.repeat per step
    _upscaled_bufs : pre-allocated 1-D buffers for the expanded values
    _out_views  : pre-computed views into _w_up/_e_up/_s_up/_n_up slices
    _col_bcast  : True → W/E (broadcast upscaled column-wise); False → S/N
    _w_up, _e_up, _s_up, _n_up : pre-allocated BC output arrays
    """

    __slots__ = (
        # single-variable fields (send side)
        '_sends', '_send_bufs', '_send_dirs',
        # single-variable fields (recv side)
        '_recv_srcs', '_recv_tags', '_recv_bufs',
        '_idx_arrs', '_upscaled_bufs', '_out_views', '_col_bcast', '_recv_dirs',
        '_w_up', '_e_up', '_s_up', '_n_up',
        # multi-variable support
        '_n_vars',           # int — 1 for single-var, N for build_*_multi plans
        '_recv_ranges',      # list of (r0,r1) for W/E or (c0,c1) for S/N per recv entry
        '_out_views_mv',     # list (n_recv) of lists (n_vars) of output-array views
        '_upscaled_bufs_mv', # list (n_recv) of lists (n_vars) of upscaled-value buffers
        '_mv_w_ups', '_mv_e_ups', '_mv_s_ups', '_mv_n_ups',  # lists of n_vars BC arrays
    )

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._sends     : list = []
        self._send_bufs : list = []
        self._send_dirs : list = []
        self._recv_srcs     : list = []
        self._recv_tags     : list = []
        self._recv_bufs     : list = []
        self._idx_arrs      : list = []
        self._upscaled_bufs : list = []
        self._out_views     : list = []
        self._col_bcast     : list = []
        self._recv_dirs     : list = []
        self._w_up = self._e_up = self._s_up = self._n_up = None
        # multi-variable
        self._n_vars             = 1
        self._recv_ranges   : list = []
        self._out_views_mv  : list = []
        self._upscaled_bufs_mv : list = []
        self._mv_w_ups = self._mv_e_ups = self._mv_s_ups = self._mv_n_ups = None

    @classmethod
    def build_send(
        cls,
        jsl: slice, isl: slice,
        ci0: int, ci1: int, cj0: int, cj1: int,
        p_nx: int, p_ny: int,
        halo: int, ratio: int,
        bc_w: int, bc_e: int, bc_s: int, bc_n: int,
        c_ny: int, c_nx: int,
        c_nrows: int, c_ncols: int,
        c_offset: int,
        tag_w: int, tag_e: int, tag_s: int, tag_n: int,
        grid: int = TGRID,
    ) -> "StripExchangePlan":
        """Build the send plan for a parent rank.

        Parameters
        ----------
        jsl, isl            : this rank's row / col slices in the parent domain
        ci0,ci1,cj0,cj1     : child region in parent cell indices
        p_nx, p_ny          : parent domain dimensions
        halo                : halo width in parent cells
        ratio               : spatial refinement ratio (child_dx = parent_dx / ratio)
        bc_w,bc_e,bc_s,bc_n : child BC widths (from bc_widths())
        c_ny, c_nx          : child domain dimensions
        c_nrows, c_ncols    : child domain tiling
        c_offset            : global rank of child's local rank 0
        tag_*               : MPI tags per direction
        """
        plan   = cls()
        halo_w = ci0 - max(ci0 - halo, 0)
        halo_e = min(ci1 + halo, p_nx) - ci1
        halo_s = cj0 - max(cj0 - halo, 0)
        halo_n = min(cj1 + halo, p_ny) - cj1

        def _add_we(r0_l, r1_l, c0_l, c1_l, tag, cr0, cr1, col_bc, east_flag, direction):
            for c_lr in range(c_nrows * c_ncols):
                jsl_c, isl_c = domain_slice(c_lr, c_nrows, c_ncols, c_ny, c_nx)
                if east_flag:
                    if isl_c.stop  <= col_bc: continue
                else:
                    if isl_c.start >= col_bc: continue
                if jsl_c.start >= cr1 or jsl_c.stop <= cr0: continue
                n = (r1_l - r0_l) * (c1_l - c0_l)
                plan._sends.append((r0_l, r1_l, c0_l, c1_l, c_offset + c_lr, tag))
                plan._send_bufs.append(np.empty(n, dtype=np.float64))
                plan._send_dirs.append(direction)

        def _add_sn(r0_l, r1_l, c0_l, c1_l, tag, cc0, cc1, bc_rows, north_flag, direction):
            for c_lr in range(c_nrows * c_ncols):
                jsl_c, isl_c = domain_slice(c_lr, c_nrows, c_ncols, c_ny, c_nx)
                if north_flag:
                    if jsl_c.stop  <= bc_rows: continue
                else:
                    if jsl_c.start >= bc_rows: continue
                if isl_c.start >= cc1 or isl_c.stop <= cc0: continue
                n = (r1_l - r0_l) * (c1_l - c0_l)
                plan._sends.append((r0_l, r1_l, c0_l, c1_l, c_offset + c_lr, tag))
                plan._send_bufs.append(np.empty(n, dtype=np.float64))
                plan._send_dirs.append(direction)

        # Innermost parent cell for each BC side — adjusted for Arakawa C-NE stagger.
        # TGRID: cell centres, no offset.
        # UGRID: east-face (x+½) — W inner shifts one step east; E one step west.
        # VGRID: north-face (y+½) — S inner shifts one step north; N one step south.
        w_col = ci0     if grid == UGRID else ci0 - 1
        e_col = ci1 - 1 if grid == UGRID else ci1
        s_row = cj0     if grid == VGRID else cj0 - 1
        n_row = cj1 - 1 if grid == VGRID else cj1

        # W
        if halo_w > 0 and bc_w > 0 and isl.start <= w_col < isl.stop:
            r0 = max(jsl.start, cj0);  r1 = min(jsl.stop, cj1)
            if r0 < r1:
                lc = w_col - isl.start
                _add_we(r0 - jsl.start, r1 - jsl.start, lc, lc + 1, tag_w,
                        (r0 - cj0) * ratio, (r1 - cj0) * ratio, bc_w, False, 'W')

        # E
        if halo_e > 0 and bc_e > 0 and isl.start <= e_col < isl.stop:
            r0 = max(jsl.start, cj0);  r1 = min(jsl.stop, cj1)
            if r0 < r1:
                lc = e_col - isl.start
                _add_we(r0 - jsl.start, r1 - jsl.start, lc, lc + 1, tag_e,
                        (r0 - cj0) * ratio, (r1 - cj0) * ratio,
                        c_nx - bc_e, True, 'E')

        # S
        if halo_s > 0 and bc_s > 0 and jsl.start <= s_row < jsl.stop:
            c0 = max(isl.start, ci0);  c1 = min(isl.stop, ci1)
            if c0 < c1:
                lr = s_row - jsl.start
                _add_sn(lr, lr + 1, c0 - isl.start, c1 - isl.start, tag_s,
                        (c0 - ci0) * ratio, (c1 - ci0) * ratio, bc_s, False, 'S')

        # N
        if halo_n > 0 and bc_n > 0 and jsl.start <= n_row < jsl.stop:
            c0 = max(isl.start, ci0);  c1 = min(isl.stop, ci1)
            if c0 < c1:
                lr = n_row - jsl.start
                _add_sn(lr, lr + 1, c0 - isl.start, c1 - isl.start, tag_n,
                        (c0 - ci0) * ratio, (c1 - ci0) * ratio,
                        c_ny - bc_n, True, 'N')

        return plan

    @classmethod
    def build_recv(
        cls,
        jsl: slice, isl: slice,
        ci0: int, ci1: int, cj0: int, cj1: int,
        p_nx: int, p_ny: int,
        halo: int, ratio: int,
        bc_w: int, bc_e: int, bc_s: int, bc_n: int,
        c_ny: int, c_nx: int,
        p_nrows: int, p_ncols: int,
        p_offset: int,
        tag_w: int, tag_e: int, tag_s: int, tag_n: int,
        grid: int = TGRID,
    ) -> "StripExchangePlan":
        """Build the recv plan for a child rank.

        Returns a plan whose :meth:`recv` method returns (w_up, e_up, s_up, n_up).
        All buffers, repeat-index arrays and output-array views are pre-allocated
        once here so :meth:`recv` performs no per-step allocations.

        Parameters
        ----------
        jsl, isl            : this rank's row / col slices in the *child* domain
        ci0,ci1,cj0,cj1     : child region in parent cell indices
        p_nx, p_ny          : parent domain dimensions
        halo                : halo width in parent cells
        ratio               : spatial refinement ratio
        bc_w,bc_e,bc_s,bc_n : child BC widths (from bc_widths())
        c_ny, c_nx          : child domain dimensions
        p_nrows, p_ncols    : parent domain tiling
        p_offset            : global rank of parent's local rank 0
        tag_*               : MPI tags per direction
        grid                : TGRID (default), UGRID, or VGRID — selects which
                              parent cell is the innermost source for each BC side
        """
        plan = cls()
        plan._w_up = np.zeros((c_ny, bc_w), dtype=np.float64)
        plan._e_up = np.zeros((c_ny, bc_e), dtype=np.float64)
        plan._s_up = np.zeros((bc_s, c_nx), dtype=np.float64)
        plan._n_up = np.zeros((bc_n, c_nx), dtype=np.float64)

        halo_w  = ci0 - max(ci0 - halo, 0)
        halo_e  = min(ci1 + halo, p_nx) - ci1
        halo_s  = cj0 - max(cj0 - halo, 0)
        halo_n  = min(cj1 + halo, p_ny) - cj1
        east_c  = c_nx - bc_e
        north_c = c_ny - bc_n

        def _register(src, tag, n_parent, out_view, col_bcast, direction, rng):
            plan._recv_srcs.append(src)
            plan._recv_tags.append(tag)
            plan._recv_bufs.append(np.empty(n_parent, dtype=np.float64))
            plan._idx_arrs.append(np.repeat(np.arange(n_parent), ratio))
            plan._upscaled_bufs.append(np.empty(n_parent * ratio, dtype=np.float64))
            plan._out_views.append(out_view)
            plan._col_bcast.append(col_bcast)
            plan._recv_dirs.append(direction)
            plan._recv_ranges.append(rng)   # (r0,r1) for W/E; (c0,c1) for S/N

        def _add_we(inner_col, tag, out_arr, col_bcast, direction):
            for p_lr in range(p_nrows * p_ncols):
                jsl_p, isl_p = domain_slice(p_lr, p_nrows, p_ncols, p_ny, p_nx)
                if not (isl_p.start <= inner_col < isl_p.stop): continue
                r0 = max(jsl_p.start, cj0);  r1 = min(jsl_p.stop, cj1)
                if r0 >= r1: continue
                cr0 = (r0 - cj0) * ratio;  cr1 = (r1 - cj0) * ratio
                if jsl.start >= cr1 or jsl.stop <= cr0: continue
                _register(p_offset + p_lr, tag, r1 - r0,
                          out_arr[cr0:cr1, :], col_bcast, direction, (cr0, cr1))

        def _add_sn(inner_row, tag, out_arr, col_bcast, direction):
            for p_lr in range(p_nrows * p_ncols):
                jsl_p, isl_p = domain_slice(p_lr, p_nrows, p_ncols, p_ny, p_nx)
                if not (jsl_p.start <= inner_row < jsl_p.stop): continue
                c0 = max(isl_p.start, ci0);  c1 = min(isl_p.stop, ci1)
                if c0 >= c1: continue
                cc0 = (c0 - ci0) * ratio;  cc1 = (c1 - ci0) * ratio
                if isl.start >= cc1 or isl.stop <= cc0: continue
                _register(p_offset + p_lr, tag, c1 - c0,
                          out_arr[:, cc0:cc1], col_bcast, direction, (cc0, cc1))

        # Innermost parent cell per BC side — same stagger logic as build_send.
        w_col = ci0     if grid == UGRID else ci0 - 1
        e_col = ci1 - 1 if grid == UGRID else ci1
        s_row = cj0     if grid == VGRID else cj0 - 1
        n_row = cj1 - 1 if grid == VGRID else cj1

        if halo_w > 0 and bc_w > 0 and isl.start < bc_w:
            _add_we(w_col, tag_w, plan._w_up, True,  'W')
        if halo_e > 0 and bc_e > 0 and isl.stop > east_c:
            _add_we(e_col, tag_e, plan._e_up, True,  'E')
        if halo_s > 0 and bc_s > 0 and jsl.start < bc_s:
            _add_sn(s_row, tag_s, plan._s_up, False, 'S')
        if halo_n > 0 and bc_n > 0 and jsl.stop > north_c:
            _add_sn(n_row, tag_n, plan._n_up, False, 'N')

        return plan

    # ── multi-variable classmethods ────────────────────────────────────────────

    @classmethod
    def build_send_multi(
        cls,
        jsl: slice, isl: slice,
        ci0: int, ci1: int, cj0: int, cj1: int,
        p_nx: int, p_ny: int,
        halo: int, ratio: int,
        bc_w: int, bc_e: int, bc_s: int, bc_n: int,
        c_ny: int, c_nx: int,
        c_nrows: int, c_ncols: int,
        c_offset: int,
        tag_w: int, tag_e: int, tag_s: int, tag_n: int,
        n_vars: int,
        grid: int = TGRID,
    ) -> "StripExchangePlan":
        """Like build_send but packs *n_vars* variables into each message.

        All n_vars variable strips are concatenated in the send buffer so that
        one ``Isend`` per direction per child rank covers all variables.  Use
        :meth:`send_multi` instead of :meth:`send`.

        When all packed variables live on the same grid, pass *grid* once.
        For mixed-grid packed exchange (e.g. T+U+V in one message) use
        sequential per-grid plans via :meth:`build_send` instead, since each
        grid requires a different innermost-cell offset.

        Parameters
        ----------
        n_vars : int — number of variables to pack into each message
        grid   : TGRID (default), UGRID, or VGRID
        (all other parameters identical to build_send)
        """
        plan = cls.build_send(
            jsl, isl, ci0, ci1, cj0, cj1, p_nx, p_ny, halo, ratio,
            bc_w, bc_e, bc_s, bc_n, c_ny, c_nx, c_nrows, c_ncols, c_offset,
            tag_w, tag_e, tag_s, tag_n, grid,
        )
        plan._n_vars = n_vars
        plan._send_bufs = [np.empty(len(b) * n_vars, dtype=np.float64)
                           for b in plan._send_bufs]
        return plan

    @classmethod
    def build_recv_multi(
        cls,
        jsl: slice, isl: slice,
        ci0: int, ci1: int, cj0: int, cj1: int,
        p_nx: int, p_ny: int,
        halo: int, ratio: int,
        bc_w: int, bc_e: int, bc_s: int, bc_n: int,
        c_ny: int, c_nx: int,
        p_nrows: int, p_ncols: int,
        p_offset: int,
        tag_w: int, tag_e: int, tag_s: int, tag_n: int,
        n_vars: int,
        grid: int = TGRID,
    ) -> "StripExchangePlan":
        """Like build_recv but unpacks *n_vars* variables from each message.

        Returns a plan whose :meth:`recv_multi` returns a list of n_vars
        ``(w_up, e_up, s_up, n_up)`` tuples.  All buffers are pre-allocated;
        no per-step allocations occur.

        Parameters
        ----------
        n_vars : int — number of variables expected in each packed message
        grid   : TGRID (default), UGRID, or VGRID
        (all other parameters identical to build_recv)
        """
        plan = cls.build_recv(
            jsl, isl, ci0, ci1, cj0, cj1, p_nx, p_ny, halo, ratio,
            bc_w, bc_e, bc_s, bc_n, c_ny, c_nx, p_nrows, p_ncols, p_offset,
            tag_w, tag_e, tag_s, tag_n, grid,
        )
        plan._n_vars = n_vars

        # Resize recv buffers to hold all n_vars per parent strip
        plan._recv_bufs = [np.empty(len(b) * n_vars, dtype=np.float64)
                           for b in plan._recv_bufs]

        # Per-variable BC output arrays (same shape as single-var arrays)
        plan._mv_w_ups = [np.zeros_like(plan._w_up) for _ in range(n_vars)]
        plan._mv_e_ups = [np.zeros_like(plan._e_up) for _ in range(n_vars)]
        plan._mv_s_ups = [np.zeros_like(plan._s_up) for _ in range(n_vars)]
        plan._mv_n_ups = [np.zeros_like(plan._n_up) for _ in range(n_vars)]

        # Pre-allocate per-variable upscaled buffers and output-array views
        plan._upscaled_bufs_mv = []
        plan._out_views_mv = []
        for direction, rng, single_up in zip(
            plan._recv_dirs, plan._recv_ranges, plan._upscaled_bufs
        ):
            n_up = len(single_up)
            plan._upscaled_bufs_mv.append(
                [np.empty(n_up, dtype=np.float64) for _ in range(n_vars)]
            )
            var_views = []
            for v in range(n_vars):
                if direction in ('W', 'E'):
                    arr = plan._mv_w_ups[v] if direction == 'W' else plan._mv_e_ups[v]
                    var_views.append(arr[rng[0]:rng[1], :])
                else:
                    arr = plan._mv_s_ups[v] if direction == 'S' else plan._mv_n_ups[v]
                    var_views.append(arr[:, rng[0]:rng[1]])
            plan._out_views_mv.append(var_views)

        return plan

    # ── diagnostics ───────────────────────────────────────────────────────────

    def describe(self, rank: "int | None" = None) -> str:
        """Return a human-readable summary of this plan.

        Parameters
        ----------
        rank : int or None
            MPI rank to include in the prefix, e.g. comm.Get_rank().
        """
        pfx   = f"[rank {rank}] " if rank is not None else ""
        lines = []

        if self._sends:
            n = len(self._sends)
            lines.append(f"{pfx}StripExchangePlan SEND  {n} entr{'y' if n == 1 else 'ies'}")
            for (r0, r1, c0, c1, dest, tag), d in zip(self._sends, self._send_dirs):
                lines.append(f"{pfx}  {d}  local[{r0}:{r1}, {c0}:{c1}]"
                             f"  → rank {dest}  tag={tag}  n={(r1-r0)*(c1-c0)}")

        if self._recv_bufs:
            n = len(self._recv_bufs)
            lines.append(f"{pfx}StripExchangePlan RECV  {n} entr{'y' if n == 1 else 'ies'}")
            for src, tag, buf, up, ov, d in zip(
                self._recv_srcs, self._recv_tags, self._recv_bufs,
                self._upscaled_bufs, self._out_views, self._recv_dirs,
            ):
                lines.append(f"{pfx}  {d}  from rank {src}  tag={tag}"
                             f"  n_parent={len(buf)}  upscaled={len(up)}"
                             f"  out_view.shape={ov.shape}")

        if not self._sends and not self._recv_bufs:
            lines.append(
                f"{pfx}StripExchangePlan  (empty — no boundary overlap on this rank)"
            )

        return "\n".join(lines)

    # ── execution ─────────────────────────────────────────────────────────────

    def send(self, comm: "MPI.Comm", local_data: np.ndarray) -> None:
        """Send boundary strips to child ranks using the precomputed plan.

        Uses non-blocking Isend + Waitall so all sends are posted before any
        blocking occurs.
        """
        reqs = []
        for (r0, r1, c0, c1, dest, tag), sbuf in zip(self._sends, self._send_bufs):
            np.copyto(sbuf, local_data[r0:r1, c0:c1].flatten())
            reqs.append(comm.Isend(sbuf, dest=dest, tag=tag))
        MPI.Request.Waitall(reqs)

    def recv(
        self, comm: "MPI.Comm",
    ) -> "tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]":
        """Receive boundary strips from parent ranks and upscale.

        Returns (w_up, e_up, s_up, n_up) — the same pre-allocated arrays are
        returned every call (zeroed then refilled).  No per-step allocations
        occur: receive buffers, index arrays, upscaled buffers and output-array
        views were all computed once in build_recv().
        """
        w_up = self._w_up;  w_up[:] = 0   # type: ignore[index]
        e_up = self._e_up;  e_up[:] = 0   # type: ignore[index]
        s_up = self._s_up;  s_up[:] = 0   # type: ignore[index]
        n_up = self._n_up;  n_up[:] = 0   # type: ignore[index]

        reqs = [comm.Irecv(buf, source=src, tag=tag)
                for src, tag, buf in zip(self._recv_srcs, self._recv_tags, self._recv_bufs)]
        MPI.Request.Waitall(reqs)

        for buf, idx, upscaled, out_view, col_bcast in zip(
            self._recv_bufs, self._idx_arrs, self._upscaled_bufs,
            self._out_views, self._col_bcast,
        ):
            np.take(buf, idx, out=upscaled)
            out_view[:] = upscaled[:, np.newaxis] if col_bcast else upscaled[np.newaxis, :]

        return w_up, e_up, s_up, n_up

    def send_multi(self, comm: "MPI.Comm", data_list: list) -> None:
        """Send *n_vars* variable strips packed into one message per direction.

        Variables are concatenated in *data_list* order within each send buffer:
        ``buf[0 : n]`` = var 0 strip, ``buf[n : 2n]`` = var 1 strip, etc.
        Build this plan with :meth:`build_send_multi`.

        Parameters
        ----------
        comm      : MPI communicator
        data_list : list of n_vars 2-D arrays shaped like the local subdomain
        """
        reqs = []
        for (r0, r1, c0, c1, dest, tag), sbuf in zip(self._sends, self._send_bufs):
            n = (r1 - r0) * (c1 - c0)
            for v, data in enumerate(data_list):
                np.copyto(sbuf[v * n: (v + 1) * n], data[r0:r1, c0:c1].flatten())
            reqs.append(comm.Isend(sbuf, dest=dest, tag=tag))
        MPI.Request.Waitall(reqs)

    def recv_multi(
        self, comm: "MPI.Comm",
    ) -> "list[tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]":
        """Receive and upscale *n_vars* variable strips from parent ranks.

        Returns a list of n_vars ``(w_up, e_up, s_up, n_up)`` tuples, one per
        variable, in the same order as *data_list* passed to :meth:`send_multi`.
        Pre-allocated buffers are reused; no per-step allocations occur.
        Build this plan with :meth:`build_recv_multi`.
        """
        for v in range(self._n_vars):
            self._mv_w_ups[v][:] = 0   # type: ignore[index,union-attr]
            self._mv_e_ups[v][:] = 0   # type: ignore[index,union-attr]
            self._mv_s_ups[v][:] = 0   # type: ignore[index,union-attr]
            self._mv_n_ups[v][:] = 0   # type: ignore[index,union-attr]

        reqs = [comm.Irecv(buf, source=src, tag=tag)
                for src, tag, buf in zip(
                    self._recv_srcs, self._recv_tags, self._recv_bufs)]
        MPI.Request.Waitall(reqs)

        for i, (buf, idx, col_bcast) in enumerate(
            zip(self._recv_bufs, self._idx_arrs, self._col_bcast)
        ):
            n_parent = len(buf) // self._n_vars
            var_upscaled = self._upscaled_bufs_mv[i]
            var_views    = self._out_views_mv[i]
            for v in range(self._n_vars):
                var_buf = buf[v * n_parent: (v + 1) * n_parent]
                np.take(var_buf, idx, out=var_upscaled[v])
                var_views[v][:] = (
                    var_upscaled[v][:, np.newaxis] if col_bcast
                    else var_upscaled[v][np.newaxis, :]
                )

        return [
            (self._mv_w_ups[v], self._mv_e_ups[v],
             self._mv_s_ups[v], self._mv_n_ups[v])
            for v in range(self._n_vars)
        ]


# ── Convenience one-shot wrappers ─────────────────────────────────────────────

def p2p_send_strips(
    comm: "MPI.Comm",
    local_data: np.ndarray,
    jsl: slice, isl: slice,
    ci0: int, ci1: int, cj0: int, cj1: int,
    p_nx: int, p_ny: int,
    halo: int, ratio: int,
    bc_w: int, bc_e: int, bc_s: int, bc_n: int,
    c_ny: int, c_nx: int,
    c_nrows: int, c_ncols: int,
    c_offset: int,
    tag_w: int, tag_e: int, tag_s: int, tag_n: int,
) -> None:
    """One-shot P2P send (builds a StripExchangePlan each call).

    Prefer building a StripExchangePlan once before the time loop and calling
    plan.send() for repeated use.
    """
    StripExchangePlan.build_send(
        jsl, isl, ci0, ci1, cj0, cj1, p_nx, p_ny, halo, ratio,
        bc_w, bc_e, bc_s, bc_n, c_ny, c_nx, c_nrows, c_ncols, c_offset,
        tag_w, tag_e, tag_s, tag_n,
    ).send(comm, local_data)


def p2p_recv_strips(
    comm: "MPI.Comm",
    jsl: slice, isl: slice,
    ci0: int, ci1: int, cj0: int, cj1: int,
    p_nx: int, p_ny: int,
    halo: int, ratio: int,
    bc_w: int, bc_e: int, bc_s: int, bc_n: int,
    c_ny: int, c_nx: int,
    p_nrows: int, p_ncols: int,
    p_offset: int,
    tag_w: int, tag_e: int, tag_s: int, tag_n: int,
) -> "tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]":
    """One-shot P2P recv (builds a StripExchangePlan each call).

    Prefer building a StripExchangePlan once before the time loop and calling
    plan.recv() for repeated use.
    """
    return StripExchangePlan.build_recv(
        jsl, isl, ci0, ci1, cj0, cj1, p_nx, p_ny, halo, ratio,
        bc_w, bc_e, bc_s, bc_n, c_ny, c_nx, p_nrows, p_ncols, p_offset,
        tag_w, tag_e, tag_s, tag_n,
    ).recv(comm)


# ── public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Arakawa C-grid constants and tag helper
    "TGRID", "UGRID", "VGRID",
    "make_tags",
    # Geometry
    "child_window",
    "bc_widths",
    # Decomposition
    "best_tiling",
    "domain_slice",
    # Masks & BC helpers
    "build_cell_mask",
    "upscale",
    "apply_bc_sub",
    # MPI collectives
    "assemble_global",
    "gather_strip",
    "gather_send_strips",
    "recv_bcast_strips",
    # P2P exchange (preferred)
    "StripExchangePlan",
    "p2p_send_strips",
    "p2p_recv_strips",
]
