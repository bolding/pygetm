"""Running a nesting hierarchy: processes, domains, exchanges.

:class:`NestingRun` places every level on its own group of MPI processes
(:func:`pygetm.nesting.split_world`), builds the level's pyGETM grids with the
ring band and the hole marked external (:attr:`pygetm.domain.Domain.external`),
shares the tilings of all levels and builds the exchange plans
(:class:`pygetm.nesting.plan.Plan`).
"""

from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
from mpi4py import MPI

from .geometry import Level
from .hierarchy import NestingContext, split_world
from .plan import LevelTiling, Plan, gather_tilings
from . import masks


def install_abort_hook(comm: MPI.Comm, label: str):
    """Make an unhandled exception on any process abort the whole run.

    The levels of a nesting run wait for each other in MPI: if one level
    stops with an exception, the others would wait forever. The hook prints
    the traceback with the level and rank, then calls ``comm.Abort``.
    ``sys.exit`` with a non-zero status or a message aborts as well (Python
    passes ``SystemExit`` to no hook; a bare ``raise SystemExit(...)`` is
    not caught: raise another exception in a nested run). Installed once
    per process."""
    import sys
    import traceback

    if getattr(sys.excepthook, "_nesting_abort", False):
        return
    previous = sys.excepthook
    previous_exit = sys.exit

    def exit_(status=None):
        if status not in (None, 0):
            if not isinstance(status, int):
                sys.stderr.write(f"{status}\n")
            sys.stderr.write(f"[nesting: {label}, rank {comm.rank}] exit with status {status!r} - aborting all levels\n")
            sys.stderr.flush()
            comm.Abort(status if isinstance(status, int) else 1)
        previous_exit(status)

    def hook(etype, value, tb):
        traceback.print_exception(etype, value, tb)
        sys.stderr.write(f"[nesting: {label}, rank {comm.rank}] unhandled exception - aborting all levels\n")
        sys.stderr.flush()
        try:
            comm.Abort(1)
        finally:
            previous(etype, value, tb)

    hook._nesting_abort = True
    sys.excepthook = hook
    sys.exit = exit_


def check_couplers(sim, coupler, world: MPI.Comm, level: str) -> None:
    """Collective, from the ``on_start`` of every nesting coupler, done by the first one of the simulation:
    every level must run the same nesting couplers in the same order and the same run type, else their
    start-up collectives would not match (a hang). Raises on every process otherwise."""
    nesting = [c for c in sim._couplers if type(c).__module__.startswith("pygetm.nesting")]
    if not nesting or nesting[0] is not coupler:
        return
    mine = (level, int(sim.runtype), tuple(type(c).__name__ for c in nesting))
    seen = {}
    for name, runtype, couplers in world.allgather(mine):
        seen.setdefault((runtype, couplers), set()).add(name)
    if len(seen) > 1:
        detail = "; ".join(f"{sorted(n)}: run type {rt}, couplers {list(c)}" for (rt, c), n in seen.items())
        raise RuntimeError(f"nesting: the levels differ in run type or nesting couplers (all need the same, "
                           f"in the same order): {detail}")

def sync_order(send_up: Callable[[], list], recv_children: Callable[[], None], send_down: Callable[[], list],
               recv_parent: Callable[[], None], *, up: bool, up_after_children: bool) -> list:
    """One sync point of a coupler, in an order without cycles over the hierarchy: (a) to the parent (what goes
    up does not depend on the children), (b) from the children, (c) to the children (this level's corrected
    state), (d) from the parent (what goes down does not depend on the parent). The sweeps up and down do not
    wait for each other. With ``up_after_children`` (:attr:`Level.sends_up_after_children`) (a) comes after
    (b). ``up``: this level exchanges with its parent at this sync. Returns the requests of the sends."""
    reqs = []
    if up and not up_after_children:
        reqs += send_up()
    recv_children()
    if up and up_after_children:
        reqs += send_up()
    reqs += send_down()
    if up:
        recv_parent()
    return reqs

class NestingRun:
    """Processes, grids and exchange plans of a nesting hierarchy (one level
    per group of processes). With ``abort_on_error`` (default), an unhandled
    exception on any process aborts all levels (:func:`install_abort_hook`)."""

    def __init__(self, levels: List[Level], ranks: Optional[Mapping[str, int]] = None, comm: Optional[MPI.Comm] = None,
                 abort_on_error: bool = True,
                 context: Optional[NestingContext] = None):
        self.levels = levels
        if context is None:
            ctx = split_world({l.name: int(ranks[l.name]) for l in levels}, comm=comm)
        else:  # e.g. from split_mpmd (one mpiexec component per level)
            ctx = context
            missing = [l.name for l in levels if l.name not in ctx.sizes]
            if missing:
                raise ValueError(f"no processes for level(s) {missing}: {dict(ctx.sizes)}")
        self.world = ctx.nesting_comm
        self.comm = ctx.comm
        self.offsets = ctx.offsets
        self.sizes = ctx.sizes
        self.level = next(l for l in levels if l.name == ctx.name)
        if abort_on_error:
            install_abort_hook(self.world, self.level.name)
        self.grid = None
        self.tilings: Dict[str, LevelTiling] = {}
        self.up: Optional[Plan] = None
        self.down: List[Plan] = []

    def build(self, make_domain: Callable, nz: int = 1, **grid_kwargs):
        """Build this level's grids. ``make_domain(level, comm)`` returns the
        pyGETM domain of the level including its band (window extended by
        ``level.band`` cells on every side). Returns the T grid (pyGETM's tiling
        gives every process a subdomain: it refuses more processes than
        subdomains with water)."""
        lev = self.level
        domain = make_domain(lev, self.comm)
        domain.external = lev.external()
        domain.tiling_mask = lev.tiling_mask()
        masks.check_dams(lev, domain)  # dams (pygetm.dams) off the interfaces
        masks.close_faces(lev, domain)  # interface faces with all child sub-faces closed
        grid_kwargs.setdefault("velocity_grids", 1)
        with self._tiling_hint():
            grid = domain.create_grids(nz, halox=lev.halo, haloy=lev.halo, **grid_kwargs)
        return self._setup(grid)

    def build_simulation(self, make_domain: Callable, make_simulation: Callable):
        """Build this level's :class:`pygetm.Simulation`. ``make_domain(level,
        comm)`` returns the level's domain including its band;
        ``make_simulation(level, domain)`` the simulation (its halo must equal
        ``level.halo``). The band and the hole are marked external before the
        simulation builds its grids; subdomains holding only hole cells that
        nothing reads are removed. Returns the simulation."""
        lev = self.level
        domain = make_domain(lev, self.comm)
        domain.external = lev.external()
        domain.tiling_mask = lev.tiling_mask()
        masks.check_dams(lev, domain)  # dams (pygetm.dams) off the interfaces
        masks.close_faces(lev, domain)  # interface faces with all child sub-faces closed
        self._assign_rivers(domain)
        with self._tiling_hint():
            sim = make_simulation(lev, domain)
        if sim.T is not None and sim.T.halox != lev.halo:
            raise ValueError(f"simulation halo {sim.T.halox} differs from nesting halo {lev.halo}")
        self._setup(sim.T)
        return sim

    def _assign_rivers(self, domain):
        """Collective. A river belongs to the finest level whose computed region holds it
        (land in a level's active region is that level's; a curvilinear child beyond its
        conforming band lies over its parent's land: the child wins); every level drops
        the rivers of the others. So a setup may give all its rivers to every level. A
        river no level holds would be lost: that raises."""
        lev = self.level
        rivers = list(domain.rivers.values())
        held = None
        if self.comm.rank == 0:
            computed = ~np.asarray(lev.external(), dtype=bool)  # domain-shaped: hole and band excluded
            held = {r.name: _river_in(domain, r, computed) for r in rivers}
        claims = self.world.allgather((lev.name, held))
        owners: Dict[str, list] = {}
        for name, h in claims:
            for river, inside in (h or {}).items():
                owners.setdefault(river, [])
                if inside:
                    owners[river].append(name)
        order = {l.name: k for k, l in enumerate(self.levels)}  # parents before children
        owner = {}
        for river, levs in owners.items():
            if not levs:
                raise ValueError(f"river {river}: no level computes its cell (its water would be lost)")
            owner[river] = max(levs, key=lambda n: (self.levels[order[n]].steps_per_root, order[n]))
        for r in rivers:
            if owner[r.name] != lev.name:
                domain.rivers.remove(r.name)
        self.river_owners = owner

    def _tiling_hint(self):
        """Context: a failing subdomain division names the level and its
        number of processes (then abort_on_error stops all levels)."""
        import contextlib

        run = self

        @contextlib.contextmanager
        def ctx():
            try:
                yield
            except Exception as e:
                if "subdomain" in str(e).lower() or "decompos" in str(e).lower():
                    raise RuntimeError(
                        f"level {run.level.name}: no subdomain division for {run.comm.size} processes"
                        f" ({run.level.nx} x {run.level.ny} cells) - use fewer processes for this level"
                        " (see nesting-balance in pygetm-nesting)") from e
                raise

        return ctx()

    def _setup(self, grid):
        lev = self.level
        self.grid = grid
        self.tilings = gather_tilings(self.levels, self.world, lev, self.grid, self.offsets)
        links = {c.name: k for k, c in enumerate(self.levels[1:])}
        til = self.tilings[lev.name]
        if lev.parent is not None:
            self.up = Plan(lev, self.tilings[lev.parent.name], til, links[lev.name])
        self.down = [Plan(c, til, self.tilings[c.name], links[c.name]) for c in lev.children]
        self._interface_comms()
        return self.grid

    def _interface_comms(self):
        """Give every plan of this process the communicator of its interface
        (``plan.comm``) and the ranks of the parent's and the child's tile 0
        in it (``plan.poff``, ``plan.coff``). Interfaces are created in the
        same (layout) order on all processes; each creation is collective
        over that interface's processes only."""
        lev = self.level
        group = self.world.Get_group()
        block = lambda name: list(range(self.offsets[name], self.offsets[name] + self.sizes[name]))
        for k, child in enumerate(self.levels[1:]):
            par = child.parent
            if lev is not par and lev is not child:
                continue
            g = group.Incl(block(par.name) + block(child.name))
            comm = self.world.Create_group(g, tag=k)
            g.Free()
            plan = self.up if lev is child else self.down[lev.children.index(child)]
            plan.comm, plan.poff, plan.coff = comm, 0, self.sizes[par.name]
        group.Free()

    @property
    def tile(self) -> int:
        """this process's subdomain in its level's tiling"""
        return self.grid.tiling.rank

    def describe(self) -> List[str]:
        """This process's nesting messages (format of check_print_plan.py)."""
        me, lev = self.world.rank, self.level
        til, t = self.tilings[lev.name], self.tile
        lines = [f"  [{lev.name} tile {t} / world {me}]  pyGETM tiling {til.map.shape}, nx_sub={til.nx_sub}"
                 f" ny_sub={til.ny_sub} offsets x={til.xoff} y={til.yoff}, removed land tiles {til.removed}"]
        if self.up is not None:
            lines += self.up.describe(me, "child", t)
        for p in self.down:
            lines += p.describe(me, "parent", t)
        return lines

    def print_plan(self):
        """Print the messages of all processes in rank order (collective)."""
        lines = self.describe()
        for rank in range(self.world.size):
            if rank == self.world.rank:
                print("\n".join(lines), flush=True)
            self.world.Barrier()

    def global_indices(self):
        """Global (j, i) of every row/column of this process's local arrays."""
        T, lev = self.grid, self.level
        y0, x0 = self.tilings[lev.name].origin(self.tile)
        h = T.halox
        return np.arange(T.ny_) - h + y0, np.arange(T.nx_) - h + x0


def ops_interp(s0, s1, a):
    return (1.0 - a) * s0 + a * s1


def _river_in(domain, river, computed: np.ndarray) -> bool:
    """Whether a river lies in a computed cell of this level's domain (root process).
    By index: that cell; by location: the cell whose centre is nearest, if the point
    lies within it (half a cell, with a margin)."""
    from ..constants import CoordinateType

    ny, nx = computed.shape
    if river.coordinate_type == CoordinateType.IJ:
        i, j = int(round(river.x)), int(round(river.y))
        return 0 <= i < nx and 0 <= j < ny and bool(computed[j, i])
    if river.coordinate_type == CoordinateType.LONLAT:
        xs, ys = domain._lon, domain._lat
    else:
        xs, ys = domain._x, domain._y
    if xs is None or ys is None:
        return False
    xc, yc = xs[1::2, 1::2], ys[1::2, 1::2]
    d2 = (xc - river.x) ** 2 + (yc - river.y) ** 2
    j, i = np.unravel_index(int(np.nanargmin(d2)), d2.shape)
    # the cell's extent from its corners
    xcor, ycor = xs[2 * j : 2 * j + 3 : 2, 2 * i : 2 * i + 3 : 2], ys[2 * j : 2 * j + 3 : 2, 2 * i : 2 * i + 3 : 2]
    half = 0.5 * max(np.ptp(xcor), np.ptp(ycor)) * 1.01
    reach = half * np.sqrt(2.0)
    if np.sqrt(d2[j, i]) > reach:
        return False  # outside this level's domain
    # a point on the edge between cells: any computed cell within reach
    return bool((computed & (d2 <= reach ** 2)).any())

