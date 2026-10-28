"""One model of a nesting run launched with one ``mpiexec`` component per
level (MPMD)::

    mpiexec -n 4 python root/run.py --nesting layout.yaml \\
          : -n 2 python child/run.py --nesting layout.yaml

Every model is an ordinary pyGETM setup with its own run script. It joins the
nesting through :class:`NestedModel`:

1. ``NestedModel(layout, name)``: the processes of this component become the
   level ``name`` (:func:`pygetm.nesting.split_mpmd`; the processes per level
   are the component sizes). :attr:`NestedModel.level` gives the geometry
   (grid edges, band) for building the level's bathymetry.
2. :meth:`NestedModel.share_depth`: every level contributes its depth once;
   the active regions and the child-led land/sea consistency
   (:func:`pygetm.nesting.masks.make_consistent`) are computed from all
   levels, identically on every process. Returns this level's depth.
3. :meth:`NestedModel.timestep`: the common time step (root step / the
   level's steps per root step) from every level's own limit.
4. :attr:`NestedModel.run`: the :class:`pygetm.nesting.runtime.NestingRun`
   (``build_simulation``, then e.g.
   :class:`pygetm.nesting.barotropic.NestedBarotropic`).

Steps 2 and 3 are collective over all processes of all components.
"""

from typing import Any, Dict, Mapping, Optional

import numpy as np
from mpi4py import MPI

from .geometry import Level, load_levels
from .hierarchy import split_mpmd
from .runtime import NestingRun, install_abort_hook
from . import masks


def _load_config(layout) -> Dict[str, Any]:
    if isinstance(layout, Mapping):
        return dict(layout)
    import yaml

    with open(layout) as f:
        return yaml.safe_load(f)


class NestedModel:
    """This process's model in an MPMD nesting run (see module docs)."""

    def __init__(self, layout, name: str, comm: Optional[MPI.Comm] = None, halo: int = 2,
                 root_band: Optional[int] = 0, abort_on_error: bool = True, interface_comms: bool = True):
        self.config = _load_config(layout)
        if name not in self.config:
            raise ValueError(f"level {name!r} is not in the layout ({[k for k in self.config if k != 'config']})")
        self.halo, self.root_band = halo, root_band
        self.interface_comms = interface_comms
        self.context = split_mpmd(name, comm)
        self.name = name
        #: communicator of this model (pass it to the pyGETM domain)
        self.comm = self.context.comm
        #: shared nesting communicator (setup, abort)
        self.world = self.context.nesting_comm
        if abort_on_error:
            install_abort_hook(self.world, name)
        specs = {k: v for k, v in self.config.items() if k != "config"}
        absent = [k for k in specs if k not in self.context.sizes]
        if absent:
            raise ValueError(f"level(s) {absent} of the layout have no mpiexec component")
        # geometry only (active regions need the wet masks of share_depth)
        self.levels = load_levels(self.config, halo=halo, root_band=root_band, active=False)
        self._run: Optional[NestingRun] = None
        self.changes: Dict[str, dict] = {}
        self.violations: Dict[str, dict] = {}

    @property
    def level(self) -> Level:
        """This model's level (after :meth:`share_depth`: with its active region)."""
        return next(l for l in self.levels if l.name == self.name)

    def share_depth(self, H: np.ndarray, active_wet: Optional[np.ndarray] = None) -> np.ndarray:
        """Collective. ``H``: this level's depth on its extended grid (window
        plus band; NaN or masked = land), the same on all processes of the
        level. ``active_wet``: the wet mask the active regions are computed
        from (default: finite ``H``). Returns this level's depth after the
        child-led consistency (NaN = land); :attr:`changes` and
        :attr:`violations` per level."""
        H = np.asarray(np.ma.filled(np.ma.asarray(H, dtype=float), np.nan))
        wet = np.isfinite(H)
        aw = wet if active_wet is None else np.asarray(active_wet, dtype=bool)
        mine = (self.name, H, wet, aw) if self.comm.rank == 0 else None
        shared = [x for x in self.world.allgather(mine) if x is not None]
        Hs = {n: h for n, h, _, _ in shared}
        wets = {n: w for n, _, w, _ in shared}
        actives = {n: a for n, _, _, a in shared}
        self.levels = load_levels(self.config, halo=self.halo, root_band=self.root_band, wet=actives)
        wets, Hs, self.changes = masks.make_consistent(self.levels, wets, Hs)
        self.violations = masks.check_consistent(self.levels, wets)
        self.wet = wets
        return np.where(wets[self.name], Hs[self.name], np.nan)

    def timestep(self, maxdt: float) -> float:
        """Collective. ``maxdt``: this level's largest stable time step
        (e.g. safety x ``Domain.cfl_check()``). Returns this level's time step:
        the root step (the smallest over levels of maxdt x steps per root
        step) divided by this level's steps per root step."""
        spr = self.level.steps_per_root
        dt_root = self.world.allreduce(float(maxdt) * spr, op=MPI.MIN)
        self.dt_root = dt_root
        return dt_root / spr

    @property
    def run(self) -> NestingRun:
        """The :class:`NestingRun` of this model (after :meth:`share_depth`)."""
        if self._run is None:
            self._run = NestingRun(self.levels, context=self.context, abort_on_error=False,
                                   interface_comms=self.interface_comms)
        return self._run
