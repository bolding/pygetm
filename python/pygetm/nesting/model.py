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
   :class:`pygetm.nesting.nested2d.Nested2D`).

Steps 2 and 3 are collective over all processes of all components.
"""

from typing import Any, Dict, Mapping, Optional

import warnings

import numpy as np
from mpi4py import MPI

from .geometry import Level, load_levels, read_layout
from .hierarchy import split_mpmd
from .runtime import NestingRun, install_abort_hook
from . import masks


def _load_config(layout) -> Dict[str, Any]:
    if isinstance(layout, Mapping):
        return dict(layout)
    return read_layout(layout)


class NestedModel:
    """This process's model in an MPMD nesting run (see module docs)."""

    def __init__(self, layout, name: str, comm: Optional[MPI.Comm] = None, halo: int = 2,
                 root_band: Optional[int] = 0, abort_on_error: bool = True):
        self.config = _load_config(layout)
        if name not in self.config:
            raise ValueError(f"level {name!r} is not in the layout ({[k for k in self.config if k != 'config']})")
        self.halo, self.root_band = halo, root_band
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
        #: per child level: least depth at its interface (masks.interface_depths)
        self.interface_depths: Dict[str, dict] = {}

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
        me = self.level
        H_dom = None
        if me.has_overhang:  # the whole domain was given: the nesting works on the view
            if H.shape != me.domain_shape:
                raise ValueError(f"{self.name}: depth has shape {H.shape}, expected the domain's {me.domain_shape}")
            H_dom, H = H, me.to_view(H).copy()
            if active_wet is not None:
                active_wet = me.to_view(np.asarray(active_wet))
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
        # curvilinear levels: the refined parent grid where water crosses the interface
        masks.check_conforming(self.levels, wets)
        # the layout keeps the interfaces out of water that floods and dries (masks.interface_depths)
        lims = {l.name: masks.interface_min_depth(self.config, l.name) for l in self.levels}
        self.interface_depths = masks.interface_depths(self.levels, wets, Hs, lims)
        for n, d in self.interface_depths.items():
            self.violations.setdefault(n, {})["interface_shallow"] = d["interface_shallow"]
        mine_iface = self.interface_depths.get(self.name)
        if mine_iface and mine_iface["interface_shallow"] and self.comm.rank == 0:
            warnings.warn(f"{self.name}: {mine_iface['interface_shallow']} interface cells shallower than "
                          f"interface_min_depth {mine_iface['interface_min_depth']} m (least: ring "
                          f"{mine_iface['min_depth_ring']:.2f} m, parent {mine_iface['min_depth_parent']:.2f} m): "
                          f"the interface may fall dry", stacklevel=2)
        self.wet = wets
        mine_H = np.where(wets[self.name], Hs[self.name], np.nan)
        if H_dom is not None:
            out = H_dom.copy()
            oj, oi = me.view_offset
            ny, nx = me.extended_shape
            # the view's ring and window from the consistency; its overhang-side band and
            # the overhang as given
            keep = me._overhang_band()
            out[oj : oj + ny, oi : oi + nx] = np.where(keep, out[oj : oj + ny, oi : oi + nx], mine_H)
            masks.check_overhang(me, np.isfinite(out), wets)
            return out
        return mine_H

    def timestep(self, maxdt: float, period: Optional[float] = 3600.0) -> float:
        """Collective. ``maxdt``: this level's largest stable time step
        (e.g. safety x ``Domain.cfl_check()``). Returns this level's time step:
        the root step divided by this level's steps per root step, the root step
        at most the smallest over levels of maxdt x steps per root step.

        ``period`` (default an hour): the root step is a simple one, ``period / n``
        for the smallest whole n that fits and gives every level a whole number
        of microseconds (:func:`simple_root_step`; e.g. 18 s with children at
        6 s), so that output at multiples of the period lands on steps and a
        restart starts at exactly the model time of the unbroken run (pyGETM
        advances its time by the step as a ``datetime.timedelta``, in
        microseconds, while its files hold steps x step). ``None``: the root step
        only rounded down to whole microseconds on all levels."""
        spr = self.level.steps_per_root
        dt_max = self.world.allreduce(float(maxdt) * spr, op=MPI.MIN)
        sprs = sorted(set(self.world.allgather(int(spr))))
        self.dt_root = simple_root_step(dt_max, sprs, period)
        return level_step(self.dt_root, spr)

    @property
    def run(self) -> NestingRun:
        """The :class:`NestingRun` of this model (after :meth:`share_depth`)."""
        if self._run is None:
            self._run = NestingRun(self.levels, context=self.context, abort_on_error=False)
        return self._run


def _lcm(values) -> int:
    import math

    out = 1
    for n in values:
        out = out * int(n) // math.gcd(out, int(n))
    return out


def simple_root_step(dt_max: float, steps_per_root, period: Optional[float] = 3600.0) -> float:
    """The root step for :meth:`NestedModel.timestep`: the largest ``period / n``
    (n whole) not above ``dt_max`` for which every level's step (root step / its
    steps per root step) is a whole number of microseconds; with ``period``
    None the largest whole number of microseconds that is a multiple of all
    steps per root step."""
    import math

    lcm = _lcm(steps_per_root)
    if period is None:
        us = int(math.floor(dt_max * 1e6 / lcm)) * lcm
        if us <= 0:
            raise ValueError(f"time step {dt_max} s too small for whole microseconds on all levels")
        return us / 1e6
    P = int(round(float(period) * 1e6))  # microseconds
    n = max(1, int(math.ceil(P / (dt_max * 1e6) - 1e-9)))
    while n <= P:
        if P % n == 0 and (P // n) % lcm == 0:
            return (P // n) / 1e6
        n += 1
    raise ValueError(f"no simple time step up to {dt_max} s for period {period} s and steps per root step {steps_per_root}")


def level_step(dt_root: float, steps_per_root: int) -> float:
    """A level's step from the root step (whole microseconds when the root step
    is a multiple of steps_per_root microseconds)."""
    us = int(round(dt_root * 1e6))
    return (us // int(steps_per_root)) / 1e6 if us % int(steps_per_root) == 0 else dt_root / steps_per_root

