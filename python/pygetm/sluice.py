"""Sluices: controlled connections between two water cells that are not open water
between them (a sluice gate, a lock, a culvert, a pumping station).

A sluice is a pair of rivers with opposite flows: water leaves cell ``a`` and
enters cell ``b`` (or the reverse), so the volume is conserved exactly. The flow
follows the water levels on both sides through the orifice law

    Q = mu * A * opening * sqrt(2 g |z_a - z_b|) * sign(z_a - z_b)      (a -> b positive)

with the gate opening (0..1) a number or a function of time and both levels, and
optionally only one direction (a flap gate). Q is limited so that one step does not
reverse the level difference.

A pipe or outlet through a dam (``invert``: the height of its outlet, m on the
model's datum): a level below the outlet counts as the outlet itself, so the head is
``max(z_a, invert) - max(z_b, invert)``. With the downstream level below the outlet
the pipe flows freely into air, driven by the upstream level above the outlet; with
it above, the outlet is submerged; with the upstream level below the outlet, nothing
flows. ``b=None``: nothing modelled downstream, the water leaves the model (a
withdrawal; with ``invert`` the free outflow of a reservoir).

Tracers go with the water: the receiving river carries the mean concentration of the
donor column over the layers the sluice draws from (its ``zl``/``zu``), while the
donor loses water at its own concentrations (pyGETM's rivers follow the cell for
outflow). Tracer content is then conserved exactly, except for the change a
biogeochemical model (FABM) makes in the donor column within one macro time step
(its sources are applied between the moment the concentration is taken and the
freshwater update).

The two cells may lie on different subdomains: their levels, areas and column means
are combined with one allreduce per step over the simulation's communicator.

    sluice = pygetm.sluice.Sluice(domain, "hvide_sande", a=(8.13, 56.00), b=(8.10, 56.00),
                                  area=50.0, mu=0.6, gate=1.0)
    sim = pygetm.Simulation(domain, ...)
    sluice.attach(sim)

``a_layers`` / ``b_layers``: keyword arguments for the rivers' vertical extent
(``zl``, ``zu``, ``vertical_position``): each end can take or give water below the
surface, as pyGETM's rivers do for reservoir outlets, e.g. a bottom outlet of a
reservoir, salt inflow near the bottom of a fjord, or a spillway at the surface.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np

from . import parallel
from .constants import GRAVITY
from .rivers import CoordinateType
from .simulation import Coupler

Gate = Union[float, Callable[..., float]]


class Sluice(Coupler):
    def __init__(
        self,
        domain,
        name: str,
        a: Tuple[float, float],
        b: Optional[Tuple[float, float]],
        area: float,
        mu: float = 0.6,
        gate: Gate = 1.0,
        direction: str = "both",
        coordinate_type: Optional[CoordinateType] = None,
        a_layers: Optional[dict] = None,
        b_layers: Optional[dict] = None,
        max_fraction: float = 0.5,
        invert: Optional[float] = None,
    ):
        """
        Args:
            domain: the domain, before the simulation is created (the rivers are added to it)
            name: name of the sluice (its rivers are ``<name>_a`` and ``<name>_b``)
            a, b: positions of the two cells (x, y in ``coordinate_type``)
            area: flow area of the open gate (m2)
            mu: discharge coefficient
            gate: opening 0..1, or a function ``gate(time, z_a, z_b) -> opening``
            direction: ``"both"``, ``"a_to_b"`` or ``"b_to_a"`` (a flap gate)
            a_layers, b_layers: vertical extent of the rivers (``zl``, ``zu``, ``vertical_position``)
            max_fraction: the flow of one step changes the level difference by at most this fraction
            invert: height of the outlet (m, model datum): levels below it count as it (a pipe in a dam)
        """
        if direction not in ("both", "a_to_b", "b_to_a"):
            raise ValueError(f"direction must be both, a_to_b or b_to_a, not {direction!r}")
        self.name = name
        self.area = float(area)
        self.mu = float(mu)
        self.gate = gate
        self.direction = direction
        self.max_fraction = float(max_fraction)
        kw = {} if coordinate_type is None else {"coordinate_type": coordinate_type}
        if b is None and invert is None:
            raise ValueError(f"sluice {name}: without a downstream cell (b=None) the outlet height invert is needed")
        self.invert = None if invert is None else float(invert)
        self._names = (f"{name}_a",) + (() if b is None else (f"{name}_b",))
        domain.rivers.add_by_location(self._names[0], *a, **kw, **(a_layers or {}))
        if b is not None:
            domain.rivers.add_by_location(self._names[1], *b, **kw, **(b_layers or {}))
        self.Q = 0.0  # m3/s, positive a -> b
        self.volume = 0.0  # m3 moved a -> b so far (with b=None: the water that left the model)
        self.content = {}  # per tracer: content moved a -> b so far (concentration x m3)
        self.opening = 0.0

    def attach(self, sim):
        sim.add_coupler(self)
        return self

    # --- helpers --------------------------------------------------------------------

    def _owned(self, river, grid) -> bool:
        return (grid.halox <= river.i < grid.halox + grid.nx) and (grid.haloy <= river.j < grid.haloy + grid.ny)

    def _locals(self, sim):
        """(index in sim.rivers, local river, owned) for both ends; None where not local."""
        names = list(sim.rivers)
        out = []
        for n in self._names:
            if n in sim.rivers:
                r = sim.rivers[n]
                out.append((names.index(n), r, self._owned(r, sim.T)))
            else:
                out.append(None)
        return out

    def _allreduce(self, sim, v: np.ndarray) -> np.ndarray:
        sim.tiling.comm.Allreduce(parallel.MPI.IN_PLACE, v, op=parallel.MPI.SUM)
        return v

    # --- coupler --------------------------------------------------------------------

    def on_start(self, sim):
        self._ends = self._locals(sim)
        v = np.zeros(2)
        for k, e in enumerate(self._ends):
            if e is not None and e[2]:
                v[k] = sim.T.area.all_values[e[1].j, e[1].i]
        self._cell_area = self._allreduce(sim, v)
        if len(self._names) == 1:
            self._cell_area[1] = np.inf  # nothing downstream
        if not (self._cell_area > 0).all():
            raise RuntimeError(f"sluice {self.name}: an end is not in a water cell")
        self._tracers = list(sim.tracers) if sim.runtype > 1 else []
        for e in self._ends:
            if e is not None:
                for t in self._tracers:
                    e[1][t.name].follow_target_cell = True
        self._update(sim)

    def after_state_update(self, sim, macro_active: bool):
        vol, content = self._pending  # the step just done moved this
        self.volume += vol
        for k, c in content.items():
            self.content[k] = self.content.get(k, 0.0) + c
        self._update(sim)

    def _update(self, sim):
        """Flow for the next step from the current levels; tracer values of both rivers."""
        nt = len(self._tracers)
        v = np.zeros(2 + 2 * nt)
        if nt:
            hall = sim.T.hn.all_values[sim.rivers.slice]
            act = sim.rivers.get_active_part_of_layers(hall)
        for k, e in enumerate(self._ends):
            if e is None or not e[2]:
                continue
            idx, r, _ = e
            v[k] = sim.T.z.all_values[r.j, r.i]
            if nt:
                w = act[:, idx]
                ws = w.sum()
                for m, t in enumerate(self._tracers):
                    c = t.all_values[:, r.j, r.i]
                    v[2 + 2 * m + k] = (c * w).sum() / ws if ws > 0 else c[-1]
        v = self._allreduce(sim, v)
        za, zb = v[0], v[1]
        if len(self._names) == 1:
            zb = -np.inf  # nothing downstream: free outflow above the outlet
        g = self.gate
        self.opening = float(g(sim.time, za, zb) if callable(g) else g)
        if self.invert is not None:
            dz = max(za, self.invert) - max(zb, self.invert)
        else:
            dz = za - zb
        Q = np.sign(dz) * self.mu * self.area * self.opening * np.sqrt(2.0 * GRAVITY * abs(dz))
        if (self.direction == "a_to_b" and Q < 0) or (self.direction == "b_to_a" and Q > 0):
            Q = 0.0
        # one step must not reverse the level difference
        # (to the outlet when the downstream level is below it, or nothing is downstream)
        qmax = self.max_fraction * abs(dz) / (sim.timestep * (1.0 / self._cell_area[0] + 1.0 / self._cell_area[1]))
        self.Q = float(np.clip(Q, -qmax, qmax))
        # what this flow moves over the next step; counted once that step is done
        donor = 0 if self.Q > 0 else 1
        self._pending = (self.Q * sim.timestep,
                         {t.name: self.Q * sim.timestep * v[2 + 2 * m + donor] for m, t in enumerate(self._tracers)})
        for k, e in enumerate(self._ends):
            if e is None:
                continue
            r = e[1]
            r.flow.all_values[...] = -self.Q if k == 0 else self.Q
            for m, t in enumerate(self._tracers):
                rt = r[t.name]
                rt.follow_target_cell = False
                rt.values[...] = v[2 + 2 * m + (1 - k)]  # the other end's column mean
