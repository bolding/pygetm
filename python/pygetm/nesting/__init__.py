"""Nesting of pyGETM domains (work in progress, `nesting` branch).

Each domain of a nested setup runs as its own :class:`pygetm.Simulation` on its own
group of MPI processes. Domains exchange information through
:class:`pygetm.simulation.Coupler` objects registered with
:meth:`pygetm.simulation.BaseSimulation.add_coupler`.

This package currently provides:

* :mod:`pygetm.nesting.exchange`: planning and execution of MPI exchanges of
  rectangular strips between independently decomposed parent and child domains
* :func:`split_world`: division of the MPI processes over the domains
* :mod:`pygetm.nesting.halo`: :class:`CellSet`, generic gather/scatter/pack of
  any 2D or 3D array at a fixed set of cells; :func:`outer_halo` (the child's
  ring outside its global domain) and :func:`hole_rim` (the parent's cells
  around the hole occupied by the child)
* :mod:`pygetm.nesting.variables`: which variables are exchanged, at which
  stage of the time step (:data:`pygetm.simulation.COUPLING_STAGES`), and why
* :mod:`pygetm.nesting.geometry`: the hierarchy (:class:`Level`,
  :func:`load_levels`): windows, ratios, holes, rims, ring cells and the faces
  at every interface; :meth:`Level.external` for
  :attr:`pygetm.domain.Domain.external`
* :mod:`pygetm.nesting.masks`: child-led land/sea consistency between levels
* :mod:`pygetm.nesting.operators`: conservative transfer operators (limited
  linear prolongation, weighted restriction, face sums for flux matching)
* :mod:`pygetm.nesting.plan`: the M:N exchange lists between a parent and a
  child on their pyGETM tilings (:class:`LevelTiling`, :class:`Plan`)

Two complementary exchange paradigms
------------------------------------
Every parent and child is an ordinary, individual pyGETM setup and uses pyGETM
directly, including its own tiling and halo exchange (:mod:`pygetm.parallel`)
on its own communicator. That halo exchange is used *within* a setup only.

This package adds the second paradigm, used *between* setups only: it fills,
conservatively, the halo zones between a parent and a child. A rank may have
zero to many partners in the other domain, with a different message size per
pair, on the separate nesting communicator.

The two paradigms do not share code. This package never calls pyGETM's halo
updaters, and pyGETM's halo code never calls this package. The nesting side
only reads a domain's :class:`pygetm.parallel.Tiling` (map, offsets,
subdomain size) to know which rank owns which cells. At a child's outer edge
pyGETM's halo exchange has no neighbour and does nothing, so the values this
package writes there are kept.
"""

from .hierarchy import NestingContext, split_world
from .halo import CellSet, outer_halo, hole_rim
from .variables import ExchangeGroup, EXCHANGE_GROUPS
from .geometry import Level, active_region, load_levels
from .plan import LevelTiling, Plan

__all__ = [
    "NestingContext",
    "split_world",
    "CellSet",
    "outer_halo",
    "hole_rim",
    "ExchangeGroup",
    "EXCHANGE_GROUPS",
    "Level",
    "load_levels",
    "LevelTiling",
    "Plan",
]
