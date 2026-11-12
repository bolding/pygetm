"""Two-way, multi-level nesting of pyGETM domains (branch ``nesting-mpmd`` of the fork bolding/pygetm).

Each domain of a nested setup runs as its own :class:`pygetm.Simulation` on its own
group of MPI processes. Domains exchange information through
:class:`pygetm.simulation.Coupler` objects registered with
:meth:`pygetm.simulation.BaseSimulation.add_coupler`.

This package currently provides:

* :func:`split_world`: division of the MPI processes over the domains;
  :func:`split_mpmd` for an MPMD launch (one ``mpiexec`` component per domain)
* :mod:`pygetm.nesting.geometry`: the hierarchy (:class:`Level`,
  :func:`load_levels`): windows, ratios, holes, rims, ring cells and the faces
  at every interface; :meth:`Level.external` for
  :attr:`pygetm.domain.Domain.external`
* :mod:`pygetm.nesting.masks`: child-led land/sea consistency between levels
* :mod:`pygetm.nesting.operators`: conservative transfer operators (limited
  linear prolongation, weighted restriction, face sums for flux matching)
* :mod:`pygetm.nesting.sponge`: the AGRIF-like sponge of a child next to its
  interface (``Nested2D(sponge_width=, sponge_alpha=)``)
* :mod:`pygetm.nesting.model`: :class:`NestedModel`, one model of a run
  launched with one ``mpiexec`` component per level (MPMD)
* :mod:`pygetm.nesting.plan`: the M:N exchange lists between a parent and a
  child on their pyGETM tilings (:class:`LevelTiling`, :class:`Plan`)
* :mod:`pygetm.nesting.runtime`: :class:`NestingRun` (processes, grids and
  plans of a run), the abort hook and the couplers' start-up check
* :mod:`pygetm.nesting.nested2d`, :mod:`pygetm.nesting.nested3d`: the couplers
  (:class:`Nested2D`: elevation and transports; :class:`Nested3D`: tracers and
  3D transports, with :mod:`pygetm.nesting.vremap` between differing layers)

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

from .hierarchy import NestingContext, split_world, split_mpmd, mpmd_blocks
from .geometry import Level, active_region, load_levels
from .plan import LevelTiling, Plan
from .model import NestedModel

__all__ = [
    "NestingContext",
    "split_world",
    "split_mpmd",
    "NestedModel",
    "mpmd_blocks",
    "Level",
    "load_levels",
    "LevelTiling",
    "Plan",
]
