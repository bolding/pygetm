"""Nesting of pyGETM domains (work in progress, `nesting` branch).

Each domain of a nested setup runs as its own :class:`pygetm.Simulation` on its own
group of MPI processes. Domains exchange information through
:class:`pygetm.simulation.Coupler` objects registered with
:meth:`pygetm.simulation.BaseSimulation.add_coupler`.

This package currently provides:

* :mod:`pygetm.nesting.exchange`: planning and execution of MPI exchanges of
  rectangular strips between independently decomposed parent and child domains
* :func:`split_world`: division of the MPI processes over the domains

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

__all__ = ["NestingContext", "split_world"]
