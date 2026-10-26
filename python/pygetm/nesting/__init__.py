"""Nesting of pyGETM domains (work in progress, `nesting` branch).

Each domain of a nested setup runs as its own :class:`pygetm.Simulation` on its own
group of MPI processes. Domains exchange information through
:class:`pygetm.simulation.Coupler` objects registered with
:meth:`pygetm.simulation.BaseSimulation.add_coupler`.

This package currently provides:

* :mod:`pygetm.nesting.exchange`: planning and execution of MPI exchanges of
  rectangular strips between independently decomposed parent and child domains
* :func:`split_world`: division of the MPI processes over the domains
"""

from .hierarchy import NestingContext, split_world

__all__ = ["NestingContext", "split_world"]
