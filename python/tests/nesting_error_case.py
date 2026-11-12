"""Failing levels for test_nesting.TestNestingErrors: the whole nested job must stop, not wait for the failed level::

    mpiexec -n 2 python nesting_error_case.py WHAT root : -n 1 python nesting_error_case.py WHAT child

WHAT (what the child does; the root runs normally):

* ``raise``: raises an exception before the run (after NestedModel);
* ``badname``: asks NestedModel for a level that is not in the layout;
* ``exit``: calls ``sys.exit(3)`` in the middle of the run;
* ``no3d``: both levels 3D, the child without Nested3D (the levels' couplers differ).
"""

import datetime
import sys

import pygetm
from pygetm import parallel
from pygetm.nesting import NestedModel
from pygetm.nesting.nested2d import Nested2D
from pygetm.nesting.nested3d import Nested3D

from nesting_mpmd_case import LAYOUT, depth, make_sim
from nesting_3d_case import make_domain


def main():
    what, name = sys.argv[1:3]
    child = name == "child"
    model = NestedModel(LAYOUT, "nowhere" if (child and what == "badname") else name)
    if child and what == "raise":
        raise RuntimeError("the child fails before the run")
    lev = model.level
    H = model.share_depth(depth(lev))
    dt = model.timestep(20.0)
    nest = model.run
    three_d = what == "no3d"

    def make_sim3(_lev, domain):
        sim = pygetm.Simulation(domain, runtype=pygetm.RunType.BAROCLINIC, airsea=pygetm.airsea.Fluxes(),
                                vertical_coordinates=pygetm.vertical_coordinates.Sigma(4),
                                logger=parallel.get_logger(level="ERROR", comm=domain.comm))
        sim.radiation.set_jerlov_type(pygetm.radiation.Jerlov.Type_II)
        for v in (sim.airsea.taux, sim.airsea.tauy, sim.airsea.sp):
            v.set(0.0)
        sim.temp.set(10.0)
        sim.salt.set(35.0)
        return sim

    sim = nest.build_simulation(lambda l, comm: make_domain(l, comm, H), make_sim3 if three_d else make_sim)
    sim.add_coupler(Nested2D(nest, sim))
    if three_d and not child:
        sim.add_coupler(Nested3D(nest, sim))
    split = 4 * lev.steps_per_root if three_d else 1
    sim.start(datetime.datetime(2000, 1, 1), timestep=dt, split_factor=split, report=0)
    for s in range(20 * lev.steps_per_root):
        sim.advance()
        if child and what == "exit" and s == 5:
            sys.exit(3)
    sim.finish()


if __name__ == "__main__":
    main()
