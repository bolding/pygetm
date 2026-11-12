"""Three levels in 2D (root, child, grandchild) for test_nesting.TestNestingThreeLevels::

    mpiexec -n 2 python nesting_levels_case.py root OUT WHERE : -n 2 python nesting_levels_case.py child OUT WHERE \\
          : -n 1 python nesting_levels_case.py grand OUT WHERE

WHERE: ``on`` (the grandchild's west side on the child's west edge, so on the child's interface with the root: its
fluxes there are passed up through the child) or ``near`` (one root cell inside that edge: the root's rim reaches into
the child's hole, so the child sends up only after its grandchild). A closed basin (the depth of nesting_mpmd_case.py),
a Gaussian hill of water across the interfaces. Every level writes OUT/<level>.npz: the water volume above rest over the
cells it computes (Nested2D.level_volume) after every root step.
"""

import datetime
import os
import sys

import numpy as np

from pygetm.nesting import NestedModel
from pygetm.nesting.nested2d import Nested2D

from nesting_mpmd_case import depth, make_domain, make_sim


def layout(where: str) -> dict:
    gi0 = {"on": 0, "near": 3}[where]  # in child cells (ratio 3): 0 on the child's edge, 3 one root cell inside
    return {
        "root": dict(nx=40, ny=30, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
        "child": dict(parent="root", istart=8, istop=32, jstart=5, jstop=25, ratio=3),
        "grand": dict(parent="child", istart=gi0, istop=gi0 + 24, jstart=15, jstop=45, ratio=2),
    }


def main():
    name, out, where = sys.argv[1:4]
    model = NestedModel(layout(where), name)
    lev = model.level
    H = model.share_depth(depth(lev))
    dt = model.timestep(20.0)
    nest = model.run
    sim = nest.build_simulation(lambda l, comm: make_domain(l, comm, H), make_sim)
    cpl = Nested2D(nest, sim)
    sim.add_coupler(cpl)
    gj, gi = nest.global_indices()
    x = lev.lon0 + (gi + 0.5) * lev.dlon
    y = lev.lat0 + (gj + 0.5) * lev.dlat
    sim.T.z.all_values[...] = 0.5 * np.exp(-((x[None, :] - 10.0) ** 2 + (y[:, None] - 13.0) ** 2) / 6.0)
    sim.T.z.all_values[sim.T.mask.all_values == 0] = 0.0
    sim.start(datetime.datetime(2000, 1, 1), timestep=dt, split_factor=1, report=0)
    vols = [model.comm.allreduce(cpl.level_volume())]
    for s in range(60 * lev.steps_per_root):
        sim.advance()
        if (s + 1) % lev.steps_per_root == 0:
            vols.append(model.comm.allreduce(cpl.level_volume()))
    sim.finish()
    if model.comm.rank == 0:
        np.savez(os.path.join(out, f"{name}.npz"), volume=np.array(vols),
                 up_after_children=lev.sends_up_after_children)


if __name__ == "__main__":
    main()
