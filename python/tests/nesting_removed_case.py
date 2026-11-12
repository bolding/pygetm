"""A parent whose tiling drops subdomains (test_nesting.TestNestingRemovedTiles): the child's window covers most of the
root, so subdomains of the root that hold only hole cells nothing reads are removed (Level.tiling_mask)::

    mpiexec -n NROOT python nesting_removed_case.py root OUT : -n 1 python nesting_removed_case.py child OUT

With NROOT = 8 the root gets a prescribed 3 x 3 tiling whose centre subdomain (cells 14-27 x 10-19, all in the
interior of the hole) is removed; with other counts pyGETM's own division (which here removes none).

The depth and the dynamics of nesting_mpmd_case.py (a closed basin, a Gaussian hill of water). Every level writes
OUT/<level>_<NROOT>.npz: its gathered elevation at the end, the water volume over the cells it computes after every
root step, and the number of subdomains its tiling removed.
"""

import datetime
import os
import sys

import numpy as np

import pygetm
from pygetm import parallel
from pygetm.nesting import NestedModel
from pygetm.nesting.nested2d import Nested2D

from nesting_mpmd_case import depth, make_domain

LAYOUT = {
    "root": dict(nx=40, ny=30, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
    "child": dict(parent="root", istart=5, istop=35, jstart=4, jstop=26, ratio=2),
}


def main():
    name, out = sys.argv[1:3]
    model = NestedModel(LAYOUT, name)
    lev = model.level
    H = model.share_depth(depth(lev))
    dt = model.timestep(20.0)
    nest = model.run

    def make_sim(lev, domain):
        tiling = None
        if lev.parent is None and domain.comm.size == 8:  # 3 x 3 with the centre (all hole) removed
            rank_map = np.array([[0, 1, 2], [3, -1, 4], [5, 6, 7]])
            tiling = parallel.Tiling(map=rank_map, comm=domain.comm)
            tiling.set_extent(lev.nx + 2 * lev.band, lev.ny + 2 * lev.band, 14, 10)
        sim = pygetm.Simulation(domain, runtype=pygetm.RunType.BAROTROPIC_2D, airsea=pygetm.airsea.Fluxes(),
                                tiling=tiling, logger=parallel.get_logger(level="ERROR", comm=domain.comm))
        for v in (sim.airsea.taux, sim.airsea.tauy, sim.airsea.sp):
            v.set(0.0)
        return sim

    sim = nest.build_simulation(lambda l, comm: make_domain(l, comm, H), make_sim)
    cpl = Nested2D(nest, sim)
    sim.add_coupler(cpl)
    gj, gi = nest.global_indices()
    x = lev.lon0 + (gi + 0.5) * lev.dlon
    y = lev.lat0 + (gj + 0.5) * lev.dlat
    sim.T.z.all_values[...] = 0.5 * np.exp(-((x[None, :] - 5.0) ** 2 + (y[:, None] - 15.0) ** 2) / 6.0)
    sim.T.z.all_values[sim.T.mask.all_values == 0] = 0.0
    sim.start(datetime.datetime(2000, 1, 1), timestep=dt, split_factor=1, report=0)
    vols = [model.comm.allreduce(cpl.level_volume())]
    for s in range(40 * lev.steps_per_root):
        sim.advance()
        if (s + 1) % lev.steps_per_root == 0:
            vols.append(model.comm.allreduce(cpl.level_volume()))
    sim.finish()
    z = sim.T.z.gather()
    nroot = nest.sizes["root"]
    if model.comm.rank == 0:
        np.savez(os.path.join(out, f"{name}_{nroot}.npz"), z=np.asarray(z), volume=np.array(vols),
                 removed=nest.tilings[lev.name].removed, world=parallel.MPI.COMM_WORLD.size)


if __name__ == "__main__":
    main()
