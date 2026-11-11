"""Two-level 3D case with FABM for test_nesting.TestNestingFABM (on demand: PYGETM_TEST_FABM=1).

The layout and depth of nesting_mpmd_case.py, 3D (root 5, child 8 sigma layers, split 4 and 12), a closed basin
at rest with a temperature front and a nutrient patch across the child's west interface, FABM's npzd + benthic predator test case
(extern/fabm/testcases/fabm-examples-npzd-benthos.yaml: total nitrogen conserved, with a bottom state variable)::

    mpiexec -n 2 python nesting_fabm_case.py root OUT : -n 1 python nesting_fabm_case.py child OUT

Each level writes OUT/<level>.npz: its total nitrogen (pelagic x volume, with the flux limiter's deferred amounts,
+ bottom x area, over the cells it computes) at every root 3D step, and the names of the tracers Nested3D exchanges.
"""

import datetime
import os
import sys

import numpy as np

import pygetm
from pygetm import parallel
from pygetm.nesting import NestedModel
from pygetm.nesting.nested2d import Nested2D
from pygetm.nesting.nested3d import Nested3D

from nesting_mpmd_case import LAYOUT, depth, make_domain

FABM_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "extern", "fabm", "testcases",
                         "fabm-examples-npzd-benthos.yaml")
NZ = {"root": 5, "child": 8}
SPLIT_ROOT = 4


def main():
    name, out = sys.argv[1:3]
    model = NestedModel(LAYOUT, name)
    lev = model.level
    H = model.share_depth(depth(lev))
    dt = model.timestep(20.0)
    split = SPLIT_ROOT * lev.steps_per_root  # every level the root's 3D step

    def make_sim(lev, domain):
        sim = pygetm.Simulation(domain, runtype=pygetm.RunType.BAROCLINIC, airsea=pygetm.airsea.Fluxes(),
                                vertical_coordinates=pygetm.vertical_coordinates.Sigma(NZ[lev.name]),
                                fabm=FABM_YAML, logger=parallel.get_logger(level="ERROR", comm=domain.comm))
        sim.radiation.set_jerlov_type(pygetm.radiation.Jerlov.Type_II)
        for v in (sim.airsea.taux, sim.airsea.tauy, sim.airsea.sp):
            v.set(0.0)
        return sim

    nest = model.run
    sim = nest.build_simulation(lambda l, comm: make_domain(l, comm, H), make_sim)
    cpl = Nested2D(nest, sim)
    sim.add_coupler(cpl)
    cpl3 = Nested3D(nest, sim)  # default tracers: temp, salt and FABM's pelagic state
    sim.add_coupler(cpl3)
    T = sim.T
    gj, gi = nest.global_indices()
    x = lev.lon0 + (gi + 0.5) * lev.dlon
    y = lev.lat0 + (gj + 0.5) * lev.dlat
    # at rest; a temperature front across the child's west interface drives the flow, and a nutrient patch
    # there is carried across it
    bump = np.exp(-((x[None, :] - 15.0) ** 2 + (y[:, None] - 14.0) ** 2) / 8.0)
    sim.temp.all_values[...] = (10.0 + 3.0 * np.tanh((x - 15.0) / 1.5))[None, None, :]
    sim.salt.all_values[...] = 35.0
    fm = sim.fabm
    pelagic = [fm._variable2array[v] for v in fm.model.interior_state_variables]
    bottom = [fm._variable2array[v] for v in fm.model.bottom_state_variables]
    nut = fm._variable2array[next(v for v in fm.model.interior_state_variables if v.name == "npzd/nut")]
    ok = np.isfinite(nut.all_values)
    nut.all_values[ok] = (nut.all_values * (1.0 + np.broadcast_to(bump, nut.all_values.shape)))[ok]

    h = T.halox
    computed = T.mask.all_values[h:-h, h:-h] == 1

    def deferred(f):
        for g, v in zip(cpl3.tracers, cpl3.deferred_content()):
            if g is f:
                return float(v)
        return 0.0

    def nitrogen():
        vol = (T.hn.all_values * T.area.all_values)[:, h:-h, h:-h]
        a = T.area.all_values[h:-h, h:-h]
        n = sum(float((f.all_values[:, h:-h, h:-h] * vol)[:, computed].sum()) + deferred(f) for f in pelagic)
        n += sum(float((f.all_values[h:-h, h:-h] * a)[computed].sum()) for f in bottom)
        V = float(vol[:, computed].sum())
        Q = float((sim.temp.all_values[:, h:-h, h:-h] * vol)[:, computed].sum()) + deferred(sim.temp)
        return [model.comm.allreduce(n), model.comm.allreduce(V), model.comm.allreduce(Q)]

    sim.start(datetime.datetime(2000, 1, 1), timestep=dt, split_factor=split, report=0)
    totals = [nitrogen()]  # after the start: the layer thicknesses include the elevation
    for s in range(40 * lev.steps_per_root):  # 10 root 3D steps
        sim.advance()
        if (s + 1) % split == 0:
            totals.append(nitrogen())
    sim.finish()
    if model.comm.rank == 0:
        tt = np.array(totals)
        np.savez(os.path.join(out, f"{name}.npz"), nitrogen=tt[:, 0], volume=tt[:, 1], heat=tt[:, 2],
                 tracers=np.array([getattr(f, "name", "") for f in cpl3.tracers]))


if __name__ == "__main__":
    main()
