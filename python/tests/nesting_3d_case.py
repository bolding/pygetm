"""Two-level 3D (baroclinic) case for test_nesting.TestNesting3D and TestNestingRestart3D.

The layout and depth of nesting_mpmd_case.py, a closed basin at rest, the root on sigma layers (5) and the child on
others (8): a temperature front across the child's west interface drives the flow, salinity is uniform (35). No surface
fluxes, so the heat content over both levels (with the flux limiter's deferred amounts) is conserved, and salinity
stays 35 everywhere::

    mpiexec -n 2 python nesting_3d_case.py MODE root OUT [OPTIONS] : -n 1 python nesting_3d_case.py MODE child OUT [OPTIONS]

OPTIONS: ``scheme=ppm|pcm`` (Nested3D remap_scheme, default ppm), ``sub`` (the child its own 3D step, a third of the
root's, and ring_time "extrapolate"; default: every level the root's 3D step, ring held).

MODE: ``run`` (2 x N root 3D steps), ``full`` (the same, restart file OUT/full_<level>.nc), ``first`` (N, OUT/first_<level>.nc),
``second`` (N from OUT/first_<level>.nc, OUT/second_<level>.nc). Every level writes OUT/<mode>_<level>.npz: at every
root 3D step its heat content (temperature x volume, with the deferred amounts, over the cells it computes), volume,
and the range of salinity over its computed wet cells.
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

from nesting_mpmd_case import LAYOUT, depth

NZ = {"root": 5, "child": 8}
Z0 = 0.001  # bottom roughness (with 0 the dissipation at the bed is infinite, which a restart's start-up check refuses)


def make_domain(lev, comm, H):
    lon, lat = lev.edges()
    return pygetm.domain.create_cartesian(lon * 2e3, lat * 2e3, interfaces=True, H=H, f=1e-4, z0=Z0, comm=comm,
                                          logger=parallel.get_logger(level="ERROR", comm=comm))
SPLIT_ROOT = 4  # the root's 3D step: 4 x its 2D step
N = 6  # root 3D steps per half of the run


def main():
    mode, name, out = sys.argv[1:4]
    opts = sys.argv[4:]
    scheme = next((o.split("=", 1)[1] for o in opts if o.startswith("scheme=")), "ppm")
    sub = "sub" in opts
    model = NestedModel(LAYOUT, name)
    lev = model.level
    H = model.share_depth(depth(lev))
    dt = model.timestep(20.0)
    split = SPLIT_ROOT * lev.steps_per_root  # the root's 3D step on every level
    if sub and lev.parent is not None:
        split //= 3  # a 3D step of its own: a third of its parent's (3D sub-stepping)

    def make_sim(lev, domain):
        sim = pygetm.Simulation(domain, runtype=pygetm.RunType.BAROCLINIC, airsea=pygetm.airsea.Fluxes(),
                                vertical_coordinates=pygetm.vertical_coordinates.Sigma(NZ[lev.name]),
                                logger=parallel.get_logger(level="ERROR", comm=domain.comm))
        sim.radiation.set_jerlov_type(pygetm.radiation.Jerlov.Type_II)
        for v in (sim.airsea.taux, sim.airsea.tauy, sim.airsea.sp):
            v.set(0.0)
        return sim

    nest = model.run
    sim = nest.build_simulation(lambda l, comm: make_domain(l, comm, H), make_sim)
    sim.add_coupler(Nested2D(nest, sim))
    cpl3 = Nested3D(nest, sim, remap_scheme=scheme, ring_time="extrapolate" if sub else "hold")
    sim.add_coupler(cpl3)
    if mode in ("full", "first", "second"):
        sim.output_manager.add_restart(os.path.join(out, f"{mode}_{name}.nc"))
    T = sim.T
    start = datetime.datetime(2000, 1, 1)
    if mode == "second":
        start = sim.load_restart(os.path.join(out, f"first_{name}.nc"))
    else:
        _, gi = nest.global_indices()
        x = lev.lon0 + (gi + 0.5) * lev.dlon
        # in the cells this level computes; the rest (land, a child's ring and band) 0: the ring gets the parent's
        # values at the first sync, and a restart file holds NaN where nothing is computed
        own = (T.mask.all_values == 1)[None]
        sim.temp.all_values[...] = np.where(own, (10.0 + 3.0 * np.tanh((x - 15.0) / 1.5))[None, None, :], 0.0)
        sim.salt.all_values[...] = np.where(own, 35.0, 0.0)

    h = T.halox
    computed = T.mask.all_values[h:-h, h:-h] == 1

    def totals():
        vol = (T.hn.all_values * T.area.all_values)[:, h:-h, h:-h]
        deferred = float(cpl3.deferred_content()[[f is sim.temp for f in cpl3.tracers].index(True)])
        Q = float((sim.temp.all_values[:, h:-h, h:-h] * vol)[:, computed].sum()) + deferred
        V = float(vol[:, computed].sum())
        s = sim.salt.all_values[:, h:-h, h:-h][:, computed & (T.H.all_values[h:-h, h:-h] > 0)]
        smin = float(s.min()) if s.size else np.inf
        smax = float(s.max()) if s.size else -np.inf
        c = model.comm
        return [c.allreduce(Q), c.allreduce(V), c.allreduce(smin, op=parallel.MPI.MIN),
                c.allreduce(smax, op=parallel.MPI.MAX)]

    sim.start(start, timestep=dt, split_factor=split, report=0)
    series = [totals()]  # after the start: the layer thicknesses include the elevation
    nsteps = (2 if mode in ("full", "run") else 1) * N * SPLIT_ROOT * lev.steps_per_root
    per_root_3d = SPLIT_ROOT * lev.steps_per_root
    for s in range(nsteps):
        sim.advance()
        if (s + 1) % per_root_3d == 0:
            series.append(totals())
    sim.finish()
    if model.comm.rank == 0:
        a = np.array(series)
        np.savez(os.path.join(out, f"{mode}_{name}.npz"), heat=a[:, 0], volume=a[:, 1], salt_min=a[:, 2],
                 salt_max=a[:, 3], ratio_3d=cpl3.ratio_3d, n_extrapolated=getattr(cpl3, "n_extrapolated", 0))


if __name__ == "__main__":
    main()
