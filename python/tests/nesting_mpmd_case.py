"""Two-level 2D barotropic case for test_nesting.TestNestingMPMD.

MPMD (one mpiexec component per level, NestedModel)::

    mpiexec -n 2 python nesting_mpmd_case.py mpmd root OUT : -n 1 python nesting_mpmd_case.py mpmd child OUT

Single program (NestingRun with rank counts), the reference::

    mpiexec -n 3 python nesting_mpmd_case.py single - OUT

Each level writes its gathered z to OUT/<mode>_<level>.npy.
"""

import datetime
import sys

import numpy as np

import pygetm
from pygetm import parallel
from pygetm.nesting import NestedModel, load_levels
from pygetm.nesting.runtime import NestingRun
from pygetm.nesting.barotropic import NestedBarotropic

LAYOUT = {
    "root": dict(nx=40, ny=30, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0),
    "child": dict(parent="root", istart=12, istop=24, jstart=8, jstop=20, ratio=3,
                  active=dict(seed=[18.0, 14.0], barriers=[{"lat_lt": 11.0, "lon_lt": 15.0}])),
}


def depth(lev):
    lon, lat = lev.edges()
    xc = 0.5 * (lon[1:] + lon[:-1])
    yc = 0.5 * (lat[1:] + lat[:-1])
    H = (20.0 + 10.0 * xc / 40.0)[None, :] * np.ones((yc.size, 1))
    H[(yc[:, None] > 16.0) & (yc[:, None] < 17.0) & (xc[None, :] > 13.0) & (xc[None, :] < 16.0)] = np.nan  # an island
    return H


def make_domain(lev, comm, H):
    lon, lat = lev.edges()
    return pygetm.domain.create_cartesian(
        lon * 2e3, lat * 2e3, interfaces=True, H=H, f=1e-4, comm=comm,
        logger=parallel.get_logger(level="ERROR", comm=comm),
    )


def make_sim(lev, domain):
    sim = pygetm.Simulation(domain, runtype=pygetm.RunType.BAROTROPIC_2D, airsea=pygetm.airsea.Fluxes(),
                            logger=parallel.get_logger(level="ERROR", comm=domain.comm))
    sim.airsea.taux.set(0.0)
    sim.airsea.tauy.set(0.0)
    sim.airsea.sp.set(0.0)
    return sim


def run(nest, lev, H, dt):
    sim = nest.build_simulation(lambda l, comm: make_domain(l, comm, H), make_sim)
    cpl = NestedBarotropic(nest, sim)
    sim.add_coupler(cpl)
    gj, gi = nest.global_indices()
    x = lev.lon0 + (gi + 0.5) * lev.dlon
    y = lev.lat0 + (gj + 0.5) * lev.dlat
    sim.T.z.all_values[...] = 0.5 * np.exp(-((x[None, :] - 18.0) ** 2 + (y[:, None] - 14.0) ** 2) / 8.0)
    sim.T.z.all_values[sim.T.mask.all_values == 0] = 0.0
    sim.start(datetime.datetime(2000, 1, 1), timestep=dt, split_factor=1, report=0)
    for _ in range(40 * lev.steps_per_root):
        sim.advance()
    sim.finish()
    return sim.T.z.gather()


def main():
    mode, name, out = sys.argv[1:4]
    if mode == "mpmd":
        model = NestedModel(LAYOUT, name)
        lev = model.level
        H = model.share_depth(depth(lev))
        dt = model.timestep(20.0)
        nest, lev = model.run, model.level
    else:
        world = parallel.MPI.COMM_WORLD
        geo = load_levels(LAYOUT, root_band=0, active=False)
        Hs = {l.name: depth(l) for l in geo}
        levels = load_levels(LAYOUT, root_band=0, wet={k: np.isfinite(v) for k, v in Hs.items()})
        from pygetm.nesting import masks

        wets, Hs, _ = masks.make_consistent(levels, {k: np.isfinite(v) for k, v in Hs.items()}, Hs)
        nest = NestingRun(levels, {"root": world.size - 1, "child": 1})
        lev = nest.level
        H = np.where(wets[lev.name], Hs[lev.name], np.nan)
        dt = 20.0 / lev.steps_per_root
    z = run(nest, lev, H, dt)
    if z is not None:
        np.save(f"{out}/{mode}_{lev.name}.npy", np.array(z))


if __name__ == "__main__":
    main()
