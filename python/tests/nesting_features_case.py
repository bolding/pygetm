"""Two levels in 2D with the features that meet the nesting (test_nesting.TestNestingFeatures)::

    mpiexec -n 2 python nesting_features_case.py WHAT root OUT : -n 1 python nesting_features_case.py WHAT child OUT

WHAT:

* ``outer``: the child's window reaches the root's east edge (an outer side: its band beyond is dry, its faces there
  are a wall, nothing is coupled across it);
* ``rivers``: a river in the root and one in the child (50 m3 s-1 each), both given to every level (each belongs to
  the level that computes its cell);
* ``dam``: a dam of zero width in each level (pygetm.dams), off the interface;
* ``sluice``: a sluice (pygetm.sluice) from the root's water to the child's, its ends on two levels;
* ``drying``: a bank inside the child, 0.05 m deep, that falls dry and floods with the Gaussian hill of water.

A closed basin (otherwise the depth of nesting_mpmd_case.py), a Gaussian hill of water. Every level writes
OUT/<level>.npz: the water volume above rest over the cells it computes after every root step, the model time, and
per variant its own diagnostics (outer sides, the sluice's moved volume, dry cells seen).
"""

import datetime
import os
import sys

import numpy as np

import pygetm
import pygetm.dams
import pygetm.sluice
from pygetm import parallel
from pygetm.nesting import NestedModel
from pygetm.nesting.nested2d import Nested2D

from nesting_mpmd_case import LAYOUT, depth, make_domain, make_sim

KM = 2e3  # make_domain's metres per layout unit


def layout(what: str) -> dict:
    lay = {k: dict(v) for k, v in LAYOUT.items()}
    lay["child"].pop("active", None)
    if what == "outer":
        lay["child"].update(istart=26, istop=40, jstart=8, jstop=20)
    else:
        lay["child"].update(istart=12, istop=24, jstart=8, jstop=20)
    return lay


def main():
    what, name, out = sys.argv[1:4]
    model = NestedModel(layout(what), name)
    lev = model.level
    H0 = depth(lev)
    if what == "drying":  # a shallow bank inside the child's window
        lon_e, lat_e = lev.edges()
        xc, yc = 0.5 * (lon_e[1:] + lon_e[:-1]), 0.5 * (lat_e[1:] + lat_e[:-1])
        bank = (xc[None, :] > 15.0) & (xc[None, :] < 19.0) & (yc[:, None] > 11.0) & (yc[:, None] < 15.0)
        H0 = np.where(bank & np.isfinite(H0), 0.05, H0)
    H = model.share_depth(H0)
    dt = model.timestep(20.0)
    nest = model.run
    sluices = []
    dam_faces = [0]

    def domain_for(l, comm):
        domain = make_domain(l, comm, H)
        if what == "rivers":  # every level gets both; each is kept by the level that computes its cell
            domain.rivers.add_by_location("in_root", 6.5 * KM, 14.5 * KM, coordinate_type=pygetm.CoordinateType.XY)
            domain.rivers.add_by_location("in_child", 17.5 * KM, 13.5 * KM, coordinate_type=pygetm.CoordinateType.XY)
        elif what == "dam":  # a partial wall in the root (west of the window) and one in the child
            for line in ([(6.0 * KM, 8.0 * KM), (6.0 * KM, 20.0 * KM)], [(18.0 * KM, 10.0 * KM), (18.0 * KM, 15.0 * KM)]):
                dam_faces[0] += pygetm.dams.close(domain, line, coordinate_type=pygetm.CoordinateType.XY)
        elif what == "sluice":
            sluices.append(pygetm.sluice.Sluice(domain, "gate", a=(6.5 * KM, 14.5 * KM), b=(17.5 * KM, 13.5 * KM),
                                                area=200.0, coordinate_type=pygetm.CoordinateType.XY))
        return domain

    sim = nest.build_simulation(domain_for, make_sim)
    cpl = Nested2D(nest, sim)
    sim.add_coupler(cpl)
    for sl in sluices:
        sl.attach(sim, nest=nest)
    if what == "rivers":
        for r in sim.rivers.values():
            r.flow.set(50.0)
    gj, gi = nest.global_indices()
    x = lev.lon0 + (gi + 0.5) * lev.dlon
    y = lev.lat0 + (gj + 0.5) * lev.dlat
    x0 = 30.0 if what == "outer" else 16.0
    sim.T.z.all_values[...] = 0.5 * np.exp(-((x[None, :] - x0) ** 2 + (y[:, None] - 14.0) ** 2) / 6.0)
    sim.T.z.all_values[sim.T.mask.all_values == 0] = 0.0
    if what == "sluice":  # a level difference across the sluice: the root's end higher
        sim.T.z.all_values[...] += np.where(x[None, :] < 11.0, 0.3, 0.0) * (sim.T.mask.all_values != 0)
    start = datetime.datetime(2000, 1, 1)
    sim.start(start, timestep=dt, split_factor=1, report=0)
    vols = [model.comm.allreduce(cpl.level_volume())]
    times = [0.0]
    h = sim.T.halox
    dry_seen = 0
    for s in range(60 * lev.steps_per_root):
        sim.advance()
        D = sim.T.D.all_values[h:-h, h:-h]
        wet = sim.T.mask.all_values[h:-h, h:-h] == 1
        dry_seen = max(dry_seen, int((wet & (D < 2.0 * sim.Dmin)).sum()))
        if (s + 1) % lev.steps_per_root == 0:
            vols.append(model.comm.allreduce(cpl.level_volume()))
            times.append((sim.time - start).total_seconds() if hasattr(sim.time - start, "total_seconds") else 0.0)
    sim.finish()
    moved = sum(float(getattr(sl, "volume", 0.0)) for sl in sluices)
    dry_seen = model.comm.allreduce(dry_seen, op=parallel.MPI.SUM)
    if model.comm.rank == 0:
        np.savez(os.path.join(out, f"{name}.npz"), volume=np.array(vols), time=np.array(times),
                 outer_sides=lev.outer_sides, rivers=len(sim.rivers), sluice_volume=moved, dry_seen=dry_seen, dam_faces=dam_faces[0])


if __name__ == "__main__":
    main()
