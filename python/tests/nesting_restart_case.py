"""Two-level 2D case for test_nesting.TestNestingRestart (the layout and depth of
nesting_mpmd_case.py; MPMD, one mpiexec component per level)::

    mpiexec -n 2 python nesting_restart_case.py MODE root OUT : -n 1 python nesting_restart_case.py MODE child OUT

MODE:

* ``full``: 40 root steps, restart file OUT/full_<level>.nc at the end;
* ``first``: 20 root steps, restart file OUT/first_<level>.nc;
* ``second``: from OUT/first_<level>.nc 20 root steps, restart file OUT/second_<level>.nc
  (must equal ``full``);
* ``mid``: 7 root steps with a restart file every 4 steps of the level (the child's last,
  after 20 of its 21 steps, is in the middle of a root step);
* ``from_mid``: from OUT/mid_<level>.nc (must be refused).
"""

import datetime
import sys

import pygetm
from pygetm.nesting import NestedModel
from pygetm.nesting.nested2d import Nested2D

from nesting_mpmd_case import LAYOUT, depth, make_domain, make_sim


def main():
    mode, name, out = sys.argv[1:4]
    model = NestedModel(LAYOUT, name)
    lev = model.level
    H = model.share_depth(depth(lev))
    dt = model.timestep(20.0)
    nest = model.run
    sim = nest.build_simulation(lambda l, comm: make_domain(l, comm, H), make_sim)
    sim.add_coupler(Nested2D(nest, sim))
    if mode == "mid":
        sim.output_manager.add_restart(f"{out}/mid_{name}.nc", interval=4)
    elif mode != "from_mid":
        sim.output_manager.add_restart(f"{out}/{mode}_{name}.nc")
    start = datetime.datetime(2000, 1, 1)
    if mode in ("second", "from_mid"):
        start = sim.load_restart(f"{out}/{'first' if mode == 'second' else 'mid'}_{name}.nc")
    else:
        import numpy as np

        gj, gi = nest.global_indices()
        x = lev.lon0 + (gi + 0.5) * lev.dlon
        y = lev.lat0 + (gj + 0.5) * lev.dlat
        sim.T.z.all_values[...] = 0.5 * np.exp(-((x[None, :] - 18.0) ** 2 + (y[:, None] - 14.0) ** 2) / 8.0)
        sim.T.z.all_values[sim.T.mask.all_values == 0] = 0.0
    sim.start(start, timestep=dt, split_factor=1, report=0)
    n = {"full": 40, "mid": 7}.get(mode, 20)
    for _ in range(n * lev.steps_per_root):
        sim.advance()
    sim.finish()


if __name__ == "__main__":
    main()
