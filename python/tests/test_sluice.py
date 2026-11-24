"""pygetm.sluice: two basins joined only by a sluice. Volume and tracer content are
conserved, water flows from the higher to the lower basin, a flap gate only one way."""
import datetime
import os
import tempfile
import unittest

import cftime
import numpy as np

import pygetm
import pygetm.sluice

START, STOP = cftime.datetime(2000, 1, 1), cftime.datetime(2000, 1, 1, 12)
TIMESTEP, SPLIT = 30.0, 10
FEED = 500.0  # m3/s into the west basin


def make(direction="both", gate=1.0, nz=10, b_layers=None, b=(21, 10), invert=None, z0=None):
    domain = pygetm.domain.create_cartesian(
        np.linspace(0, 40e3, 41), np.linspace(0, 20e3, 21), H=20.0, interfaces=True, f=0.0,
        logger=pygetm.parallel.get_logger(level="ERROR"), **({} if z0 is None else {"z0": z0}),
    )
    domain.logger.parent.setLevel("ERROR")
    domain.mask[:, 40:43] = 0  # supergrid: a wall of T cells 20 between the basins (0-19 west, 21-39 east)
    domain.rivers.add_by_index("feed", 5, 10)
    sluice = pygetm.sluice.Sluice(domain, "s", a=(19, 10), b=b, area=20.0, gate=gate, direction=direction,
                                  coordinate_type=pygetm.rivers.CoordinateType.IJ, b_layers=b_layers, invert=invert)
    sim = pygetm.Simulation(domain, airsea=pygetm.airsea.Fluxes(), radiation=pygetm.radiation.Radiation(),
                            vertical_coordinates=pygetm.vertical_coordinates.Sigma(nz))
    sim.rivers["feed"].flow.set(FEED)
    dye = sim.tracers.add("dye")
    sim.tracer_totals.append(pygetm.tracer.TracerTotal(dye))
    dye.all_values[...] = np.where(sim.T.x.all_values < 20e3, 1.0, 0.0)
    sim.rivers["feed"]["dye"].follow_target_cell = True
    sluice.attach(sim)
    return sim, sluice


def run(sim):
    sim.start(START, TIMESTEP, SPLIT, report=datetime.timedelta(days=1))
    v0, t0 = sim.totals
    while sim.time < STOP:
        sim.advance()
    v1, t1 = sim.totals
    sim.finish()
    return v0, t0[-1][1], v1, t1[-1][1]


def basin_levels(sim):
    z = sim.T.z.values
    return float(z[:, :20].mean()), float(z[:, 21:].mean())


class TestSluice(unittest.TestCase):
    def test_conservation_and_flow(self):
        sim, sl = make()
        v0, c0, v1, c1 = run(sim)
        dt = (STOP - START).total_seconds()
        # all water and dye accounted for: the sluice only moves them
        self.assertAlmostEqual(v1 / (v0 + FEED * dt), 1.0, delta=1e-13)
        self.assertAlmostEqual(c1 / (c0 + FEED * dt), 1.0, delta=1e-12)  # the feed brings dye at the west's 1
        self.assertGreater(sl.volume, 0.0)       # west -> east
        zw, ze = basin_levels(sim)
        self.assertGreater(zw, ze)
        self.assertGreater(ze, 0.0)              # the east basin got water
        self.assertGreater(float(sim["dye"].values[:, :, 21:].sum()), 0.0)  # and dye

    def test_closed_gate(self):
        sim, sl = make(gate=0.0)
        run(sim)
        self.assertEqual(sl.volume, 0.0)
        self.assertAlmostEqual(basin_levels(sim)[1], 0.0, delta=1e-12)

    def test_flap_gate(self):
        sim, sl = make(direction="b_to_a")  # west is higher: nothing may flow
        run(sim)
        self.assertEqual(sl.volume, 0.0)

    def test_bottom_inflow(self):
        sim, sl = make(b_layers=dict(zl=0.0, zu=0.0, vertical_position=pygetm.rivers.VerticalPosition.DistanceFromBottom))
        run(sim)
        dye = sim["dye"].values[:, 10, 21]
        self.assertGreater(dye[0], dye[-1])      # enters the bottom layer (layer 0) of the east cell

    def test_pipe_in_dam(self):
        # outlet at +0.05 m: the east basin (at 0) is below it, the pipe flows freely,
        # driven by the west level above the outlet only
        sim, sl = make(invert=0.005)
        run(sim)
        self.assertGreater(sl.volume, 0.0)
        zw, ze = basin_levels(sim)
        self.assertGreater(zw, 0.005)

    def test_outflow_to_air(self):
        # nothing downstream: the reservoir drains above its outlet, the water leaves the model
        sim, sl = make(b=None, invert=0.0)
        v0, c0, v1, c1 = run(sim)
        dt = (STOP - START).total_seconds()
        self.assertAlmostEqual(v1 / (v0 + FEED * dt - sl.volume), 1.0, delta=1e-12)
        self.assertGreater(sl.volume, 0.0)
        self.assertAlmostEqual(basin_levels(sim)[1], 0.0, delta=1e-12)  # east untouched
        self.assertAlmostEqual(c1 / (c0 + FEED * dt - sl.content["dye"]), 1.0, delta=1e-12)  # the dye that left is accounted for

    def test_shallow_donor(self):
        # review F8: a donor holding little water. A basin of one cell (0.5 m deep) drains through a pipe whose
        # outlet is 3 m lower: the level limit alone would let one step take more than the cell holds; the depth
        # limit lets it approach Dmin and never go below
        domain = pygetm.domain.create_cartesian(
            np.linspace(0, 10e3, 11), np.linspace(0, 5e3, 6), H=0.5, interfaces=True, f=0.0,
            logger=pygetm.parallel.get_logger(level="ERROR"),
        )
        domain.mask[...] = 0
        domain.mask[5, 3] = 1  # supergrid: T cell (1, 2) alone
        domain.mask[1:10, 9:20] = 1  # another basin, not connected
        sl = pygetm.sluice.Sluice(domain, "s", a=(1, 2), b=None, area=1000.0, invert=-3.0,
                                  coordinate_type=pygetm.rivers.CoordinateType.IJ)
        sim = pygetm.Simulation(domain, runtype=pygetm.BAROTROPIC_2D, airsea=pygetm.airsea.Fluxes())
        sl.attach(sim)
        sim.start(START, TIMESTEP, 1, report=datetime.timedelta(days=1))
        depths = []
        for _ in range(60):
            sim.advance()
            depths.append(float(sim.T.H.values[2, 1] + sim.T.z.values[2, 1]))
        sim.finish()
        self.assertGreaterEqual(min(depths), sim.Dmin - 1e-12)
        self.assertAlmostEqual(depths[-1], sim.Dmin, delta=1e-6)
        area = float(sim.T.area.values[2, 1])
        self.assertAlmostEqual(sl.volume / ((0.5 - sim.Dmin) * area), 1.0, delta=1e-6)  # all above Dmin, no more

    def test_restart(self):
        # review F9: the volume and content moved so far continue after a restart, and the run continues as the
        # unbroken one (z0 > 0: a restart with z0 = 0 stops at pyGETM's check of eps)
        mid = cftime.datetime(2000, 1, 1, 6)
        with tempfile.TemporaryDirectory() as out:
            full, sl_full = make(z0=0.01)
            run(full)

            first, sl_first = make(z0=0.01)
            first.output_manager.add_restart(os.path.join(out, "first.nc"))
            first.start(START, TIMESTEP, SPLIT, report=datetime.timedelta(days=1))
            while first.time < mid:
                first.advance()
            first.finish()
            self.assertGreater(sl_first.volume, 0.0)

            second, sl_second = make(z0=0.01)
            start = second.load_restart(os.path.join(out, "first.nc"))
            second.start(start, TIMESTEP, SPLIT, report=datetime.timedelta(days=1))
            while second.time < STOP:
                second.advance()
            second.finish()
        self.assertGreater(sl_second.volume, sl_first.volume)
        self.assertAlmostEqual(sl_second.volume / sl_full.volume, 1.0, delta=1e-12)
        self.assertAlmostEqual(sl_second.content["dye"] / sl_full.content["dye"], 1.0, delta=1e-12)
        np.testing.assert_array_equal(second.T.z.values, full.T.z.values)


if __name__ == "__main__":
    unittest.main()
