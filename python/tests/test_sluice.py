"""pygetm.sluice: two basins joined only by a sluice. Volume and tracer content are
conserved, water flows from the higher to the lower basin, a flap gate only one way."""
import datetime
import unittest

import cftime
import numpy as np

import pygetm
import pygetm.sluice

START, STOP = cftime.datetime(2000, 1, 1), cftime.datetime(2000, 1, 1, 12)
TIMESTEP, SPLIT = 30.0, 10
FEED = 500.0  # m3/s into the west basin


def make(direction="both", gate=1.0, nz=10, b_layers=None, b=(21, 10), invert=None):
    domain = pygetm.domain.create_cartesian(
        np.linspace(0, 40e3, 41), np.linspace(0, 20e3, 21), H=20.0, interfaces=True, f=0.0,
        logger=pygetm.parallel.get_logger(level="ERROR"),
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


if __name__ == "__main__":
    unittest.main()
