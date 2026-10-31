import datetime
import unittest

import numpy as np

import pygetm
from pygetm.constants import CENTERS

DX = 1000.0  # grid spacing (m)
CS = 0.2  # Smagorinsky coefficient


def create_simulation(Cs=CS, Am=0.0, runtype=pygetm.BAROTROPIC_2D, nz=3, nx=40, ny=30):
    domain = pygetm.domain.create_cartesian(
        np.arange(nx + 1) * DX,
        np.arange(ny + 1) * DX,
        interfaces=True,
        f=0.0,
        H=20.0,
        logger=pygetm.parallel.get_logger(level="ERROR"),
    )
    kwargs = {}
    if runtype != pygetm.BAROTROPIC_2D:
        kwargs["vertical_coordinates"] = pygetm.vertical_coordinates.Sigma(nz)
    return pygetm.Simulation(
        domain,
        runtype=runtype,
        momentum=pygetm.momentum.Momentum(Am=Am, Cs=Cs),
        airsea=pygetm.airsea.Fluxes(),
        **kwargs,
    )


def velocity(sim, ufun, vfun, z=None):
    """u, v on the U and V grids from functions of (x, y); 3D: the same in every layer"""
    u = sim.U.array(fill=0.0, z=z)
    v = sim.V.array(fill=0.0, z=z)
    u.all_values[...] = ufun(sim.U.x.all_values, sim.U.y.all_values)
    v.all_values[...] = vfun(sim.V.x.all_values, sim.V.y.all_values)
    return u, v


def smagorinsky(sim, u, v, Cs=CS, Am0=0.0):
    m = sim.momentum
    z = CENTERS if u.z else None
    Am = (
        m.uua.grid.array(fill=np.nan, z=z),
        m.uva.grid.array(fill=np.nan, z=z),
        m.vva.grid.array(fill=np.nan, z=z),
    )
    pygetm._pygetm.smagorinsky(u, v, Cs, Am0, *Am)
    return Am


def interior(a, n=3):
    """values away from the closed boundaries (n points), where all neighbours are water"""
    return a.values[..., n:-n, n:-n]


class TestSmagorinsky(unittest.TestCase):
    def test_analytic(self):
        sim = create_simulation()
        a = 1e-4  # s-1
        scale = CS**2 * DX * DX
        cases = (
            # u, v, expected |D|
            ("simple shear", lambda x, y: a * y, lambda x, y: 0.0 * x, a),
            ("pure strain", lambda x, y: a * x, lambda x, y: -a * y, 2.0 * a),
            ("uniform flow", lambda x, y: 0.3 + 0.0 * x, lambda x, y: -0.2 + 0.0 * y, 0.0),
            ("rotation", lambda x, y: -a * (y - 15e3), lambda x, y: a * (x - 20e3), 0.0),
        )
        for name, ufun, vfun, D in cases:
            u, v = velocity(sim, ufun, vfun)
            for Am in smagorinsky(sim, u, v):
                np.testing.assert_allclose(
                    interior(Am), scale * D, rtol=1e-10, atol=1e-12 * scale * a, err_msg=name
                )

    def test_background(self):
        sim = create_simulation()
        u, v = velocity(sim, lambda x, y: 1e-4 * y, lambda x, y: 0.0 * x)
        for Am in smagorinsky(sim, u, v, Cs=0.0, Am0=7.0):
            self.assertTrue((Am.all_values == 7.0).all())
        for Am in smagorinsky(sim, u, v, Cs=CS, Am0=7.0):
            np.testing.assert_allclose(interior(Am), 7.0 + CS**2 * DX * DX * 1e-4, rtol=1e-12)

    def test_layers(self):
        """3D: the same profile in every layer gives the 2D result in every layer"""
        sim = create_simulation(runtype=pygetm.BAROTROPIC_3D, nz=4)
        ufun = lambda x, y: 1e-4 * x + 2e-5 * y
        vfun = lambda x, y: 3e-5 * x - 5e-5 * y
        Am2d = smagorinsky(sim, *velocity(sim, ufun, vfun))
        Am3d = smagorinsky(sim, *velocity(sim, ufun, vfun, z=CENTERS))
        for a2, a3 in zip(Am2d, Am3d):
            for k in range(a3.all_values.shape[0]):
                np.testing.assert_array_equal(a3.all_values[k], a2.all_values)

    def test_constant_unchanged(self):
        """diffusion with Am arrays filled with Am0 is identical to the constant-Am call"""
        sim = create_simulation(Cs=0.0, Am=50.0)
        m = sim.momentum
        rng = np.random.default_rng(1)
        u, v = velocity(
            sim,
            lambda x, y: rng.normal(size=x.shape) * 0.1,
            lambda x, y: rng.normal(size=x.shape) * 0.1,
        )
        u.all_values[sim.U.mask.all_values == 0] = 0.0
        v.all_values[sim.V.mask.all_values == 0] = 0.0
        D = (m.uua.grid.D, m.uva.grid.D, m.vua.grid.D, m.vva.grid.D)
        du1, dv1 = sim.U.array(fill=0.0), sim.V.array(fill=0.0)
        du2, dv2 = sim.U.array(fill=0.0), sim.V.array(fill=0.0)
        pygetm._pygetm.momentum_diffusion(*D, u, v, 50.0, du1, dv1)
        Am = (m.uua.grid.array(fill=50.0), m.uva.grid.array(fill=50.0), m.vva.grid.array(fill=50.0))
        pygetm._pygetm.momentum_diffusion(*D, u, v, 0.0, du2, dv2, *Am)
        np.testing.assert_array_equal(du1.all_values, du2.all_values)
        np.testing.assert_array_equal(dv1.all_values, dv2.all_values)
        self.assertGreater(np.abs(du1.values).max(), 0.0)

    def test_jet(self):
        """a sheared jet in a closed basin: finite, volume unchanged, Smagorinsky
        viscosity largest on the jet's flanks and more energy loss than without"""
        timestep, nstep = 10.0, 200
        ke = {}
        for Cs in (0.0, CS):
            sim = create_simulation(Cs=Cs)
            m = sim.momentum
            yc, w = 15e3, 3e3
            m.U.all_values[...] = np.where(
                sim.U.mask.all_values != 0,
                0.5 * sim.U.D.all_values * np.exp(-(((sim.U.y.all_values - yc) / w) ** 2)),
                0.0,
            )
            vol0 = (sim.T.z * sim.T.area).global_sum(where=sim.T.mask != 0)
            sim.start(datetime.datetime(2000, 1, 1), timestep=timestep, report=0)
            for istep in range(nstep):
                sim.advance()
                if istep == 0 and Cs > 0.0:
                    # along the middle of the basin (away from the walls the jet runs
                    # into later) the shear of the Gaussian jet, and so Am, is largest
                    # on its flanks, at yc +- w / sqrt(2)
                    i = m.Am_uv.values.shape[1] // 2
                    Am = m.Am_uv.values[:, i]
                    y = m.uva.grid.y.values[:, i]
                    for side in (y < yc, y > yc):
                        ymax = y[side][np.argmax(Am[side])]
                        self.assertLess(abs(abs(ymax - yc) - w / np.sqrt(2.0)), 1.5 * DX)
                    self.assertLess(Am[np.argmin(abs(y - yc))], 0.2 * Am.max())  # small at the jet axis
            sim.finish()
            self.assertTrue(np.isfinite(m.U.all_values).all())
            vol = (sim.T.z * sim.T.area).global_sum(where=sim.T.mask != 0)
            self.assertLess(abs(vol - vol0), 1e-6 * sim.T.area.values.sum())
            ke[Cs] = float(((m.U.values**2).sum() + (m.V.values**2).sum()))
        self.assertLess(ke[CS], ke[0.0])


if __name__ == "__main__":
    unittest.main()
