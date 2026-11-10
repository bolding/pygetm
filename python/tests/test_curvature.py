"""Curvilinear grids: the curvature term of the momentum equations (legacy GETM's
cord_curv), its default per grid type, exact cell areas and legacy grid_type 3."""

import datetime
import os
import tempfile
import unittest

import netCDF4
import numpy as np

import pygetm
import pygetm.legacy

G = 9.81


def polar_supergrid(nx, ny, r0, r1):
    """Supergrid of an annulus: i clockwise, j outward (right-handed)."""
    ii, jj = np.meshgrid(np.arange(2 * nx + 1) / 2.0, np.arange(2 * ny + 1) / 2.0)
    th = -ii * 2 * np.pi / nx
    r = r0 + jj * (r1 - r0) / ny
    return r * np.cos(th), r * np.sin(th)


def annulus_error(curvature, hours=2.0, dt=20.0):
    """Largest |z - z_exact| (m) of cyclostrophic flow (u = 1 m/s, f = 0, no
    friction) in an annulus, which is analytically steady."""
    nx, ny, r0, r1, H, u0 = 72, 20, 50e3, 150e3, 50.0, 1.0
    x, y = polar_supergrid(nx, ny, r0, r1)
    domain = pygetm.domain.Domain(nx, ny, x=x, y=y, H=H, f=0.0, periodic_x=True)
    sim = pygetm.Simulation(
        domain,
        runtype=pygetm.RunType.BAROTROPIC_2D,
        momentum=pygetm.momentum.Momentum(curvature=curvature),
        airsea=pygetm.airsea.Fluxes(taux=0.0, tauy=0.0),
        log_level="ERROR",
    )
    sim.momentum.apply_bottom_friction = False
    T, U = sim.T, sim.U
    rT = np.hypot(T.x.all_values, T.y.all_values)
    rU = np.hypot(U.x.all_values, U.y.all_values)
    z0 = u0**2 / G * np.log(rT / 100e3)
    wet = T.mask.all_values > 0
    T.z.all_values[wet] = z0[wet]
    mu = U.mask.all_values > 0
    sim.momentum.U.all_values[mu] = (u0 * (H + u0**2 / G * np.log(rU / 100e3)))[mu]
    sim.start(datetime.datetime(2000, 1, 1), timestep=dt, report=0)
    for _ in range(int(hours * 3600 / dt)):
        sim.advance()
    h = T.halox
    err = np.abs(T.z.all_values - z0)[h:-h, h:-h][wet[h:-h, h:-h]].max()
    sim.finish()
    return err, sim


def curvature_used(domain, curvature=None):
    sim = pygetm.Simulation(
        domain,
        runtype=pygetm.RunType.BAROTROPIC_2D,
        momentum=pygetm.momentum.Momentum(curvature=curvature),
        airsea=pygetm.airsea.Fluxes(taux=0.0, tauy=0.0),
        log_level="ERROR",
    )
    return sim.momentum._curv is not None


class TestCurvature(unittest.TestCase):
    def test_annulus(self):
        err_on, _ = annulus_error(None)
        err_off, _ = annulus_error(False)
        # without the term the flow is not steady (about 0.06 m after 2 h); with it
        # the error is set by the spatial discretisation (about 1e-4 m)
        self.assertGreater(err_off, 0.03)
        self.assertLess(err_on, 1e-3)

    def test_defaults(self):
        # regular Cartesian: the term is zero, not used
        d = pygetm.domain.create_cartesian(np.arange(11) * 1e3, np.arange(9) * 1e3, H=10.0, f=1e-4)
        self.assertFalse(curvature_used(d))
        # regular lon/lat: off by default (as upstream), on if asked
        lon, lat = np.linspace(0.0, 2.0, 11), np.linspace(54.0, 55.0, 9)
        self.assertFalse(curvature_used(pygetm.domain.create_spherical(lon, lat, H=10.0)))
        self.assertTrue(curvature_used(pygetm.domain.create_spherical(lon, lat, H=10.0), True))
        self.assertFalse(curvature_used(pygetm.domain.create_spherical(lon, lat, H=10.0), False))
        # curvilinear (polar): on by default
        x, y = polar_supergrid(36, 10, 50e3, 150e3)
        self.assertTrue(curvature_used(pygetm.domain.Domain(36, 10, x=x, y=y, H=10.0, f=0.0, periodic_x=True)))

    def test_exact_area(self):
        nx, ny, r0, r1 = 36, 10, 50e3, 150e3
        x, y = polar_supergrid(nx, ny, r0, r1)
        d = pygetm.domain.Domain(nx, ny, x=x, y=y, H=10.0, f=0.0, periodic_x=True, exact_area=True)
        sim = pygetm.Simulation(d, runtype=pygetm.RunType.BAROTROPIC_2D, airsea=pygetm.airsea.Fluxes(taux=0.0, tauy=0.0), log_level="ERROR")
        h = sim.T.halox
        area = sim.T.area.all_values[h:-h, h:-h]
        ra = r0 + np.arange(ny) * (r1 - r0) / ny
        rb = ra + (r1 - r0) / ny
        quad = 0.5 * np.sin(2 * np.pi / nx) * (rb**2 - ra**2)  # the corner quadrilaterals
        np.testing.assert_allclose(area, np.broadcast_to(quad[:, None], area.shape), rtol=1e-12)

    def test_legacy_grid_type_3(self):
        nx, ny, r0, r1 = 36, 10, 50e3, 150e3
        ic, jc = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
        th = -ic * 2 * np.pi / nx
        r = r0 + jc * (r1 - r0) / ny
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "topo.nc")
            with netCDF4.Dataset(path, "w") as nc:
                nc.createDimension("x", nx)
                nc.createDimension("y", ny)
                nc.createDimension("xx", nx + 1)
                nc.createDimension("yx", ny + 1)
                nc.createVariable("grid_type", "i4")[...] = 3
                nc.createVariable("bathymetry", "f8", ("y", "x"))[...] = 50.0
                nc.createVariable("xx", "f8", ("yx", "xx"))[...] = r * np.cos(th)
                nc.createVariable("yx", "f8", ("yx", "xx"))[...] = r * np.sin(th)
            d = pygetm.legacy.domain_from_topo(path, z0=0.001, f=0.0)
            sim = pygetm.Simulation(d, runtype=pygetm.RunType.BAROTROPIC_2D, airsea=pygetm.airsea.Fluxes(taux=0.0, tauy=0.0), log_level="ERROR")
        h = sim.T.halox
        dx = sim.T.dx.all_values[h:-h, h:-h]
        rc = r0 + (np.arange(ny) + 0.5) * (r1 - r0) / ny
        chord = 2 * rc * np.sin(np.pi / nx)  # between the U points, halfway between corners
        np.testing.assert_allclose(dx, np.broadcast_to(chord[:, None], dx.shape), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
