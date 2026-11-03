import numpy as np
import pytest

from pygetm.nesting import vremap as vr

rng = np.random.default_rng(1)


def columns(n, nz, D):
    h = rng.uniform(0.2, 1.0, (n, nz))
    return h * (np.asarray(D, float)[..., None] / h.sum(-1, keepdims=True))


def test_remap_conserves_and_is_monotone():
    D = rng.uniform(5, 50, 200)
    hs, hd = columns(200, 30, D), columns(200, 17, D)
    c = rng.normal(size=(200, 30))
    cd = vr.remap(hs, c, hd)
    np.testing.assert_allclose((cd * hd).sum(-1), (c * hs).sum(-1), rtol=1e-12, atol=1e-12)
    assert (cd.max(-1) <= c.max(-1) + 1e-12).all() and (cd.min(-1) >= c.min(-1) - 1e-12).all()


def test_remap_identity_and_uniform():
    h = columns(50, 20, rng.uniform(5, 50, 50))
    c = rng.normal(size=(50, 20))
    np.testing.assert_allclose(vr.remap(h, c, h), c, rtol=1e-12, atol=1e-12)
    hd = columns(50, 7, h.sum(-1))
    np.testing.assert_allclose(vr.remap(h, np.full_like(c, 3.0), hd), 3.0, rtol=1e-12)


def test_shared_prepare_many_tracers():
    D = rng.uniform(5, 50, 40)
    hs, hd = columns(40, 25, D), columns(40, 10, D)
    c = rng.normal(size=(6, 40, 25))
    prep = vr.prepare(vr.interfaces(hs), vr.interfaces(hd))
    many = vr.apply(prep, c, hs) / hd
    for k in range(6):
        np.testing.assert_array_equal(many[k], vr.apply(prep, c[k], hs) / hd)


@pytest.mark.parametrize("keep", [0.0, 0.8])
def test_rim_conserves(keep):
    n, m, nzc, nzp = 30, 9, 30, 20
    Dc = rng.uniform(10, 45, (n, m))
    a = rng.uniform(0.9, 1.1, (n, m))
    a[:, 0] = 0.0  # one land column
    w = a / a.sum(-1, keepdims=True)
    Dp = (w * Dc).sum(-1)  # consistent depths
    hc, hp = columns(n * m, nzc, Dc.ravel()).reshape(n, m, nzc), columns(n, nzp, Dp)
    c = rng.normal(size=(2, n, m, nzc))
    c2 = c ** 2  # positive, for the multiplicative fix
    content = lambda x: (w[..., None] * x * hc).sum((-1, -2))
    sh = vr.rim_columns_prepare(hc, a, hp, keep)
    for x, cp in ((c, vr.rim_average_apply(vr.rim_average_prepare(hc, a, hp), c, hc, hp)),
                  (c, vr.rim_columns_apply(sh, c, hc, hp)),
                  (c, vr.rim_columns_apply(sh, c, hc, hp, by_volume=True, fix="add")),
                  (c2, vr.rim_columns_apply(sh, c2, hc, hp, by_volume=True, fix="scale"))):
        np.testing.assert_allclose((cp * hp).sum(-1), content(x), rtol=1e-12)
    # a uniform field stays uniform: by volume (B', B'') always; by content (B) with the linear stretch
    u = np.full((n, m, nzc), 2.5)
    for kw in (dict(by_volume=True), dict(by_volume=True, fix="add"), dict(by_volume=True, fix="scale")):
        np.testing.assert_allclose(vr.rim_columns_apply(sh, u, hc, hp, **kw), 2.5, rtol=1e-12)
    if keep == 0.0:
        np.testing.assert_allclose(vr.rim_columns_apply(sh, u, hc, hp), 2.5, rtol=1e-12)
        np.testing.assert_allclose(sh[2], hp, rtol=1e-12)


def test_sigma_child_a_equals_b():
    n, m, nzc, nzp = 10, 9, 30, 20
    Dc = rng.uniform(10, 45, (n, m))
    a = np.ones((n, m))
    Dp = Dc.mean(-1)
    sig = rng.uniform(0.5, 1.0, nzc)
    sig /= sig.sum()
    hc, hp = Dc[..., None] * sig, columns(n, nzp, Dp)
    c = rng.normal(size=(n, m, nzc))
    A = vr.rim_average_apply(vr.rim_average_prepare(hc, a, hp), c, hc, hp)
    B = vr.rim_columns_apply(vr.rim_columns_prepare(hc, a, hp), c, hc, hp)
    np.testing.assert_allclose(A, B, rtol=1e-11, atol=1e-12)


def test_rim_uniform_with_volume_mismatch():
    """NestedTracers3D's rim: child columns whose total volume differs from the
    parent cell's (elevation not yet synchronised) keep a uniform field uniform
    when concentrations are formed by the child volume."""
    n, m, nzc, nzp = 50, 9, 20, 10
    Dp = rng.uniform(5.0, 50.0, n)
    Dc = Dp[:, None] * rng.uniform(0.6, 1.4, (n, m)) + rng.uniform(-0.3, 0.3, (n, m))
    h_c = rng.uniform(0.2, 1.0, (n, m, nzc))
    h_c *= (Dc / h_c.sum(-1))[..., None]
    h_p = columns(n, nzp, Dp)
    a = rng.uniform(0.5, 1.0, (n, m))
    for keep in (0.0, 0.8):
        shared = vr.rim_columns_prepare(h_c, a, h_p, keep=keep)
        c = vr.rim_columns_apply(shared, np.full((2, n, m, nzc), 7.5), h_c, h_p, by_volume=True)
        assert np.abs(c - 7.5).max() < 1e-12


def test_ring_uniform():
    """NestedTracers3D's ring: one parent column onto a child column of another
    depth and layer count keeps a uniform field uniform."""
    n, nzp, nzc = 40, 10, 25
    Dp = rng.uniform(5.0, 50.0, n)
    hs = columns(n, nzp, Dp)
    hd = columns(n, nzc, Dp * rng.uniform(0.7, 1.3, n))
    shared = vr.rim_columns_prepare(hs[:, None, :], np.ones((n, 1)), hd, keep=0.8)
    c = vr.rim_columns_apply(shared, np.full((1, n, 1, nzp), 3.0), hs[:, None, :], hd, by_volume=True)
    assert np.abs(c - 3.0).max() < 1e-12
