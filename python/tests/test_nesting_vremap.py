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
    """Nested3D's rim: child columns whose total volume differs from the
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
    """Nested3D's ring: one parent column onto a child column of another
    depth and layer count keeps a uniform field uniform."""
    n, nzp, nzc = 40, 10, 25
    Dp = rng.uniform(5.0, 50.0, n)
    hs = columns(n, nzp, Dp)
    hd = columns(n, nzc, Dp * rng.uniform(0.7, 1.3, n))
    shared = vr.rim_columns_prepare(hs[:, None, :], np.ones((n, 1)), hd, keep=0.8)
    c = vr.rim_columns_apply(shared, np.full((1, n, 1, nzp), 3.0), hs[:, None, :], hd, by_volume=True)
    assert np.abs(c - 3.0).max() < 1e-12


def test_ppm_conserves_and_is_monotone():
    h = rng.uniform(0.2, 3.0, (300, 17))
    c = rng.normal(size=(300, 17))
    hd = rng.uniform(0.1, 2.0, (300, 23))
    hd *= (h.sum(-1) / hd.sum(-1))[:, None]
    out = vr.remap(h, c, hd, "ppm")
    assert np.abs((out * hd).sum(-1) - (c * h).sum(-1)).max() < 1e-12
    assert (out.max(-1) - c.max(-1)).max() < 1e-12 and (c.min(-1) - out.min(-1)).max() < 1e-12
    assert np.abs(vr.remap(h, np.full_like(c, 4.2), hd, "ppm") - 4.2).max() < 1e-12


def test_ppm_more_accurate_than_pcm():
    """Layer means of a thin pycnocline (tanh) from 10 onto 20 layers in 35 m."""
    lnc = lambda x: np.abs(x) + np.log1p(np.exp(-2 * np.abs(x))) - np.log(2)
    means = lambda z: 12 - 3 * np.diff(3 * lnc((z - 15) / 3), axis=-1) / np.diff(z, axis=-1)
    hs, hd = np.full((1, 10), 3.5), np.full((1, 20), 1.75)
    ref = means(vr.interfaces(hd))
    e_pcm = np.abs(vr.remap(hs, means(vr.interfaces(hs)), hd, "pcm") - ref).max()
    e_ppm = np.abs(vr.remap(hs, means(vr.interfaces(hs)), hd, "ppm") - ref).max()
    assert e_ppm < 0.2 * e_pcm


@pytest.mark.parametrize("scheme", ["pcm", "ppm"])
def test_compiled_kernel_matches_numpy(scheme):
    """vremap.contents (compiled kernel) against prepare + apply (numpy)."""
    pytest.importorskip("pygetm._pygetm")
    from pygetm import _pygetm

    if not hasattr(_pygetm, "vertical_remap"):
        pytest.skip("kernel not built")
    n, ns, nd = 400, 13, 21
    hs = rng.uniform(0.0, 2.0, (n, ns))
    hs[rng.uniform(size=hs.shape) < 0.1] = 0.0  # empty layers
    hs[:, 0] += 0.1
    zs = vr.interfaces(hs)
    hd = rng.uniform(0.1, 2.0, (n, nd))
    zd = vr.interfaces(hd) * (zs[:, -1] / vr.interfaces(hd)[:, -1])[:, None]
    zd[:50] = zs[:50, :1] + np.linspace(0, 1, nd + 1)[None] * zs[:50, -1:] * 1.3  # beyond the source
    zd[50:100, : ns + 1] = zs[50:100]  # coinciding interfaces
    zd[50:100, ns + 1:] = zs[50:100, -1:]
    zd[50:100] = np.maximum.accumulate(zd[50:100], axis=1)
    c = rng.normal(size=(3, n, ns))
    ref = vr.apply(vr.prepare(zs, zd), c, hs, scheme)
    out = vr.contents(zs, zd, c, hs, scheme)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-11


def test_gather_scatter_columns():
    arr = rng.normal(size=(12, 9, 11))
    jj = rng.integers(0, 9, 40)
    ii = rng.integers(0, 11, 40)
    out = vr.gather_columns(arr, jj, ii)
    assert np.array_equal(out, arr[::-1, jj, ii].T)
    dst = arr.copy()
    sel = np.unique(jj * 11 + ii, return_index=True)[1]  # distinct cells
    vals = rng.normal(size=(sel.size, 12))
    vr.scatter_columns(dst, jj[sel], ii[sel], vals)
    ref = arr.copy()
    ref[::-1, jj[sel], ii[sel]] = vals.T
    assert np.array_equal(dst, ref)


def test_numpy_switch(monkeypatch):
    """PYGETM_NESTING_REMAP=numpy: the pure numpy path, the same results."""
    zs = vr.interfaces(rng.uniform(0.5, 2.0, (30, 8)))
    zd = vr.interfaces(rng.uniform(0.5, 2.0, (30, 12)))
    zd *= (zs[:, -1] / zd[:, -1])[:, None]
    c = rng.normal(size=(2, 30, 8))
    h = np.diff(zs, axis=-1)
    a = vr.contents(zs, zd, c, h, "ppm")
    monkeypatch.setenv("PYGETM_NESTING_REMAP", "numpy")
    assert vr._kernel("vertical_remap") is None
    b = vr.contents(zs, zd, c, h, "ppm")
    assert np.abs(a - b).max() < 1e-12


@pytest.mark.parametrize("compiled", [False, True])
def test_ppm_linear_exact_in_second_layers(compiled, monkeypatch):
    # a linear profile on uniform layers, the interfaces shifted by a fraction of a layer:
    # with the edges next to the outermost layers from the two layer means, the parabola
    # is exact in the second layers too (IDEAL V2: a pycnocline in the bottom layers)
    if not compiled:
        monkeypatch.setenv("PYGETM_NESTING_REMAP", "numpy")
    nz, D = 15, 20.0
    hs = np.full((1, nz), D / nz)
    zs = np.concatenate([[0.0], np.cumsum(hs[0])])
    c = (0.5 * (zs[1:] + zs[:-1]) * 0.3 + 1.0)[None]  # layer means of 1 + 0.3 z
    zd = zs + 0.25 * D / nz  # every interface shifted down by a quarter layer
    zd[0], zd[-1] = 0.0, D
    out = vr.contents(zs[None], zd[None], c[None], hs, scheme="ppm")[0, 0] / np.diff(zd)
    exact = 1.0 + 0.3 * 0.5 * (zd[1:] + zd[:-1])
    # target layers 1 .. nz-3 lie within source layers 1 .. nz-2 (not the outermost ones)
    np.testing.assert_allclose(out[1:-2], exact[1:-2], rtol=0, atol=1e-12)


def test_macro_ratios():
    """3D sub-stepping: a child's 3D step is its parent's divided by a whole number."""
    from pygetm.nesting.nested3d import macro_ratios

    # the shared 3D step (split = parent's x time ratio), then the strait's own (a third)
    shared = {"c": (None, 180.0, 10, 1), "s": ("c", 180.0, 10, 1), "t": ("s", 180.0, 30, 3)}
    assert macro_ratios(shared) == {"s": 1, "t": 1}
    own = {"c": (None, 180.0, 10, 1), "s": ("c", 180.0, 10, 1), "t": ("s", 60.0, 10, 3)}
    assert macro_ratios(own) == {"s": 1, "t": 3}
    # a chain: both levels sub-step
    chain = {"c": (None, 180.0, 10, 1), "s": ("c", 90.0, 5, 1), "t": ("s", 30.0, 5, 3)}
    assert macro_ratios(chain) == {"s": 2, "t": 3}
    for bad in ({"c": (None, 180.0, 10, 1), "t": ("c", 72.0, 12, 3)},  # 30 / 12: no whole number
                {"c": (None, 180.0, 10, 1), "t": ("c", 360.0, 60, 3)},  # longer than the parent's
                {"c": (None, 180.0, 10, 1), "t": ("c", 50.0, 10, 3)}):  # time steps that do not fit
        with pytest.raises(ValueError):
            macro_ratios(bad)
