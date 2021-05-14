import numpy as np
import pytest
import esutil as eu

def test_coords_conversions_smoke():
    rng = np.random.RandomState(18)
    num = 10000

    ra, dec = eu.coords.randsphere(num, rng=rng)

    x, y, z = eu.coords.eq2xyz(ra, dec)
    ra_new, dec_new = eu.coords.xyz2eq(x, y, z)

    assert np.allclose(ra, ra_new)
    assert np.allclose(dec, dec_new)

    clambda, ceta = eu.coords.eq2sdss(ra, dec)
    ra_new, dec_new = eu.coords.sdss2eq(clambda, ceta)

    assert np.allclose(ra, ra_new)
    assert np.allclose(dec, dec_new)


@pytest.mark.parametrize('b1950', [False, True])
def test_coords_conversions_smoke_equinox(b1950):
    rng = np.random.RandomState(91)
    num = 10000

    ra, dec = eu.coords.randsphere(num, rng=rng)

    gal_l, gal_b = eu.coords.eq2gal(ra, dec, b1950=b1950)
    ra_new, dec_new = eu.coords.gal2eq(gal_l, gal_b, b1950=b1950)

    assert np.allclose(ra, ra_new)
    assert np.allclose(dec, dec_new)

    lam, beta = eu.coords.eq2ec(ra, dec, b1950=b1950)
    ra_new, dec_new = eu.coords.ec2eq(lam, beta, b1950=b1950)

    assert np.allclose(ra, ra_new)
    assert np.allclose(dec, dec_new)


@pytest.mark.parametrize('ra_cen', [0.0, 20.0, 300.0, 360.0])
@pytest.mark.parametrize('dec_cen', [-90.0, -15.0, 25.0, 90.0])
@pytest.mark.parametrize('radius', [0.01, 0.1])
def test_coords_gcirc(ra_cen, dec_cen, radius):
    rng = np.random.RandomState(91)
    num = 10000
    ra, dec = eu.coords.randcap(
        num, ra_cen, dec_cen, radius, rng=rng,
    )

    dis = eu.coords.sphdist(ra_cen, dec_cen, ra, dec)
    assert np.all(dis < radius)

    dis = eu.coords.gcirc(ra_cen, dec_cen, ra, dec)
    assert np.all(dis < radius)

    dis = eu.coords.sphdist(ra, dec, ra, dec)
    assert np.all(dis == 0)

    dis = eu.coords.gcirc(ra, dec, ra, dec)
    assert np.all(dis == 0)


@pytest.mark.parametrize('ra_range', [None, (0, 360), (200, 220)])
@pytest.mark.parametrize('dec_range', [None, (-90, 90), (18, 25)])
def test_coords_randsphere(ra_range, dec_range):
    rng = np.random.RandomState(781)
    num = 10000
    ra, dec = eu.coords.randsphere(
        num, rng=rng, ra_range=ra_range, dec_range=dec_range,
    )

    if ra_range is None:
        ra_range = [0.0, 360.0]
    if dec_range is None:
        dec_range = [-90.0, 90.0]

    assert ra.min() >= ra_range[0]
    assert ra.max() <= ra_range[1]
    assert dec.min() >= dec_range[0]
    assert dec.max() <= dec_range[1]

    rng = np.random.RandomState(781)
    ranew, decnew = eu.coords.randsphere(
        num, rng=rng, ra_range=ra_range, dec_range=dec_range,
    )
    assert np.all(ra == ranew)
    assert np.all(dec == decnew)


@pytest.mark.parametrize('ra_cen', [0.0, 20.0, 300.0, 360.0])
@pytest.mark.parametrize('dec_cen', [-90.0, -15.0, 25.0, 90.0])
@pytest.mark.parametrize('radius', [0.01, 0.1])
def test_coords_randcap(ra_cen, dec_cen, radius):
    rng = np.random.RandomState(91)
    num = 10000
    ra, dec = eu.coords.randcap(
        num, ra_cen, dec_cen, radius, rng=rng,
    )

    dis = eu.coords.sphdist(ra_cen, dec_cen, ra, dec)
    assert np.all(dis < radius)
