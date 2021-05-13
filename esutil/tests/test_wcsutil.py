import pytest
import numpy as np

from esutil.wcsutil import (
    wrap_ra_diff,
    make_xy_grid,
    Invert2DPolynomial,
    pack_coeffs,
    Apply2DPolynomial,
)


def test_wrap_dra_array():
    dra = np.array([-350, -170, 0, 350, 350 + 360 * 10, -350 - 360 * 10])
    ans = np.array([10, -170, 0, -10, -10, 10])
    assert np.allclose(wrap_ra_diff(dra), ans)

    for _dra, _ans in zip(dra, ans):
        assert np.allclose(wrap_ra_diff(_dra), _ans)


def test_wrap_dra_scalar_nan_inf():
    assert np.isnan(wrap_ra_diff(np.nan))
    assert np.isinf(wrap_ra_diff(np.inf))


def test_wrap_dra_array_nan_inf():
    dra = np.array(
        [np.nan, np.inf, -350, -170, 0, 350, 350 + 360 * 10, -350 - 360 * 10]
    )
    ans = np.array([np.nan, np.inf, 10, -170, 0, -10, -10, 10])
    msk = np.isfinite(dra)
    assert np.allclose(wrap_ra_diff(dra[msk]), ans[msk])
    assert np.isnan(ans[0])
    assert np.isinf(ans[1])


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("constant", [False, True])
def test_invert_2dpoly(inverse, constant):

    # total number of constraints should be at least equal to the
    # number of coeffs.  If constant term is included, ncoeff is
    #  (order+1)*(order+2)//2 < (order+2)^2//2 < (order+2)^2
    # So let's do a lot: 20*(order+2)^2

    porder = 2
    fac = 5
    if porder > 3:
        raise ValueError("Only testing up to order 3 right now")

    # in making the grid we will square this n
    n = 2 * (porder + 2)
    n *= fac

    cen = [500.0, 1000.0]
    u, v = make_xy_grid(n, [1.0, 1000.0], [1.0, 2000.0])
    u -= cen[0]
    v -= cen[1]

    if constant:
        x0 = 2.0
        y0 = 3.0
        start = 0
    else:
        start = 1
        x0 = 0.0
        y0 = 0.0

    ucoeffs_in = np.array(
        [x0, 0.1, 0.2, 0.05, 0.03, 0.04, 0.005, 0.004, 0.001, 0.0009],
        dtype="f8",
    )
    vcoeffs_in = np.array(
        [y0, 0.3, 0.5, 0.06, 0.05, 0.06, 0.004, 0.008, 0.003, 0.002],
        dtype="f8",
    )
    ucoeffs_in = np.array(
        [x0, 1.0, 1.0e-2, 5.0e-3, 3.0e-3, 4.0e-3, 0.000, 0.000, 0.000, 0.0000],
        dtype="f8",
    )
    vcoeffs_in = np.array(
        [y0, 1.0, 2.0e-2, 6.0e-3, 5.5e-3, 4.0e-3, 0.000, 0.000, 0.000, 0.000],
        dtype="f8",
    )

    # number to actuall use
    ncoeff = (porder + 1) * (porder + 2) // 2
    keep = np.arange(start, ncoeff)
    ucoeffs_in = ucoeffs_in[keep]
    vcoeffs_in = vcoeffs_in[keep]

    ain, bin = pack_coeffs(ucoeffs_in, vcoeffs_in, porder, constant=constant)
    x = Apply2DPolynomial(ain, u, v)
    y = Apply2DPolynomial(bin, u, v)

    if not inverse:
        # get poly from u,v to x,y
        ucoeffs, vcoeffs = Invert2DPolynomial(
            u, v, x, y, porder, pack=False, constant=constant
        )
        ucoeffsp, vcoeffsp = Invert2DPolynomial(
            u, v, x, y, porder, pack=True, constant=constant
        )
        newx = Apply2DPolynomial(ucoeffsp, u, v)
        newy = Apply2DPolynomial(vcoeffsp, u, v)

        w, = np.where((np.abs(x) > 5) & (np.abs(y) > 5))

        assert np.allclose(x[w], newx[w])
        assert np.allclose(y[w], newy[w])

    else:
        # Now test the inverse, from x,y to u,v
        # smoke only
        xcoeffs, ycoeffs = Invert2DPolynomial(
            x, y, u, v, porder, pack=False, constant=constant
        )
        xcoeffsp, ycoeffsp = Invert2DPolynomial(
            x, y, u, v, porder, pack=True, constant=constant
        )
        newu = Apply2DPolynomial(xcoeffsp, x, y)
        newv = Apply2DPolynomial(ycoeffsp, x, y)

        print("%s" % u[0:25])
        print("%s" % newu[0:25])

        w, = np.where((np.abs(u) > 5) & (np.abs(v) > 5))
        ufrac = (u[w] - newu[w]) / u[w]
        vfrac = (v[w] - newv[w]) / v[w]

        print("xcoeffs%s" % xcoeffs)
        print("ycoeffs%s\n" % ycoeffs)
        print("median(ufracerr)%s" % np.median(ufrac))
        print("median(abs(ufracerr))%s" % np.median(np.abs(ufrac)))
        print("sdev(ufracerr)%s" % ufrac.std())
        print("median(vfracerr)%s" % np.median(vfrac))
        print("median(abs(vfracerr))%s" % np.median(np.abs(vfrac)))
        print("sdev(vfracerr)%s\n" % vfrac.std())
