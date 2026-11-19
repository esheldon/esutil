import pytest
import numpy as np

from esutil.wcsutil import (
    wrap_ra_diff,
    make_xy_grid,
    Invert2DPolynomial,
    pack_coeffs,
    Apply2DPolynomial,
    WCS,
)


TEST_HEADER = """\
{'simple': True, 'bitpix': -32, 'naxis': 2, 'naxis1': 2048, 'naxis2': 4096, 'extend': True, 'gs_xmin': 1, 'gs_ymin': 1, 'gs_wcs': 'GSFitsWCS', 'ctype1': 'RA---TPV', 'ctype2': 'DEC--TPV', 'crpix1': -4617.70016427, 'crpix2': -8609.62886415, 'cd1_1': -1.994451915262e-07, 'cd1_2': 7.289650536209e-05, 'cd2_1': -7.300436190616e-05, 'cd2_2': -3.286398079503e-08, 'cunit1': 'deg', 'cunit2': 'deg', 'crval1': 358.5617798256, 'crval2': -57.81766737102, 'pv1_0': -0.009146940164239, 'pv2_0': 0.002473995478717, 'pv1_1': 1.027766536793, 'pv2_1': 0.9994322877373, 'pv1_2': -0.01398090700451, 'pv2_2': -0.009716336331065, 'pv1_4': -0.02952351234152, 'pv2_4': -0.0193351226434, 'pv1_5': 0.02138414797012, 'pv2_5': -0.01497696747962, 'pv1_6': -0.01585161167923, 'pv2_6': 0.009830021177779, 'pv1_7': 0.01013118530504, 'pv2_7': -0.02110018166535, 'pv1_8': -0.01181821848945, 'pv2_8': -0.006334653583388, 'pv1_9': 0.006892844593728, 'pv2_9': 0.005586126448887, 'pv1_10': -0.007570308721554, 'pv2_10': -0.003747169969, 'extname': 'sci', 'proctype': 'RAW', 'prodtype': 'image', 'pixscal1': 0.27, 'pixscal2': 0.27, 'obs-long': 70.81489, 'telescop': 'CTIO 4.0-m telescope', 'observat': 'CTIO', 'obs-lat': -30.16606, 'obs-elev': 2215.0, 'instrume': 'DECam', 'expreq': 90.0, 'exptime': 90.0, 'darktime': 90.5598199, 'obsid': 'ct4m20161127t032848', 'date-obs': '2016-11-27T03:28:48.037808', 'time-obs': '03:28:48.037808', 'mjd-obs': 57719.14500044, 'openshut': '2016-11-27T03:28:47.537400', 'timesys': 'UTC', 'expnum': 596832, 'object': 'DES survey hex -32-582 tiling 7', 'obstype': 'raw_obj', 'camshut': 'Open', 'program': 'survey', 'observer': 'Agnes Ferte, Yuanyuan Zhan, Stephanie Hamiltong', 'proposer': 'Frieman', 'dtpi': 'Frieman', 'propid': '2012B-0001', 'excluded': '', 'hex': 2882, 'tiling': 7, 'seqid': '-32-582 enqueued on 2016-11-27 03:20:49Z by SurveyTacticianY1', 'seqnum': 1, 'seqtot': 1, 'aos': True, 'bcam': False, 'guider': 1, 'skystat': True, 'filter': 'g DECam SDSS c0001 4720.0 1520.0', 'filtpos': 'cassette_2', 'instance': 'DECam_20161126', 'errors': 'None', 'telequin': 2000.0, 'telstat': 'Track', 'ra': '23:54:11.624', 'dec': '-57:49:10.589', 'telra': '23:54:11.196', 'teldec': '-57:49:08.195', 'ha': '03:15:55.520', 'zd': 43.25, 'az': 215.9592, 'domeaz': 213.74, 'zpdelra': -66.8317, 'zpdeldec': -67.7, 'telfocus': '2757.18,1483.88,2788.46,-95.43,56.97, 0.00', 'vsub': True, 'gskyphot': True, 'lskyphot': False, 'windspd': 12.07, 'winddir': 344.0, 'humidity': 48.0, 'pressure': 778.0, 'dimmsee': 0.702, 'dimm2see': 0.71, 'mass2': 'NaN', 'astig1': 0.02, 'astig2': 0.06, 'outtemp': 10.8, 'airmass': 1.37, 'gskyvar': 0.01, 'gskyhot': 0.001, 'lskyvar': 'NaN', 'lskyhot': 'NaN', 'lskypow': 'NaN', 'msurtemp': 8.825, 'mairtemp': 9.3, 'uptrtemp': 11.02, 'lwtrtemp': 'NaN', 'pmostemp': 8.9, 'utn-temp': 11.265, 'uts-temp': 10.765, 'utw-temp': 10.99, 'ute-temp': 11.06, 'pmn-temp': 8.4, 'pms-temp': 8.6, 'pmw-temp': 9.0, 'pme-temp': 9.3, 'domelow': 'NaN', 'domehigh': 10.8, 'domeflor': 6.4, 'g-meanx': 0.0276, 'g-meany': 0.0552, 'donutfs4': '[1.22,1.55,-8.79,-0.11,-0.22,-0.04,-0.14,-0.02,-0.36,]', 'donutfs3': '[0.33,0.94,9.50,0.00,-0.02,0.09,-0.17,0.08,-0.23,]', 'donutfs2': '[]', 'donutfs1': '[1.92,1.31,8.68,0.29,0.17,0.34,-0.22,0.28,0.12,]', 'g-flxvar': 1456035.199, 'g-meanxy': -0.004535, 'donutfn1': '[0.32,0.16,-8.80,-0.27,-0.01,0.08,0.02,0.21,-0.21,]', 'donutfn2': '[1.72,1.87,8.67,-0.35,-0.06,0.24,0.13,0.23,-0.03,]', 'donutfn3': '[0.62,1.24,-8.69,0.11,-0.13,0.02,0.08,0.01,0.36,]', 'donutfn4': '[2.02,1.72,8.66,0.10,-0.26,0.13,0.13,-0.03,0.22,]', 'time_recorded': '2016-11-27T03:30:40.736263', 'g-feedbk': '10, 5', 'g-ccdnum': 4, 'doxt': 0.07, 'g-maxx': 0.3378, 'fadz': -13.18, 'fady': -39.02, 'fadx': -228.46, 'g-mode': 'auto', 'fayt': -13.18, 'dodz': -13.18, 'dody': -1.17, 'dodx': -1.25, 'multiexp': False, 'skyupdat': '2016-11-27T03:28:17', 'g-seeing': 1.808, 'g-transp': 0.555, 'g-meany2': 0.010972, 'doyt': 0.03, 'g-latenc': 1.289, 'lutver': 'v20160516', 'faxt': -6.56, 'g-maxy': 0.298, 'g-meanx2': 0.011114, 'sispiver': 'trunk', 'constver': 'DECAM:55', 'hdrver': '13', 'dtsite': 'ct', 'dttelesc': 'ct4m', 'dtinstru': 'decam', 'dtcaldat': '2016-11-26', 'odateobs': '', 'dtutc': '2016-11-27T03:31:08', 'dtobserv': 'NOAO', 'dtpropid': '2012B-0001', 'dtpiaffl': '', 'dttitle': '', 'dtcopyri': 'AURA', 'dtacquis': 'pipeline3.ctio.noao.edu', 'dtaccoun': 'sispi', 'dtacqnam': '/data_local/images/DTS/2012B-0001/DECam_00596832.fits.fz', 'dtnsanam': 'c4d_161127_033108_ori.fits', 'dt_rtnam': 'c4d_161127_033108_ori', 'dtqueue': 'des', 'dtstatus': 'done', 'sb_host': 'pipeline3.ctio.noao.edu', 'sb_accou': 'sispi', 'sb_site': 'ct', 'sb_local': 'dec', 'sb_dir1': '20161126', 'sb_dir2': 'ct4m', 'sb_dir3': '2012B-0001', 'sb_recno': 456430, 'sb_id': 'dec456430', 'sb_name': 'c4d_161127_033108_ori.fits', 'sb_rtnam': 'c4d_161127_033108_ori', 'rmcount': 0, 'recno': 456430, 'bunit': 'electrons', 'wcsaxes': 2, 'detsize': '[1:29400,1:29050]', 'datasec': '[1:2048,1:4096]', 'detsec': '[18433:20480,22528:26623]', 'ccdsec': '[1:2048,1:4096]', 'detseca': '[18433:19456,22528:26623]', 'ccdseca': '[1:1024,1:4096]', 'ampseca': '[1:1024,1:4096]', 'dataseca': '[1:1024,1:4096]', 'detsecb': '[19457:20480,22528:26623]', 'ccdsecb': '[1025:2048,1:4096]', 'ampsecb': '[2048:1025,1:4096]', 'datasecb': '[1025:2048,1:4096]', 'detector': 'S3-235_135959-22-2', 'ccdnum': 50, 'detpos': 'N19', 'gaina': 1.0471003469739, 'rdnoisea': 5.806, 'saturata': 175948.657196455, 'gainb': 1.04612227936184, 'rdnoiseb': 6.114, 'saturatb': 161091.045783673, 'fpa': 'DECAM_BKP5', 'ccdbin1': 1, 'ccdbin2': 1, 'dheinf': 'MNSN fermi hardware', 'dhefirm': 'demo30', 'slot00': 'MCB 7 5.210000', 'slot01': 'DESCB 23 4.010000', 'slot02': 'DESCB 10 4.010000', 'slot03': 'CCD12 3 4.080000', 'slot04': 'CCD12 23 4.080000', 'slot05': 'CCD12 13 4.080000', 'radesys': 'ICRS', 'equinox': 2000.0, 'ltm2_2': 1.0, 'ltm2_1': 0.0, 'pv2_3': 0.0, 'ltm1_1': 1.0, 'pv1_3': 0.0, 'ltm1_2': 0.0, 'ltv2': 0.0, 'ltv1': 0.0, 'valida': True, 'validb': True, 'ndonuts': 0, 'photflag': 1, 'desdcxtk': 'Sat Oct 13 16:24:19 2018', 'xtalkfil': 'DECam_20130606.xtalk', 'desoscn': 'Sat Oct 13 16:24:19 2018', 'fzalgor': 'RICE_1', 'fzqmethd': 'SUBTRACTIVE_DITHER_2', 'fzqvalue': 16, 'fzdthrsd': 'CHECKSUM', 'band': 'g', 'camsym': 'D', 'nite': '20161126', 'desfname': 'D00596832_g_c50_r3630p01_immasked.fits', 'pipeline': 'finalcut', 'unitname': 'D00596832', 'attnum': 1, 'eupsprod': 'finalcut', 'eupsver': 'Y5A1+3', 'reqnum': 3630, 'biasfil': 'D_n20160921t1003_c50_r2877p05_biascor.fits', 'desbias': 'Sat Oct 13 16:34:54 2018', 'lincfil': 'lin_tbl_v0.4.fits', 'deslinc': 'Sat Oct 13 16:34:57 2018', 'dessat': 'Sat Oct 13 16:34:57 2018', 'nsatpix': 323, 'desbpm': 'Sat Oct 13 16:34:57 2018', 'bpmfil': 'D_n20160921t1003_c50_r2901p03_bpm.fits', 'flatmeda': 1.0471003469739, 'flatmedb': 1.04612227936184, 'saturate': 175948.657196455, 'desgainc': 'Sat Oct 13 16:34:57 2018', 'fixcfil': 'D_n20160921t1003_c50_r2901p03_bpm.fits', 'desfixc': 'Sat Oct 13 16:34:57 2018', 'bfcfil': 'D_n20170406_r2959p01_bf.fits', 'desbfc': 'Sat Oct 13 16:35:00 2018', 'flatfil': 'D_n20160921t1003_g_c50_r2877p05_norm-dflatcor.fits', 'desflat': 'Sat Oct 13 16:35:00 2018', 'ra_cent': 0.0347611268912457, 'dec_cent': -58.2213049493389, 'rac1': 359.749291645411, 'decc1': -58.1495089297263, 'rac2': 359.753406915197, 'decc2': -58.2990136873104, 'rac3': 0.321503005309369, 'decc3': -58.2925345335529, 'rac4': 0.314815880180505, 'decc4': -58.1431129833383, 'racmin': 359.749291645411, 'racmax': 0.321503005309353, 'deccmin': -58.2990136873104, 'deccmax': -58.1431129833383, 'crossra0': 'Y', 'fwhm': 4.0811, 'scampchi': 11.9157, 'elliptic': 0.0738, 'scampnum': 3317, 'scampref': 'GAIA-DR2', 'desepoch': 'Y4E1.5', 'desbleed': 'Sat Oct 13 16:59:27 2018', 'nbleed': 1305, 'starmask': 'Sat Oct 13 16:59:27 2018', 'des_ext': 'IMAGE', 'skysbfil': 'Y4A1_20160801t1215_g_c50_r2930p02_skypca-tmpl.fits', 'skypc00': 428.925098953531, 'skypc01': -2.10815630785595, 'skypc02': -1.0848112878144, 'skypc03': 0.309499447822962, 'skyvara': 428.656870617396, 'skyvarb': 430.854771365326, 'skysigma': 20.7307280537364, 'skybrite': 416.442384289468, 'desskysb': 'Sat Oct 13 17:12:37 2018', 'starfil': 'Y4A1_20160801t1215_g_c50_r2931p01_starflat.fits', 'desstar': 'Sat Oct 13 17:19:22 2018', 'desncray': 84, 'desnstrk': 0, 'desimmsk': 'Sat Oct 13 17:33:56 2018', 'descncts': 'Sat Oct 13 17:36:36 2018 Mask 0 new streaks', 'zdither0': 9289, 'checksum': 'kJA9nH67kHA7kH37', 'datasum': '2533260435'}
"""  # noqa


def test_wcs_invert():
    hdr = eval(TEST_HEADER)
    wcs = WCS(hdr)
    ra, dec = 0.00010227173946750857, -58.251664526579034
    np.testing.assert_allclose(
        wcs.sky2image(ra, dec),
        (1434.318593617967, 1790.0870452056042),
    )


def test_wcsutil_wrap_dra_array():
    dra = np.array([-350, -170, 0, 350, 350 + 360 * 10, -350 - 360 * 10])
    ans = np.array([10, -170, 0, -10, -10, 10])
    assert np.allclose(wrap_ra_diff(dra), ans)

    for _dra, _ans in zip(dra, ans):
        assert np.allclose(wrap_ra_diff(_dra), _ans)


def test_wcsutil_wrap_dra_scalar_nan_inf():
    assert np.isnan(wrap_ra_diff(np.nan))
    assert np.isinf(wrap_ra_diff(np.inf))


def test_wcsutil_wrap_dra_array_nan_inf():
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
def test_wcsutil_invert_2dpoly(inverse, constant):

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
