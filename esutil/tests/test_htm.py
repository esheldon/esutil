import os
import numpy as np
import tempfile
import pickle

import esutil as eu


def test_hist_smoke():
    depth = 10
    h = eu.htm.HTM(depth)

    expected = 10518792
    htmid = h.lookup_id(200.0, 0.0)
    assert htmid == expected

    ra = np.array([200.0, 115.25])
    dec = np.array([0.0, 24.3])
    expected = np.array([10518792, 15368378])
    htmid = h.lookup_id(ra, dec)

    assert np.all(htmid == expected)


def test_intersect():
    depth = 10
    h = eu.htm.HTM(depth)

    expected = np.array(
        [10518792, 10518793, 10518795, 10518820, 10518823, 10518836,
         10518839, 14172676, 14172678, 14172679, 14172696, 14172699,
         14172728, 14172731],
    )
    idlist = h.intersect(200.0, 0.0, 0.1)
    assert np.all(idlist == expected)


def test_match():
    depth = 10
    h = eu.htm.HTM(depth)

    two = 2.0/3600.

    # offset second list by fraction of 2 arcsec in dec
    # not last ones don'e match at all

    ra1 = np.array([200.0, 200.0, 200.0, 175.23, 21.36])
    dec1 = np.array([24.3, 24.3, 24.3, -28.25, -15.32])

    # make one of them big endian to check byte swapping

    ra2 = np.array([200.0, 200.0, 200.0, 175.23, 55.25], dtype='>f8')
    dec2 = np.array([24.3+0.75*two, 24.3 + 0.25*two, 24.3 - 0.33*two, -28.25 + 0.58*two, 75.22])  # noqa

    for maxmatch, expected in [(0, 10), (1, 4), (2, 7)]:
        m1, m2, d12 = h.match(
            ra1, dec1, ra2, dec2, two, maxmatch=maxmatch,
        )

        assert m1.size == expected

    fname = tempfile.mktemp(suffix='.rec')
    maxmatch = 0
    expected = 10

    with tempfile.TemporaryDirectory() as tdir:
        fname = os.path.join(tdir, 'test.rec')

        nmatch = h.match(
            ra1, dec1, ra2, dec2, two, file=fname, maxmatch=maxmatch,
        )

        assert nmatch == expected

        pairs = h.read(fname)
        assert pairs.size == expected


def test_bincount():

    depth = 10
    h = eu.htm.HTM(depth)

    two = 2.0/3600.
    ten = 10.0/3600.
    eighty = 30.0/3600.0
    hundred = 100.0/3600.0
    ra1 = np.array([200.0, 200.0, 200.0, 175.23,  21.36])
    dec1 = np.array([24.3, 24.3, 24.3, -28.25, -15.32])
    ra2 = np.array([200.0, 200.0, 200.0, 175.23,  55.25])
    dec2 = np.array([24.3+two, 24.3 + ten, 24.3 - eighty, -28.25 + hundred,  75.22])  # noqa

    # these must be in degrees unless scale is sent, in which case it is in
    # units of radians*scale
    rmin = 5/3600.
    rmax = 150/3600.
    nbin = 10

    lower, upper, counts = h.bincount(
        rmin, rmax, nbin,
        ra1, dec1, ra2, dec2,
        getbins=True,
    )
    counts_truth = np.array([0, 0, 3, 0, 0, 3, 0, 0, 1, 0], dtype='i8')

    assert np.all(counts == counts_truth)

    # now radii in Mpc
    dmpc = 412.0  # Mpc

    rmin = 0.01  # 10 kpc
    rmax = 0.3  # 300 kpc
    nbin = 10

    lower, upper, counts = h.bincount(
        rmin, rmax, nbin,
        ra1, dec1, ra2, dec2,
        getbins=True,
        scale=dmpc,
    )

    assert np.all(counts == counts_truth)

    dmpc = np.array([dmpc]*ra1.size)
    lower, upper, counts = h.bincount(
        rmin, rmax, nbin,
        ra1, dec1, ra2, dec2,
        getbins=True,
        scale=dmpc,
    )

    assert np.all(counts == counts_truth)


def test_pickle():
    h = eu.htm.HTM(5)
    pickled = pickle.dumps(h)
    result = pickle.loads(pickled)
    assert h.get_depth() == result.get_depth()
