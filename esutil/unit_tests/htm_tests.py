from __future__ import with_statement, print_function
import sys, os
import tempfile
import warnings
import numpy
from numpy import arange, array

import tempfile
import unittest

from .. import htm



def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHTM)
    unittest.TextTestRunner(verbosity=2).run(suite)

class TestHTM(unittest.TestCase):
    def setUp(self):
        self.depth=10

    def testCreate(self):
        try:
            h=htm.HTM(self.depth)
            ok=True
        except:
            ok=False

        self.assertTrue(ok,"creating HTM object with default %d" % self.depth)

        outd = h.get_depth()
        self.assertEqual(outd, self.depth, "checking depth can be gotten")

    def testLookup(self):
 
        try:
            h=htm.HTM(self.depth)
            ok=True
        except:
            ok=False

        self.assertTrue(ok,"creating HTM object")
        if not ok:
            skipTest("cannot test result if HTM object creation fails")

        expected = 10518792
        htmid = h.lookup_id(200.0, 0.0)
        self.assertEqual(htmid, expected, "looking htm id of scalar ra,dec")

        ra = numpy.array([200.0, 115.25])
        dec = numpy.array([0.0, 24.3])
        expected = numpy.array([10518792, 15368378])
        htmid = h.lookup_id(ra,dec)

        self.assertTrue(numpy.all(htmid==expected), "looking htm id of array ra,dec")

    def testIntersect(self):
        try:
            h=htm.HTM(self.depth)
            ok=True
        except:
            ok=False

        self.assertTrue(ok,"creating HTM object")
        if not ok:
            skipTest("cannot test result if HTM object creation fails")

        expected=numpy.array([10518792, 10518793, 10518795, 10518820, 10518823, 10518836,
                              10518839, 14172676, 14172678, 14172679, 14172696, 14172699,
                              14172728, 14172731])
        idlist=h.intersect(200.0, 0.0, 0.1)
        self.assertTrue(numpy.all(idlist==expected), "intersect")

    def testMatch(self):
 
        try:
            h=htm.HTM(self.depth)
            ok=True
        except:
            ok=False

        self.assertTrue(ok,"creating HTM object")

        if not ok:
            skipTest("cannot test result if HTM object creation fails")

        two = 2.0/3600.
        # offset second list by fraction of 2 arcsec in dec
        # not last ones don'e match at all
        ra1 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 21.36])
        dec1 = numpy.array( [24.3,          24.3,            24.3,  -28.25, -15.32])
        # make one of them big endian to check byte swapping
        ra2 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 55.25], dtype='>f8')
        dec2 = numpy.array( [24.3+0.75*two, 24.3 + 0.25*two, 24.3 - 0.33*two, -28.25 + 0.58*two, 75.22])

        for maxmatch, expected in [(0,10),(1,4),(2,7)]:
            m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,two,maxmatch=maxmatch)

            self.assertEqual(
                m1.size,
                expected,
                "expected %d matches with maxmatch=%d, got %d" % (expected,maxmatch,m1.size),
            )


        fname=tempfile.mktemp(suffix='.rec')
        maxmatch=0
        expected=10

        try:
            nmatch = h.match(ra1,dec1,ra2,dec2,two,file=fname,maxmatch=maxmatch)

            self.assertEqual(
                nmatch,
                expected,
                "file write, expected %d matches with maxmatch=%d, got %d" % (expected,maxmatch,nmatch),
            )

            pairs = h.read(fname)
            self.assertEqual(
                pairs.size,
                expected,
                "file read, expected to read %d matches got %d" % (expected,nmatch),
            )


        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def testBincount(self):

        try:
            h=htm.HTM(self.depth)
            ok=True
        except:
            ok=False

        self.assertTrue(ok,"creating HTM object")

        if not ok:
            skipTest("cannot test result if HTM object creation fails")

        two = 2.0/3600.
        ten=10.0/3600.
        eighty=30.0/3600.0
        hundred=100.0/3600.0
        ra1 = numpy.array(  [200.0,         200.0,         200.0,           175.23,  21.36])
        dec1 = numpy.array( [24.3,           24.3,          24.3,           -28.25, -15.32])
        ra2 = numpy.array(  [200.0,         200.0,         200.0,           175.23,  55.25])
        dec2 = numpy.array( [24.3+two, 24.3 + ten, 24.3 - eighty, -28.25 + hundred,  75.22])

        # these must be in degrees unless scale is sent, in which case it is in
        # units of radians*scale
        rmin = 5/3600.
        rmax = 150/3600.
        nbin = 10

        lower,upper,counts = h.bincount(
            rmin,rmax,nbin,
            ra1,dec1,ra2,dec2,
            getbins=True,
        )
        counts_truth = numpy.array([0, 0, 3, 0, 0, 3, 0, 0, 1, 0],dtype='i8')


        wbad, = numpy.where(counts != counts_truth)

        self.assertTrue(numpy.all(counts == counts_truth),
                        "expected %s counts in bincount, got %s" % (counts_truth,counts))


