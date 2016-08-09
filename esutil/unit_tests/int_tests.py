from __future__ import with_statement, print_function
import sys, os
import tempfile
import warnings
import numpy
from numpy import arange, array

import tempfile
import unittest

from .. import integrate

def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegrate)
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestIntegrate(unittest.TestCase):

    def testCGAuleg(self):
        """
        make sure the gauleg code runs
        """

        self.assertTrue(integrate.have_cgauleg,"have C gauleg code")

        try:
            npts=10
            x,w = integrate.gauleg(1.0, 2.0, npts)
            ok=True
        except:
            ok=False

        self.assertTrue(ok,"gauleg runs")
        self.assertEqual(x.size,npts,"x is right size")
        self.assertEqual(w.size,npts,"w is right size")
