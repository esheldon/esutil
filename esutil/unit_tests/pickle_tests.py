from __future__ import with_statement, print_function
import sys, os
import tempfile
import warnings

from ..cosmology import Cosmo
import tempfile
import pickle

import unittest


def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPickleCosmo)
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestPickleCosmo(unittest.TestCase):

    def _pickle_compare(self, cosmo):
        with tempfile.TemporaryDirectory() as temp_dir:
            fn = os.path.join(temp_dir, 'cosmo.pkl')
            with open(fn, 'wb') as f:
                pickle.dump(cosmo, f)
            with open(fn, 'rb') as f:
                _cosmo = pickle.load(f)

            self.assertEqual(cosmo.DH(), _cosmo.DH())
            self.assertEqual(cosmo.flat(), _cosmo.flat())
            self.assertEqual(cosmo.omega_m(), _cosmo.omega_m())
            self.assertEqual(cosmo.omega_l(), _cosmo.omega_l())
            self.assertEqual(cosmo.omega_k(), _cosmo.omega_k())

    def test_pickle_compare1(self):
        """
        Test case 1
        """
        cosmo = Cosmo(h=None, H0=0.7, flat=True, omega_m=0.27, omega_l=0.73,
                      omega_k=0.0)
        self._pickle_compare(cosmo=cosmo)

    def test_pickle_compare2(self):
        """
        Test case 2
        """
        cosmo = Cosmo(h=None, H0=0.67, flat=False, omega_m=0.32, omega_l=0.67,
                      omega_k=0.01)
        self._pickle_compare(cosmo=cosmo)

    def test_pickle_compare3(self):
        """
        Test case 3
        """
        cosmo = Cosmo(h=1, H0=None, flat=None, omega_m=0.3, omega_l=0.7,
                      omega_k=None)
        self._pickle_compare(cosmo=cosmo)
