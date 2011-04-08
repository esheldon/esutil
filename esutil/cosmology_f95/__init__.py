"""
A package to calculate cosmological distances.  

This is an implementation of Hogg, D., Distance measures in cosmology,
astro-ph/9905116

The implementation is a python class Cosmo, which is a wrapper for fast
routines written in fortran 90.  

See docs for the Cosmo class for details, e.g.  in ipython

    from esutil import cosmology
    cosmology.Cosmo?

"""
from . import cosmology
from .cosmology import Cosmo, test

