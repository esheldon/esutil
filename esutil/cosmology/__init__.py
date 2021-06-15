"""
A package for calculating  cosmological distances.  

Classes
-------
Cosmo:
    This is the class that does all the calculating.  See docs on cosmology.Cosmo
    for more details.

    Cosmo Class Methods
    --------------
    DH: Return the hubble distance.
    Dc: Comoving distance.
    Dm: Transverse comoving distance.
    Da: Angular diameter distance.
    Dl: Luminosity distance.
    dV: Volume element.
    V:  Volume between two redshifts.
    distmod: Distance modulus.
    sigmacritinv: Inverse critical density for lensing.

    Ez_inverse: Calculate 1/E(z)
    Ezinv_integral: Calculate the integral of 1/E(z) from zmin to zmax


"""

# flake8: noqa

from . import cosmology
from .cosmology import Cosmo

__version__="1.1.0"
