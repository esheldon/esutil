from . import _cosmolib

import numpy
from numpy import isscalar, log10

class Cosmo(dict):
    """
    A Class for calculating  cosmological distances.  

    This is an implementation of Hogg, D., Distance measures in cosmology,
    astro-ph/9905116 The python class is a wrapper for fast fortran 90
    routines.

    Class Name
    ----------
    Cosmo

    Construction Parameters
    -----------------------
    H0, h: float, optional
        Hubble constant in units of m/s/Mpc.  You can send either H0 or little h.
        Default is H0=100
    flat: boolean, optional
        Force a flat geometry.  Default is True
    omega_m: float, optional
        Matter density relative to the critical density.  Default is 0.3
    omega_l: float, optional
        Dark energy density relative to the critical density.  If flat is True,
        omega_l = 1-omega_m
    omega_k: float, optional
        Curvature in units of the critical density. If flat, omega_k=0


    Methods
    -------
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

    Examples:
        import cosmology
        c=cosmology.Cosmo()

        # comoving distance to z=0.5
        c.Dc(0.0, 0.5) 

        # angular diameter distance between z=0.5 and z=0.9
        c.Da(0.5, 0.9)

        # luminosity distance between z=0.2 and a sequence of redshifts
        c.Dl(0.2, [0.3, 0.4, 0.5])

        # new cosmology
        c=cosmology.Cosmo(H0=70.0, omega_m=0.25)

        # inverse critical density for lensing, lens at 0.2 and
        # source at 0.3
        c.sigmacritinv(0.2, 0.3)

    Internal fixed parameters
    -------------------------
    npts: integer
        Number of points used for integrating 1/E(z).  npts=5 is good to 1.e-8,
        as E(z) is a very slow function.
    vnpts: integer
        Number of points to use for volume integrations. Default is 10, good to
        1.e-11 for example between 0.2 and 1.0

    """
    def __init__(self, 
                 H0=100.0,
                 h=None, # can send either h or H0
                 flat=True,
                 omega_m=0.3, 
                 omega_l=0.7,
                 omega_k=None):

        flat, omega_m, omega_l, omega_k = \
                self.extract_parms(omega_m,omega_l,omega_k,flat)

        if h is not None:
            H0 = 100.0*h

        self.H0 = H0 
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.omega_k = omega_k
        self.flat = flat
        _cosmolib.cosmolib.cosmo_init()

        self.DH = float(_cosmolib.cosmolib.c/H0)

        self.Distmod = self.distmod

    def __repr__(self):
        m="""
        H0:      {H0}
        flat:    {flat}
        omega_m: {omega_m}
        omega_l: {omega_l}
        omega_k: {omega_k}
        """.format(H0=self.H0,
                   flat=self.flat,
                   omega_m=self.omega_m,
                   omega_l=self.omega_l,
                   omega_k=self.omega_k)
        return m

    def extract_parms(self, omega_m, omega_l, omega_k, flat):
        if omega_k is not None:
            # if omega_k is 0.0, we will set flat=True to simplify
            # the calculations
            if omega_k == 0.0:
                flat=True
            else:
                flat=False

        if omega_k is None:
            # without omega_k set we default to flat
            flat=True
            omega_k=0.0
        elif flat:
            # finally, if flat is set we always put omega_k = 0
            omega_k=0.0

        if flat:
            omega_l = 1.0-omega_m

        return flat, omega_m, omega_l, omega_k


    def Dc(self, zmin, zmax):
        """
        Calculate the comoving distance from zmin to zmax in units of Mpc.

        Parameters
        ----------
        zmin, zmax: scalars or arrays
            The following combinations are supported
                1) Two scalars
                2) zmin a scalar and zmax an array
                2) Both arrays of the same length.

        """

        if isscalar(zmin) and isscalar(zmax):
            # two scalars of any kind.
            dc = _cosmolib.cosmolib.cdist(zmin, zmax,
                                          self.DH, self.flat, 
                                          self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            dc=numpy.zeros(len(zmin), dtype='f8')
            _cosmolib.cosmolib.cdist_vec1(zmin, zmax, dc,
                                             self.DH, self.flat, 
                                             self.omega_m, self.omega_l, self.omega_k)

        elif isscalar(zmin) and not isscalar(zmax):
            # array for zmin, scalar zmax
            dc=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cosmolib.cdist_vec2(zmin, zmax, dc,
                                         self.DH, self.flat, 
                                         self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            dc=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cosmolib.cdist_2vec(zmin, zmax, dc,
                                          self.DH, self.flat, 
                                          self.omega_m, self.omega_l, self.omega_k)
        else:
            raise ValueError("zmin,zmax should be two scalars, zmin scalar zmax array, or both arrays")

        return dc

    def Dm(self, zmin, zmax):
        """
        Calculate the transvers comoving distance from zmin to zmax in units of Mpc.


        Useful for calculating transverse comoving distance at zmax.  When zmin
        is not zero, useful in calculating angular diameter distances

        Parameters
        ----------
        zmin, zmax: scalars or arrays
            The following combinations are supported
                1) Two scalars
                2) zmin a scalar and zmax an array
                2) Both arrays of the same length.

        """


        if isscalar(zmin) and isscalar(zmax):
            # two scalars of any kind.
            dm = _cosmolib.cosmolib.tcdist(zmin, zmax,
                                           self.DH, self.flat, 
                                           self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            dm=numpy.zeros(len(zmin), dtype='f8')
            _cosmolib.cosmolib.tcdist_vec1(zmin, zmax, dm,
                                           self.DH, self.flat, 
                                           self.omega_m, self.omega_l, self.omega_k)

        elif isscalar(zmin) and not isscalar(zmax):
            # scalar for zmin, array for zmax
            dm=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cosmolib.tcdist_vec2(zmin, zmax, dm,
                                           self.DH, self.flat, 
                                           self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            dm=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cosmolib.tcdist_2vec(zmin, zmax, dm,
                                           self.DH, self.flat, 
                                           self.omega_m, self.omega_l, self.omega_k)
        else:
            raise ValueError("zmin,zmax should be two scalars, zmin scalar zmax array, or both arrays")

        return dm

    def Da(self, zmin, zmax):
        """
        Calculate the angular diameter distance from zmin to zmax in units of Mpc.


        Parameters
        ----------
        zmin, zmax: scalars or arrays
            The following combinations are supported
                1) Two scalars
                2) zmin a scalar and zmax an array
                2) Both arrays of the same length.

        """


        if isscalar(zmin) and isscalar(zmax):
            # two scalars of any kind.
            da = _cosmolib.cosmolib.angdist(zmin, zmax,
                                            self.DH, self.flat, 
                                            self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            da=numpy.zeros(len(zmin), dtype='f8')
            _cosmolib.cosmolib.angdist_vec1(zmin, zmax, da,
                                            self.DH, self.flat, 
                                            self.omega_m, self.omega_l, self.omega_k)


        elif isscalar(zmin) and not isscalar(zmax):
            # scalar for zmin, array for zmax
            da=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cosmolib.angdist_vec2(zmin, zmax, da,
                                            self.DH, self.flat, 
                                            self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            da=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cosmolib.angdist_2vec(zmin, zmax, da,
                                            self.DH, self.flat, 
                                            self.omega_m, self.omega_l, self.omega_k)
        else:
            raise ValueError("zmin,zmax should be two scalars, zmin scalar zmax array, or both arrays")

        return da

    def Dl(self, zmin, zmax):
        """
        Calculate the luminosity distance from zmin to zmax in units of Mpc.


        Parameters
        ----------
        zmin, zmax: scalars or arrays
            The following combinations are supported
                1) Two scalars
                2) zmin a scalar and zmax an array
                2) Both arrays of the same length.

        """

        if isscalar(zmin) and isscalar(zmax):
            # two scalars of any kind.
            da = _cosmolib.cosmolib.lumdist(zmin, zmax,
                                            self.DH, self.flat, 
                                            self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            da=numpy.zeros(len(zmin), dtype='f8')
            _cosmolib.cosmolib.lumdist_vec1(zmin, zmax, da,
                                            self.DH, self.flat, 
                                            self.omega_m, self.omega_l, self.omega_k)

        elif isscalar(zmin) and not isscalar(zmax):
            # scalar for zmin, array for zmax
            da=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cosmolib.lumdist_vec2(zmin, zmax, da,
                                            self.DH, self.flat, 
                                            self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            da=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cosmolib.lumdist_2vec(zmin, zmax, da,
                                            self.DH, self.flat, 
                                            self.omega_m, self.omega_l, self.omega_k)
        else:
            raise ValueError("zmin,zmax should be two scalars, zmin scalar zmax array, or both arrays")

        return da



    def dV(self, z):
        """
        Calculate the volume element at redshift z

        Parameters
        ----------
        z: scalar or array
            Redshift
        """
        if isscalar(z):
            dv = _cosmolib.cosmolib.dv(z, 
                                       self.DH, self.flat, 
                                       self.omega_m, self.omega_l, self.omega_k)
        else:
            dv = numpy.zeros(len(z), dtype='f8')
            _cosmolib.cosmolib.dv_vec(z,dv,
                                      self.DH, self.flat, 
                                      self.omega_m, self.omega_l, self.omega_k)

        return dv

    def V(self, zmin, zmax):
        """
        Calculate the comoving volume between zmin and zmax.

        Parameters
        ----------
        zmin, zmax: scalars
            min and max redshifts
        """
        return _cosmolib.cosmolib.volume(zmin, zmax,
                                         self.DH, self.flat, 
                                         self.omega_m, self.omega_l, self.omega_k)

    def distmod(self, z):
        """
        Calculate the distance modulus to the given redshift.

        Parameters
        ----------
        z: scalar or array
            The redshift 
        """

        dmpc = self.Dl(0.0, z)
        dpc = dmpc*1.e6
        dm = 5.0*log10(dpc/10.0)
        return dm      


    def sigmacritinv(self, zl, zs):
        """
        Calculate the inverse critical density for the lens and source redshifts


        Parameters
        ----------
        zl, zs: scalars or arrays
            The following combinations are supported
                1) Two scalars
                2) zl a scalar and zs an array
                2) Both arrays of the same length.

        """


        if isscalar(zl) and isscalar(zs):
            # two scalars of any kind.
            scinv = _cosmolib.cosmolib.scinv(zl, zs,
                                             self.DH, self.flat, 
                                             self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zl) and isscalar(zs):
            # scalar for zl, array for zs
            scinv=numpy.zeros(len(zl), dtype='f8')
            _cosmolib.cosmolib.scinv_vec1(zl, zs, scinv,
                                          self.DH, self.flat, 
                                          self.omega_m, self.omega_l, self.omega_k)

        elif isscalar(zl) and not isscalar(zs):
            # scalar for zl, array for zs
            scinv=numpy.zeros(len(zs), dtype='f8')
            _cosmolib.cosmolib.scinv_vec2(zl, zs, scinv,
                                          self.DH, self.flat, 
                                          self.omega_m, self.omega_l, self.omega_k)

        elif not isscalar(zl) and not isscalar(zs):
            # both arrays: must be same length
            if len(zl) != len(zs):
                raise ValueError("If zl and zs are arrays, they must be same length")
            scinv=numpy.zeros(len(zs), dtype='f8')
            _cosmolib.cosmolib.scinv_2vec(zl, zs, scinv,
                                          self.DH, self.flat, 
                                          self.omega_m, self.omega_l, self.omega_k)
        else:
            raise ValueError("zl,zs should be two scalars, zl scalar zs array, or both arrays")

        return scinv


    def Ez_inverse(self, z):
        """
        Integrate kernel 1/E(z) from 0 to z.
        
        1/E(z) is used for distance calculations in FRW.

        Parameters
        ----------
        z: scalar or array
            The redshift 
        """

        if isscalar(z):
            ez = _cosmolib.cosmolib.ez_inverse(z, self.flat, 
                                               self.omega_m, self.omega_l, self.omega_k)
        else:
            ez = numpy.zeros(len(z), dtype='f8')
            _cosmolib.cosmolib.ez_inverse_vec(z, ez, self.flat, 
                                              self.omega_m, self.omega_l, self.omega_k)

        return ez

    def Ezinv_integral(self, zmin, zmax):
        """
        Integrate kernel 1/E(z) from zmin to zmax.

        1/E(z) is used for distance calculations in FRW.

        Parameters
        ----------
        zmin,zmax: scalars
            The redshifts
        """

        return _cosmolib.cosmolib.ez_inverse_integral(zmin, zmax, self.flat, 
                                                      self.omega_m, self.omega_l, self.omega_k)



def test_cosmo(omega_k=None):
    if omega_k is not None:
        flat=False
    else:
        flat=True
        omega_k=0.0

    c=Cosmo(flat=flat, omega_k=omega_k)

    print "  Testing Da,Dl"
    da=c.Da(0.1, 0.5)
    da=c.Da(0.1, [0.4, 0.5])
    da=c.Da([0.1,0.2], [0.4, 0.5])

    dl=c.Dl(0.1, 0.5)
    dl=c.Dl(0.1, [0.4, 0.5])
    dl=c.Dl([0.1,0.2], [0.4, 0.5])

    print "  Testing sicmacrit inverse"
    scinv=c.sigmacritinv(0.1, 0.5)
    scinv=c.sigmacritinv(0.1, [0.4, 0.5])
    scinv=c.sigmacritinv([0.1,0.2], [0.4, 0.5])


def test():
    print "Testing flat"
    test_cosmo()

    omega_k=0.05
    print "Testing non-flat, omega_k:",omega_k
    test_cosmo(omega_k=omega_k)
