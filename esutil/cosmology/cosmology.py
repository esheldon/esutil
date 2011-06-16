main_docs="""
A Class for calculating  cosmological distances.  

This is an implementation of Hogg, D., Distance measures in cosmology,
astro-ph/9905116 The python class is a wrapper for fast C routines.

Class Name
----------
Cosmo

import esutil
c=esutil.cosmology.Cosmo()

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

flat(): return if universe is flat
omega_m(): value of omega matter
omega_l(): value of omega lambda
omega_k(): value of omega curvature

Optional Construction Keywords
------------------------------
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

Notes:
    Don't call the c codes directly, they do very little error checking.  Error
    checking is dealt with in the Cosmo python class.

Modification:
    Early 2011: Complete Re-write without using SWIG.
"""

__doc__=main_docs
import numpy
from numpy import isscalar, linspace
import _cosmolib


_CLIGHT=2.99792458e5

class Cosmo:
    __doc__=main_docs
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

        DH = _CLIGHT/H0

        self._cosmo =  _cosmolib.cosmo(DH, flat, omega_m, omega_l, omega_k)

        self.Distmod = self.distmod

        self._H0 = H0

    def H0(self):
        return self._H0
    def DH(self):
        return self._cosmo.DH()
    def flat(self):
        return self._cosmo.flat()
    def omega_m(self):
        return self._cosmo.omega_m()
    def omega_l(self):
        return self._cosmo.omega_l()
    def omega_k(self):
        return self._cosmo.omega_k()

    def Dc(self, zmin, zmax):
        """
        Calculate the comoving distance from zmin to zmax in units of Mpc.

        Parameters
        ----------
        zmin, zmax: scalars or arrays
            The following combinations are supported
                1) Two scalars
                2) zmin a scalar and zmax an array
                3) zmin an array and zmax a scalar
                4) Both arrays of the same length.

        """

        if isscalar(zmin) and isscalar(zmax):
            # two scalars of any kind.
            d = self._cosmo.Dc(zmin, zmax)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            zmin = numpy.array(zmin, dtype='f8', copy=False, order='C')
            d=self._cosmo.Dc_vec1(zmin, zmax)

        elif isscalar(zmin) and not isscalar(zmax):
            # array for zmin, scalar zmax
            zmax = numpy.array(zmax, dtype='f8', copy=False, order='C')
            d=self._cosmo.Dc_vec2( zmin, zmax)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            zmin = numpy.array(zmin, dtype='f8', copy=False, order='C')
            zmax = numpy.array(zmax, dtype='f8', copy=False, order='C')
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            d=self._cosmo.Dc_2vec(zmin, zmax)
        else:
            raise ValueError("zmin,zmax should be two scalars, zmin scalar zmax array, or both arrays")

        return d

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
                3) zmin an array and zmax a scalar
                4) Both arrays of the same length.

        """

        if isscalar(zmin) and isscalar(zmax):
            # two scalars of any kind.
            d = self._cosmo.Dm(zmin, zmax)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            zmin = numpy.array(zmin, dtype='f8', copy=False, order='C')
            d=self._cosmo.Dm_vec1(zmin, zmax)

        elif isscalar(zmin) and not isscalar(zmax):
            # array for zmin, scalar zmax
            zmax = numpy.array(zmax, dtype='f8', copy=False, order='C')
            d=self._cosmo.Dm_vec2( zmin, zmax)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            zmin = numpy.array(zmin, dtype='f8', copy=False, order='C')
            zmax = numpy.array(zmax, dtype='f8', copy=False, order='C')
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            d=self._cosmo.Dm_2vec(zmin, zmax)
        else:
            raise ValueError("zmin,zmax should be two scalars, zmin scalar zmax array, or both arrays")

        return d


    def Da(self, zmin, zmax):
        """
        Calculate the angular diameter distance from zmin to zmax in units of Mpc.


        Parameters
        ----------
        zmin, zmax: scalars or arrays
            The following combinations are supported
                1) Two scalars
                2) zmin a scalar and zmax an array
                3) zmin an array and zmax a scalar
                4) Both arrays of the same length.

        """

        if isscalar(zmin) and isscalar(zmax):
            # two scalars of any kind.
            d = self._cosmo.Da(zmin, zmax)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            zmin = numpy.array(zmin, dtype='f8', copy=False, order='C')
            d=self._cosmo.Da_vec1(zmin, zmax)

        elif isscalar(zmin) and not isscalar(zmax):
            # array for zmin, scalar zmax
            zmax = numpy.array(zmax, dtype='f8', copy=False, order='C')
            d=self._cosmo.Da_vec2( zmin, zmax)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            zmin = numpy.array(zmin, dtype='f8', copy=False, order='C')
            zmax = numpy.array(zmax, dtype='f8', copy=False, order='C')
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            d=self._cosmo.Da_2vec(zmin, zmax)
        else:
            raise ValueError("zmin,zmax should be two scalars, zmin scalar zmax array, or both arrays")

        return d


    def Dl(self, zmin, zmax):
        """
        Calculate the luminosity distance from zmin to zmax in units of Mpc.


        Parameters
        ----------
        zmin, zmax: scalars or arrays
            The following combinations are supported
                1) Two scalars
                2) zmin a scalar and zmax an array
                3) zmin an array and zmax a scalar
                4) Both arrays of the same length.

        """

        if isscalar(zmin) and isscalar(zmax):
            # two scalars of any kind.
            d = self._cosmo.Dl(zmin, zmax)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            zmin = numpy.array(zmin, dtype='f8', copy=False, order='C')
            d=self._cosmo.Dl_vec1(zmin, zmax)

        elif isscalar(zmin) and not isscalar(zmax):
            # array for zmin, scalar zmax
            zmax = numpy.array(zmax, dtype='f8', copy=False, order='C')
            d=self._cosmo.Dl_vec2( zmin, zmax)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            zmin = numpy.array(zmin, dtype='f8', copy=False, order='C')
            zmax = numpy.array(zmax, dtype='f8', copy=False, order='C')
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            d=self._cosmo.Dl_2vec(zmin, zmax)
        else:
            raise ValueError("zmin,zmax should be two scalars, zmin scalar zmax array, or both arrays")

        return d




    def dV(self, z):
        """
        Calculate the volume element at redshift z

        Parameters
        ----------
        z: scalar or array
            Redshift
        """
        if isscalar(z):
            dv = self._cosmo.dV(z)
        else:
            z = numpy.array(z, dtype='f8', copy=False, order='C')
            dv = self._cosmo.dV_vec(z)

        return dv

    def V(self, zmin, zmax):
        """
        Calculate the comoving volume between zmin and zmax.

        Parameters
        ----------
        zmin, zmax: scalars
            min and max redshifts
        """
        return self._cosmo.V(zmin, zmax)

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
                2) zmin a scalar and zmax an array
                3) zmin an array and zmax a scalar
                4) Both arrays of the same length.

        """

        if isscalar(zl) and isscalar(zs):
            # two scalars of any kind.
            scinv = self._cosmo.scinv(zl, zs)

        elif not isscalar(zl) and isscalar(zs):
            # scalar for zl, array for zs
            zl = numpy.array(zl, dtype='f8', copy=False, order='C')
            scinv=self._cosmo.scinv_vec1(zl, zs)

        elif isscalar(zl) and not isscalar(zs):
            # array for zl, scalar zs
            zs = numpy.array(zs, dtype='f8', copy=False, order='C')
            scinv=self._cosmo.scinv_vec2( zl, zs)

        elif not isscalar(zl) and not isscalar(zs):
            # both arrays: must be same length
            zl = numpy.array(zl, dtype='f8', copy=False, order='C')
            zs = numpy.array(zs, dtype='f8', copy=False, order='C')
            if len(zl) != len(zs):
                raise ValueError("If zl and zs are arrays, they must be same length")
            scinv=self._cosmo.scinv_2vec(zl, zs)
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
            ez = self._cosmo.ez_inverse(z)
        else:
            z=numpy.array(z, dtype='f8', copy=False, order='C')
            ez=self._cosmo.ez_inverse_vec(z)

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

        return self._cosmo.ez_inverse_integral(zmin, zmax)


    def __repr__(self):
        m="""H0:      %s
flat:    %s
omega_m: %s
omega_l: %s
omega_k: %s
        """ % (self._H0, self.flat(), self.omega_m(), self.omega_l(), self.omega_k())
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

    def test(self):

        print "ez_inverse:"
        print "     ",self.Ez_inverse(0.2)

        print "ez_inverse vec:"
        print "     ",self.Ez_inverse([0.2,0.4])

        print "ez_inverse_integral"
        print "     ",self.Ezinv_integral(0.2, 0.4)

        print "\nDc"
        print "     ",self.Dc(0.2, 0.4)

        print "Dc vec1"
        print "     ",self.Dc([0.2,0.3], 0.4)

        print "Dc vec2"
        print "     ",self.Dc(0.1, [0.2,0.3])

        print "Dc 2 vec"
        print "     ",self.Dc([0.1,0.1], [0.2,0.3])




        print "\nDm"
        print "     ",self.Dm(0.2, 0.4)

        print "Dm vec1"
        print "     ",self.Dm([0.2,0.3], 0.4)

        print "Dm vec2"
        print "     ",self.Dm(0.1, [0.2,0.3])

        print "Dm 2 vec"
        print "     ",self.Dm([0.1,0.1], [0.2,0.3])


        print "\nDa"
        print "     ",self.Da(0.2, 0.4)

        print "Da vec1"
        print "     ",self.Da([0.2,0.3], 0.4)

        print "Da vec2"
        print "     ",self.Da(0.1, [0.2,0.3])

        print "Da 2 vec"
        print "     ",self.Da([0.1,0.1], [0.2,0.3])


        print "\nDl"
        print "     ",self.Dl(0.2, 0.4)

        print "Dl vec1"
        print "     ",self.Dl([0.2,0.3], 0.4)

        print "Dl vec2"
        print "     ",self.Dl(0.1, [0.2,0.3])

        print "Dl 2 vec"
        print "     ",self.Dl([0.1,0.1], [0.2,0.3])


        print "\ndV"
        print "     ",self.dV(0.4)

        print "dV vec1"
        print "     ",self.dV([0.2,0.3])


        print "\nV"
        print "     ",self.V(0.1,0.4)


        print "\nsigmacritinv"
        print "     ",self.sigmacritinv(0.1,0.4)

        print "sigmacritinv vec1"
        print "     ",self.sigmacritinv([0.2,0.3], 0.4)

        print "sigmacritinv vec2"
        print "     ",self.sigmacritinv(0.1, [0.2,0.3])

        print "sigmacritinv 2 vec"
        print "     ",self.sigmacritinv([0.1,0.1], [0.2,0.3])


    def test_vs_purepy(self, ntime=0):
        import time
        import esutil as eu
        from esutil import cosmology_purepy
        cpy = cosmology_purepy.Cosmo(H0=self.H0,
                                     flat=self.flat,
                                     omega_m=self.omega_m, 
                                     omega_l=self.omega_l,
                                     omega_k=self.omega_k)



        print "Comparing ez_inverse:"
        print "  this:  ",self.Ez_inverse(0.2)
        print "  purepy:",cpy.Ez_inverse(0.2)

        print "Comparing ez_inverse vec:"
        print "  this:  ",self.Ez_inverse([0.2,0.4])
        print "  purepy:",cpy.Ez_inverse([0.2,0.4])

        print "Comparing ez_inverse_integral"
        print "  this:  ",self.Ezinv_integral(0.2, 0.4)
        print "  purepy:",cpy.Ezinv_integral(0.2, 0.4)

        print "\nComparing Dc"
        print "  this:  ",self.Dc(0.2, 0.4)
        print "  purepy:",cpy.Dc(0.2, 0.4)[0]

        print "Comparing Dc vec1"
        print "  this:  ",self.Dc([0.2,0.3], 0.4)
        print "  purepy:",cpy.Dc([0.2,0.3], 0.4)

        print "Comparing Dc vec2"
        print "  this:  ",self.Dc(0.1, [0.2,0.3])
        print "  purepy:",cpy.Dc(0.1, [0.2,0.3])

        print "Comparing Dc 2 vec"
        print "  this:  ",self.Dc([0.1,0.1], [0.2,0.3])
        print "  purepy:",cpy.Dc([0.1,0.1], [0.2,0.3])




        print "\nComparing Dm"
        print "  this:  ",self.Dm(0.2, 0.4)
        print "  purepy:",cpy.Dm(0.2, 0.4)[0]

        print "Comparing Dm vec1"
        print "  this:  ",self.Dm([0.2,0.3], 0.4)
        print "  purepy:",cpy.Dm([0.2,0.3], 0.4)

        print "Comparing Dm vec2"
        print "  this:  ",self.Dm(0.1, [0.2,0.3])
        print "  purepy:",cpy.Dm(0.1, [0.2,0.3])

        print "Comparing Dm 2 vec"
        print "  this:  ",self.Dm([0.1,0.1], [0.2,0.3])
        print "  purepy:",cpy.Dm([0.1,0.1], [0.2,0.3])


        print "\nComparing Da"
        print "  this:  ",self.Da(0.2, 0.4)
        print "  purepy:",cpy.Da(0.2, 0.4)[0]

        print "Comparing Da vec1"
        print "  this:  ",self.Da([0.2,0.3], 0.4)
        print "  purepy:",cpy.Da([0.2,0.3], 0.4)

        print "Comparing Da vec2"
        print "  this:  ",self.Da(0.1, [0.2,0.3])
        print "  purepy:",cpy.Da(0.1, [0.2,0.3])

        print "Comparing Da 2 vec"
        print "  this:  ",self.Da([0.1,0.1], [0.2,0.3])
        print "  purepy:",cpy.Da([0.1,0.1], [0.2,0.3])


        print "\nComparing Dl"
        print "  this:  ",self.Dl(0.2, 0.4)
        print "  purepy:",cpy.Dl(0.2, 0.4)[0]

        print "Comparing Dl vec1"
        print "  this:  ",self.Dl([0.2,0.3], 0.4)
        print "  purepy:",cpy.Dl([0.2,0.3], 0.4)

        print "Comparing Dl vec2"
        print "  this:  ",self.Dl(0.1, [0.2,0.3])
        print "  purepy:",cpy.Dl(0.1, [0.2,0.3])

        print "Comparing Dl 2 vec"
        print "  this:  ",self.Dl([0.1,0.1], [0.2,0.3])
        print "  purepy:",cpy.Dl([0.1,0.1], [0.2,0.3])


        print "\nComparing dV"
        print "  this:  ",self.dV(0.4)
        print "  purepy:",cpy.dV(0.4)[0]

        print "Comparing dV vec1"
        print "  this:  ",self.dV([0.2,0.3])
        print "  purepy:",cpy.dV([0.2,0.3])


        print "\nComparing V"
        print "  this:  ",self.V(0.1,0.4)
        print "  purepy:",cpy.V(0.1,0.4)[0]


        print "\nComparing sigmacritinv"
        print "  this:  ",self.sigmacritinv(0.1,0.4)
        print "  purepy:",cpy.sigmacritinv(0.1,0.4)[0]

        print "Comparing sigmacritinv vec1"
        print "  this:  ",self.sigmacritinv([0.2,0.3], 0.4)
        print "  purepy:",cpy.sigmacritinv([0.2,0.3], 0.4)

        print "Comparing sigmacritinv vec2"
        print "  this:  ",self.sigmacritinv(0.1, [0.2,0.3])
        print "  purepy:",cpy.sigmacritinv(0.1, [0.2,0.3])

        print "Comparing sigmacritinv 2 vec"
        print "  this:  ",self.sigmacritinv([0.1,0.1], [0.2,0.3])
        print "  purepy:",cpy.sigmacritinv([0.1,0.1], [0.2,0.3])


        if ntime > 0:
            print '\nComparing timings for sigmacritinv'
            tm=0.0
            tmpy=0.0
            print '   doing c code'
            for i in xrange(ntime):
                tm0=time.time()
                da = self.Da(0.0, linspace(0.1, 0.2, 100000))
                tm += time.time()-tm0

            print '   doing python code'
            for i in xrange(ntime):
                tm0=time.time()
                da = cpy.Da(0.0, linspace(0.1, 0.2, 100000))
                tmpy += time.time()-tm0

            print 'C code:',tm
            print 'pure py code:',tmpy
            print 'C code is',tmpy/tm,'faster'
