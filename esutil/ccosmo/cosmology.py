import numpy
from numpy import isscalar, linspace
from . import _cosmolib


_CLIGHT=2.99792458e5

class Cosmo(dict):
    """
    A Class for calculating  cosmological distances.  

    This is an implementation of Hogg, D., Distance measures in cosmology,
    astro-ph/9905116 The python class is a wrapper for fast c routines.

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

        self.DH = _CLIGHT/H0

        self._cosmo =  _cosmolib.new_cosmo(self.DH, flat, omega_m, omega_l, omega_k)

        self.H0 = H0 
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.omega_k = omega_k
        self.flat = flat


        self.Distmod = self.distmod

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
            dc = _cosmolib.cdist(self._cosmo, zmin, zmax)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax: note order reversed in call for SWIG reasons
            dc=numpy.zeros(len(zmin), dtype='f8')
            _cosmolib.cdist_vec1(self._cosmo, zmin, zmax, dc)

        elif isscalar(zmin) and not isscalar(zmax):
            # array for zmin, scalar zmax
            dc=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cdist_vec2(self._cosmo, zmin, zmax, dc)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            dc=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.cdist_2vec(self._cosmo, zmin, zmax, dc)
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
            dm = _cosmolib.tcdist(self._cosmo, zmin, zmax)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            dm=numpy.zeros(len(zmin), dtype='f8')
            _cosmolib.tcdist_vec1(self._cosmo, zmin, zmax, dm)

        elif isscalar(zmin) and not isscalar(zmax):
            # scalar for zmin, array for zmax
            dm=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.tcdist_vec2(self._cosmo, zmin, zmax, dm)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            dm=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.tcdist_2vec(self._cosmo, zmin, zmax, dm)
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
            da = _cosmolib.angdist(self._cosmo, zmin, zmax)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            da=numpy.zeros(len(zmin), dtype='f8')
            _cosmolib.angdist_vec1(self._cosmo, zmin, zmax, da)


        elif isscalar(zmin) and not isscalar(zmax):
            # scalar for zmin, array for zmax
            da=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.angdist_vec2(self._cosmo, zmin, zmax, da)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            da=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.angdist_2vec(self._cosmo, zmin, zmax, da)
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
            dl = _cosmolib.lumdist(self._cosmo, zmin, zmax)

        elif not isscalar(zmin) and isscalar(zmax):
            # scalar for zmin, array for zmax
            dl=numpy.zeros(len(zmin), dtype='f8')
            _cosmolib.lumdist_vec1(self._cosmo, zmin, zmax, dl)

        elif isscalar(zmin) and not isscalar(zmax):
            # scalar for zmin, array for zmax
            dl=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.lumdist_vec2(self._cosmo, zmin, zmax, dl)

        elif not isscalar(zmin) and not isscalar(zmax):
            # both arrays: must be same length
            if len(zmin) != len(zmax):
                raise ValueError("If zmin and zmax are arrays, they must be same length")
            dl=numpy.zeros(len(zmax), dtype='f8')
            _cosmolib.lumdist_2vec(self._cosmo, zmin, zmax, dl)
        else:
            raise ValueError("zmin,zmax should be two scalars, zmin scalar zmax array, or both arrays")

        return dl



    def dV(self, z):
        """
        Calculate the volume element at redshift z

        Parameters
        ----------
        z: scalar or array
            Redshift
        """
        if isscalar(z):
            dv = _cosmolib.dV(self._cosmo, z)
        else:
            dv = numpy.zeros(len(z), dtype='f8')
            _cosmolib.dV_vec(self._cosmo, z,dv)

        return dv

    def V(self, zmin, zmax):
        """
        Calculate the comoving volume between zmin and zmax.

        Parameters
        ----------
        zmin, zmax: scalars
            min and max redshifts
        """
        return _cosmolib.volume(self._cosmo, zmin, zmax)

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
            scinv = _cosmolib.scinv(self._cosmo, zl, zs)

        elif not isscalar(zl) and isscalar(zs):
            # scalar for zl, array for zs
            scinv=numpy.zeros(len(zl), dtype='f8')
            _cosmolib.scinv_vec1(self._cosmo, zl, zs, scinv)

        elif isscalar(zl) and not isscalar(zs):
            # scalar for zl, array for zs
            scinv=numpy.zeros(len(zs), dtype='f8')
            _cosmolib.scinv_vec2(self._cosmo, zl, zs, scinv)

        elif not isscalar(zl) and not isscalar(zs):
            # both arrays: must be same length
            if len(zl) != len(zs):
                raise ValueError("If zl and zs are arrays, they must be same length")
            scinv=numpy.zeros(len(zs), dtype='f8')
            _cosmolib.scinv_2vec(self._cosmo, zl, zs, scinv)
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
            ez = _cosmolib.ez_inverse(self._cosmo, z)
        else:
            ez = numpy.zeros(len(z), dtype='f8')
            z=numpy.array(z, dtype='f8', copy=False)
            _cosmolib.ez_inverse_vec(self._cosmo, z, ez)

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

        return _cosmolib.ez_inverse_integral(self._cosmo, zmin, zmax)


    def __del__(self):
        _cosmolib.delete_cosmo(self._cosmo)

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


    def test(self, ntime=0):
        import time
        from .. import cosmology
        from .. import cosmology_purepy
        cold = cosmology.Cosmo(H0=self.H0,
                               flat=self.flat,
                               omega_m=self.omega_m, 
                               omega_l=self.omega_l,
                               omega_k=self.omega_k)
        cpy = cosmology_purepy.Cosmo(H0=self.H0,
                                     flat=self.flat,
                                     omega_m=self.omega_m, 
                                     omega_l=self.omega_l,
                                     omega_k=self.omega_k)



        print "Comparing ez_inverse:"
        print "  new:   ",self.Ez_inverse(0.2)
        print "  purepy:",cpy.Ez_inverse(0.2)

        print "Comparing ez_inverse vec:"
        print "  new:   ",self.Ez_inverse([0.2,0.4])
        print "  purepy:",cpy.Ez_inverse([0.2,0.4])

        print "Comparing ez_inverse_integral"
        print "  new:   ",self.Ezinv_integral(0.2, 0.4)
        print "  purepy:",cpy.Ezinv_integral(0.2, 0.4)

        print "\nComparing Dc"
        print "  new:   ",self.Dc(0.2, 0.4)
        print "  purepy:",cpy.Dc(0.2, 0.4)[0]

        print "Comparing Dc vec1"
        print "  new:   ",self.Dc([0.2,0.3], 0.4)
        print "  purepy:",cpy.Dc([0.2,0.3], 0.4)

        print "Comparing Dc vec2"
        print "  new:   ",self.Dc(0.1, [0.2,0.3])
        print "  purepy:",cpy.Dc(0.1, [0.2,0.3])

        print "Comparing Dc 2 vec"
        print "  new:   ",self.Dc([0.1,0.1], [0.2,0.3])
        print "  purepy:",cpy.Dc([0.1,0.1], [0.2,0.3])




        print "\nComparing Dm"
        print "  new:   ",self.Dm(0.2, 0.4)
        print "  purepy:",cpy.Dm(0.2, 0.4)[0]

        print "Comparing Dm vec1"
        print "  new:   ",self.Dm([0.2,0.3], 0.4)
        print "  purepy:",cpy.Dm([0.2,0.3], 0.4)

        print "Comparing Dm vec2"
        print "  new:   ",self.Dm(0.1, [0.2,0.3])
        print "  purepy:",cpy.Dm(0.1, [0.2,0.3])

        print "Comparing Dm 2 vec"
        print "  new:   ",self.Dm([0.1,0.1], [0.2,0.3])
        print "  purepy:",cpy.Dm([0.1,0.1], [0.2,0.3])


        print "\nComparing Da"
        print "  new:   ",self.Da(0.2, 0.4)
        print "  purepy:",cpy.Da(0.2, 0.4)[0]

        print "Comparing Da vec1"
        print "  new:   ",self.Da([0.2,0.3], 0.4)
        print "  purepy:",cpy.Da([0.2,0.3], 0.4)

        print "Comparing Da vec2"
        print "  new:   ",self.Da(0.1, [0.2,0.3])
        print "  purepy:",cpy.Da(0.1, [0.2,0.3])

        print "Comparing Da 2 vec"
        print "  new:   ",self.Da([0.1,0.1], [0.2,0.3])
        print "  purepy:",cpy.Da([0.1,0.1], [0.2,0.3])


        print "\nComparing Dl"
        print "  new:   ",self.Dl(0.2, 0.4)
        print "  purepy:",cpy.Dl(0.2, 0.4)[0]

        print "Comparing Dl vec1"
        print "  new:   ",self.Dl([0.2,0.3], 0.4)
        print "  purepy:",cpy.Dl([0.2,0.3], 0.4)

        print "Comparing Dl vec2"
        print "  new:   ",self.Dl(0.1, [0.2,0.3])
        print "  purepy:",cpy.Dl(0.1, [0.2,0.3])

        print "Comparing Dl 2 vec"
        print "  new:   ",self.Dl([0.1,0.1], [0.2,0.3])
        print "  purepy:",cpy.Dl([0.1,0.1], [0.2,0.3])


        print "\nComparing dV"
        print "  new:   ",self.dV(0.4)
        print "  purepy:",cpy.dV(0.4)[0]

        print "Comparing dV vec1"
        print "  new:   ",self.dV([0.2,0.3])
        print "  purepy:",cpy.dV([0.2,0.3])


        print "\nComparing V"
        print "  new:   ",self.V(0.1,0.4)
        print "  purepy:",cpy.V(0.1,0.4)[0]


        print "\nComparing sigmacritinv"
        print "  new:   ",self.sigmacritinv(0.1,0.4)
        print "  purepy:",cpy.sigmacritinv(0.1,0.4)[0]

        print "Comparing sigmacritinv vec1"
        print "  new:   ",self.sigmacritinv([0.2,0.3], 0.4)
        print "  purepy:",cpy.sigmacritinv([0.2,0.3], 0.4)

        print "Comparing sigmacritinv vec2"
        print "  new:   ",self.sigmacritinv(0.1, [0.2,0.3])
        print "  purepy:",cpy.sigmacritinv(0.1, [0.2,0.3])

        print "Comparing sigmacritinv 2 vec"
        print "  new:   ",self.sigmacritinv([0.1,0.1], [0.2,0.3])
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
