"""
    Module Name:
        cosmology

    Purpose:
        A set of tools for calculating distances in an expanding universe.
        These routines are completely general for any specified omega_m,
        omega_k, and cosmological constant omega_l.  This code follows the
        conventions of Hogg astro-ph/9905116.
        
        All distances are in units of Mpc/h unless h is specified. Volumes are
        in (Mpc/h)**3.  All return values are arrays.

    Classes:
        Cosmo :  This class is instantiated with the desired cosmology and
            all subsequent calculations are in that cosmology.

            Instantiation:
                import esutil
                cosmo=esutil.cosmology.Cosmo(omega_m=0.3,
                                             omega_l=0.7,
                                             omega_k=0.0,
                                             h=1.0,
                                             flat=True,
                                             npts=5,
                                             vnpts=10)

            All parameters are optional.  For the defaults simply use
                cosmo=esutil.cosmology.Cosmo()

            Methods (see method docs for more details):

                Da(zmin, zmax) : angular diameter distance.
                Dl(zmin, zmax) : luminosity distance.
                distmod(z): Distance modulus.
                dV(z, comoving=True): Volume element.
                V(zmin, zmax, comoving=True):  Volume between two redshifts.
                Dc(zmin,zmax): Comoving distance.
                Dm(zmin,zmax): Transverse comoving distance.
                DH: Hubble distance c/H. 
                Ez_inverse(z): 
                    1/sqrt( omega_m*(1+z)**3 + omega_k*(1+z)**2 + omega_l)
                Ezinv_integral(z1,z2): 
                    Integral of Ez_inverse over a range of redshifts.

    The module also provides these Convenience Functions.  These are called in
    the same way as the class methods listed above, but each also takes in the
    cosmological keywords omega_m,omega_l,omega_k,h,flat as well as appropriate
    integration parameters.

        Da: angular diameter distance.
        Dl: luminosity distance.
        Distmod: Distance modulus.
        dV: Volume element.
        V:  Volume between two redshifts.
        Dc: Comoving distance.
        Dm: Transverse comoving distance.
        DH: Hubble distance c/H. 
        Ez_inverse: 1/sqrt( omega_m*(1+z)**3 + omega_k*(1+z)**2 + omega_l)
        Ezinv_integral: Integral of Ez_inverse over a range of redshifts.


    Examples:
        # using the Cosmo class.  
        >>> import esutil
        >>> cosmo=esutil.cosmology.Cosmo(omega_m=0.24,h=0.7)
        >>> cosmo.Da(0.0, 0.35)
        array([ 1034.76013423])
        # using a convenience function
        >>> esutil.cosmology.Da(0.0,0.35,omega_m=0.24,h=0.7)
        array([ 1034.76013423])

    Requirements:
        NumPy

    Revision History:
        Copied from IDL routines.  2006-11-07, Erin Sheldon, NYU
        Converted to using faster Gauss-Legendre integration 
            2007-05-17, Erin Sheldon, NYU
        Cleaned up imports so the module can be imported without
            numpy/scipy even though nothing will work.  2009-11-01. E.S.S. BNL

        Added Cosmo class for more convenient usage.
            2010-02-18, Erin Sheldon, BNL

"""

license="""
  Copyright (C) 2009-10  Erin Sheldon

    This program is free software; you can redistribute it and/or modify it
    under the terms of version 2 of the GNU General Public License as
    published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
"""

from sys import stdout

try:
    import numpy
    from numpy import sqrt, sin, sinh, log10, isscalar

    # Global variables for Ez integration.  
    _EZI_XXi=numpy.array([])
    _EZI_WWi=numpy.array([])

    # Global variables for volume integration.  
    _VI_XXi=numpy.array([])
    _VI_WWi=numpy.array([])


except:
    stdout.write("Could not import numpy")

import esutil.integrate

class Cosmo(object):
    def __init__(self, 
                 omega_m=0.3, 
                 omega_l=0.7,
                 omega_k=0.0,
                 H0=100.0,
                 h=None,
                 flat=True,
                 npts=5,
                 vnpts=10):

        # If flat is specified, make sure omega_l = 1-omega_m
        # and omega_k=0

        if h is not None:
            H0 = 100.0*h
        else:
            h = H0/100.0

        flat, omega_m, omega_l, omega_k = \
                self.extract_parms(omega_m,omega_l,omega_k,flat)

        self.flat = flat

        self.omega_m=omega_m
        self.omega_l=omega_l
        self.omega_k=omega_k
        self.h=h
        self.flat=flat
        self.npts=npts
        self.vnpts=vnpts

        # Will only change if npts changes 
        self._ezi_run_gauleg()
        self._vi_run_gauleg()

        self._four_pi_G_over_c_squared = four_pi_G_over_c_squared(dunits='Mpc')

    def __repr__(self):
        rep="""
        H0:      %s
        omega_m: %s
        omega_l: %s
        omega_k: %s
        flat:    %s\n""" % (self.h*100., self.omega_m, self.omega_l, self.omega_k, self.flat)
        return rep


        # it used to be capilalized
        self.Distmod=self.distmod

    def DH(self):
        """
        NAME:
            DH
        PURPOSE:
            Calculate the Hubble distance in Mpc/h
        CALLING SEQUENCE:
            import esutil
            cosmo=esutil.cosmology.Cosmo(omega_m=0.3,
                                         omega_l=0.7,
                                         omega_k=0.0,
                                         h=1.0,
                                         flat=True,
                                         npts=5,
                                         vnpts=10)
            d = cosmo.DH()
        """
        return 2.99792458e5/100.0/self.h


    def Dc(self, z1in, z2in):
        """
        NAME:
            Dc
        PURPOSE:
            Calculate the comoving distance between redshifts z1 and z2 in
            a FRW universe. Units: Mpc 
        CALLING SEQUENCE:
            import esutil
            cosmo=esutil.cosmology.Cosmo(omega_m=0.3,
                                         omega_l=0.7,
                                         omega_k=0.0,
                                         h=1.0,
                                         flat=True,
                                         npts=5,
                                         vnpts=10)
            d=cosmo.Dc(z1, z2)
        INPUTS:
            z1, z2: The redshifts.  These must either be 
                1) Two scalars
                2) A scalar and an array.
                3) Two arrays of the same length.
        """
  
        # Make sure they are arrays, but don't copy if already an array
        z1 = numpy.array(z1in, ndmin=1, copy=False)
        z2 = numpy.array(z2in, ndmin=1, copy=False)

        # All the permutations of inputs
        dh=self.DH()
        if z1.size == z2.size:
            if z1.size == 1:
                return dh*self.Ezinv_integral(z1,z2)
            else:
                dc = numpy.zeros(z1.size)
                for i in numpy.arange(z1.size):
                    dc[i] = dh*self.Ezinv_integral(z1[i],z2[i])
        else:
            if z1.size == 1:
                dc = numpy.zeros(z2.size)
                for i in numpy.arange(z2.size):
                    dc[i] = dh*self.Ezinv_integral(z1,z2[i])
            elif z2.size == 1:
                dc = numpy.zeros(z1.size)
                for i in numpy.arange(z1.size):
                    dc[i] = dh*self.Ezinv_integral(z1[i],z2)
            else:
                raise ValueError("z1,z2: Must be same length or one a scalar")

        return dc

    def Dm(self, zmin, zmax):
        """
        NAME:
            Dm

        PURPOSE:
            Calculate the transverse comoving distance between two objects at
            the same redshift in a a FRW universe.  Units: Mpc.

        CALLING SEQUENCE:
            import esutil
            cosmo=esutil.cosmology.Cosmo(omega_m=0.3,
                                         omega_l=0.7,
                                         omega_k=0.0,
                                         h=1.0,
                                         flat=True,
                                         npts=5,
                                         vnpts=10)
            d=cosmo.Dm(zmin, zmax)
        INPUTS:
            zmin, zmax: The redshifts.  
                Note, to interpret as the transverse distance between objects
                at the same redshift as viewed by a redshift zero observer,
                zmin=0.0  It is useful to allow zmin != 0 when measuring for
                example angular diameter distances between two non zero
                redshifts, as in lensing calculations.  These redshifts must
                either be 

                1) Two scalars
                2) A scalar and an array.
                3) Two arrays of the same length.

        """

        dh = self.DH()
        dc=self.Dc(zmin, zmax)

        if self.flat:
            return dc
        elif self.omega_k > 0:
            return dh/sqrt(self.omega_k)*sinh( sqrt(self.omega_k)*dc/dh )
        else:
            return dh/sqrt(-self.omega_k)*sin( sqrt(-self.omega_k)*dc/dh )


    def Da(self, zmin, zmax):
        """
        NAME:
            Da 
        PURPOSE:
            Calculate the angular diameter distance between z1 and z2 in a 
            FRW universe. Units: Mpc.
        CALLING SEQUENCE:
            import esutil
            cosmo=esutil.cosmology.Cosmo(omega_m=0.3,
                                         omega_l=0.7,
                                         omega_k=0.0,
                                         h=1.0,
                                         flat=True,
                                         npts=5,
                                         vnpts=10)
            d=cosmo.Da(zmin, zmax)
        INPUTS:
            zmin, zmax: The redshifts.  These must either be 
                1) Two scalars
                2) A scalar and an array.
                3) Two arrays of the same length.
        """

        z1 = numpy.array(zmin, ndmin=1, copy=False)
        z2 = numpy.array(zmax, ndmin=1, copy=False)
        d = self.Dm(z1, z2)

        da = numpy.where( z1 < z2, d/(1.0+z2), d/(1.0+z1) )

        return da


    def Dl(self, zmin, zmax):
        """
        NAME:
            Dl
        PURPOSE:
            Calculate the luminosity distance between z1 and z2 in a 
            FRW universe. Units: Mpc.
        CALLING SEQUENCE:
            import esutil
            cosmo=esutil.cosmology.Cosmo(omega_m=0.3,
                                         omega_l=0.7,
                                         omega_k=0.0,
                                         h=1.0,
                                         flat=True,
                                         npts=5,
                                         vnpts=10)
            d=cosmo.Dl(zmin, zmax)
        INPUTS:
            zmin, zmax: The redshifts.  These must either be 
                1) Two scalars
                2) A scalar and an array.
                3) Two arrays of the same length.
        """

        z1 = numpy.array(zmin, ndmin=1, copy=False)
        z2 = numpy.array(zmax, ndmin=1, copy=False)
        return self.Da(z1,z2)*(1.0+z2)**2

    def distmod(self, z):
        """
        NAME:
            distmod
        PURPOSE:
            Calculate the distance modulus to redshift z.
        CALLING SEQUENCE:
            import esutil
            cosmo=esutil.cosmology.Cosmo(omega_m=0.3,
                                         omega_l=0.7,
                                         omega_k=0.0,
                                         h=1.0,
                                         flat=True,
                                         npts=5,
                                         vnpts=10)
            d=cosmo.Distmod(z)
        INPUTS:
            z: The redshift(s).
        """

        dmpc = self.Dl(0.0, z)
        dpc = dmpc*1.e6
        dm = 5.0*log10(dpc/10.0)
        return dm      


    def dV(self, z_input, comoving=True):
        """
        NAME:
            dV
        PURPOSE:
            Calculate the volume elementd dV in a FRW universe. Units: Mpc**3
        CALLING SEQUENCE:
            import esutil
            cosmo=esutil.cosmology.Cosmo(omega_m=0.3,
                                         omega_l=0.7,
                                         omega_k=0.0,
                                         h=1.0,
                                         flat=True,
                                         npts=5,
                                         vnpts=10)
            dv = cosmo.dV(z, comoving=True)
        INPUTS:
            z: The redshift
            comoving=True: Use comoving coords, default True.
        """

        z = numpy.array(z_input, ndmin=1, copy=False)

        dh = self.DH()
        da = self.Da(0.0, z)
        Ez = 1.0/self.Ez_inverse(z)
        if comoving:
            dv = dh*da**2/Ez*(1.0+z)**2
        else:
            dv = dh*da**2/Ez*(1.0+z)

        return dv

    def V(self, zmin, zmax, comoving=True):
        """
        NAME:
            V
        PURPOSE:
            Calculate the volume between zmin and zmax in an FRW universe.
            Units: Mpc**3
        CALLING SEQUENCE:
            import esutil
            cosmo=esutil.cosmology.Cosmo(omega_m=0.3,
                                         omega_l=0.7,
                                         omega_k=0.0,
                                         h=1.0,
                                         flat=True,
                                         npts=5,
                                         vnpts=10)
            v = cosmo.V(zmin, zmax, comoving=True)
        INPUTS:
            zmin, zmax The redshift limits.  
            comoving: Use comoving coords, default True.
        """

        # these needed for coordinate transformation
        f1 = (zmax-zmin)/2.
        f2 = (zmax+zmin)/2.

        zvals = self.vxxi*f1 + f2
        ezivals = self.dV(zvals, comoving=comoving)

        v =  f1 * ((ezivals*self.vwwi).sum())
        v = numpy.array(v, ndmin=1)
        return v


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



    def _ezi_run_gauleg(self):
        self.xxi, self.wwi = esutil.integrate.gauleg(-1.0,1.0,self.npts)
    def _vi_run_gauleg(self):
        self.vxxi, self.vwwi = esutil.integrate.gauleg(-1.0,1.0,self.vnpts)




    def Ez_inverse(self,z):
        """
        NAME:
            Ez_inverse
        PURPOSE:
            Calculate kernel 1/E(z) for distance integrals in FRW universe.
        CALLING SEQUENCE:
            ezi = cosmo.Ez_inverse(z)
        """
        if not isscalar(z):
            z = numpy.array(z, copy=False)
        arg=self.omega_m*(1.0+z)**3 + self.omega_k*(1.0+z)**2 + self.omega_l
        return 1.0/sqrt(arg)



    def Ezinv_integral(self, z1, z2):
        """
        NAME:
            Ezinv_integral
        PURPOSE:
            Integrate kernel 1/E(z) used for distance calculations in
            FRW universe. Gauss-legendre integration. Default of npts=5
            is actually good to 1.e-8 between redshift 0-1 because it is
            such a slow function.

        CALLING SEQUENCE:
            ezint = Ezinv_integral(z1, z2)
        INPUTS:
            z1, z2: The redshift interval, scalars.
        """


        f1 = (z2-z1)/2.
        f2 = (z2+z1)/2.

        zvals = self.xxi*f1 + f2
        ezivals = self.Ez_inverse(zvals)

        ezint = f1 * ((ezivals*self.wwi).sum())
        return abs(ezint)


    def sigmacritinv(self, zl, zs):
        """
        Method:
            sigmacritinv
        Purpose:
            Calculate the inverse critical density for lensing. The units
            are pc^2/Msun
        usage: 
            c=Cosmo(keywords..)
            sc = c.sigmacritinv(zl, zs)

        Possible inputs:
            zl scalar, zs scalar
            zl scalar, zs vector
            zl vector, zs scalar
            zl vector, zs vector of the same length

        """
        
        zl = numpy.array(zl, ndmin=1, copy=False)
        zs = numpy.array(zs, ndmin=1, copy=False)

        if (zl.size != 1) and (zs.size != 1):
            if zl.size != zs.size:
                raise ValueError("Possible z input:\n"
                                 "  zl scalar, zs scalar"
                                 "  zl scalar, zs vector"
                                 "  zl vector, zs scalar"
                                 "  zl vector, zs vector of the same length")


        # units are Mpc
        dl  = self.Da(0.0, zl)
        ds  = self.Da(0.0, zs)
        dls = self.Da(zl, zs)

        D = dls*dl/ds   # Mpc/h
        scinv = D*self._four_pi_G_over_c_squared

        w,=numpy.where(zs <= zl)
        if w.size > 0:
            scinv[w] = 0.0

        return scinv

def four_pi_G_over_c_squared(dunits='Mpc'):
    """
    4*pi*G/c^2 Dl * Dls/Ds has units of m^2/kg

    we want units of pc^2/kg and Dl in units specified
    by the dunits keyword
    """

    from math import pi as PI

    # we want the formula to return pc^2/Msun
    C=2.99792458e8 # m/s
    GNEWTON=6.67428e-11 # m^3/kg/s^2

    KG_PER_SUN=1.98892e30 # kg
    M_PER_PARSEC=3.08568025e16

    # m^2/kg
    fourpiGoverc2 = 4.0*PI*GNEWTON/(C**2)

    if dunits == 'meters':
        return fourpiGoverc2
    else:
        # pc^2/msun, but would require Dl in parsecs
        fourpiGoverc2 *= KG_PER_SUN/M_PER_PARSEC

        if dunits == 'kpc':
            return fourpiGoverc2*1.e3
        elif dunits == 'Mpc':
            return fourpiGoverc2*1.e6
        elif dunits == 'Gpc':
            return fourpiGoverc2*1.e9
        else:
            raise ValueError("Don't support dunits='%s'" % dunits)



def DH(h=1.0):
    """
    NAME:
        DH
    PURPOSE:
        Calculate the Hubble distance in Mpc/h
    CALLING SEQUENCE:
        d = DH(h=1.0)
    """
    return 2.9979e5/100.0/h

def Ez_inverse(z, omega_m, omega_l, omega_k):
    """
    NAME:
        Ez_inverse
    PURPOSE:
        Calculate kernel 1/E(z) for distance integrals in FRW universe.
    CALLING SEQUENCE:
        ezi = Ez_inverse(z, omega_m, omega_l, omega_k)
    """
    return 1.0/sqrt( omega_m*(1.0+z)**3 + omega_k*(1.0+z)**2 + omega_l)

# Old slower version using scipy integrator
def Ezinv_integral_old(z1, z2, omega_m, omega_l, omega_k):
    """
    NAME:
        Ezinv_integral
    PURPOSE:
        Integrate kernel 1/E(z) used for distance calculations in FRW
        universe. Uses the "quad" integrator in scipy, which calls
        the fortran library QUADPACK.
    CALLING SEQUENCE:
        ezint = Ezinv_integral(z1, z2, omega_m, omega_l, omega_k)
    """
    # just import here since we don't use this old version any more
    import scipy.integrate
    (val, err) = scipy.integrate.quad(Ez_inverse, z1, z2, 
                                      args=(omega_m,omega_l,omega_k))
    return numpy.abs(val)


def _ezi_run_gauleg(npts):
    if _EZI_XXi.size != npts:
        globals()['_EZI_XXi'], globals()['_EZI_WWi'] = esutil.integrate.gauleg(-1.0,1.0,npts)

def Ezinv_integral(z1, z2, omega_m, omega_l, omega_k, npts=5):
    """
    NAME:
        Ezinv_integral
    PURPOSE:
        Integrate kernel 1/E(z) used for distance calculations in FRW
        universe. Gauss-legendre integration.  Defaults to npts=5 which
        is actually good to 1.e-8 to redshift 1 because it is such a slow
        function.
    CALLING SEQUENCE:
        ezint = Ezinv_integral(z1, z2, omega_m, omega_l, omega_k, npts=5)
    INPUTS:
        z1, z2: The redshift interval, scalars.
        omega_m, omega_l, omega_k: Density parameters relative to critical.
        h: Hubble parameter. Default 1.0
        npts: Number of points in the integration. Default 5, good to 1.e-8
            to redshift 1.
    """

    # Will only change if npts changes 
    _ezi_run_gauleg(npts)

    f1 = (z2-z1)/2.
    f2 = (z2+z1)/2.

    zvals = _EZI_XXi*f1 + f2
    ezivals = Ez_inverse(zvals, omega_m, omega_l, omega_k)

    ezint = f1 * ((ezivals*_EZI_WWi).sum())
    return abs(ezint)


def _extract_omegas(omega_m, omega_l, omega_k, flat):
    if flat:
        omega_l = 1.0-omega_m
        omega_k = 0.0
    return (omega_m, omega_l, omega_k)


def Dc(z1in, z2in, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, flat=True, 
       npts=5):
    """
    NAME:
        Dc
    PURPOSE:
        Calculate the comoving distance between redshifts z1 and z2 in
        a FRW universe. Units: Mpc 
    CALLING SEQUENCE:
        d=Dc(z1, z2, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
             flat=True, npts=5)
    INPUTS:
        z1, z2: The redshifts.  These must either be 
           1) Two scalars
           2) A scalar and an array.
           3) Two arrays of the same length.
        omega_m, omega_l, omega_k: Density parameters relative to critical.
          If flat=True, then only omega_m is used, omega_l is set to
          1.0 - omega_m, and omega_k=0.0.   Defaults, 0.3, 0.7, 0.0
        h: Hubble parameter. Default 1.0
        flat: Should we assume a flat cosmology?  Default True.
        npts: Number of points in the integration. Default 5, good to 1.e-8
            to redshift 1.
    """
  
    (omega_m, omega_l, omega_k) = _extract_omegas(omega_m, omega_l, omega_k, 
                                                  flat)
    
    # Make sure they are arrays, but don't copy if already an array
    z1 = numpy.array(z1in, ndmin=1, copy=False)
    z2 = numpy.array(z2in, ndmin=1, copy=False)

    # All the permutations of inputs
    dh=DH(h=h)
    if z1.size == z2.size:
        if z1.size == 1:
            return dh*Ezinv_integral(z1,z2,omega_m,omega_l,omega_k,npts=npts)
        else:
            dc = numpy.zeros(z1.size)
            for i in numpy.arange(z1.size):
                dc[i] = dh*Ezinv_integral(z1[i],z2[i],
                                          omega_m,omega_l,omega_k,npts=npts)
    else:
        if z1.size == 1:
            dc = numpy.zeros(z2.size)
            for i in numpy.arange(z2.size):
                dc[i] = dh*Ezinv_integral(z1,z2[i],
                                          omega_m,omega_l,omega_k,npts=npts)
        elif z2.size == 1:
            dc = numpy.zeros(z1.size)
            for i in numpy.arange(z1.size):
                dc[i] = dh*Ezinv_integral(z1[i],z2,
                                          omega_m,omega_l,omega_k,npts=npts)
        else:
            raise ValueError("z1,z2: Must be same length or one a scalar")

    return dc

def Dm(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, flat=True, 
       npts=5):
    """
    NAME:
        Dm
    PURPOSE:
        Calculate the transverse comoving distance between two objects at the
        same redshift in a a FRW universe.  Units: Mpc.
    CALLING SEQUENCE:
        d=Dm(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
             flat=True, npts=5)
    INPUTS:
        zmin, zmax: The redshifts.  Note, to interpret as the transverse
          distance between objects at the same redshift as viewed by a redshift
          zero observer, zmin=0.0  It is useful to allow zmin != 0 when 
          measuring for example angular diameter distances between two non
          zero redshifts, as in lensing calculations.  These redshifts must 
          either be 
            1) Two scalars
            2) A scalar and an array.
            3) Two arrays of the same length.
        omega_m, omega_l, omega_k: Density parameters relative to critical.
          If flat=True, then only omega_m is used, omega_l is set to
          1.0 - omega_m, and omega_k=0.0.   Defaults, 0.3, 0.7, 0.0
        h: Hubble parameter. Default 1.0
        flat: Should we assume a flat cosmology?  Default True.
        npts: Number of points in the integration. Default 5, good to 1.e-8
            to redshift 1.
    """
    (omega_m, omega_l, omega_k) = _extract_omegas(omega_m, omega_l, omega_k, 
                                                  flat)

    dh = DH(h=h)
    dc=Dc(zmin, zmax, omega_m, omega_l, omega_k, h=h, flat=flat,npts=npts)

    if omega_k == 0:
        return dc
    elif omega_k > 0:
        return dh/sqrt(omega_l)*sinh( sqrt(omega_k)*dc/dh )
    else:
        return dh/sqrt(omega_l)*sin( sqrt(omega_k)*dc/dh )

def Da(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, flat=True,
       npts=5):
    """
    NAME:
        Da 
    PURPOSE:
        Calculate the angular diameter distance between z1 and z2 in a 
        FRW universe. Units: Mpc.
    CALLING SEQUENCE:
        d=Da(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
             flat=True, npts=5)
    INPUTS:
        zmin, zmax: The redshifts.  These must either be 
           1) Two scalars
           2) A scalar and an array.
           3) Two arrays of the same length.
        omega_m, omega_l, omega_k: Density parameters relative to critical.
          If flat=True, then only omega_m is used, omega_l is set to
          1.0 - omega_m, and omega_k=0.0.   Defaults, 0.3, 0.7, 0.0
        h: Hubble parameter. Default 1.0
        flat: Should we assume a flat cosmology?  Default True.
        npts: Number of points in the integration. Default 5, good to 1.e-8
            to redshift 1.
    """
    z1 = numpy.array(zmin, ndmin=1, copy=False)
    z2 = numpy.array(zmax, ndmin=1, copy=False)
    d = Dm(z1, z2, omega_m, omega_l, omega_k, h=h, flat=flat, npts=npts)

    da = numpy.where( z1 < z2, d/(1.0+z2), d/(1.0+z1) )

    return da

def Dl(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, flat=True,
       npts=5):
    """
    NAME:
        Dl
    PURPOSE:
        Calculate the luminosity distance between z1 and z2 in a 
        FRW universe. Units: Mpc.
    CALLING SEQUENCE:
        d=Dl(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
             flat=True, npts=5)
    INPUTS:
        zmin, zmax: The redshifts.  These must either be 
           1) Two scalars
           2) A scalar and an array.
           3) Two arrays of the same length.
        omega_m, omega_l, omega_k: Density parameters relative to critical.
          If flat=True, then only omega_m is used, omega_l is set to
          1.0 - omega_m, and omega_k=0.0.   Defaults, 0.3, 0.7, 0.0
        h: Hubble parameter. Default 1.0
        flat: Should we assume a flat cosmology?  Default True.
        npts: Number of points in the integration. Default 5, good to 1.e-8
            to redshift 1.
    """
    return Da(zmin,zmax,omega_m,omega_l,omega_k,h,flat,npts)*(1.0+zmax)**2

def Distmod(z, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, flat=True, 
            npts=5):
    """
    NAME:
        Distmod
    PURPOSE:
        Calculate the distance modulus to redshift z.
    CALLING SEQUENCE:
        d=Distmod(z, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
                  flat=True, npts=5)
    INPUTS:
        z: The redshift(s).
        omega_m, omega_l, omega_k: Density parameters relative to critical.
          If flat=True, then only omega_m is used, omega_l is set to
          1.0 - omega_m, and omega_k=0.0.   Defaults, 0.3, 0.7, 0.0
        h: Hubble parameter. Default 1.0
        flat: Should we assume a flat cosmology?  Default True.
        npts: Number of points in the integration. Default 5, good to 1.e-8
            to redshift 1.
    """

    dmpc = Dl(0.0, z, omega_m=omega_m, omega_l=omega_l, omega_k=omega_k, 
              h=h, flat=flat, npts=npts)      
    dpc = dmpc*1.e6
    dm = 5.0*log10(dpc/10.0)
    return dm      


def dV(z, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, flat=True, 
       npts=5, comoving=True):
    """
    NAME:
        dV
    PURPOSE:
        Calculate the volume elementd dV in a FRW universe. Units: Mpc**3
    CALLING SEQUENCE:
        dv = dV(z, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
                flat=True, npts=5, comoving=True)
    INPUTS:
        z: The redshift
        omega_m, omega_l, omega_k: Density parameters relative to critical.
          If flat=True, then only omega_m is used, omega_l is set to
          1.0 - omega_m, and omega_k=0.0.   Defaults, 0.3, 0.7, 0.0
        h: Hubble parameter. Default 1.0
        flat: Should we assume a flat cosmology?  Default True.
        npts: Number of points in the integration. Default 5, good to 1.e-8
            to redshift 1.
    """

    dh = DH(h=h)
    da = Da(0.0, z, omega_m, omega_l, omega_k, h=h, flat=flat, npts=npts)
    Ez = 1.0/Ez_inverse(z, omega_m, omega_l, omega_k)
    if comoving:
        dv = dh*da**2/Ez*(1.0+z)**2
    else:
        dv = dh*da**2/Ez*(1.0+z)

    return dv

# This is about a factor of 3 slower than the new one
def Vold(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
         flat=True, npts=5, comoving=True):
    """
    NAME:
        V
    PURPOSE:
        Calculate the volume between zmin and zmax in an FRW universe.
        Units: Mpc**3
    CALLING SEQUENCE:
        v = V(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
              flat=True, npts=5, comoving=True)
    INPUTS:
        zmin, zmax The redshift limits.  
        omega_m, omega_l, omega_k: Density parameters relative to critical.
          If flat=True, then only omega_m is used, omega_l is set to
          1.0 - omega_m, and omega_k=0.0.   Defaults, 0.3, 0.7, 0.0
        h: Hubble parameter. Default 1.0
        flat: Should we assume a flat cosmology?  Default True.
        npts: Number of points in the distance integration. Default 5, good to 
            1.e-8 to redshift 1.
    """

    # just import here since we don't use this old version any more
    import scipy.integrate
    (v,err) = scipy.integrate.quad(dV, zmin, zmax, 
                                   args=(omega_m,omega_l,omega_k,h,flat,npts))
    return v


def _vi_run_gauleg(npts):
    if _VI_XXi.size != npts:
        globals()['_VI_XXi'], globals()['_VI_WWi'] = esutil.integrate.gauleg(-1.0,1.0,npts)
 
def V(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
      flat=True, npts=5, vnpts=10, comoving=True):
    """
    NAME:
        V
    PURPOSE:
        Calculate the volume between zmin and zmax in an FRW universe.
        Units: Mpc**3
    CALLING SEQUENCE:
        v = V(zmin, zmax, omega_m=0.3, omega_l=0.7, omega_k=0.0, h=1.0, 
              flat=True, npts=5, vnpts=100, comoving=True)
    INPUTS:
        zmin, zmax The redshift limits.  
        omega_m, omega_l, omega_k: Density parameters relative to critical.
          If flat=True, then only omega_m is used, omega_l is set to
          1.0 - omega_m, and omega_k=0.0.   Defaults, 0.3, 0.7, 0.0
        h: Hubble parameter. Default 1.0
        flat: Should we assume a flat cosmology?  Default True.
        npts: Number of points in the distance integration. Default 5, good to 
            1.e-8 to redshift 1.
        vnpts: Number of points in the volume integration. Default is 10
        comoving: Use comoving coords, default True.
    """

    # Will only change if npts changes 
    _vi_run_gauleg(vnpts)

    # these needed for coordinate transformation
    f1 = (zmax-zmin)/2.
    f2 = (zmax+zmin)/2.

    zvals = _VI_XXi*f1 + f2
    ezivals = dV(zvals, omega_m, omega_l, omega_k, h, flat, 
                 comoving=comoving, npts=npts)

    v =  f1 * ((ezivals*_VI_WWi).sum())
    v = numpy.array(v, ndmin=1)
    return v





def test_sigmacritinv_npts(epsfile=None):
    """
    Test accuracy of sigmacrit inv as a function of number
    of points
    """

    from biggles import FramedPlot, PlotKey, PlotLabel, Curve, FramedArray, Table

    c1000 = Cosmo(npts=1000)
    c5 = Cosmo(npts=5)
    c4 = Cosmo(npts=4)
    c3 = Cosmo(npts=3)
    c2 = Cosmo(npts=2)


    tab = Table( 2, 2 )
    tab.uniform_limits = 0
    tab.cellspacing = 1

    p5 = FramedPlot()
    p5.xlabel = 'zsource'
    p5.ylabel = '% diff'
    p4 = FramedPlot()
    p4.xlabel = 'zsource'
    p4.ylabel = '% diff'
    p3 = FramedPlot()
    p3.xlabel = 'zsource'
    p3.ylabel = '% diff'
    p2 = FramedPlot()
    p2.xlabel = 'zsource'
    p2.ylabel = '% diff'
    
    l5 = PlotLabel(0.2,0.7,'npts = 5')
    p5.add(l5)
    l4 = PlotLabel(0.2,0.1,'npts = 4')
    p4.add(l4)
    l3 = PlotLabel(0.2,0.9,'npts = 3')
    p3.add(l3)
    l2 = PlotLabel(0.2,0.9,'npts = 2')
    p2.add(l2)


    colors = ['black','violet','blue','green','orange','magenta','firebrick','red']
    zlvals = numpy.arange(0.1,0.9,0.1)
    if zlvals.size != len(colors):
        raise ValueError("mismatch colors and zlvals")



    i=0

    allc5 = []
    for zl in zlvals:

        zs = numpy.arange(zl+0.01, 2.0, 0.01)

        scinv = c1000.sigmacritinv(zl, zs)
        scinv5 = c5.sigmacritinv(zl, zs)
        scinv4 = c4.sigmacritinv(zl, zs)
        scinv3 = c3.sigmacritinv(zl, zs)
        scinv2 = c2.sigmacritinv(zl, zs)

        curve5 = Curve(zs, 100*(scinv-scinv5)/scinv, color=colors[i])
        curve5.label = 'zlens: %0.2f' % zl
        allc5.append(curve5)
        p5.add(curve5)

        curve4 = Curve(zs, 100*(scinv-scinv4)/scinv, color=colors[i])
        p4.add(curve4)

        curve3 = Curve(zs, 100*(scinv-scinv3)/scinv, color=colors[i])
        p3.add(curve3)

        curve2 = Curve(zs, 100*(scinv-scinv2)/scinv, color=colors[i])
        p2.add(curve2)

        i+=1

    key = PlotKey(0.15,0.5,allc5)
    p5.add(key)


    tab[0,0] = p5
    tab[0,1] = p4
    tab[1,0] = p3
    tab[1,1] = p2

    tab.show()

    if epsfile is not None:
        tab.write_eps(epsfile)
