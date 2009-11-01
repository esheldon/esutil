"""
    NAME:
        cosmology
    PURPOSE:
        A set of tools for calculating distances in an expanding universe.
        These routines are completely general for any specified omega_m,
        omega_k, and cosmological constant omega_l.  This code follows the
        conventions of Hogg astro-ph/9905116.
        
        All distances are in units of Mpc/h unless h is specified. Volumes are
        in (Mpc/h)**3.  All return values are arrays.

    FUNCTIONS:
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

    EXAMPLES:
        >>> import cosmology as c
        >>> c.Da(0.0, 0.35, omega_m=0.24, h=0.7)
        array([ 1034.76013423])

    REQUIREMENTS:
        NumPy
        SciPy for integrations
    REVISION HISTORY:
        Copied from IDL routines.  2006-11-07, Erin Sheldon, NYU
        Converted to using faster Gauss-Legendre integration 
            2007-05-17, Erin Sheldon, NYU


"""

license="""
  Copyright (C) 2009  Erin Sheldon

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

try:
    import numpy
    from numpy import sqrt, sin, sinh, log10

    have_numpy=True

    # Global variables for Ez integration.  
    _EZI_XXi=numpy.array([])
    _EZI_WWi=numpy.array([])

    # Global variables for volume integration.  
    _VI_XXi=numpy.array([])
    _VI_WWi=numpy.array([])


except:
    have_numpy=False

try:
    import scipy.integrate
    have_scipy=True
except:
    have_scipy=False

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
    (val, err) = scipy.integrate.quad(Ez_inverse, z1, z2, 
                                      args=(omega_m,omega_l,omega_k))
    return numpy.abs(val)


def _ezi_run_gauleg(npts):
    if _EZI_XXi.size != npts:
        globals()['_EZI_XXi'], globals()['_EZI_WWi'] = gauleg(-1.0,1.0,npts)

def Ezinv_integral(z1, z2, omega_m, omega_l, omega_k, npts=5):
    """
    NAME:
        Ezinv_integral
    PURPOSE:
        Integrate kernel 1/E(z) used for distance calculations in FRW
        universe. Gauss-legendre integration.  Defaults to npts=5 which
        is actually good to 0.05% to redshift 1 because it is such a slow
        function.
    CALLING SEQUENCE:
        ezint = Ezinv_integral(z1, z2, omega_m, omega_l, omega_k, npts=5)
    INPUTS:
        z1, z2: The redshift interval, scalars.
        omega_m, omega_l, omega_k: Density parameters relative to critical.
        h: Hubble parameter. Default 1.0
        npts: Number of points in the integration. Default 5, good to 0.05%
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
        npts: Number of points in the integration. Default 5, good to 0.05%
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
        npts: Number of points in the integration. Default 5, good to 0.05%
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
        npts: Number of points in the integration. Default 5, good to 0.05%
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
        npts: Number of points in the integration. Default 5, good to 0.05%
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
        npts: Number of points in the integration. Default 5, good to 0.05%
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
        npts: Number of points in the integration. Default 5, good to 0.05%
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
            0.05% to redshift 1.
    """

    (v,err) = scipy.integrate.quad(dV, zmin, zmax, 
                                   args=(omega_m,omega_l,omega_k,h,flat,npts))
    return v


def _vi_run_gauleg(npts):
    if _VI_XXi.size != npts:
        globals()['_VI_XXi'], globals()['_VI_WWi'] = gauleg(-1.0,1.0,npts)
 
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
            0.05% to redshift 1.
        vnpts: Number of points in the volume integration. Default is 10
        comoving: Use comoving coords, default True.
    """

    # Will only change if npts changes 
    _vi_run_gauleg(vnpts)

    # these needed for coordinate transformation
    f1 = (zmax-zmin)/2.
    f2 = (zmax+zmin)/2.

    zvals = _VI_XXi*f1 + f2
    ezivals = dV(zvals, omega_m, omega_l, omega_k, h, flat, comoving, npts=npts)

    v =  f1 * ((ezivals*_VI_WWi).sum())
    v = numpy.array(v, ndmin=1)
    return v





def gauleg(x1, x2, npts):
    """
    NAME:
      gauleg()
      
    PURPOSE:
      Calculate the weights and abscissa for Gauss-Legendre integration.
    
    CALLING SEQUENCE:
      x,w = gauleg(x1,x2,npts)

    INPUTS:
      x1,x2: The range for the integration.
      npts: Number of points to use in the integration.

    REVISION HISTORY:
      Created: 2006-10-24. Adapted from Numerial recipes in C. Uses
        scipy.weave.inline for the C loops.  2006-10-24 Erin Sheldon NYU
    """

    try:
        from scipy import weave
    except:
        raise ImportError("scipy.weave could not be imported")
    # outputs
    x = numpy.zeros(npts, dtype='f8')
    w = numpy.zeros(npts, dtype='f8')

    # Code string for weave
    code = \
         """
         int i, j, m;
         double xm, xl, z1, z, p1, p2, p3, pp, pi, EPS, abszdiff;
         
         EPS = 3.e-11;
         pi=3.1415927;

         m = (npts + 1)/2;

         xm = (x1 + x2)/2.0;
         xl = (x2 - x1)/2.0;
         z1 = 0.0;

         for (i=1; i<= m; ++i) 
         {
      
           z=cos( pi*(i-0.25)/(npts+.5) );

           abszdiff = fabs(z-z1);

           while (abszdiff > EPS) 
           {
             p1 = 1.0;
             p2 = 0.0;
             for (j=1; j <= npts;++j)
             {
                p3 = p2;
                p2 = p1;
                p1 = ( (2.0*j - 1.0)*z*p2 - (j-1.0)*p3 )/j;
             }
             pp = npts*(z*p1 - p2)/(z*z -1.);
             z1=z;
             z=z1 - p1/pp;

             abszdiff = fabs(z-z1);

           }
      
           x(i-1) = xm - xl*z;
           x(npts+1-i-1) = xm + xl*z;
           w(i-1) = 2.0*xl/( (1.-z*z)*pp*pp );
           w(npts+1-i-1) = w(i-1);


         }

         return_val = 0;
         

         """
    
    weave.inline(code, ['x1', 'x2', 'npts', 'x', 'w'],
                       type_converters = weave.converters.blitz)

    return x,w


