"""
    NAME
        coords
    PURPOSE
        A set of astronomical utilities for dealing with coordinates and
        coordinate transformations.

    COORDINATE TRANSFORMATIONS
        euler:  
            A generic routine for transforming between Galactic, Celestial,
            and ecliptic coords.  The following wrapper routines are also
            supplied for convenience:

        l,b = eq2gal(ra, dec, b1950=False, dtype='f8')
            Convert equatorial to glactic coordinates.

        # The following use the same interface:
        gal2eq
            Convert galactic to equatorial coordinates.
        eq2ec
            Convert equatorial to ecliptic coordinates.
        ec2eq
            Convert ecliptic to equatorial coordinates.
        ec2gal
            Convert ecliptic to galactic coordinates.
        gal2ec
            Convert galactic to ecliptic coordinates.

        # These SDSS specific functions do not use euler
        eq2sdss
            Convert between equatorial and corrected SDSS survey coords.
        sdss2eq
            Convert between corrected SDSS survey and equatorial coords.

        eq2xyz: Convert equatorial to x,y,z on the sphere according to
            the following transform:
                    x = sin(pi/2-dec)*cos(ra)
                    y = sin(pi/2-dec)*sin(ra)
                    z = cos(pi/2-dec)

        xyz2eq: 
            inverse of eq2xyz

        sphdist:
            Calculate the arc length between two sets of points on the sphere.
            Currently only takes ra,dec.

        shiftlon:
            shift the input longitude.  By default wrap the coordinate to
            -180,180.  If a shift is entered, return the new value
            lon-shift such that the range is still [0,360)
            
        shiftra:
            shift right ascension.  This just calls shiftlon

        radec2aitoff:
            Convert ra,dec to aitoff coordinates.

        dec_parse(decstring)
            parse a colon separated string representing declination ito
            degrees.
        ra_parse(decstring)
            parse a colon separated string representing right ascension ito
            degrees.

        randsphere(numrand, system='lonlat'):
            Generate random points on the sphere.  By default lon,lat are
            returned.  If system='xyz' then x,y,z are returned.

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
    from numpy import where, sin, cos, arccos, arcsin, arctan2, sqrt
    have_numpy=True
except:
    have_numpy=False

import math
PI=math.pi
D2R = PI/180.0
R2D = 1.0/D2R

_sdsspar = {}
_sdsspar['center_ra'] = 185.0
_sdsspar['center_dec'] = 32.5
_sdsspar['node'] = (_sdsspar['center_ra'] - 90.0)*D2R
_sdsspar['etapole'] = _sdsspar['center_dec']*D2R
_sdsspar['etaoffset'] = 91.25

_sdsspar['doc'] = """
    A set of transformation functions for use with SDSS coordinate systems.

    eq2sdss(): Convert between equatorial and corrected SDSS survey coords.
    sdss2eq(): Convert between corrected SDSS survey and equatorial coords.

    Don't use these unless you have to, as these are the old coordinates
        eq2survey(): Convert between equatorial and uncorrected survey coords.
        survey2eq(): Convert between uncorrected survey and equatorial coords.

    Adapted from astrotools
        Erin Sheldon, NYU, 2006-03-11
    Force data type and allow selection of dtype through keyword.
        Erin Sheldon, NYU, 2007-05-23

"""


def euler(ai_in, bi_in, select, b1950=False, dtype='f8'):
    """
    NAME:
        euler
    PURPOSE:
        Transform between Galactic, celestial, and ecliptic coordinates.

    CALLING SEQUENCE:
        long_out, lat_out = 
            euler(long_in, lat_in, type, b1950=False, dtype='f8')

    INPUTS:
       long_in - Input Longitude in DEGREES, scalar or vector.  
       lat_in  - Input Latitude in DEGREES
       select  - Integer (1-6) specifying type of coordinate transformation.  

      select   From          To        |   select      From            To
       1     RA-Dec (2000)  Galactic   |     4       Ecliptic      RA-Dec    
       2     Galactic       RA-DEC     |     5       Ecliptic      Galactic  
       3     RA-Dec         Ecliptic   |     6       Galactic      Ecliptic  

      Celestial coordinates (RA, Dec) should be given in equinox J2000 
      unless the b1950=True keyword is set.

    OUTPUTS:
       long_out - Output Longitude in DEGREES
       lat_out  - Output Latitude in DEGREES

    INPUT KEYWORD:
       b1950 - If this keyword is true then input and output 
             celestial and ecliptic coordinates should be given in equinox 
             B1950.
    REVISION HISTORY:
       Written W. Landsman,  February 1987
       Adapted from Fortran by Daryl Yentis NRL
       Converted to IDL V5.0   W. Landsman   September 1997
       Made J2000 the default, added /FK4 keyword  W. Landsman December 1998
       Add option to specify SELECT as a keyword W. Landsman March 2003

       Converted from IDL to numerical Python: Erin Sheldon, NYU, 2008-07-02

    """

    # Make a copy as an array. ndmin=1 to avoid messed up scalar arrays
    ai = numpy.array(ai_in, ndmin=1, copy=True, dtype=dtype)
    bi = numpy.array(bi_in, ndmin=1, copy=True, dtype=dtype)

    twopi   =   2.0*PI
    fourpi  =   4.0*PI

    #   J2000 coordinate conversions are based on the following constants
    #   (see the Hipparcos explanatory supplement).
    #  eps = 23.4392911111d           Obliquity of the ecliptic
    #  alphaG = 192.85948d            Right Ascension of Galactic North Pole
    #  deltaG = 27.12825d             Declination of Galactic North Pole
    #  lomega = 32.93192d             Galactic longitude of celestial equator  
    #  alphaE = 180.02322d            Ecliptic longitude of Galactic North Pole
    #  deltaE = 29.811438523d         Ecliptic latitude of Galactic North Pole
    #  Eomega  = 6.3839743d           Galactic longitude of ecliptic equator              
    # Parameters for all the different conversions
    if b1950:

        equinox = '(B1950)' 
        psi    = numpy.array([ 0.57595865315, 4.9261918136,  
                              0.00000000000, 0.0000000000,  
                              0.11129056012, 4.7005372834], dtype=dtype)
        stheta = numpy.array([ 0.88781538514,-0.88781538514, 
                              0.39788119938,-0.39788119938, 
                              0.86766174755,-0.86766174755], dtype=dtype)
        ctheta = numpy.array([ 0.46019978478, 0.46019978478, 
                              0.91743694670, 0.91743694670, 
                              0.49715499774, 0.49715499774], dtype=dtype) 
        phi  =   numpy.array([ 4.9261918136,  0.57595865315, 
                              0.0000000000, 0.00000000000, 
                              4.7005372834, 0.11129056012], dtype=dtype)

    else:

        equinox = '(J2000)'

        psi    = numpy.array([ 0.57477043300, 4.9368292465,  
                              0.00000000000, 0.0000000000,    
                              0.11142137093, 4.71279419371], dtype=dtype)
        stheta = numpy.array([ 0.88998808748,-0.88998808748, 
                              0.39777715593,-0.39777715593, 
                              0.86766622025,-0.86766622025], dtype=dtype)
        ctheta = numpy.array([ 0.45598377618, 0.45598377618, 
                              0.91748206207, 0.91748206207, 
                              0.49714719172, 0.49714719172], dtype=dtype)
        phi    = numpy.array([ 4.9368292465,  0.57477043300, 
                              0.0000000000, 0.00000000000, 
                              4.71279419371, 0.11142137093], dtype=dtype)

    # zero offset
    i  = select - 1 
    a  = ai*D2R - phi[i]
    
    b = bi*D2R
    sb = sin(b)
    cb = cos(b)
    cbsa = cb * sin(a)
    b  = -stheta[i] * cbsa + ctheta[i] * sb
    w, = numpy.where(b > 1.0)
    if w.size > 0:
        b[w] = 1.0
    bo    = arcsin(b)*R2D

    a =  arctan2( ctheta[i] * cbsa + stheta[i] * sb, cb * cos(a) )

    ao = ( (a+psi[i]+fourpi) % twopi) * R2D

    return ao, bo



#
# Some clearer shortcut functions which call Euler
#
def eq2gal(ra, dec, b1950=False, dtype='f8'):
    """
    NAME
        eq2gal
    PURPOSE
        Convert from equatorial to galactic coordinates in units of degrees.
    CALLING SEQUENCE
        l,b = eq2gal(ra, dec, b1950=False, dtype='f8')
    INPUTS
        ra, dec: Equatorial coordinates.  May be Numpy arrays, sequences, or
            scalars as long as they are all the same length.  They must be
            convertible to a Numpy array with the specified datatype.
    KEYWORDS
        b1950:  If True, use b1950 coordiates.  By default j2000 are used.
        dtype:  The datatype of the output arrays.  Default is f8
    OUTPUTS
        l, b:  Galactic longitude and latitude.  The returned value is always
            a Numpy array with the specified dtype
    REVISION HISTORY
        Created Erin Sheldon, NYU, 2008-07-02
    """
    return euler(ra, dec, 1, b1950=b1950, dtype=dtype)


def gal2eq(l, b, b1950=False, dtype='f8'):
    """
    NAME
        gal2eq
    PURPOSE
        Convert from galactice to equatorial coordinates in units of degrees.
    CALLING SEQUENCE
        ra,dec = gal2eq(l, b, b1950=False, dtype='f8')
    INPUTS
        l, b: Galactic coordinates.  May be Numpy arrays, sequences, or
            scalars as long as they are all the same length.  They must be
            convertible to a Numpy array with the specified datatype.
    KEYWORDS
        b1950:  If True, use b1950 coordiates.  By default j2000 are used.
        dtype:  The datatype of the output arrays.  Default is f8
    OUTPUTS
        ra, dec:  Equatorial longitude and latitude.  The returned value is 
            always a Numpy array with the specified dtype
    REVISION HISTORY
        Created Erin Sheldon, NYU, 2008-07-02
    """

    return euler(l, b, 2, b1950=b1950, dtype=dtype)

def eq2ec(ra, dec, b1950=False, dtype='f8'):
    """
    NAME
        eq2ec
    PURPOSE
        Convert from equatorial to ecliptic coordinates in units of degrees.
    CALLING SEQUENCE
        lam,beta = eq2ec(ra, dec, b1950=False, dtype='f8')
    INPUTS
        ra, dec: Equatorial coordinates.  May be Numpy arrays, sequences, or
            scalars as long as they are all the same length.  They must be
            convertible to a Numpy array with the specified datatype.
    KEYWORDS
        b1950:  If True, use b1950 coordiates.  By default j2000 are used.
        dtype:  The datatype of the output arrays.  Default is f8
    OUTPUTS
        lam, beta:  Ecliptic longitude and latitude.  The returned value is 
            always a Numpy array with the specified dtype
    REVISION HISTORY
        Created Erin Sheldon, NYU, 2008-07-02
    """

    return euler(ra, dec, 3, b1950=b1950, dtype=dtype)

def ec2eq(lam, beta, b1950=False, dtype='f8'):
    """
    NAME
        ec2eq
    PURPOSE
        Convert from ecliptic to equatorial coordinates in units of degrees.
    CALLING SEQUENCE
        ra,dec = eq2gal(lam, beta, b1950=False, dtype='f8')
    INPUTS
        lam,beta: Ecliptic coordinates.  May be Numpy arrays, sequences, or
            scalars as long as they are all the same length.  They must be
            convertible to a Numpy array with the specified datatype.
    KEYWORDS
        b1950:  If True, use b1950 coordiates.  By default j2000 are used.
        dtype:  The datatype of the output arrays.  Default is f8
    OUTPUTS
        ra,dec:  Equatorial longitude and latitude.  The returned value is 
            always a Numpy array with the specified dtype
    REVISION HISTORY
        Created Erin Sheldon, NYU, 2008-07-02
    """

    return euler(lam, beta, 4, b1950=b1950, dtype=dtype)

def ec2gal(lam, beta, b1950=False, dtype='f8'):
    """
    NAME
        ec2gal
    PURPOSE
        Convert from ecliptic to galactic coordinates in units of degrees.
    CALLING SEQUENCE
        l,b = eq2gal(lam, beta, b1950=False, dtype='f8')
    INPUTS
        lam, beta: Ecliptic coordinates.  May be Numpy arrays, sequences, or
            scalars as long as they are all the same length.  They must be
            convertible to a Numpy array with the specified datatype.
    KEYWORDS
        b1950:  If True, use b1950 coordiates.  By default j2000 are used.
        dtype:  The datatype of the output arrays.  Default is f8
    OUTPUTS
        l, b:  Galactic longitude and latitude.  The returned value is always
            a Numpy array with the specified dtype
    REVISION HISTORY
        Created Erin Sheldon, NYU, 2008-07-02
    """

    return euler(lam, beta, 5, b1950=b1950, dtype=dtype)

def gal2ec(l, b, b1950=False, dtype='f8'):
    """
    NAME
        gal2ec
    PURPOSE
        Convert from Galactic to Ecliptic coordinates in units of degrees.
    CALLING SEQUENCE
        lam,beta = eq2gal(l, b, b1950=False, dtype='f8')
    INPUTS
        l, b: Galactic coordinates.  May be Numpy arrays, sequences, or
            scalars as long as they are all the same length.  They must be
            convertible to a Numpy array with the specified datatype.
    KEYWORDS
        b1950:  If True, use b1950 coordiates.  By default j2000 are used.
        dtype:  The datatype of the output arrays.  Default is f8
    OUTPUTS
        lam,beta:  Ecliptic longitude and latitude.  The returned value is 
            always a Numpy array with the specified dtype
    REVISION HISTORY
        Created Erin Sheldon, NYU, 2008-07-02
    """

    return euler(l, b, 6, b1950=b1950, dtype=dtype)


def _thetaphi2xyz(theta, phi):
    """
    theta and phi in radians relative to the SDSS node at ra=95 degrees
    """
    x = cos(theta)*cos(phi)
    y = sin(theta)*cos(phi);
    z = sin(phi);

    return x,y,z

def _xyz2thetaphi(x,y,z):
    """
    returns theta, phi in radians relative to the SDSS node at ra=95 degrees
    """
    phi = arcsin(z)
    theta = arctan2(y,x)

    return theta, phi

def eq2xyz(ra, dec, dtype='f8'):
    """
    Name:
        eq2xyz
    Purpose:
        Convert equatorial coordinates RA and DEC to x,y,z on
        the unit sphere
    Calling Sequence:
        x,y,z = eq2xyz(ra,dec)

    Notes:
        This follows the same convention as the STOMP package.
    """

    theta = numpy.array(ra, ndmin=1, copy=False, dtype=dtype)
    phi = numpy.array(dec, ndmin=1, copy=False, dtype=dtype)

    # in place is more efficient
    theta *= D2R
    phi *= D2R

    theta -= _sdsspar['node']

    return _thetaphi2xyz(theta, phi)

def xyz2eq(xin,yin,zin):
    x = numpy.array(xin, ndmin=1, copy=False)
    y = numpy.array(yin, ndmin=1, copy=False)
    z = numpy.array(zin, ndmin=1, copy=False)

    theta,phi = _xyz2thetaphi(x,y,z)
    theta += _sdsspar['node']


    theta *= R2D
    phi *= R2D

    atbound(theta, 0.0, 360.0)

    # theta->ra, phi->dec
    return theta,phi

   

# Not good for lensing, we also need the angle in the ra-dec or whatever
# coords
def sphdist(ra1, dec1, ra2, dec2):
    """
    Name:
        sphdist
    Purpose:
        Get the arc length between two points on the unit sphere
    Calling Sequence:
        d = sphdist(ra1,dec1,ra2,dec2)
    Inputs:
        ra1,dec1,ra2,dec2: Scalars or arrays in degrees.  Must be
            the same length
    """
    
    x1,y1,z1 = eq2xyz(ra1, dec1)
    x2,y2,z2 = eq2xyz(ra2, dec2)

    costheta = x1*x2 + y1*y2 + z1*z2
    w,=where(costheta > 1.0)
    if w.size > 0:
        costheta[w] = 1.0

    w,=where(costheta < -1.0)
    if w.size > 0:
        costheta[w] = -1.0
    theta = arccos(costheta)

    return theta



#
# SDSS specific conversions
#

# utility functions
def atbound(longitude, minval, maxval):
    w, = numpy.where(longitude < minval)
    while w.size > 0:
        longitude[w] += 360.0
        w, = numpy.where( longitude < minval )

    w, = numpy.where(longitude > maxval)
    while w.size > 0:
        longitude[w] -= 360.0
        w, = numpy.where( longitude > maxval )

    return

def atbound2(theta, phi):

    atbound(theta, -180.0, 180.0)

    w, = numpy.where( numpy.abs(theta) > 90.0 )
    if w.size > 0:
        theta[w] = 180.0 - theta[w]
        phi[w] += 180.0

    atbound(theta, -180.0, 180.0)
    atbound(phi, 0.0, 360.0)

    w, = numpy.where( numpy.abs(theta) == 90.0 )
    if w.size > 0:
        phi[w] = 0.0
           

def eq2sdss(ra_in, dec_in, dtype='f8'):
    """
    NAME:
      eq2sdss
    PURPOSE:
       Convert from ra, dec to the corrected clambda, ceta 
       SDSS survey coordinate system.  It is corrected so that the
       longitude eta ranges from [-180.0, 180.0] and the latitude
       lambda ranges from [-90.0,90.0].  The standard lambda/eta 
       both range from [-180.0,180.0] which doesn't make sense.
       NOTE: lambda is often referred to as longitude but this
       is incorrect since it has poles at [-90,90]

    CALLING SEQUENCE:
      from esutil import coords
      (clambda, ceta) = coords.eq2sdss(ra, dec, dtype='f8')

    INPUTS: 
      ra: Equatorial latitude in degrees. 
      dec: Equatorial longitude in degrees. 
    OPTIONAL INPUTS:
        dtype: The data type of output.  Default is 'f8'. See 
        numpy.typeDict for a list of possible types.
        dtype: The data type of output.  Default is 'f8'.

    OUTPUTS: 
      clambda: Corrected Survey longitude (actually lattitude) in degrees
      ceta: Corrected Survey latitude (actually logitude) in degrees
      
    REVISION HISTORY:
      Written: 11-March-2006  Converted from IDL program.
    """

    # Make a copy as an array. ndmin=1 to avoid messed up scalar arrays
    ra = numpy.array(ra_in, ndmin=1, copy=True, dtype=dtype)
    dec = numpy.array(dec_in, ndmin=1, copy=True, dtype=dtype)

    if (ra.size != dec.size):
        raise ValueError("RA, DEC must be same size")

    # range checking
    if (ra.min() < 0.0) | (ra.max() > 360.0):
        raise ValueError('RA must we within [0,360]')
    if (dec.min() < -90.0) | (dec.max() > 90.0):
        raise ValueError('DEC must we within [-90,90]')

    ra *= D2R
    dec *= D2R
    ra -= _sdsspar['node']

    # generate x,y,z on unit sphere, clearing memory as we go
    cdec = cos(dec)
    
    x = cos(ra)*cdec
    y = sin(ra)*cdec

    ra = 0; cdec = 0 # mem 
    
    z = numpy.sin(dec)
    
    dec = 0 # mem

    # generate clambda, ceta
    # do things in place to save memory

    # clambda = -arcsin( x ) (not a copy clambda=x)
    arcsin(x, x); clambda=x
    clambda *= -1

    
    arctan2( z, y, z); ceta = z
    ceta -= _sdsspar['etapole']

    clambda *= R2D
    ceta *= R2D

    atbound(ceta, -180.0, 180.0)
    
    return (clambda, ceta)

def sdss2eq(clambda_in, ceta_in, dtype='f8'):
    """
    NAME:
      sdss2eq
    PURPOSE:
       Convert corrected clambda, ceta SDSS survey coordinate system t
       equatorial coords.  

    CALLING SEQUENCE:
      from esutil import coords
      (ra, dec) = coords.sdss2eq(clambda, ceta, dtype='f8')

    INPUTS: 
      clambda: Corrected Survey longitude (actually lattitude) in degrees
      ceta: Corrected Survey latitude (actually logitude) in degrees
    OPTIONAL INPUTS:
        dtype: The data type of output.  Default is 'f8'. See 
        numpy.typeDict for a list of possible types.

    OUTPUTS: 
      ra: Equatorial latitude in degrees. 
      dec: Equatorial longitude in degrees. 
      
    REVISION HISTORY:
      Written: 11-March-2006  Converted from IDL program.
    """
    
    # Make a copy as an array. ndmin=1 to avoid messed up scalar arrays
    clambda = numpy.array(clambda_in, ndmin=1, copy=True, dtype=dtype)
    ceta = numpy.array(ceta_in, ndmin=1, copy=True, dtype=dtype)

    # range checking
    if (clambda.min() < -90.0) | (clambda.max() > 90.0):
        raise ValueError('CLAMBDA must we within [-90,90]')
    if (ceta.min() < -180.0) | (ceta.max() > 180.0):
        raise ValueError('CETA must we within [-180,180]')

    clambda *= D2R
    ceta    *= D2R

    x = -sin(clambda)
    y = cos(ceta + _sdsspar['etapole'])*cos(clambda)
    z = sin(ceta + _sdsspar['etapole'])*cos(clambda)
    
    ra = arctan2( y, x ) + _sdsspar['node']
    dec = arcsin(z)

    ra  *= R2D
    dec *= R2D
    atbound2(dec, ra)

    return (ra,dec)


def _eq2survey(ra_in, dec_in, dtype='f8'):
    """
    NAME:
      _eq2survey
    PURPOSE:
       Convert from ra, dec to the lambda, eta 
       SDSS survey coordinate system.  Note this coordinate system is
       not well defined.  Recommend you use csurvey coords.

    CALLING SEQUENCE:
      from esutil import coords
      (lambda, eta) = coords._eq2survey(ra, dec, dtype='f8')

    INPUTS: 
      ra: Equatorial latitude in degrees. 
      dec: Equatorial longitude in degrees. 
    OPTIONAL INPUTS:
        dtype: The data type of output.  Default is 'f8'. See 
        numpy.typeDict for a list of possible types.

    OUTPUTS: 
      lambda: SDSS Survey longitude (actually lattitude) in degrees
      eta: SDSS Survey latitude (actually logitude) in degrees
      
    REVISION HISTORY:
      Written: 11-March-2006  Converted from IDL program.
    """

    # Make a copy as an array. ndmin=1 to avoid messed up scalar arrays
    ra = numpy.array(ra_in, ndmin=1, copy=True, dtype=dtype)
    dec = numpy.array(dec_in, ndmin=1, copy=True, dtype=dtype)

    if (ra.size != dec.size):
        raise ValueError("RA, DEC must be same size")

    # range checking
    if (ra.min() < 0.0) | (ra.max() > 360.0):
        raise ValueError('RA must we within [0,360]')
    if (dec.min() < -90.0) | (dec.max() > 90.0):
        raise ValueError('DEC must we within [-90,90]')



    ra *= D2R
    dec *= D2R
    ra -= _sdsspar['node']

    # generate x,y,z on unit sphere, clearing memory as we go
    cdec = cos(dec)
    
    x = cos(ra)*cdec
    y = sin(ra)*cdec

    ra = 0; cdec = 0 # mem 
    
    z = sin(dec)
    
    dec = 0 # mem

    # generate lam, eta
    # do things in place to save memory

    # lam = -arcsin( x ) (not a copy lam=x)
    arcsin(x, x); lam=x
    lam *= -1

    
    arctan2( z, y, z); eta = z
    eta -= _sdsspar['etapole']

    lam *= R2D
    eta *= R2D

    atbound2(lam, eta)
    atbound(eta, -180.0, 180.0)

    w, = numpy.where( eta > (90.0 - _sdsspar['center_dec']) )
    if w.size > 0:
        eta[w] -= 180.0
        lam[w]  = 180.0 - lam[w]

    atbound(lam, -180.0, 180.0)

    return (lam, eta)


def _survey2eq(ra, dec, dtype='f8'):
    """
    NAME:
      _survey2eq
    PURPOSE:
       Convert clambda, ceta SDSS survey coordinate system to
       equatorial coords.  

    CALLING SEQUENCE:
      from esutil import coords
      (ra, dec) = coords._survey2eq(lam, eta, dtype='f8')

    INPUTS: 
      lambda: Survey longitude (actually lattitude) in degrees
      eta:    Survey latitude (actually logitude) in degrees
    OPTIONAL INPUTS:
        dtype: The data type of output.  Default is 'f8'. See 
        numpy.typeDict for a list of possible types.
      
    OUTPUTS: 
      ra: Equatorial latitude in degrees. 
      dec: Equatorial longitude in degrees. 
      
    REVISION HISTORY:
      Written: 11-March-2006  Converted from IDL program.
    """

    return csurvey2eq(ra,dec, dtype=dtype)


def dec_parse(decstring):
    """
    dec = dec_parse(decstring)

    parse a colon separated string representing declination ito
    degrees.
    """
    dec = 0.0

    ds = decstring.split(':')
    lds = len(ds)
    if lds >= 1:
        deg = float(ds[0])
        dec += deg
    if lds >= 2:
        minutes = float(ds[1])
        dec += minutes/60.0
    if lds >= 3:
        sec = float(ds[2])
        dec += sec/3600.0
    return dec

def ra_parse(rastring, hours=True):
    """
    ra = ra_parse(decstring)

    parse a colon separated string representing right ascension ito
    degrees.
    """
    ra = 0.0

    rs = rastring.split(':')
    lrs = len(rs)
    if lrs >= 1:
        deg = float(rs[0])
        ra += deg
        if hours:
            ra *= 15
    if lrs >= 2:
        minutes = float(rs[1])
        ra += minutes/60.0
    if lrs >= 3:
        sec = float(rs[2])
        ra += sec/3600.0
    return ra


def fitsheader2dict(hdr, ext=0):
    """
    Convert a fits header object into a dict.  A dict provides more expected
    interface to the data but cannot be written back to a fits file without
    transformation.
    """

    hdict={}
    for key in hdr:
        hdict[key.lower()] = hdr[key]

    return hdict



def shiftlon(lon_input, shift=None, wrap=True):
    """
    Name:
        shiftlon
    Calling Sequence:
        newlon = shiftlon(longitude, wrap=True, shift=0.0)

    Purpose:

        Shift the value of a longitude.  By default, the value is "wrapped" to
        be [-180,180] instead of [0,360]

        If the shift keyword is sent, then the longitude is simply shifted by
        the input value and then constrained to be again on the [0,360) range.
    
    Input:
        A longitude or array of longitudes on the range [0,360)

    Keywords:
        shift: 
            If shift is sent, then lon-shift is returned, constrained to still
            be on [0,360).
        
        wrap: 
            If shift is not sent, and wrap is True, wrap the range to
            [-180,180]

    """
    lon = numpy.array(lon_input, ndmin=1, copy=True, dtype='f8')

    if shift is not None:
        negshift = False
        if shift < 0:
            negshift=True

        abs_shift = abs(shift)

        # make sure in range [0,360)
        abs_shift = abs_shift % 360.0

        if negshift:
            lon += abs_shift

            w, = numpy.where(lon > 360.0)
            if w.size > 0:
                lon[w] -= 360.0
        else:
            lon -= abs_shift

            w, = numpy.where(lon < 0.0)
            if w.size > 0:
                lon[w] += 360.0

    elif wrap:
        w,=where(lon > 180)
        if w.size > 0:
            lon[w] -= 360

    return lon



def shiftra(ra, shift=None, wrap=True):
    """
    Name:
        shiftra
    Calling Sequence:
        newra = shiftra(ra, wrap=True, shift=0.0)

    Purpose:

        Shift the value of a longitude RA.  By default, the value is "wrapped"
        to be [-180,180] instead of [0,360]

        If the shift keyword is sent, then the longitude is simply shifted by
        the input value and then constrained to be again on the [0,360) range.
    
    Input:
        ra or any other longitude on the range [0,360)

    Keywords:
        shift: 

            If shift is sent, then ra-shift is returned, constrained to still
            be on [0,360).
        
        wrap: 
            If shift is not sent, and wrap is True, wrap the range to
            [-180,180]

    """
    return shiftlon(ra, shift=shift, wrap=wrap)



def radec2aitoff(ra, dec):
    """
    Take the ra/dec into aitoff coords
    """

    r2 = numpy.sqrt(2.0)
    f = 2.*r2/PI


    sra = shiftra(ra)

    alpha2 = sra/2.*D2R
    delta = dec*D2R

    cdec = cos(delta)
    
    denom = sqrt(1.0 + cdec*cos(alpha2))

    x = cdec*sin(alpha2)*2.*r2/denom
    y = sin(delta)*r2/denom
    x = x*R2D/f
    y = y*R2D/f

    crap="""
    sa = l
    if N_elements(sa) eq 1 then sa = fltarr(1) + sa
    x180 = where (sa gt 180.0)
    if x180[0] ne -1 then sa[x180]  = sa[x180] - 360.
    alpha2 = sa/(2*!RADEG)
    delta = b/!RADEG   
    r2 = sqrt(2.)    
    f = 2*r2/!PI     
    cdec = cos(delta)    
    denom =sqrt(1. + cdec*cos(alpha2))
    x = cdec*sin(alpha2)*2.*r2/denom
    y = sin(delta)*r2/denom
    x = x*!radeg/f
    y = y*!radeg/f
    """

    return x,y

def randsphere(num, system='lonlat'):
    """
    Name:
        randsphere
    Purpose:
        Generate random points all over the sphere

    Calling Sequence:
        lon, lat = randsphere(num, system='lonlat')

    Inputs:
        num: The number of randoms to generate
    Keywords:
        system: Default is 'lonlat' for longitude
            and latitude system.  Can also be 'xyz'.

    Outputs:
        longitude on [0,360) and ,latitude on [-90,90)
        or x,y,z for system='xyz'

        lon,lat = randsphere(2000)
        x,y,z = randsphere(2000, system='xyz')

    """

    # longitude [0,360)
    lon = numpy.random.random(num)*360.0

    # number [-1,1)
    v = numpy.random.random(num)
    v *= 2
    v -= 1

    # Now this generates on [0,pi)
    lat = numpy.arccos(v)

    # convert to degrees
    numpy.rad2deg(lat,lat)
    # now in range [-90,90.0)
    lat -= 90.0
    
    if system == 'xyz':
        x,y,z = eq2xyz(lon, lat)
        return x,y,z
    else:
        return lon, lat


