import esutil
import sys,os
from sys import stdout

try:
    import stomp
    have_stomp=True

    INSIDE_MAP=stomp.cvar.Map_INSIDE_MAP
    FIRST_QUADRANT_OK=stomp.cvar.Map_FIRST_QUADRANT_OK
    SECOND_QUADRANT_OK=stomp.cvar.Map_SECOND_QUADRANT_OK
    THIRD_QUADRANT_OK=stomp.cvar.Map_THIRD_QUADRANT_OK
    FOURTH_QUADRANT_OK=stomp.cvar.Map_FOURTH_QUADRANT_OK

except:
    have_stomp=False

try:
    import numpy
    have_numpy=True
except:
    have_numpy=False


def in_window(stomp_map, 
              ra=None, 
              dec=None,
              clambda=None, 
              ceta=None, 
              x1=None, 
              x2=None, 
              x3=None,
              radius=None,
              system=None):
    """
    Module:
        stomp_util
    Name:
        in_window
    Purpose:
        Check if points are contained within the input stomp map.
        If radius= is sent, also check against edges and return a
        tuple (inwindow,edgeflags).

    """

    if ra is not None:
        if dec is None:
            raise ValueError("Send both ra and dec")

        x1=numpy.array(ra, ndmin=1, copy=False, dtype='f8')
        x2=numpy.array(dec, ndmin=1, copy=False, dtype='f8')
        system = 'eq'
    elif clambda is not None:
        if ceta is None:
            raise ValueError("Send both clambda and ceta")

        x1=numpy.array(ra, ndmin=1, copy=False, dtype='f8')
        x2=numpy.array(dec, ndmin=1, copy=False, dtype='f8')
        system = 'survey'
    elif b is not None:
        if l is None:
            raise ValueError("Send both l and b")
        x1=numpy.array(l, ndmin=1, copy=False, dtype='f8')
        x2=numpy.array(b, ndmin=1, copy=False, dtype='f8')
        system = 'gal'
    elif x1 is not None:
        if x2 is None:
            raise ValueError("Send both x1 and x2 (possibly x3)")
        x1=numpy.array(x1, ndmin=1, copy=False, dtype='f8')
        x2=numpy.array(x2, ndmin=1, copy=False, dtype='f8')
        if x3 is not None:
            x3=numpy.array(x3, ndmin=1, copy=False, dtype='f8')
            system = 'sphere'
        else:
            if system is None:
                raise ValueError("If sending x1,x2 you must also send sytem=")
    else:
        raise ValueError("A good set of coordinates not found")


    if x1.size != x2.size:
        raise ValueError("all coords must be same size")

    if x3 is not None:
        x3=numpy.array(x3, ndmin=1, copy=False)
        if x3.size != x2.size:
            raise ValueError("all coords must be same size")
        system='sphere'

    nrad=0
    if radius is not None:
        rad = numpy.array(radius,ndmin=1,copy=False,dtype='f8')
        if rad.size == 1:
            nrad = 1
            thisrad = rad[0]
        elif rad.size == x1.size:
            nrad = x1.size
        else:
            raise ValueError("radius must be size 1 or same size as coords")


    system=system.lower()
    stomp_system = getsystem(system)

    if system == 'unitsphere' or system == 'sphere':
        # significantly faster to use SetUnitSphereCoordinates
        # so we will make separate code branch for that
        dosphere=True
    else:
        dosphere=False

    maskflags = numpy.zeros(x1.size, dtype='i1')
    ang = stomp.AngularCoordinate()
    for i in xrange(x1.size):
        if dosphere:
            ang.SetUnitSphereCoordinates(x1[i],x2[i],x3[i])
        else:
            ang.Set(x1[i],x2[i],stomp_system)

        is_in_win = stomp_map.Contains(ang)
        if is_in_win:
            maskflags[i] += INSIDE_MAP

            # If radius was sent and we are inside the map,
            # then do a full quadrant search
            if nrad > 0:
                if nrad > 1:
                    thisrad=rad[i]
                maskflags[i] += quad_check(stomp_map, ang, thisrad, stomp_system)


    return maskflags

def quad_check(stomp_map, ang, rad, system, res=2048):
    flags = 0

    angle1=0.0
    angle2=90.0
    wedge_bound=stomp.WedgeBound(ang, rad, angle1, angle2, system)
    mw = stomp.Map(wedge_bound, 1.0, res)
    if stomp_map.Contains(mw):
        flags += FIRST_QUADRANT_OK

    angle1=90.0
    angle2=180.0
    wedge_bound=stomp.WedgeBound(ang, rad, angle1, angle2, system)
    mw = stomp.Map(wedge_bound, 1.0, res)
    if stomp_map.Contains(mw):
        flags += SECOND_QUADRANT_OK

    angle1=180.0
    angle2=270.0
    wedge_bound=stomp.WedgeBound(ang, rad, angle1, angle2, system)
    mw = stomp.Map(wedge_bound, 1.0, res)
    if stomp_map.Contains(mw):
        flags += THIRD_QUADRANT_OK

    angle1=270.0
    angle2=360.0
    wedge_bound=stomp.WedgeBound(ang, rad, angle1, angle2, system)
    mw = stomp.Map(wedge_bound, 1.0, res)
    if stomp_map.Contains(mw):
        flags += FOURTH_QUADRANT_OK

    return flags




def getsystem(system):
    if isinstance(system, (str,unicode)):
        system=system.lower()
        if system == 'eq' or system=='equatorial':
            return stomp.AngularCoordinate.Equatorial
        elif system=='sdss' or system=='survey':
            return stomp.AngularCoordinate.Survey
        elif system == 'galactic' or system == 'gal':
            return stomp.AngularCoordinate.Galactic
        else:
            raise ValueError("Can't interpret 2d system: '%s'" % system)
    else:
        raise ValueError("Expected a string for system indicated, got %s" % system)

 


def toAngularVector(filename=None, 
                    ra=None, dec=None, 
                    clambda=None, ceta=None, 
                    x1=None, x2=None, x3=None,
                    system=None):

    if filename is not None:
        # loading from a file
        ang = stomp.AngularVector()
        stomp.AngularCoordinate.ToAngularVector(filename, ang)
        return ang
        
    if ra is not None:
        if dec is None:
            raise ValueError("Send both ra and dec")

        x1=numpy.array(ra, ndmin=1, copy=False)
        x2=numpy.array(dec, ndmin=1, copy=False)
        system = 'eq'
    elif clambda is not None:
        if ceta is None:
            raise ValueError("Send both clambda and ceta")

        x1=numpy.array(ra, ndmin=1, copy=False)
        x2=numpy.array(dec, ndmin=1, copy=False)
        system = 'survey'
    elif b is not None:
        if l is None:
            raise ValueError("Send both l and b")
        x1=numpy.array(l, ndmin=1, copy=False)
        x2=numpy.array(b, ndmin=1, copy=False)
        system = 'gal'
    elif x1 is not None:
        if x2 is None:
            raise ValueError("Send both x1 and x2 (possibly x3)")
        x1=numpy.array(x1, ndmin=1, copy=False)
        x2=numpy.array(x2, ndmin=1, copy=False)
        if x3 is not None:
            x3=numpy.array(x3, ndmin=1, copy=False)
            system = 'sphere'
        else:
            if system is None:
                raise ValueError("If sending x1,x2 you must also send sytem=")
    else:
        raise ValueError("A good set of coordinates not found")


    if x1.size != x2.size:
        raise ValueError("all coords must be same size")

    if x3 is not None:
        x3=numpy.array(x3, ndmin=1, copy=False)
        if x3.size != x2.size:
            raise ValueError("all coords must be same size")
        system='sphere'

    
    ang = stomp.AngularVector(x1.size)

    system=system.lower()
    i=0
    if system == 'equatorial' or system == 'eq':
        while i < x1.size:
            ang[i].SetEquatorialCoordinates(x1[i], x2[i])
            i+=1

    elif system == 'sdss' or system == 'survey':
        while i < x1.size:
            ang[i].SetSurveyCoordinates(x1[i], x2[i])
            i+=1

    elif system == 'galactic' or system == 'gal':
        while i < x1.size:
            ang[i].SetGalacticCoordinates(x1[i], x2[i])
            i+=1

    elif system == 'unitshere' or system == 'sphere':
        while i < x1.size:
            ang[i].SetUnitSphereCoordinates(x1[i], x2[i], x3[i])
            i+=1

    return ang


def toFile(filename, x1, x2):
    import recfile

    x1=numpy.array(x1, ndmin=1, copy=False)
    x2=numpy.array(x2, ndmin=1, copy=False)
    if x1.size != x2.size:
        raise ValueError("all coords must be same size")

    stdout.write("creating record array for writing\n")
    stdout.flush()

    rec=numpy.zeros(x1.size, dtype=[('x1','f8'),('x2','f8')])
    rec['x1'] = x1
    rec['x2'] = x2

    stdout.write("Opening file: %s\n" % filename)
    stdout.flush()
    recf = recfile.Open(filename, 'w', delim=' ')
    
    stdout.write("Writing data...")
    stdout.flush()
    recf.write(rec)

    recf.close()
    stdout.write("Done\n")
    stdout.flush()

    del rec


def test_speed(create_each_time=False, explicit=False):
    from time import time
    import columns

    c=columns.Columns('~/sweeps_reduce/gal01.cols')
    ra=c['ra'][0:500000]
    dec=c['dec'][0:500000]

    iw = numpy.zeros(ra.size,dtype=numpy.bool_)

    stomp_map=stomp.Map('/home/users/esheldon/masks/stomp-sdss/stripe_dr7.hmap_basic')

    tm0=time()
    if create_each_time:
        stdout.write("Testing creation of angular coord each time\n")
        i=0
        while i < ra.size:
            ang = stomp.AngularCoordinate(ra[i], dec[i], 
                                          stomp.AngularCoordinate.Equatorial)
            iw[i] = stomp_map.Contains(ang)
            del ang
            i += 1
    else:
        stdout.write("Testing re-setting values of same object\n")
        i=0
        ang = stomp.AngularCoordinate()

        if explicit:
            while i < ra.size:
                ang.SetEquatorialCoordinates(ra[i], dec[i])
                iw[i] = stomp_map.Contains(ang)
                i += 1
        else:
            system=stomp.AngularCoordinate.Equatorial
            while i < ra.size:
                ang.Set(ra[i], dec[i],system)
                iw[i] = stomp_map.Contains(ang)
                i += 1


    tm1=time()
    esutil.misc.ptime(tm1-tm0)


