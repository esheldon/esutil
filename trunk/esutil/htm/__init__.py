"""

Module:
    htm


Classes:
    HTM

Purpose:

    This is a Class to deal with the Hierarchical Triangular Mesh, which is a
    method for breaking the unit sphere into a tree structure where each node
    in the tree is represented by a spherical triangle.  The "depth" of the
    tree determines the size of the smallest triangle, with higher depths
    producing smaller triangles.  Currently depths up to 13 are supported,
    which corresponds to an area of 0.28 square arcminutes, which is limited by
    using 32-bit integers for the indices.

    A primary advantage of the HTM over other schemes is that it deals
    perfectly well with the poles.  

    The HTM was developed by astrophysicists at JHU, see their page for a full
    explanation:

        http://www.sdss.jhu.edu/htm/

    At this point a few tasks can be peformed with this code: 
    
        1) Find the id of the triangle a point or set of points belongs to.

        2) Calculate the area of triangles at the current depth.

        3) Match two sets of points to one another, returning lists of matches
        and the separation distance, or alternatively write the data to a file.


Methods:

    lookup_id(ra, dec):  

        Return the index of the input ra/dec at the current htm depth.
        ra/dec may be arrays.

    area():
        Return the mean area of triangles at the current depth. The units
        are square degrees.

    match(ra1,dec1,ra2,dec2,radius,
          maxmach=0,
          htmid2=None,
          htmrev2=None,
          minid=None,
          maxid=None,
          file=None)

        Match two sets of ra/dec points using the Hierarchical Triangular
        Mesh code.  This is very efficient for large search angles and
        large lists.  May seem slow otherwise due to overhead creating htm
        indices.  You can optionally write the results to a file.
    
    read(filename)
        Read the binary file format written by the match() code.
        
    See the docs for each method for more details.  For example, in ipython:
        >>> import esutil
        >>> h=esutil.htm.HTM(depth)
        >>> h.lookup_id?
        >>> h.match?
        >>> h.area?

Examples:

    >>> import esutil
    >>> depth=10
    >>> h = esutil.htm.HTM(depth)
    >>> h.area()*3600 # area in square arcminutes
    17.703850328673052


    # try looking up the id of an ra/dec position.  Note
    # ra/dec can also be arrays
    >>> ra = 200.0
    >>> dec = 0.0
    >>> h.lookup_id(ra,dec)
    array([10518792], dtype=int32)



    # try the matching two lists of ra/dec points
    # Matching by ra/dec, expect 10 matches ordered by distance....

    # match within two arcseconds
    two = 2.0/3600.

    # offset second list by fraction of 2 arcsec in dec
    # but last one won't match anything
    ra1 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 21.36])
    dec1 = numpy.array( [24.3,          24.3,            24.3,  -28.25, -15.32])
    ra2 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 55.25])
    dec2 = numpy.array( [24.3+0.75*two, 24.3 + 0.25*two, 24.3 - 0.33*two, -28.25 + 0.58*two, 75.22])

    m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,two)

    for i in range(m1.size):
        stdout.write('%s %s %s\\n' % (m1[i],m2[i],d12[i]))

    # this produces
    0 1 0.00013888984367
    0 2 0.00018333285694
    0 0 0.000416666032158
    1 1 0.00013888984367
    1 2 0.00018333285694
    1 0 0.000416666032158
    2 1 0.00013888984367
    2 2 0.00018333285694
    2 0 0.000416666032158
    3 3 0.000322221232243



To Perform Unit Tests:
    import esutil
    esutil.htm.unit_tests.test()


MODIFICATION HISTORY:
    SWIG Wrapper working 2010-03-03, Erin Sheldon, BNL.

"""



import htm
from htm import HTM
import unit_tests
