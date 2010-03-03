"""

This is a Class to deal with the Heirarchical Triangular Mesh, which is a
method for breaking the unit sphere into a tree structure where each node in
the tree is represented by a spherical triangle.  The "depth" of the tree
determines the size of the smallest triangle, with higher depths meaning
smaller triangles.  Currently depths up to 13 are supported, which 
corresponds to an area of

    http://www.sdss.jhu.edu/htm/

At this point two tasks can be peformed with this code: 
    
    1) Find the id of the triangle a point or set of points belongs to.

    2) Match two sets of points to one another, returning lists of matches and
    the separation distance, or alternatively write the data to a file.


Examples:

    >>> import esutil
    >>> h = esutil.htm.HTM(

"""
import htmc
from esutil import stat
import numpy
from sys import stdout

class HTM(htmc.HTMC):

    def area(self):
        pi=numpy.pi
        area0=4.0*pi/8.0

        areadiv = 4.0**self.depth
        area = 60.0**2 * area0/areadiv*(180.0/pi)**2
        return area

    def match(self, ra1, dec1, ra2, dec2, radius, distance=False,
              maxmatch=0, 
              htmid2=None, 
              htmrev2=None,
              minid=None,
              maxid=None,
              file=None):

        if (len(ra1) != len(dec1)) or (len(ra2) != len(dec2)):
            raise ValueError("require len(ra)==len(dec) for "
                             "both sets of inputs")

        if htmid2 is None:
            htmid2 = self.lookup_id(ra2, dec2)
            minid = htmid2.min()
            maxid = htmid2.max()
        else:
            if minid is None:
                minid = htmid2.min()
            if maxid is None:
                maxid = htmid2.max()

        if htmrev2 is None:
            hist2, htmrev2 = stat.histogram(htmid2-minid,rev=True)

        return self.cmatch(radius,
                           ra1,
                           dec1,
                           ra2,
                           dec2,
                           htmrev2,
                           minid,
                           maxid,
                           maxmatch,
                           file)

    def read(self, filename, verbose=False):
        """
        Read the binary file format written by the ra,dec matching code
        """

        fobj=open(filename,'r')

        nrows=numpy.fromfile(fobj,count=1,dtype='i8')

        if verbose:
            stdout.write("Reading %s rows from file: %s\n" % (nrows[0],filename))

        dtype=[('i1','i8'),('i2','i8'),('d12','f8')]

        data = numpy.fromfile(fobj, count=nrows[0], dtype=dtype)

        fobj.close()

        return data
