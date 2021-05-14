"""

Module:
    htm


Classes:
    HTM, Matcher

HTM
---
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

    get_depth(): get the depth of the HTM tree

    lookup_id(ra, dec):

        Return the index of the input ra/dec at the current htm depth.
        ra/dec may be arrays.

    intersect(ra, dec, radius, inclusive=True):
        look up all triangles that are contained within or intersect a circle
        centered on the input point.

    area():
        Return the mean area of triangles at the current depth. The units
        are square degrees.

    match(ra1,dec1,ra2,dec2,radius,
          maxmatch=1,
          htmid2=None,
          htmrev2=None,
          minid=None,
          maxid=None,
          file=None)

        Match two sets of ra/dec points using the Hierarchical Triangular Mesh
        code.  This is a wrapper using a Matcher object. This is very efficient
        for large search angles and large lists.  May seem slow otherwise due
        to overhead creating htm indices.  You can optionally write the results
        to a file.

        If you need to match the same set multiple times, use a Matcher
        object

    read(filename)
        Read the pairs from a file written by the match() code.

    See the docs for each method for more details.  For example, in ipython:
        >>> import esutil
        >>> h=esutil.htm.HTM(depth)
        >>> h.lookup_id?
        >>> h.match?
        >>> h.area?

Matcher
-------

Class to match sets of ra,dec points.  One set is loaded and put
into a tree structure, and can then be matched quickly to other
sets of ra,dec points

methods
-------

get_depth(): get the depth of the HTM tree
match(): match against a set of ra,dec points

"""

# flake8: noqa


from . import htm
from .htm import HTM, Matcher, read_pairs
