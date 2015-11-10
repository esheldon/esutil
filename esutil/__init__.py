"""
Package:
    esutil

Sub-packages and modules:
    numpy_util:
        A large number of functions for working with numerical python arrays.
        The focus is primarily on structures, aka recarrays or structured
        arrays.

    io: 
        File input/output convenience functions.  Read and write many file
        formats using the same read() write() interface.

    integrate:
        Tools for integration of data and functions.  Currently contains the QGauss
        class for gauss-legendre integration, which relies on the gauleg C++ extension.

    fits:
        A module wrappying pyfits that uses the recfile C extension (see below) to
        read subset of rows and columns from binary tables.

    recfile:
        Contains the class Recfile for efficiently reading and writing
        structured numpy arrays to and from binary and ascii files.  Individual
        columns and rows can be selected.   Underlying code is C++ linked
        as an extension.

    stat:
        This packages contains tools for statistical analysis, including an IDL-like
        histogram function.  The histogram function is written in C++ and linked
        as an extension.

    cosmology:
        A set of tools for calculating distances in an expanding universe.
        These routines are completely general for any specified omega_m,
        omega_k, and cosmological constant omega_l.  This code follows the
        conventions of Hogg astro-ph/9905116.  The underlying calculations
        are done in an extension module written in C

    coords:
        A set of astronomical utilities for dealing with coordinates and
        coordinate transformations.

    wcsutil:
        Fast tools for working with the World Coordinat System used in astronomy to
        convert instrument coordinates to sky coordinates.

    htm:  
        Tools for working with the Hierarchical Triangular Mesh, whic his a
        method for breaking the unit sphere into a tree structure where each
        node in the tree is represented by a spherical triangle.   This can be
        used for fast searching of the sphere and matching lists of points.

        The underlying code is C++ linked as an extension.



    json_util:
        Convienience functions for working with JSON files
        http://en.wikipedia.org/wiki/JSON

    misc:
        Miscellaneous usefule tools, such as a tool for printing variables
        in column format, pretty printing elapsed time, executing system
        processes, sub-selecting from a dictionary, etc.


    sqlite_util
        Tools for working with an sqlite database, including the ability to write
        record arrays to tables and read from tables into rec arrays.

    random:
        A class to generate random numbers from arbitrary distributions.


    stomp_util
    ostools
    plotting
    sfile
    xmltools
    oracle_util

"""
# version info
# You need to run 'svn propset svn:keywords HeadURL' on the file and commit
# before this works.
#
# Don't edit these svn properties by hand

_property_headurl='$HeadURL$'

import sys

__version__ = "0.5.3"

def version():
    return __version__

def get_python_version(numerical=False):
    if numerical:
        v=sys.version_info[0:3]
        pyvers=v[0] + 0.1*v[1] + 0.01*v[2]
    else:
        pyvers='v%s.%s.%s' % sys.version_info[0:3]
    return pyvers


import algorithm
import xmltools
import ostools
import misc
import integrate
import json_util
import stat
import numpy_util
import oracle_util
import sfile
import io
import wcsutil
import cosmology
import coords
import coords as astro_util
import stomp_util
import plotting
import hdfs

try:
    import sqlite_util
except:
    pass
import random

try:
    import htm
except:
    pass

import fits
try:
    import pyfitspatch
except:
    pass

try:
    import pyfitspatch240
except:
    pass



try:
    import recfile
except:
    pass

