"""
Package:
    esutil

Sub-packages:

    htm:  
        Tools for working with the Hierarchical Triangular Mesh, whic his a
        method for breaking the unit sphere into a tree structure where each
        node in the tree is represented by a spherical triangle.   This can be
        used for fast searching of the sphere and matching lists of points.

        The underlying code is C++ linked as an extension.

    cosmology:
        A set of tools for calculating distances in an expanding universe.
        These routines are completely general for any specified omega_m,
        omega_k, and cosmological constant omega_l.  This code follows the
        conventions of Hogg astro-ph/9905116.  The underlying calculations
        are done in an extension module written in fortran 95

    integrate:
        Tools for integration of data and functions.  Currently contains the QGauss
        class for gauss-legendre integration, which relies on the gauleg C++ extension.

    pyfitspatch:
        A patched version of pyfits that allows writing structures (numpy arrays with
        fields, recarrays) to disk in a single call.

    recfile:
        Contains the class Recfile for efficiently reading and writing
        structured numpy arrays to and from binary and ascii files.  Individual
        columns and rows can be selected.   Underlying code is C++ linked
        as an extension.

    stat:
        This packages contains tools for statistical analysis, including an IDL-like
        histogram function.  The histogram function is written in C++ and linked
        as an extension.

Sub-modules:
    coords:
        A set of astronomical utilities for dealing with coordinates and
        coordinate transformations.

    wcsutil:
        Fast tools for working with the World Coordinat System used in astronomy to
        convert instrument coordinates to sky coordinates.

    io: 
        File input/output convenience functions.  Read and write many file
        formats using the same read() write() interface.

    json_util:
        Convienience functions for working with JSON files
        http://en.wikipedia.org/wiki/JSON

    misc:
        Miscellaneous usefule tools, such as a tool for printing variables
        in column format, pretty printing elapsed time, executing system
        processes, sub-selecting from a dictionary, etc.

    numpy_util:
        A large number of functions for working with numerical python arrays.
        The focus is primarily on structures, aka recarrays or structured
        arrays.  E.g. 
            ahelp: an IDL style help function to show what is
                contained in the structure.
            aprint: printing structures.
            arrscl: scale the range of an array
            combine_arrlist: Combine a list of structures into one big array.
            copy_fields: Copy the fields from on structure to another.
            extract_fields: Extract a subset of fields from a structure, recarray.
            remove_fields: Remove a subset of fields.
            add_fields:  Add new fields to a structure/recarray.
            reorder_fields: Re-order the fields in a structure/recarray.
            split_fields: Split the fields of a structure/recarray into a tuple
                of arrays. No copy is if data is made.
            compare_arrays: Compare structures/recarrays field by field.
            is_big_endian: Check if the array is big endian byte order
            is_little_endian: Check if the array is little endian byte order.
            to_big_endian: convert to big endian
            to_little_endian: convert to little endian
            to_native: Convert to native byte order.
            byteswap: byteswap array with ability to automatically update the
                type descriptor.
            unique: Return unique elements or indices (note now there is a built
                numpy routine for this...)
            match: Match the elements two arrays.
            dict2array: Convert a dict to a structure/recarray.
            splitarray: Split rows of the array into chunks of at least a given size.
            randind: Get random indices with replacement in the open range [0,nmax)
            random_subset: Generate a random subset of integers in a range.  Like
                randind but *without* replacement.

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

def version():
    from sys import stderr

    thisname='/esutil/__init__.py'
    badvers="NOTAG: unparseable"

    psplit=_property_headurl.split()
    if len(psplit) != 3:
        mess="headurl did not split into 3: '%s'\n" % _property_headurl
        stderr.write(mess)
        return badvers

    url=psplit[1]

    if url.find(thisname) == -1:
        mess="url '%s' does not contain string '%s'\n" % \
                (_property_headurl, thisname)
        stderr.write(mess)
        return badvers

    urlfront = url.replace(thisname, '')

    tag=urlfront.split('/')[-1]
    return tag

def get_python_version(numerical=False):
    if numerical:
        v=sys.version_info[0:3]
        pyvers=v[0] + 0.1*v[1] + 0.01*v[2]
    else:
        pyvers='v%s.%s.%s' % sys.version_info[0:3]
    return pyvers


from . import algorithm

from . import xmltools
from . import ostools
from . import misc
from . import integrate
from . import json_util
from . import stat
from . import numpy_util
from . import oracle_util
from . import sfile
from . import io
from . import wcsutil

from . import stomp_util
from . import plotting
from . import random
from . import fits

from . import coords
# old name for this module
from . import coords as astro_util


# extensions might not compile
from . import cosmology_purepy
try:
    from . import cosmology
except:
    pass
    cosmology = cosmology_purepy
try:
    from . import sqlite_util
except:
    pass

try:
    from . import htm
except:
    pass

try:
    from . import pyfitspatch
except:
    pass

try:
    from . import recfile
except:
    pass


