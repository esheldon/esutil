"""
Some functions to simplify input and output to various file types.

The functions most users will use:

    read():
        Provide a single interface to read from a variety of file types.
        Supports reading from a list of files.

    write()
        Provide a single interface to write a variety of file types.
        Not yet implemented.

    Useful functions called by the above:

        read_fits(): 
            A convenience function wrapping pyfits.getdata.  Adds some extra
            functionality such as getting subsets of rows and columns for
            binary tables, ensuring native byte ordering, and forcing the case
            of binary table columns to be upper/lower.

Created late 2009 Erin Sheldon, Brookhaven National Laboratory.  See docs
for individual methods for revision history.

"""

license="""
  Copyright (C) 2010  Erin Sheldon

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




from esutil import numpy_util
from esutil import json_util
from esutil import xmltools
from esutil import sfile
from esutil import ostools

from sys import stdout, stderr

try:
    import pyfitspatch as pyfits
    have_pyfits=True
except:
    try:
        import pyfits
        have_pyfits=True
    except:
        have_pyfits=False


def read(fobj, **keywords): 
    """
    Name:
        io.read

    Usage:
        read(filename/fileobject,
             typ=None,
             ext=0,
             rows=None, fields=None, columns=None,
             header=False, 
             combine=False, 
             view=None,
             lower=False, upper=False,
             noroot=True, seproot=False,
             verbose=False, 
             ensure_native=False)

    Purpose:
        Provide a single interface to read from a variety of file types.
        Supports reading from a list of files.

    Currently Supported File Types:
        FITS
            Flexible Image Transport System
        REC
            Simple ascii header followed by data in binary or text form. These
            files can be written/read using the esutil.sfile module.  Supports
            writing in faster native byte ordering for binary data and
            writing/reading of recarrays (numpy arrays with fields) to ascii
            and binary.  Supports reading sub-selections of rows and columns.
        XML
            Extensible Markup Language
        JSON
            JavaScript Object Notation.  Less flexible than XML but more useful
            in most practical situations such as storing inhomogeneous data in
            a portable way. 

    Inputs:
        filename/fileobject:  
            File name or an open file object.  Can also be a sequence.  If a
            sequence is input, the return value will, by default, be a list of
            results.  If the return types are numpy arrays, one can send the
            combine=True keyword to combine them into a single array as long
            as the data types match.

    Keywords:
        type: A string describing the file type.  If this is not sent, then
              the file type is determined from the file extension.
        ext: 
            The file extension.  If multiple extensions are supported by the
            file type, such as for FITS, then use this keyword to select which
            is to be read. Default is 0, the first extension.

        rows:  
            For numpy record-type files such as FITS binary tables or simple
            REC files, setting this keyword will return a subset of the rows.
            For FITS, this requires reading the entire file and selecting a
            subset.  For REC files only the requested rows are read from disk
            by using the recfile package.  Default is all rows.

        fields=, columns=:  
            For numpy record-type files such as FITS binary tables or simple
            REC files, return a subset of the columns or fields.  The keywords
            "fields" and "columns" are synonyms.  For FITS, this requires
            reading the entire file and selecting a subset.  For REC files only
            the requested rows are read from disk by using the recfile package.
            Default is all columns.

        header:  If True, and the file type supports header+data, return a 
            tuple  (data, header).  Default is False.
        combine:  If a list of filenames/fileobjects is sent, the default
            behavior is to return a list of data.  If combine=True and the
            data are numpy arrays, attempt to combine them into a single
            array.  Only works if the data types match.
        view:  If the result is derived from a numpy array, set this to
            pick the view.  E.g. pyfits returns a special pyfits type for
            binary table.  You can request a simple numpy array with fields
            by setting view=numpy.ndarray, or a numpy recarray type with
            view=numpy.recarray

        lower,upper:  For FITS files, if true convert the case of the
            fields to all lower or all upper.  Certain FITS writers
            tend to write all fields names as capitals which can result
            in annoyance.

        noroot:  For XML files, do not return the root name as the base
            name in the dictionary.  Default is True
        seproot: For XML files, return a tuple (data, rootname) instead of
            just the data under the root.

        ensure_native: For numpy arrays, make sure data is in native
            byte ordering.

    Revision History:
        Use **keywords for input and for sending to all called methods. Much
        more flexible when adding new keywords and file types.
        2010
    """


    verbose = keywords.get('verbose', False)

    # If input is a sequence, read them all.
    if isinstance(fobj, (list,tuple)):
        combine = keywords.get('combine', False)

        # a list was given
        alldata = []
        for f in fobj:
            # note, only fields/columns is begin passed on but not rows
            # also note seproot is not being passed on
            data = read(f, **keywords) 
            alldata.append(data)

        if combine:
            if len(fobj) == 1:
                alldata = alldata[0]
            else:
                fn,type = _get_fname_ftype_from_inputs(fobj[0], **keywords)
                if type == 'fits' or type == 'rec':
                    # this will only work if the all data has the 
                    # same structure
                    if verbose:
                        stdout.write("Combining arrays\n")
                    alldata = numpy_util.combine_arrlist(alldata)
        return alldata

    # a scalar was input
    fname,type=_get_fname_ftype_from_inputs(fobj, **keywords)
    if verbose:
        stdout.write("Reading: %s\n" % fname)

    # pick the right reader based on type
    if type == 'fits':
        data = read_fits(fobj, **keywords)
    elif type == 'json':
        data = json_util.read(fobj)
    elif type == 'rec':
        data = read_rec(fobj, **keywords)
    elif type == 'xml':
        data = read_xml(fobj, **keywords)
    else:
        raise ValueError("Don't know about file type '%s'" % type)

    return data



def read_fits(fobj, **keywords):
    """
    Name:
        read_fits
    Purpose:
        Read data from a single fits file.
    Calling Sequence:
        data=read_fits(fobj, **keywords)
    Inputs:
        fobj: The file name/file object for a fits file.
    Keywords:
        ext: Which extension, or HDU, to read.  Default 0.
        view: What view of the data to return. Default is numpy.ndarray
        header:  Return the data,header tuple?  Default False.
        rows:  Subset of the rows to return if reading a binary table extension.
        columns:  Subset of the columns to return if reading a binary table.
        fields: synonymous with columns
        lower: Force the field names to be lower case.
        upper: Force the field names to be upper case.
        ensure_native:  FITS always stores big-endian byte order.  Sending
            ensure_native=True forces the byte ordering to be machine native.
    Example:
        import esutil
        data=esutil.io.read('test.fits', ext=1, )
    """

    if not have_pyfits:
        raise ImportError("Could not import pyfits")
    import numpy

    ext=keywords.get('ext',0)
    view = keywords.get('view', numpy.ndarray)
    header = keywords.get('header', False)
    rows=keywords.get('rows',None)
    fields = keywords.get('fields', None)
    columns = keywords.get('columns', None)
    lower= keywords.get('lower',False)
    upper= keywords.get('upper',False)
    ensure_native = keywords.get('ensure_native',False)

    if fields is None:
        if columns is not None:
            # allow columns to be synonymous with fields
            fields=columns

    if view is None:
        import numpy
        view = numpy.ndarray

    if isinstance(fobj,(str,unicode)):
        fobj=ostools.expand_filename(fobj)
    if header:
        d,h = pyfits.getdata(fobj, ext=ext, header=header)
    else:
        d = pyfits.getdata(fobj, ext=ext)

    if lower:
        d.dtype.names = [n.lower() for n in d.dtype.names]
    elif upper:
        d.dtype.names = [n.upper() for n in d.dtype.names]

    # extract subsets of the data
    if rows is not None:
        d = d[rows]
    if fields is not None:
        d = numpy_util.extract_fields(d, fields)

    d=d.view(view)

    if ensure_native:
        numpy_util.to_native(d, inplace=True)

    if header:
        return d,h
    else:
        return d


def read_rec(fobj, **keywords):
    import numpy
    header=keywords.get('header',False)
    view=keywords.get('view',numpy.ndarray)
    rows=keywords.get('rows',None)
    columns=keywords.get('columns',None)
    fields=keywords.get('fields',None)
    ensure_native = keywords.get('ensure_native',False)

    data = sfile.read(fobj, header=header, view=view, 
                      rows=rows, fields=fields, columns=columns)
    if ensure_native:
        numpy_util.to_native(data, inplace=True)

    return data


def read_xml(fobj, **keywords):
    noroot=keywords.get('noroot',True)
    seproot=keywords.get('seproot',False)
    data = xmltools.xml2dict(fobj, noroot=noroot, seproot=seproot)
    return data

def ftype2fext(ftype_input):
    ftype=ftype_input.lower()

    if ftype == 'fits' or ftype == 'fit':
        return 'fits'
    elif ftype == 'rec' or ftype == 'pya':
        return 'rec'
    elif ftype == 'json':
        return 'json'
    elif ftype == 'xml':
        return 'xml'
    else:
        raise ValueError("Don't know about '%s' files" % ftype)

def fext2ftype(fext_input):

    fext=fext_input.lower()

    if fext == 'fits' or fext == 'fit':
        return 'fits'
    elif fext == 'rec' or fext == 'pya':
        return 'rec'
    elif fext == 'json':
        return 'json'
    elif fext == 'xml':
        return 'xml'
    else:
        raise ValueError("Don't know about files with '%s' extension" % fext)

def _get_fname_ftype_from_inputs(fobj, **keywords):

    if isinstance(fobj, file):
        fname = fobj.name
    elif isinstance(fobj, (str,unicode)):
        # make sure we expand all ~username and other variables
        fname=ostools.expand_filename(fobj)
    else:
        raise ValueError("Input must be a string or file object, or a "
                         "list thereof")


    ftype=None
    if 'type' in keywords:
        ftype=keywords['type']
    elif 'typ' in keywords:
        ftype = keywords['typ']

    if ftype is None:
        ftype=get_ftype(fname)
    ftype = ftype.lower()

    return fname, ftype


def get_ftype(filename):
    fsplit = filename.split('.')
    fext = fsplit[-1]
    if (fext == 'gz' or fext == 'bz' or fext == 'bz2') and len(fsplit) > 2:
        fext = fsplit[-2]
    typ = fext2ftype(fext)
    return typ
