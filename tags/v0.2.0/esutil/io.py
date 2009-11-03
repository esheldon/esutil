"""
Some functions to simplify input and output to various file types.

The functions most users will use:

    read():
        Provide a single interface to read from a variety of file types.
        Supports reading from a list of files.

    write()
        Provide a single interface to write a variety of file types.
        Not yet implemented.

"""
from esutil import numpy_util
from esutil import json_util
from esutil import xmltools
from esutil import sfile
from esutil import ostools

from sys import stdout, stderr

try:
    import pyfits
    have_pyfits=True
except:
    have_pyfits=False


def read(fobj, 
         typ=None, 
         ext=0,
         rows=None, fields=None, columns=None,
         header=False, 
         combine=False, view=None,
         norecfile=False,
         lower=False, upper=False,
         noroot=True, seproot=False,
         verbose=False):
    """
    Name:
        io.read

    Usage:
        read(filename/fileobject,
             typ=None,
             ext=0,
             rows=None, fields=None, columns=None,
             norecfile=False,
             header=False, combine=False, view=None,
             lower=False, upper=False,
             noroot=True, seproot=False,
             verbose=False)

    Purpose:
        Provide a single interface to read from a variety of file types.
        Supports reading from a list of files.

    Currently Supported File Types:
        FITS
        REC
            Simple ascii header followed by data in binary or text form. These
            files can be written/read using the esutil.sfile module.  Supports
            writing in faster native byte ordering for binary data and
            writing/reading of recarrays (numpy arrays with fields) to ascii
            and binary.  Supports reading sub-selections of rows and columns.
        XML
        JSON

    Inputs:
        filename/fileobject:  
            File name or an open file object.  Can also be a sequence.  If a
            sequence is input, the return value will, by default, be a list of
            results.  If the return types are numpy arrays, one can send the
            combine=True keyword to combine them into a single array as long
            as the data types match.

    Keywords:
        typ:  A string describing the file type.  If this is not sent, then
            the file type is determined from the file extension.
        ext: The file extension.  If multiple extensions are supported by
            the file type, then use this keyword to select which is to be
            read. Default is 0, the first extension.
        rows:  
            For record-type files such as FITS binary tables or simple REC
            files, setting this keyword will return a subset of the rows.
            For FITS, this requires reading the entire file and selecting
            a subset.  For REC files only the requested rows are read from
            disk by using the recfile package.  That behaviour can be turned
            off with the norecfile=True keyword.  Default is all rows.

        fields=, columns=:  
            For record-type files such as FITS binary tables or simple REC
            files, return a subset of the columns or fields.  The keywords
            "fields" and "columns" are synonyms.  For FITS, this requires
            reading the entire file and selecting a subset.  For REC files
            only the requested rows are read from disk by using the recfile
            package.  That behaviour can be turned off with the norecfile=True
            keyword.  Default is all columns.

        norecfile:  Do not use recfile to read subsets of rows/columns for
            REC file type.  This can be faster in certain situations, such
            as when all rows are being read and a subset of columns is 
            requested.  The tradeoff is higher memory usage.


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

    """

    # If input is a sequence, read them all.
    if isinstance(fobj, list) or isinstance(fobj, tuple):
        alldata = []
        for f in fobj:
            # note, only fields/columns is begin passed on but not rows
            # also note seproot is not being passed on
            data = read(f, 
                        typ=typ, 
                        ext=ext,
                        fields=fields, columns=columns, view=view,
                        norecfile=norecfile,
                        noroot=noroot, verbose=verbose)
            alldata.append(data)
        if combine:
            # only attempt this if we are dealing with numpy arrays
            if typ is None:
                typ=get_ftype(f)

            if typ == 'fits' or typ == 'rec':
                # this will only work if the all data has the 
                # same structure
                if verbose:
                    stdout.write("Combining arrays\n")
                alldata = numpy_util.combine_arrlist(alldata)
        return alldata


    # a scalar was input
    if isinstance(fobj, file):
        fname = fobj.name
    elif isinstance(fobj, str) or isinstance(fobj, unicode):
        # make sure we expand all ~username and other variables
        fname=ostools.expand_filename(fobj)
        if verbose:
            stdout.write("Reading: %s\n" % fobj)
    else:
        raise ValueError("Input must be a string or file object, or a "
                         "list thereof")

    # we can guess the filetype from the extension if not input
    if typ is None:
        typ=get_ftype(fname)

    typ = typ.lower()

    # pick the right reader based on typ
    if typ == 'fits':
        return read_fits(fobj, ext=ext, header=header, view=view,
                         lower=lower,upper=upper)

    elif typ == 'json':
        return json_util.read(fobj)

    elif typ == 'rec':
        return sfile.read(fobj, header=header, view=view, 
                          rows=rows, fields=fields, columns=columns,
                          norecfile=norecfile)

    elif typ == 'xml':
        return xmltools.xml2dict(fobj, noroot=noroot, seproot=seproot)

    else:
        raise ValueError("Don't know about file type '%s'" %ftype)




def read_fits(fobj, ext=0, view=None, header=False, 
              rows=None, fields=None, columns=None,
              lower=False, upper=False):
    if not have_pyfits:
        raise ImportError("Could not import pyfits")

    if fields is None:
        if columns is not None:
            # allow columns to be synonymous with fields
            fields=columns

    if view is None:
        import numpy
        view = numpy.ndarray

    if header:
        d,h = pyfits.getdata(fobj, ext=ext, header=header, 
                             lower=lower, upper=upper)
    else:
        d = pyfits.getdata(fobj, ext=ext, lower=lower, upper=upper)


    # extract subsets of the data
    if rows is not None:
        d = d[rows]
    if fields is not None:
        d = numpy_util.extract_fields(d, fields)

    d=d.view(view)

    if header:
        return d,h
    else:
        return d




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

def get_ftype(filename):
    fext=filename.split('.')[-1]
    typ = fext2ftype(fext)
    return typ
