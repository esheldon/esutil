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


def read_fits(fobj, ext=0, view=None, header=False, 
              rows=None, fields=None, columns=None):
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
        d,h = pyfits.getdata(fobj, ext=ext, header=header)
    else:
        d = pyfits.getdata(fobj, ext=ext)


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


def read(fobj, 
         typ=None, 
         ext=0, header=False, combine=False, view=None,
         rows=None, fields=None, columns=None,
         norecfile=False,
         noroot=True, seproot=False,
         verbose=False):

    # If input is a list, read them all.
    if isinstance(fobj, list):
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

    if isinstance(fobj, file):
        if typ is None:
            raise ValueError("You must send typ= if input is a file object")
    else:
        if verbose:
            stdout.write("Reading: %s\n" % fobj)

        # for strings we can guess the filetype from the extension
        if typ is None:
            typ=get_ftype(fobj)

        # make sure we expand all ~username and other variables
        if isinstance(fobj, str):
            fobj=ostools.expand_filename(fobj)



    typ = typ.lower()


    # pick the right reader based on typ
    if typ == 'fits':
        return read_fits(fobj, ext=ext, header=header, view=view)

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
