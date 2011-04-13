import numpy

try:
    import pyfits
    have_pyfits=True
except:
    have_pyfits=False

from . import recfile
from . import numpy_util

modemap={'r':'readonly'}

def read(filename, ext, **keys):
    """
    Read data from a FITS file using the FITS class.

    Currently this is a wrapper for pyfits designed to allow reading subsets of
    rows and columns.  The recfile package is used for this purpose.  It can be
    found in esutil, where most of the development occurs, and it's own package
    on google code.

    Parameters
    ----------
    ext: number 
        The extension

    rows: scalar,sequence,array
        A scalar, sequence or array indicating a subset of rows to read.
    fields or columns: scalar, sequence, array 
        A subset of field to read. fields and columns mean the same thing.
    view: type object
        Specify an alternative view of the data.  Should be a subclass
        of numpy.ndarray

    lower,upper: boolean
        If lower, all names are converted to lower case.
        If upper, all names are converted to upper case.

    split: boolean
        Return a tuple of results rather than a rec array. Note the data
        are still stored in one big chunk, this is just an alternative access
        method.  E.g.

        # this might return a rec array with fields accessed
        # such as data['x'] data['y'] data['index']
        data = r.read()
        # this returns a tuple with an element for each
        x,y,index = r.read(split=True)

    Limitations
    -----------
    .gz files will always be read fully into memory.


    """
    f=FITS(filename, **keys)
    return f.read(ext, **keys)

class FITS(list):
    """
    A class for working with fits files.

    Currently this is a wrapper for pyfits designed to allow reading subsets of
    rows and columns.  The recfile package is used for this purpose.  It can be
    found in esutil, where most of the development occurs, and it's own package
    on google code.

    Examples
    ----------
        f = FITS(filename)
        # read extension 1
        data = f.read(1)

        # read a subset of rows and columns in a binary table extension
        data = f.read(1, rows=[1,235,881], columns=['ra','dec'])

        # read a header
        h = f.read_header(1)

    Limitations
    -----------
    .gz files will always be read fully into memory.


    """
    def __init__(self, filename=None, mode='r', **keys):
        self.filename=filename
        if filename is not None:
            self.open(filename, mode, **keys)

    def open(self, filename, mode='r', **keys):
        while len(self) > 0:
            self.pop()

        if mode not in modemap:
            raise ValueError('only support modes: %s' % list(modemap.keys()))

        if mode == 'r':
            mode='readonly'
        hdulist = pyfits.core.open(filename,mode)
        for i in xrange(len(hdulist)):
            self.append(hdulist[i])


    def read(self, ext, **keys):
        """
        Read data from a fits file

        Table Parameters
        -----------------
        ext: number 
            The extension

        rows: scalar,sequence,array
            A scalar, sequence or array indicating a subset of rows to read.
        fields or columns: scalar, sequence, array 
            A subset of field to read. fields and columns mean the same thing.
        view: type object
            Specify an alternative view of the data.  Should be a subclass
            of numpy.ndarray

        lower,upper: boolean
            If lower, all names are converted to lower case.
            If upper, all names are converted to upper case.

        split: boolean
            Return a tuple of results rather than a rec array. Note the data
            are still stored in one big chunk, this is just an alternative access
            method.  E.g.

            # this might return a rec array with fields accessed
            # such as data['x'] data['y'] data['index']
            data = r.read()
            # this returns a tuple with an element for each
            x,y,index = r.read(split=True)

        Limitations
        -----------
        .gz files will always be read fully into memory.

        """
        self._check_ext(ext)
        if isinstance(self[ext], pyfits.BinTableHDU):
            return self.read_table(ext, **keys)
        else:
            return self[ext].data

    def read_table(self, ext, **keys):
        """
        Read data from a fits binary table, possibly selecting rows and columns.

        Parameters
        ----------
        ext: number 
            The extension

        rows: scalar,sequence,array
            A scalar, sequence or array indicating a subset of rows to read.
        fields or columns: scalar, sequence, array 
            A subset of field to read. fields and columns mean the same thing.
        view: type object
            Specify an alternative view of the data.  Should be a subclass
            of numpy.ndarray

        lower,upper: boolean
            If lower, all names are converted to lower case.
            If upper, all names are converted to upper case.

        split: boolean
            Return a tuple of results rather than a rec array. Note the data
            are still stored in one big chunk, this is just an alternative access
            method.  E.g.

            # this might return a rec array with fields accessed
            # such as data['x'] data['y'] data['index']
            data = r.read()
            # this returns a tuple with an element for each
            x,y,index = r.read(split=True)

        Limitations
        -----------
        .gz files will always be read fully into memory.

        """
        self._check_ext(ext)

        if not isinstance(self[ext], pyfits.BinTableHDU):
            raise ValueError("Extension %s is not a BinTableHDU" % ext)

        hdu = self[ext] 
        hdu._file.seek(hdu._datLoc)
        
        dtype=self.get_dtype(hdu, **keys)
        nrows = hdu.size()/dtype.itemsize
        robj = recfile.Recfile(hdu._file, dtype=dtype, nrows=nrows)

        res = robj.read(**keys)
        
        return res
        

    def read_header(self, ext, **keys):
        self._check_ext(ext)
        return self[ext].header

    def get_dtype(self, hdu, **keys):
        """
        Copied from pyfits.core._get_tbdata and modified
        """

        lower=keys.get('lower',False)
        upper=keys.get('upper',False)

        # get the right shape for the data part of the random group,
        # since binary table does not support ND yet
        if isinstance(hdu, pyfits.GroupsHDU):
            f = `hdu._dimShape()[:-1]` + tmp._dat_format
            dtype = pyfits.core._convert_format(f)
        else:
            dtype=[]
            for c in hdu.columns:
                name = c.name
                if lower:
                    name=name.lower()
                elif upper:
                    name=name.upper()

                format = c.format
                dt = pyfits.core._convert_format(format)
                dtype.append( (name, dt) )
        
        dtype = numpy.dtype(dtype)

        # if this is a little endian machine, swap the byteorder of our dtype
        if numpy.little_endian:
            dtype = dtype.newbyteorder('>')
        return dtype


    def _check_ext(self, ext):
        if len(self) == 0:
            raise ValueError("Open a file first")
        maxext = len(self)-1
        if ext > maxext:
            raise ValueError("ext is outside of range [%s,%s]" % (0,maxext))
