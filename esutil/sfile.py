"""
Module:
    sfile

Purpose:

    Read and write numpy arrays to a simple file format.  The format is a
    simple ascii header followed by data, either in binary form or ascii.  For
    details of the ascii header, see the documentation for the read_header()
    function.
    
    The power of this code comes from two unique features of the RECFILE
    package: 
        
        1) The recfile package understands the structure of arrays with
        fields, aka recarrays or structured arrays.  Unlike the memmap class
        distributed with numpy, individual fields from these types of arrays
        can be read without reading the whole file into memory.  This is
        accomplished using a C++ class linked to python.

        2) Structured arrays can be written to and read from delimited ascii
        files.  This is done efficiently using the same C++ class linked to
        python.

    Another nice feature not supported by the format module of numpy is the
    ability to append data to the file.

    Although the non-memory map functiontionality looks very much like a memory
    map, the underlying class is *not* true memory map.  It does not support
    writing data in place in the file, only writing new files or appending.


    Note, a memory map is actually used when appropriate, e.g. for binary files
    when no sub-fields are being selected.  Also, you can use the get_memmap
    method of the SFile class to get a memmap to the underlying data.


Convenience functions:
    Open(): Get an instance of the SFile class.  Using an SFile instance
        is the most powerful method.  This can also return a memory map.

    read_header():  A convenience function to read the header from the simple
        file format.  Uses an SFile instance internally.
    read():  A convenience function to read data and optionally the header
        from the simple file format. Uses an SFile instance internally.
    write(): A convenience function to write data to the simple file format,
        including an ascii header. Uses an SFile instance internally.

    For more docs, check the docs for the individual functions.


Class SFile

    Examples of usage.  See docs for sfile.SFile for more details.  Note if
    using this as part of esutil, import with "from esutil import sfile"


    Examples with record arrays
        import sfile

        # Open and read from a .rec file
        sf = sfile.Open(filename_or_object)

        # see what's in the file
        print(sf)
        filename: 'test.rec'
        mode: 'r'
        size: 64526192
        dtype:
          [('x', '<f4'),
           ('y', '<f4'),
           ('flux', '<f8'),
           ('name', '|S20')]
        hdr:
          {'dataset': 'dc4',
           'pyvers':'v2.6.2'}


        # get the header
        hdr = sf.read_header()

        #
        # A few ways to read all the data
        #
        data = sf[:]
        data = sf.read()

        # reading subsets of the data

        # read an entire column name 'x' from the file.  Only data for that
        # column are read
        data = sf['x'][:]
        data = sf.read(columns='x')

        # read a list of columns.  Any list, tuple or numpy array of strings
        # can be sent.
        data = sf[ ['x','y'] ][:]
        data = sf.read(columns=column_list)

        # read a subset of rows.  Can use slices, single numbers for rows,
        # or list/array of row numbers.

        data = sf[35]
        data = sf[35:100]
        data = sf[ [35,66,22,23432] ]
        data = sf[ row_list ]
        data =sf.read(rows=row_list)
        
        # read a subset of rows and columns.  Only those rows/columns are
        # read from the file.

        data = sf[columns][rows]
        data = sf[ 35:5012 ][ ['x','y'] ]
        data = sf.read(colums=columns, rows=rows)


        # writing data to a CSV file
        data[0:5]
            array([(0, 0.13933435082435608), 
                   (1, 0.78211665153503418),
                   (2, 0.13652685284614563), 
                   (3, 0.54205995798110962),
                   (4, 0.2745462954044342)],
                      dtype=[('ind', '<i4'), ('rnd', '<f4')])

        sf = Open('test.rec', 'w+', delim=',')
        sf.write(data)
        sf.write(more_data)

        # check the first row of the data we have written
        sf[0]
            array([(0, 0.13933435082435608)],
                     dtype=[('ind', '<i4'), ('rnd', '<f4')])

        sf.write(more_data)
        sf.close()

    Examples with ordinary arrays:
        # With ordinary arrays, appending is not supported, nor is the 
        [ ] notation.  Use a memmap or the basic read() function instead.

        data = numpy.random.random(n*m).reshape( (n,m) )
        sf=sfile.Open('randmatrix.rec', 'w+')
        sf.write(data)

        tmp = sf.read()
        print(tmp[2,3])
        0.54422426143267832
        
        mmap = sf.get_memmap()
        print(mmap[2,3])
        0.54422426143267832

Modification History:
    This section will track major changes.  See docs for SFile and individual
    functions for smaller changes.


    Created: 2007-05-25, Erin Sheldon, NYU

    2010-02-18  Added '_VERSION' to the header, allowing future changes
        while still supporting backwards compatibility.  
            Erin Sheldon, BNL

"""
# vim: set filetype=python :
import sys
from sys import stdout,stderr
import os
import pprint
import copy

try:
    import numpy
    # importing array will allow array types in the header to be properly
    # eval()uated.  Not sure if this works for all dtypes.
    from numpy import array
    have_numpy=True
except:
    have_numpy=False

try:
    try:
        # look for local install of recfile first
        from esutil import recfile
        have_recfile=True
    except:
        # might be installed elsewhere
        import recfile
        have_recfile=True
except:
    # ascii reading/writing not supported
    have_recfile=False



SFILE_VERSION='1.0'


def Open(fobj, mode='r', delim=None, verbose=False, memmap=False):
    sf = SFile(fobj, mode=mode, delim=delim, verbose=verbose)
    if memmap and delim is None:
        return sf.get_memmap()
    else:
        return sf

class SFile(object):
    """
    Class SFile

    This class implements a simple file format for holding numerical python
    arrays.  The format is a simple ascii header followed by data, either in
    binary form or ascii.  For details of the ascii header, see the
    documentation for the read_header() function.


    Examples with record arrays
        import sfile

        # Open and read from a .rec file
        sf = sfile.Open(filename_or_object)

        # see what's in the file
        print(sf)
        filename: 'test.rec'
        mode: 'r'
        size: 64526192
        dtype:
          [('x', '<f4'),
           ('y', '<f4'),
           ('flux', '<f8'),
           ('name', '|S20')]
        hdr:
          {'dataset': 'dc4',
           'pyvers':'v2.6.2'}


        # get the header
        hdr = sf.read_header()

        #
        # A few ways to read all the data
        #
        data = sf[:]
        data = sf.read()

        # reading subsets of the data

        # read an entire column name 'x' from the file.  Only data for that
        # column are read
        data = sf['x'][:]
        data = sf.read(columns='x')

        # read a list of columns.  Any list, tuple or numpy array of strings
        # can be sent.
        data = sf[ ['x','y'] ][:]
        data = sf.read(columns=column_list)

        # read a subset of rows.  Can use slices, single numbers for rows,
        # or list/array of row numbers.

        data = sf[35]
        data = sf[35:100]
        data = sf[ [35,66,22,23432] ]
        data = sf[ row_list ]
        data =sf.read(rows=row_list)
        
        # read a subset of rows and columns.  Only those rows/columns are
        # read from the file.

        data = sf[columns][rows]
        data = sf[ 35:5012 ][ ['x','y'] ]
        data = sf.read(colums=columns, rows=rows)


        # writing data to a CSV file
        data[0:5]
            array([(0, 0.13933435082435608), 
                   (1, 0.78211665153503418),
                   (2, 0.13652685284614563), 
                   (3, 0.54205995798110962),
                   (4, 0.2745462954044342)],
                      dtype=[('ind', '<i4'), ('rnd', '<f4')])

        sf = Open('test.rec', 'w+', delim=',')
        sf.write(data)
        sf.write(more_data)

        # check the first row of the data we have written
        sf[0]
            array([(0, 0.13933435082435608)],
                     dtype=[('ind', '<i4'), ('rnd', '<f4')])

        sf.write(more_data)
        sf.close()

    Examples with ordinary arrays:
        # With ordinary arrays, appending is not supported, nor is the 
        [ ] notation.  Use a memmap or the basic read() function instead.

        data = numpy.random.random(n*m).reshape( (n,m) )
        sf=sfile.Open('randmatrix.rec', 'w+')
        sf.write(data)

        tmp = sf.read()
        print(tmp[2,3])
        0.54422426143267832
        
        mmap = sf.get_memmap()
        print(mmap[2,3])
        0.54422426143267832
    """
    def __init__(self, fobj=None, mode='r', delim=None, 
                 padnull=False, ignorenull=False, verbose=False):
        self.open(fobj, mode=mode, delim=delim, verbose=verbose)

    def open(self, fobj, mode='r', delim=None, verbose=False,
             padnull=False, ignorenull=False):
        """
        Open the file.  If the file already exists and the mode is 'r*' then
        a read of the header is attempted.  If this succeeds, delim is gotten
        from the header and the delim= keyword is ignored.
        """

        if not have_numpy:
            raise ImportError("numpy could not be imported")

        self.verbose=verbose

        self.close()
        self.padnull=padnull
        self.ignorenull=ignorenull

        if fobj is None:
            return

        if isstring(fobj):
            # expand shortcut variables
            fpath = os.path.expanduser(fobj)
            fpath = os.path.expandvars(fpath)

            if self.verbose:
                stdout.write("\nOpening file: %s\n" % fpath)

            if mode == 'r+' and not os.path.exists(fobj):
                # path doesn't exist but we want to append.  Change the
                # mode to write
                if self.verbose:
                    stdout.write("Requested append on non-existent file: "
                                 "Will create a new file\n")
                mode = 'w+'

            self.fobj = open(fpath, mode)
        elif isinstance(fobj, file):
            #elif hasattr(fobj,'readline'):
            self.fobj = fobj
        else:
            raise ValueError("Only support filenames and file objects "
                             "as input")

        
        crap="""
        import gzip
        reading=False
        if isinstance(self.fobj,gzip.GzipFile):
            if self.fobj.mode == 1:
                reading=True

        elif isinstance(self.fobj,file):
            if self.fobj.mode[0] == 'r':
                reading=True
        """


        if self.fobj.mode[0] == 'r':
            #if reading:
            # For 'r' and 'r+' try to read the header
            if self.verbose:
                stdout.write("\tReading header\n")
            self.hdr = self.read_header()
            
            self.data_start = self.fobj.tell()

            self.delim = _match_key(self.hdr, '_delim')
            self.size = _match_key(self.hdr, '_size', require=True)
            self.descr = _match_key(self.hdr, '_dtype', require=True)
            self.dtype = numpy.dtype(self.descr)
            self.has_fields = (self.dtype.names is not None)
            # will be None unless no fields
            self.shape = self.get_shape()
        else:
            # get delim from the keyword.  This will be used for writing later
            self.delim=delim


    def close(self):
        """
        Close any open file object.  Make sure fobj, _hdr, and delim are None
        """
        self.hdr=None
        self.data_start=None
        self.delim=None
        self.size=0
        self.descr=None
        self.dtype=None
        self.has_fields=None
        self.shape=None
        if hasattr(self, 'fobj'):
            if self.fobj is not None:
                if isinstance(self.fobj, file):
                    self.fobj.close()
        self.fobj=None

        self.padnull=False
        self.ignorenull=False

    def __repr__(self):
        s = []

        if isinstance(self.fobj,file):
            s += ["filename: '%s'" % self.fobj.name]
            s += ["mode: '%s'" % self.fobj.mode]
        if self.delim is not None:
            s=["delim: '%s'" % self.delim]

        s += ["size: %s" % self.size]

        if self.shape is not None:
            s += ["shape: " + pprint.pformat(self.shape)]
        if self.descr is not None:
            drepr=pprint.pformat(self.descr)
            drepr = '  '+drepr.replace('\n','\n  ')
            s += ["dtype: \n"+drepr]

        if self.hdr is not None:
            hs = self.h2string()
            if hs != '':
                hs = '  '+hs.replace('\n','\n  ')
                if hs != '':
                    s += ["hdr: \n"+hs]
        s = "\n".join(s)
        return s

    def h2string(self, strip=True):
        if self.hdr is None:
            return ''
        newd = {}

        skipkeys=['_VERSION','_DTYPE','_SIZE','_NROWS',
                  '_HAS_FIELDS','_DELIM','_SHAPE']
        for key in self.hdr:
            if strip:
                if key not in skipkeys:
                    newd[key] = self.hdr[key]
            else:
                news[key] = self.hdr[key]

        if len(newd) == 0:
            return ''
        return pprint.pformat(newd)


    def write(self, data, header=None):
        """
        """

        # Write or update the header. If we are updating the file, this will
        # just alter the number of rows marked in the header and in self.size.
        # Either way, self.fobj will point to the end of the file, ready to
        # write the data

        if self.fobj.mode[0] != 'w' and '+' not in self.fobj.mode:
            raise ValueError("You must open with 'w*' or 'r+' to write")

        self._write_header(data, header=header)

        if data.dtype.names is None:
            self._write_simple(data)
        else:
            self._write_structured(data)


    def _write_structured(self, data):
        if self.verbose:
            stdout.write("Writing structured %s: %s\n" % \
                            (data.size,pprint.pformat(data.dtype.descr)))
        if (self.delim is not None) and (have_recfile is not True):
            stdout.write("delim='"+self.delim+"'\n")
            stdout.write("have_recfile=%s\n" % have_recfile)
            raise ImportError("recfile package not found.  recfile is "
                              "required for writing ascii records")
        elif (self.delim is not None):
            # let recfile deal with ascii writing
            r = recfile.Open(self.fobj, mode='r+', delim=self.delim, 
                             dtype=data.dtype,
                             padnull=self.padnull, ignorenull=self.ignorenull)
            # simple view of data
            dataview = data.view(numpy.ndarray) 
            r.Write(dataview)
        else:
            # Write data out as a binary chunk
            data.tofile(self.fobj)

    def _write_simple(self, data):
        if self.verbose:
            stdout.write("Writing simple %s: '%s'\n" % (data.size,data.dtype.str))
        # got to wrap this because tofile does not support delim=None
        if self.delim is not None:
            data.tofile(self.fobj, sep=self.delim)
        else:
            data.tofile(self.fobj)


    def flush(self):
        if hasattr(self,'fobj'):
            if isinstance(self.fobj, file):
                self.fobj.flush()


    def get_subset(self, rows=None, fields=None, columns=None):
        robj = recfile.Open(self.fobj, nrows=self.size, mode='r', 
                            offset=self.fobj.tell(),
                            dtype=self.dtype, delim=self.delim)
        return robj.get_subset(rows=rows, fields=fields, columns=columns)

    def get_memmap(self, mode='r', view=None, header=False):

        if self.delim is not None:
            raise ValueError("Cannot memory map ascii files")

        if self.fobj.tell() != self.data_start:
            self.fobj.seek(self.data_start)


        if not self.has_fields and self.shape is not None:
            shape = self.shape
        else:
            shape = (self.size,)

        result = numpy.memmap(self.fobj, dtype=self.dtype, shape=shape, 
                              mode=mode, offset=self.fobj.tell())
        if view is not None:
            result = result.view(view)
        if header:
            return result, self.hdr
        else:
            return result



    def read(self, rows=None, fields=None, columns=None, header=False, 
             view=None, split=False, reduce=False):
        """
        Read the data into memory.
        """
        
        if self.fobj.tell() != self.data_start:
            self.fobj.seek(self.data_start)

        if not self.has_fields:
            result = self._read_simple()
        else:
            result = self._read_structured(rows=rows, 
                                           fields=fields, columns=columns)
        if view is not None:
            result = result.view(view)
        elif split:
            result = split_fields(result)
        elif reduce:
            result = reduce_array(result)

        if header:
            return result, self.hdr
        else:
            return result

    def __getitem__(self, arg):
        """

        # read subsets of columns and/or rows from the file.  This only works
        # for record types
        sf = SFile(....)


        # read subsets of rows
        data = sf[ 35 ]
        data = sf[ 35:88 ]
        data = sf[ [3,234,5551,.. ] ]

        # read subsets of columns
        data = sf['fieldname'][:]
        data = sf[ ['field1','field2',...] ][:]

        # read subset of rows *and* columns.
        data = sf['fieldname'][3:58]
        data = sf[fieldlist][rowlist]
        """
        if not self.has_fields:
            raise RuntimeError("For simple arrays, use a read() or a memmap "
                               "object returned by get_memmap()")

        if self.fobj.tell() != self.data_start:
            self.fobj.seek(self.data_start)

        robj = recfile.Open(self.fobj, nrows=self.size, mode='r', 
                            offset=self.fobj.tell(),
                            dtype=self.dtype, delim=self.delim)
        return robj[arg]


    def get_shape(self):
        if self.has_fields:
            return None
        shape = _match_key(self.hdr, '_shape')
        if shape is None:
            return None
        ashape = numpy.array(shape,ndmin=1)
        if ashape.prod() == self.size:
            return shape
        else:
            return None

    def _read_structured(self, rows=None, fields=None, columns=None):
        if self.verbose:
            stdout.write("\tReading as a structured array\n")


        rows2read = self._get_rows2read(rows)
        fields2read = self._get_fields2read(fields, columns=columns)

        if fields2read is None and rows2read is None and self.delim is None:
            # Its binary and all, just use fromfile
            result = numpy.fromfile(self.fobj,dtype=self.dtype)
        else:
            if have_recfile:
                result = self._recfile_read(rows=rows2read, fields=fields2read)
            else:
                result = self._memmap_read(rows=rows2read, fields=fields2read)

        return result

    def _recfile_read(self, rows=None, fields=None):
        robj = recfile.Open(self.fobj, nrows=self.size, mode='r', 
                            offset=self.fobj.tell(),
                            dtype=self.dtype, delim=self.delim)
        return robj.Read(rows=rows, fields=fields)

    def _memmap_read(self,rows=None, fields=None):
        raise RuntimeError("Not Yet Implemented")

    def _read_simple(self):
        if self.verbose:
            stdout.write("\tReading as a simple array\n")
        # delim can't be None, so case it
        if self.delim is None:
            result = numpy.fromfile(self.fobj, dtype=self.dtype, 
                                    count=self.size)
        else:
            result = numpy.fromfile(self.fobj, dtype=self.dtype, 
                                    count=self.size, sep=self.delim)
        return self._try_reshape(result) 

    def _try_reshape(self, data):
        shape = _match_key(self.hdr, '_shape')
        if shape is None:
            return data

        # Only use it if the lengths match
        try:
            ashape = numpy.array(shape, ndmin=1)
        except:
            return data
        if ashape.prod() == data.size:
            if self.verbose:
                stdout.write("\tReshaping array to "
                             "%s\n" % pprint.pformat(shape))
            return data.reshape(shape)
        else:
            if self.verbose:
                stdout.write("_shape number of elements does not match "
                             "array size: not reshaping\n")
            return data

    def _get_rows2read(self, rows):
        if rows is None:
            return None
        rows2read = numpy.array(rows,ndmin=1,copy=False)
        if rows2read is not None and self.verbose:
            stdout.write("\t\tReading %s rows\n" % len(rows2read))

        return rows2read

    def _get_fields2read(self, fields, columns=None):
        if fields is None:
            fields=columns

        if fields is None:
            return None
        elif isinstance(fields, (list,numpy.ndarray)):
            f=fields
        elif isinstance(fields,tuple):
            f=list(fields)
        elif isstring(fields):
            f=[fields]
        else:
            raise ValueError('fields must be list,tuple,string or array')

        if not isstring(f[0]):
            # this is probably a list of column numbers, convert to strings
            allnames = self.dtype.names
            f = [allnames[i] for i in f]

        if self.verbose:
            _out = (len(f), pprint.pformat(f))
            stdout.write("\t\tReading %s fields: %s\n" % _out)

        return f





    def _make_header(self, data, header=None):
            if header is None:
                head={}
            else:
                head=copy.deepcopy(header)

            for key in ['_size','_nrows','_delim','_shape','_has_fields']:
                if key in head: del head[key]
                if key.upper() in head: del head[key.upper()]

            has_fields = (data.dtype.names is not None)
            if has_fields:
                descr = data.dtype.descr
            else:
                descr = str(data.dtype.str)
                head['_SHAPE'] = data.shape

            # this is not needed
            #head['_HAS_FIELDS'] = has_fields

            if self.delim is not None:
                head['_DELIM'] = self.delim

                # Text file. Remove the byte order specification. 
                descr = self._remove_byteorder(descr)

            head['_DTYPE'] = descr
            head['_VERSION'] = SFILE_VERSION

            return head

    def _write_header(self, data, header=None):

        if self.hdr is not None:
            # we are appending data.  Make sure we are dealing with a
            # structured array, or else we will abort the append.

            # Just update the nrows and move to the end
            self._update_size(data.size)
            if self.verbose:
                stdout.write("\tSeeking to end of file\n")
            self.fobj.seek(0,2) # Seek to end of file
        else:

            if self.verbose:
                stdout.write("\tCreating new header\n")
            self.hdr = self._make_header(data, header=header)
            self.descr = self.hdr['_DTYPE']
            self.dtype = numpy.dtype(self.descr)
            self.has_fields = (self.dtype.names is not None)

            self.fobj.seek(0)

            self._write_size(data.size)
            self.size = data.size

            # As long as the dict contains types that can be represented as
            # constants, this pretty printing can be eval()d.

            pprint.pprint(self.hdr, stream=self.fobj)

            self.fobj.write('END\n')
            self.fobj.write('\n')

            self.data_start = self.fobj.tell()

    def _remove_byteorder(self, descr):
        if isstring(descr):
            return descr[1:]

        new_descr = []
        for d in descr:
            # d is a tuple, make it a list
            newd = list( copy.copy(d) )

            tdef = newd[1]
            tdef = tdef[1:]
            newd[1] = tdef
            newd = tuple(newd)
            new_descr.append(newd)

        return new_descr

    def _update_size(self, size_add):
        size_current = self.size
        if size_current is None:
            raise RuntimeError('Attempting to update size but not found in header')
        size_new = size_current + size_add

        if self.verbose:
            stdout.write("\tUpdating size to %s\n" % size_new)

        self._write_size(size_new)
        self.size = size_new
        self.hdr['_SIZE'] = size_new

    def _write_size(self, size):
        # Go to the beginning of the file
        self.fobj.seek(0)
        # Specially formatted fixed-length for updating later
        out = 'SIZE = %20d\n' % (size,)
        self.fobj.write(out)


    def _extract_size(self, header):
        if not isinstance(header,dict):
            raise RuntimeError('header must be a dict')

        size = _match_key(header,'_size')
        if size is None:
            size = _match_key(header,'_nrows')
            if size is None:
                raise RuntimeError("Header must contain the size in first "
                                   "row as 'SIZE = %20d'")
        return long(size)


    def read_header(self):
        """
        Name:
            read_header()

        Calling Sequence:
            sf = sfile.Open(file)
            hdr = sf.read_header()

        Read the header from a simple self-describing file format with an
        ascii header.  See the write() function for information about reading
        this file format, and read() for reading.


        The file format:
          First line:
              SIZE = --------------number

        where if possible the number should be formatted as %20d.  This is
        large enough to hold a 64-bit number.  This exact formatting is
        required so SIZE can be updated *in place* when appending rows to a
        file holding structured arrays.  Note the file can always be read as
        long as the first line reads SIZE = some_number but appending requires
        the exact format.


        Last two lines of the header region must be:
                END
                blank line
        case does not matter.

          
        In between the SIZE and END lines is the header data.  This is a
        string that must eval() to a dictionary.  It must contain the
        following entry:

              _DTYPE = array data type description in list of tuples or
                string form (case does not matter, can also be called _dtype).
                  
                for a structured array.
                    [('field1', 'f8'), ('f2','2i4')]
                for a simple array:
                    '<f4'

        As noted above, only files holding structured arrays (recarray, array
        with fields) can be appended.  If the data is a simple array or matrix
        no appending is supported.

        There should also be a _VERSION tag.

              '_VERSION': '1.0'

        If '_VERSION' is not present, it is assumed that the version is 1.0,
        but you should always set this.  If you use this module to write data,
        it will always be set.


        If the file holds a simple array, and the dtype field is a simple
        string, then the following keyword, if present, will be used to
        reshape the array:

              '_SHAPE'

        If the total elements in the _shape field matches the size then it
        will be used to reshape the array before returning or when using
        memory maps.

        If the data are ascii then delimiter must be given in the keyword

              _DELIM  
              
        This can be for example ',', ' ', or a tab character.  Again, case does
        not matter.  

        The rest of the keywords can by any variable can be used as long as it
        can be eval()d.

        An example header:
            SIZE =                   10
            {'_VERSION': '1.0',
             '_DELIM': ',',
             '_DTYPE': [('x', 'f4'),
                        ('y', 'f4'),
                        ('ra', 'f8'),
                        ('dec', 'f8'),
                        ('exposurename', 'S20'),
                        ('ccd', 'i1'),
                        ('size_flags', 'i4'),
                        ('magi', 'f4'),
                        ('sigma0', 'f4'),
                        ('star_flag', 'i4'),
                        ('shear_flags', 'i4'),
                        ('shapelet_sigma', 'f4'),
                        ('shear1', 'f4'),
                        ('shear2', 'f4'),
                        ('shear_cov00', 'f4'),
                        ('shear_cov01', 'f4'),
                        ('shear_cov11', 'f4')],
             'listvar': [1, 2, 3],
             'subd': {'subd1': 'subfield', 'sublist': [8.5, 6.6]},
             'svar': 'hello',
             'test1': 35}
            END

            -- data begins --

        Modification History:
            Created: 2007-05-25, Erin Sheldon, NYU
            Allow continuation characters "\\" to continue header keywords
                onto the next line.  2009-05-05

            Switched header to be simply a dictionary representation.
                SIZE = %20d
                { a dictionary representation }
                END
                blank line
            This is much more powerful and simple.  Anything that can be eval()d
            can be in the header, including numpy arrays.  
                        2009-10-28. Erin Sheldon, BNL

            Allow simple arrays (without fields) to be written.  Switch to
            using SIZE = %20d on the first row instead of NROWS, this makes
            more sense for both structured and simple arrays.  Note the
            old style can still be read, just not appended.
                2009-11-16, ESS
            
            2010-02-18  Added '_VERSION' to the header, allowing future changes
                while still supporting backwards compatibility.  
                    Erin Sheldon, BNL
        """

        if self.fobj is None:
            raise ValueError("You have not yet opened a file")

        if not have_numpy:
            raise ImportError("numpy could not be imported")

        # read first line, which should be 
        # SIZE = .....
        # or
        # NROWS = ...

        self.fobj.seek(0)
        line = self.fobj.readline().strip()
        lsplit = line.split('=')
        if len(lsplit) != 2:
            raise ValueError("First line of header must be SIZE = %20d")
        fname=lsplit[0].strip()
        if fname.upper() != 'SIZE' and fname.upper() != 'NROWS':
            raise ValueError("First line of header must be SIZE = %20d")

        if fname.upper() == 'NROWS':
            self.cannot_append_oldheader=True
        else:
            self.cannot_append_oldheader=False

        size = eval(lsplit[1])


        # Read through the header until we hit "END"
        lines = []
        line=self.fobj.readline().strip()
        while line.upper() != 'END':
            lines.append(line)
            line=self.fobj.readline().strip()

        # read one more line, which should be blank
        line = self.fobj.readline()
        if line.strip() != '':
            raise RuntimeError("Header should end with END on it's own line "
                               "followed by a blank line")

        hdrstring = ' '.join(lines)
        hdr = eval(hdrstring)

        hdr['_SIZE'] = size

        # this will leave open the possibility of changing the header or other
        # details later
        if '_version' in hdr or '_VERSION' in hdr:
            pass
        else:
            hdr['_VERSION'] = '1.0'

        return hdr

    def header(self):
        return self.hdr




def write(array, outfile, header=None, delim=None, 
          padnull=False, ignorenull=False, append=False, verbose=False):
    """
    Name:
        sfile.write()

    Calling Sequence:
        sfile.write(array, outfile, header=None, delim=None, 
                    padnull=False, ignorenull=False, append=False, 
                    verbose=False)

    Write a numpy array into a simple self-describing file format with an ascii
    header.  See the read() function for information about reading this file
    format.  See the docs for the SFile class for an idea of the full
    functionality wrapped by this covenience function.


    Inputs:
        array: A numpy array.
        outfile: A string or file pointer for the output file.

    Optional Inputs:
        header=: A dictionary containing keyword-value pairs to be added to
            the header.

        delim=None: Delimiter between fields.  Default is None for binary.  For
            ascii can be any string, e.g. ',', ' ', or a tab character.

        padnull=False:  
            When writing ascii, replace Null characters with spaces.  This is
            useful when writing files to be read in by programs that do not
            recognize null characters, e.g. sqlite databases.  But note, if
            read back in these fields will not compare equal with the original
            data!

        ignorenull=False:  
            When writing ascii, ignore Null characters.  This is useful when
            writing files to be read in by programs that do not recognize null
            characters, e.g. sqlite databases.  But note you will not be
            able to read the data back in with sfile.read() becuase the fields
            are no longer the correct length!

        append=False: Append to the file. Default is False. If set to True,
            then what happens is situation dependent:
                1) if the input is a file object then it is assumed there is 
                    an existing header.  The header is updated to reflect the
                    new appended rows after writing.
                2) if the input is a string and the file exists, then the file
                    is opened with mode "r+", it is assumed the header
                    exists, and the header is updated to reflext the new
                    appended rows after writing.
                3) if the input is a string and the file does *not* exist,
                    then the file is opened with mode "w" and the request
                    to append is ignored.

        verbose=False: Write informative messages.

    Examples:
        import sfile
        hdr={'date': '2007-05-12','age': 33} 
        sfile.write(arr, 'test.rec', header=hdr)

        sfile.write(arr2, 'test.rec', append=True)

        If this is part of the esutil package, use
            import esutil
            esutil.sfile.write(...)

    File Format:
        The file format is an ascii header followed by data in binary or rows
        of ascii.  The data columns must be fixed length in order to map onto
        numpy arrays.  See the documentation for read_header() for details
        about the header format.

    Modification History:
        Created: 2007-05-25, Erin Sheldon, NYU
        Ignore append=True when file does not yet exist.
        Allow continuation characters "\\" to continue header keywords
        onto the next line.  2009-05-05

        Moved to object-oriented approach using the SFile class.
            2009-11-16, ESS, BNL

    """



    #def write(array, outfile, header=None, delim=None, 
    #      padnull=False, ignorenull=False, append=False):

    if append:
        # if file doesn't yet exist, this will be changed to 'w+' internally.
        mode = 'r+'
    else:
        mode = 'w'

    sf=SFile(outfile, mode=mode, delim=delim, 
             padnull=padnull, ignorenull=ignorenull, verbose=verbose)
    sf.write(array, header=header)

    return

def read(infile, rows=None, fields=None, columns=None, 
         header=False, view=None, memmap=False, verbose=False):
    """
    sfile.read()

    Read a numpy array from a simple self-describing file format with an ascii
    header.  See the write() function for information about this file format.
    For ascii files the recfile package is used for reading and writing.

    Calling Sequence:
        arr = sfile.read(infile, rows=None, columns=None, 
                         header=False, view=numpy.ndarray)

    Inputs:
        infile: A string or file pointer for the input file.
    Optional Inputs:
        rows=: A scalar, array, list or tuple of row numbers to read.  
            Default is all.
        columns=: A scalar, list or tuple of strings naming columns to read 
            from the file.  Default is all.
        fields=:  Same as sending columns=.
        header=False:  If True, return both the array and the header dict in
            a tuple.
        view=:  How to view the array.  Default is numpy.ndarray.

    Examples:
        import sfile
        arr=sfile.read('test.rec')

        arr2, hdr = sfile.read('test.rec', 
                               rows=[3,4], fields=['ra', 'dec'], header=True)

    File Format:
        The file format is an ascii header followed by data in binary or rows
        of ascii.  The data columns must be fixed length in order to map onto
        numpy arrays.  See the documentation for read_header() for details
        about the header format.


    Modification History:
        Created: 2007-05-25, Erin Sheldon, NYU
        Allow continuation characters "\\" to continue header keywords
        onto the next line.  2009-05-05

        Switched over to new header format.  2009-10-38, Erin Sheldon, BNL.
    """

    #def read(infile, rows=None, fields=None, columns=None, norecfile=False,
    #         header=False, view=None, memmap=False):

    sf = SFile(infile, verbose=verbose)
    if memmap:
        return sf.get_memmap(view=view, header=header)
    else:
        return sf.read(rows=rows, fields=fields, columns=columns, 
                       view=view, header=header)

    

def read_header(infile, verbose=False):
    """
    Name:
        read_header()

    Calling Sequence:
        hdr = sfile.read_header(infile)

    Inputs:
        infile: A string or file pointer for the input file.

    Read the header from a simple self-describing file format with an
    ascii header.  See the write() function for information about reading
    this file format, and read() for reading.


    The file format:
      First line:
          SIZE = --------------number

    where if possible the number should be formatted as %20d.  This is
    large enough to hold a 64-bit number.  This exact formatting is
    required so SIZE can be updated *in place* when appending rows to a
    file holding structured arrays.  Note the file can always be read as
    long as the first line reads SIZE = some_number but appending requires
    the exact format.


    Last two lines of the header region must be:
            END
            blank line
    case does not matter.

      
    In between the SIZE and END lines is the header data.  This is a
    string that must eval() to a dictionary.  It must contain the
    following entry:

          _DTYPE = array data type description in list of tuples or
            string form (case does not matter).
              
            for a structured array.
                [('field1', 'f8'), ('f2','2i4')]
            for a simple array:
                '<f4'

    As noted above, only files holding structured arrays (recarray, array
    with fields) can be appended.  If the data is a simple array or matrix
    no appending is supported.

    There should also be a _VERSION tag.

        '_VERSION': '1.0'

    Case does not matter. If '_VERSION' is not present, it is assumed that the
    version is 1.0, but you should always set this.  If you use this module to
    write data, it will always be set.


    If the file holds a simple array, and the dtype field is a simple
    string, then the following keyword, if present, will be used to
    reshape the array:

          '_SHAPE'

    If the total elements in the _shape field matches the size then it
    will be used to reshape the array before returning or when using
    memory maps.

    If the data are ascii then delimiter must be given in the keyword

          _DELIM  
          
    This can be for example ',', ' ', or a tab character.  Again, case does
    not matter.  

    The rest of the keywords can by any variable can be used as long as it
    can be eval()d.

    An example header:
        SIZE =                   10
        {'_VERSION': '1.0',
         '_DELIM': ',',
         '_DTYPE': [('x', 'f4'),
                    ('y', 'f4'),
                    ('ra', 'f8'),
                    ('dec', 'f8'),
                    ('exposurename', 'S20'),
                    ('ccd', 'i1'),
                    ('size_flags', 'i4'),
                    ('magi', 'f4'),
                    ('sigma0', 'f4'),
                    ('star_flag', 'i4'),
                    ('shear_flags', 'i4'),
                    ('shapelet_sigma', 'f4'),
                    ('shear1', 'f4'),
                    ('shear2', 'f4'),
                    ('shear_cov00', 'f4'),
                    ('shear_cov01', 'f4'),
                    ('shear_cov11', 'f4')],
         'listvar': [1, 2, 3],
         'subd': {'subd1': 'subfield', 'sublist': [8.5, 6.6]},
         'svar': 'hello',
         'test1': 35}
        END

        -- data begins --

    Modification History:
        Created: 2007-05-25, Erin Sheldon, NYU
        Allow continuation characters "\\" to continue header keywords
            onto the next line.  2009-05-05

        Switched header to be simply a dictionary representation.
            SIZE = %20d
            { a dictionary representation }
            END
            blank line
        This is much more powerful and simple.  Anything that can be eval()d
        can be in the header, including numpy arrays.  
                    2009-10-28. Erin Sheldon, BNL

        Allow simple arrays (without fields) to be written.  Switch to
        using SIZE = %20d on the first row instead of NROWS, this makes
        more sense for both structured and simple arrays.  Note the
        old style can still be read, just not appended.
            2009-11-16, ESS
        
        2010-02-18  Added '_VERSION' to the header, allowing future changes
            while still supporting backwards compatibility.  Erin Sheldon, BNL

    """

    sf = SFile(infile, verbose=verbose)
    return sf.hdr


def split_fields(data, fields=None, getnames=False):
    """
    Name:
        split_fields

    Calling Sequence:
        The standard calling sequence is:
            field_tuple = split_fields(data, fields=)
            f1,f2,f3,.. = split_fields(data, fields=)

        You can also return a list of the extracted names
            field_tuple, names = split_fields(data, fields=, getnames=True)

    Purpose:
        Get a tuple of references to the individual fields in a structured
        array (aka recarray).  If fields= is sent, just return those
        fields.  If getnames=True, return a tuple of the names extracted
        also.

        If you want to extract a set of fields into a new structured array
        by copying the data, see esutil.numpy_util.extract_fields

    Inputs:
        data: An array with fields.  Can be a normal numpy array with fields
            or the recarray or another subclass.
    Optional Inputs:
        fields: A list of fields to extract. Default is to extract all.
        getnames:  If True, return a tuple of (field_tuple, names)

    """

    outlist = []
    allfields = data.dtype.fields

    if allfields is None:
        if fields is not None:
            raise ValueError("Could not extract fields: data has "
                             "no fields")
        return (data,)
    
    if fields is None:
        fields = allfields
    else:
        if isinstance(fields, (str,unicode)):
            fields=[fields]

    for field in fields:
        if field not in allfields:
            raise ValueError("Field not found: '%s'" % field)
        outlist.append( data[field] )

    output = tuple(outlist)
    if getnames:
        return output, fields
    else:
        return output

def reduce_array(data):
    # if this is a structured array with fields, and only has a single
    # field, return a simple array view of that field, e.g. data[fieldname]
    if hasattr(data, 'dtype'):
        if data.dtype.names is not None:
            if len(data.dtype.names) == 1:
                # get a simpler view
                return data[data.dtype.names[0]]
    else:
        return data

def _match_key(d, key, require=False):
    """
    Match the key in a case-insensitive way and return the value. Return None
    if not found or raise an error if require=True
    """
    if not isinstance(d,dict):
        raise RuntimeError('Input object must be a dict')
    keys = list( d.keys() )
    keyslow = [k.lower() for k in keys]
    keylow = key.lower()
    if keyslow.count(keylow) != 0:
        ind = keyslow.index(keylow)
        return d[keys[ind]]
    else:
        if not require:
            return None
        else:
            raise RuntimeError("Could not find required key: '%s'" % key)
 

def isstring(obj):
    string_types=(basestring, numpy.string_)

    if isinstance(obj, string_types):
        return True
    else:
        return False

def _fix_range(i, maxval):
    if i < 0:
        i=maxval-i
    if i > maxval:
        i=maxval
    return i



def test():
    """
    very simple test
    """
    import tempfile
    tmpfile = tempfile.TemporaryFile(suffix='.sf')
    x=numpy.array([(1.0, 3),(4.5,2)], dtype=[('fcol','f4'),('icol','i4')])

    write(x, tmpfile)
    tmpfile.seek(0)
    read(tmpfile)

    tmpfile.close()





















