"""
TODO
    - fix docs
    - bools, complex (should work for binary)
"""
from __future__ import print_function
import numpy
import sys
import os
import math
import pprint

from . import records

try:
    xrange=xrange
except:
    xrange=range

def write(filename, data, mode='w', **keys):
    """
    write data into a records file

    For more information about keywords, see docs for recfile.Recfile
    """

    if mode not in ['w','w+','r+']:
        raise ValueError("to write, mode must be one of w, w+, r+")
    with Recfile(filename, mode=mode, **keys) as robj:
        robj.write(data)

def read(filename, dtype, **keys):
    """
    read data from a recfile

    For more information about keywords, see docs for recfile.Recfile
    """

    with Recfile(filename, dtype=dtype, mode='r', **keys) as robj:
        data=robj.read(**keys)

    return data

def Open(filename, mode='r', **keys):
    """
    just instantiates a Recfile object

    it is generally better to use a with context
    """
    # doc string generated dynamically below

    # make sure it's a dtype and not just a descr
    return Recfile(filename, mode=mode, **keys)

class Recfile(object):
    """
    Class to read and write to files with fixed lenght records

    parameters
    -----------
    filename: string
        path to the file

    mode: The file mode.  Default is "r" but can also be "r+","w","w+".
    delim: The delimiter used in the file.  Use None for
        binary files.  Default is None. Can also be any string such
        as ",", "\\t", etc.

    dtype:  numpy dtype object, optional
        REQUIRED FOR READING.
        For example:
            [('field1', 'i4'),('field2', 'f8')]
            array.dtype
        the type must contain fields.

    nrows: integer, optional
        The number of rows in the file.  If rows is not sent, it will
        be determined from the newlines for ascii or from file data and
        dtype for binary.

    offset:  integer, optional
        Offset into the file where reading will begin.  Used if opening with
        "r" or "r+".  For "w" or "w+" offset is zero by definition.

    padnull: bool, optional
        When writing ascii, replace nulls in strings with spaces.  Useful for
        programs that don't understand nulls like sqlite databases.

    ignorenull: bool, optional
        When writing ascii, just ignore nulls when writing strings. Note this
        will not result in fixed length data so you cannot generally read it
        back in using recfile. Useful for programs that don't understand nulls,
        like sqlite databases.

    bracket_arrays: bool, optional

        If True and writing ascii, arrays are written thus: 
            {el1,el2,....}
        Currently the delimiter is forced to be a comma because the authors
        were implementing postgres input files.

    examples
    --------
    Instantiate a new Recfile class
        For writing:
            from recfile import Recfile
            with Recfile(fname, mode="w") as robj:
                robj.write(data1)
                robj.write(data2) # append

            for updating use mode="r+"

            # you can also use a convenience function
            import recfile
            recfile.write(fname, data)

        For reading

            fname='test.bin'
            dtype=[('field1','f8'),('field2','2i4'),('field3','i8')]
            with Recfile(fname, mode="r", dtype=dtype) as robj:

                data = robj[:]
                data = robj.read()

                # read a subset of rows using slice notation
                data = robj[3500:5238]
                data = robj[ 10:1234:3 ]

                # specifying rows explicitly
                row_list = [35,88,217]
                data = robj[row_list]
                data = robj.read(rows=row_list)


                # read a subset of columns.
                column_list = ['field2','field3']
                data = robj.read(columns=column_list)

                # In bracket notation, you must specify rows to read the data.
                data = robj['field2'][:]
                data = robj[column_list][rowlist]
                data = robj['field3'].read()


            # Read from a CSV file of the same structure, and only read a subset
            # of the data.  Nrows can be specified to speed up reading if known,
            # otherwise they will be counted, which is slow for text.

            fname='test.csv'
            delim=','
            with Recfile(fname, mode="r", dtype=dtype, delim=delim) as robj:
                # etc

    """
    def __init__(self, filename, mode='r', **keys):
        self.open(filename, mode=mode, **keys)

    def open(self, filename, mode='r', **keys):
        """
        see docs for the Recfile class
        """

        self.close()

        self.mode=mode

        dtype=keys.get('dtype',None)
        nrows=keys.get('nrows',-9999)

        self.bracket_arrays=keys.get('bracket_arrays',False)

        self.padnull=keys.get('padnull',False)
        self.ignorenull=keys.get('ignorenull',False)

        self.delim = keys.get('delim',None)
        self.skiplines=keys.get('skiplines',None)
        self.offset=keys.get('offset',None)

        if self.skiplines is not None:
            raise RuntimeError("skiplines is no longer supported")

        if self.offset in [None,""]:
            self.offset=0
        else:
            self.offset=int(self.offset)
            if self.offset < 0:
                self.offset=0

        if self.delim == "":
            self.delim=None

        if self.delim is not None:
            self.is_ascii = True
        else:
            self.is_ascii = False

        # expand shortcut variables
        filename = os.path.expanduser(filename)
        self.filename = os.path.expandvars(filename)

        if self.mode not in ['r','r+','w','w+']:
            raise ValueError("bad mode: '%s'" % self.mode)

        if self.mode == 'r+' and not os.path.exists(filename):
            raise RuntimeError("opened with 'r+' but file does not exist")

        if self.mode[0] == 'r':
            if dtype is None:
                raise ValueError("You must enter dtype when reading")

            self.dtype = numpy.dtype(dtype)
            if self.is_ascii:
                # we don't care about byte order for ascii
                nbo = remove_dtype_byteorder(self.dtype)
                self.dtype = numpy.dtype(nbo)

            self.colnames = numpy.array(self.dtype.names)
            self.ncols = self.colnames.size

            if nrows is None or nrows < 0:
                self.nrows = self._count_nrows()
            else:
                self.nrows=int(nrows)

            self.robj = records.Records(
                self.filename,
                mode=self.mode,
                delim=self.delim,
                dtype=self.dtype,
                nrows=self.nrows,
                offset=self.offset,
                padnull=self.padnull,
                ignorenull=self.ignorenull,
            )
        else:
            self.robj = records.Records(
                filename,
                mode=self.mode,
                delim=self.delim,
                bracket_arrays=self.bracket_arrays,
                padnull=self.padnull,
                ignorenull=self.ignorenull,
            )

    def close(self):
        """
        Close any open file object.  Make sure various things are None
        """
        self.offset=0
        self.delim=None
        self.nrows=0
        self.dtype=None
        if hasattr(self, 'robj'):
            if self.robj is not None:
                self.robj.close()
        self.robj=None

        self.padnull=False
        self.ignorenull=False



    def _count_nrows(self):
        """
        get the number of rows in the file
        """
        with open(self.filename) as fobj:
            if self.offset > 0:
                fobj.seek(self.offset)

            if self.delim is not None:
                # for ascii this can be slow
                nrows = 0
                for line in fobj:
                    nrows += 1
            else:
                # For binary, try to figure out the number of rows based on
                # the number of bytes

                rowsize=self.dtype.itemsize
                # go to end
                fobj.seek(0,2)
                datasize = fobj.tell() - self.offset
                nrows = datasize//rowsize

        return nrows


    def __repr__(self):
        s = []

        s += ["filename: '%s'" % self.filename]
        s += ["mode: '%s'" % self.mode]
        if self.delim is not None:
            s += ['filetype: TEXT']
            s=["delim: '%s'" % self.delim]
        else:
            s += ['filetype: BINARY']

        s += ["nrows: %s" % self.nrows]

        if self.dtype is not None:
            drepr=pprint.pformat(self.dtype.descr)
            drepr = '  '+drepr.replace('\n','\n  ')
            s += ["dtype: \n"+drepr]

        s = "\n".join(s)
        return s


    def get_colnum(self, colname):
        """
        get the column number for the input column name
        """

        if not numpy.isscalar(colname):
            raise ValueError("column name should be a string, "
                             "got %s" % str(colname))

        w,=numpy.where(self.colnames == colname)
        if w.size == 0:
            raise ValueError("column '%s' not found" % colname)
        return w[0]

    def get_colnums(self, colnames):
        """
        get the column number for the input column name
        """

        colnames = numpy.array(colnames, ndmin=1, copy=False)

        colnums = numpy.zeros(colnames.size, dtype='i8')

        for i in xrange(colnames.size):
            colnums[i] = self.get_colnum(colnames[i])

        return numpy.unique( colnums )

    def read(self, rows=None, fields=None, columns=None,
             split=False, **keys):
        """
        Class:
            Recfile
        Method:
            read
        Purpose:
            read records from the opened file.
        Syntax:
            r=recfile.Open(...)
            data = r.read(rows=None,
                          fields=None, columns=None,
                          split=False)

            If no arguments are given, all data are read.

        Inputs:
            rows: A scalar, sequence or array indicating a subset
                of rows to read.
            fields or columns: A scalar, sequence, or array indicating
                a subset of field to read. fields and columns mean the
                same thing.
            split: Return a tuple of results rather than a rec array. Note
                the data are still stored in one big chunk, this is just
                an alternative access method.  E.g.

                # this might return a rec array with fields accessed
                # such as data['x'] data['y'] data['index']
                data = r.read()
                # this returns a tuple with an element for each
                x,y,index = r.read(split=True)
        """

        if self.robj is None:
            raise ValueError("You have not yet opened a file")

        rows = self._get_rows2read(rows)
        colnums, isscalar = self._get_colnums_to_read(fields, columns=columns)

        read_all_rows = (rows is None) or (rows.size == self.nrows)
        read_all_cols = (colnums is None) or (colnums.size==self.ncols)

        if self.is_ascii:
            # we always use the same code for ascii
            result = self._read_columns(colnums, rows)
        else:
            # we have specialized codes for binary
            if read_all_cols and read_all_rows:
                result = self._read_binary_slice(slice(0,self.nrows,1))

            #elif read_all_cols:
            #    # read some row subset
            #    result = self.robj.read_binary_rows(rows)

            else:
                result = self._read_columns(colnums,rows)

        if isscalar:
            result = result[columns]
        elif split:
            result=split_fields(result)

        return result

    Read=read

    def _read_columns(self, colnums, rows):
        """
        read a set of columns from the file, possibly a subset of the rows

        parameters
        ----------
        colnums: string
            string column numbes
        rows: array, optional
            Subset of rows to read
        """

        if self.robj is None:
            raise ValueError("You have not yet opened a file")

        if rows is not None:
            nrows = rows.size
        else:
            nrows = self.nrows

        if colnums is not None:
            dtype=[]
            for colnum in colnums:
                dtype.append( self.dtype.descr[colnum] )
        else:
            dtype=self.dtype

        data=numpy.zeros(nrows, dtype=dtype)

        self.robj.read_columns(data,colnums,rows)

        return data



    def write(self, data):
        """
        Write data to the file.

        The dtype of the data must match for successive calls to write.

        parameters
        -----------
        data: array
            array with fields
        """
        if self.robj is None:
            raise ValueError("You have not yet opened a file")


        dataview = data.view(numpy.ndarray)

        if self.is_ascii:
            # for ascii, make sure the data are in native format.  This greatly
            # simplifies the C code
            to_native_inplace(dataview)

        self.robj.Write(dataview)

        # update nrows to reflect the write
        self.nrows += dataview.size

    Write=write

    def __getitem__(self, arg):
        """
        sf = Recfile(....)

        # read subsets of columns and/or rows from the file.  Rows and
        # columns can be lists/tuples/arrays

        # read subsets of rows
        data = sf[:]
        data = sf[ 35 ]
        data = sf[ 35:88 ]
        data = sf[ [3,234,5551,.. ] ]

        # read subsets of columns
        data = sf['fieldname'][:]
        data = sf[ ['field1','field2',...] ][row_list]


        # read subset of rows *and* columns.
        data = sf['fieldname'][3:58]
        data = sf[fieldlist][rowlist]

        # Note, if you send just columns, a RecfileColumnSubset object is
        # returned
        sub = sf['fieldname']
        data = sub.read(rows=)
        """

        if self.robj is None:
            raise ValueError("You have not yet opened a file")

        if self.is_ascii:
            unpack=True
        else:
            unpack=False

        res, isrows, isslice = self._process_args_as_rows_or_columns(arg, unpack=unpack)
        if isrows:
            # rows were entered: read all columns
            if isslice:
                return self._read_binary_slice(res)
            else:
                rows=res
                return self.read(rows=rows)
        else:
            # columns was entered.  Return a subset objects
            return RecfileColumnSubset(self, columns=res)

    def _get_slice_nrows(self, arg):
        """
        we have already done error checking on the slice
        """

        rowdiff = arg.stop-arg.start
        extra = 0
        if (rowdiff % arg.step) != 0:
            extra = 1

        nrows = rowdiff//arg.step + extra

        return nrows

    def _read_binary_slice(self, arg, split=False):
        """
        read a slice of rows

        we have already done error checking on the slice
        """
        if self.robj is None:
            raise ValueError("You have not yet opened a file")

        nrows = self._get_slice_nrows(arg)

        data = numpy.zeros(nrows, dtype=self.dtype)

        self.robj.read_binary_slice(
            data,
            int(arg.start),
            int(arg.stop),
            int(arg.step),
        )

        if split:
            return split_fields(data)
        else:
            return data


    def get_memmap(self, **keys):
        """
        no longer supported
        """
        raise RuntimeError("memmap is no longer supported")


    def get_subset(self, rows=None, fields=None, columns=None):
        """
        sub = rf.get_subset(rows=None, fields=None, columns=None)

        Get a RecfileSubset object with the specified rows/columns. See
        the docs for RecfileSubset for more info.
        """
        return RecfileSubset(self, rows=rows, fields=fields, columns=columns)

    def _process_args_as_rows_or_columns(self, arg, unpack=False):
        """

        args must be a tuple.  Only the first one or two args are used.

        We must be able to interpret the args as as either a column name or
        row number, or sequences thereof.  Numpy arrays and slices are also
        fine.

        Examples:
            'field'
            35
            [35,55,86]
            ['f1',f2',...]
        Can also be tuples or arrays.

        """

        isslice = False
        isrows = False
        result=arg
        if isinstance(arg, (tuple,list,numpy.ndarray)):
            # a sequence was entered
            if isstring(arg[0]):
                pass
            else:
                isrows=True
                result = arg
        elif isstring(arg):
            # a single string was entered
            pass
        elif isinstance(arg, slice):
            isrows=True
            if unpack:
                isslice=False
                result = self._slice2rows(arg.start, arg.stop, arg.step)
            else:
                isslice=True
                result = self._process_slice(arg)
        else:
            # a single object was entered.  Probably should apply some more 
            # checking on this
            isrows=True

        return result, isrows, isslice

    def _process_slice(self, arg):
        start = arg.start
        stop = arg.stop
        step = arg.step

        if step is None:
            step=1
        if start is None:
            start = 0
        if stop is None:
            stop = self.nrows
        elif stop > self.nrows:
            stop=self.nrows

        if start < 0:
            start = self.nrows + start
            if start < 0:
                raise IndexError("Index out of bounds")

        if stop < 0:
            stop = self.nrows + stop

        if stop < start:
            # will return an empty struct
            stop = start

        return slice(start, stop, step)


    def _slice2rows(self, start, stop, step=None):
        if start is None:
            start=0
        if stop is None:
            stop=self.nrows
        if step is None:
            step=1

        tstart = self._fix_range(start)
        tstop  = self._fix_range(stop)
        #if tstart == 0 and tstop == self.nrows:
        #    # this is faster: if all fields are also requested, then a 
        #    # single fread will be done
        #    return None
        if tstop < tstart:
            raise ValueError("start is greater than stop in slice")
        return numpy.arange(tstart, tstop, step, dtype='i8')

    def _fix_range(self, num, isslice=True):
        """
        If el=True, then don't treat as a slice element
        """

        if isslice:
            # include the end
            if num < 0:
                num=self.nrows + (1+num)
            elif num > self.nrows:
                num=self.nrows
        else:
            # single element
            if num < 0:
                num=self.nrows + num
            elif num > (self.nrows-1):
                num=self.nrows-1

        return num



    def _get_rows2read(self, rows):
        if rows is None:
            return None
        try:
            # a sequence entered
            rowlen=len(rows)

            rows2read = numpy.array(rows, ndmin=1, copy=False, dtype='i8')
            if rows2read.size == 1:
                rows2read[0] = self._fix_range(rows2read[0], isslice=False)
        except:
            # single object entered
            rows2read = self._fix_range(rows, isslice=False)
            rows2read = numpy.array([rows2read], dtype='i8')


        # should we do this sort, or assume sorted?

        rows2read = numpy.unique(rows2read)

        rmin = rows2read[0]
        rmax = rows2read[-1]
        if rmin < 0 or rmax >= self.nrows:
            raise ValueError("Requested rows range from %s->%s: out of "
                             "range %s->%s" % (rmin,rmax,0,self.nrows-1))

        return rows2read

    def _get_colnums_to_read(self, fields, columns=None):
        if fields is None:
            fields=columns

        if fields is None:
            # return None, False
            colnums = numpy.arange(self.colnames.size)
            return colnums, False

        is_scalar=numpy.isscalar(fields)
        if is_scalar:
            fields=[fields]

        if isinstance(fields, (list,numpy.ndarray)):
            f2read=fields
        elif isinstance(fields,tuple):
            f2read=list(fields)
        elif isstring(fields):
            f2read=fields
        else:
            raise ValueError('fields must be list,tuple,string or array')

        colnums = self.get_colnums(f2read)

        return colnums, is_scalar

    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
    def __len__(self):
        return self.nrows



class RecfileSubset(object):
    """
    A class representing a subset of the data on disk.  Useful for chaining
    together selections. e.g.

        sf = recfile.Open(fname, dtype=dtype)

        sub = sf.get_subset(rows=rowlist, columns=colliset)

        sub1 = sf.get_subset(rows=rows)
        sub2 = sub1.get_subset(columns=columns)
        or simpley call the object
        sub2 = sub1(columns=columns)

        data = sub2.read()

        data = sf.get_subset(rows=rows)(columns=cols).read()

    Useful because subsets can be passed around to functions.
    """
    def __init__(self, rf, fields=None, columns=None, rows=None):
        """
        Input is the SFile instance and a list of column names.
        """

        if columns is None:
            columns=fields

        self.recfile = rf
        self.rows=self.recfile._get_rows2read(rows)
        self.columns = columns

        # alias
        self.__call__ = self.get_subset

    def read(self, split=False):
        """
        Read the data from disk and return as a numpy array
        """

        return self.recfile.read(rows=self.rows, columns=self.columns, 
                                 split=split)


    def get_subset(self, fields=None, columns=None, rows=None):
        """
        Specify subsets of the data.  Returns a new RecfileSubset object.
        """
        if columns is not None:
            self.columns=columns
        elif fields is not None:
            self.columns=fields
        if rows is not None:
            self.rows=rows

        return self.recfile.get_subset(columns=self.columns, rows=self.rows)

    def __repr__(self):
        s=[]
        if self.columns is not None:
            c=pprint.pformat(self.columns)
            c="  "+c.replace('\n','\n  ')
            s += ["column subset:\n" + c]
        if self.rows is not None:
            r=pprint.pformat(self.rows)
            r="  "+r.replace('\n','\n  ')
            s += ["row subset:\n" + r]
        if len(s) > 0:
            s += ['\nfrom file:']
        s+=[self.recfile.__repr__()]
        s = "\n".join(s)
        return s

class RecfileColumnSubset(object):
    """

    A class representing a subset of the the columns on disk.  When called
    with .read() or [ rows ]  the data are read from disk.

    Useful because subsets can be passed around to functions, or chained
    with a row selection.

    This class is returned when using [ ] notation to specify fields in the
    recfile class

        sf = recfile.Open(fname, dtype=dtype)
        colsub = sf[field_list]

    returns aa RecfileColumnSubset object.  To read rows:

        data = colsub[row_list]
        data = colsub.read(rows=row_list)
    to read all, use .read() with no args or [:]
    """

    def __init__(self, rf, fields=None, columns=None):
        """
        Input is the SFile instance and a list of column names.
        """

        if columns is None:
            columns=fields

        self.recfile = rf
        self.columns = columns


    def read(self, rows=None, split=False):
        """
        Read the data from disk and return as a numpy array
        """

        return self.recfile.read(rows=rows, columns=self.columns,
                                 split=split)

    def __getitem__(self, arg):
        """
        If columns are sent, then the columns will just get reset and
        we'll return a new object

        If rows are sent, they are read and the result returned.
        """

        # we have to unpack the rows if we are reading a subset
        # of the columns because our slice operator only works
        # on whole rows.  We could allow rows= keyword to
        # be a slice...

        res, isrows, isslice = \
            self.recfile._process_args_as_rows_or_columns(arg, unpack=True)
        if isrows:
            # rows was entered: read all current column subset
            return self.read(rows=res)

        # columns was entered.  Return a subset objects
        return RecfileColumnSubset(self, columns=res)



    def __repr__(self):
        s=[]
        if self.columns is not None:
            c=pprint.pformat(self.columns)
            c="  "+c.replace('\n','\n  ')
            s += ["column subset:\n" + c]
        if len(s) > 0:
            s += ['\nfrom file:']
        s+=[self.recfile.__repr__()]
        s = "\n".join(s)
        return s





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


_major_pyvers = int( sys.version_info[0] )
def isstring(obj):
    if _major_pyvers >= 3:
        string_types=(str, numpy.string_)
    else:
        string_types=(str, unicode, numpy.string_)

    if isinstance(obj, string_types):
        return True
    else:
        return False

def remove_dtype_byteorder(dtype):

    newdt=[]
    for dt in dtype.descr:

        typestr = dt[1][1:]
        if len(dt) == 3:
            dt = (dt[0], typestr, dt[2])
        else:
            dt = (dt[0], typestr)

        newdt.append(dt)

    return newdt

def to_native_inplace(array):
    """
    Convert to native byte ordering in place
    """

    if numpy.little_endian:
        machine_little=True
    else:
        machine_little=False

    data_little=False
    if array.dtype.names is None:
        data_little = is_little_endian(array.dtype)
    else:
        # assume all are same byte order: we only need to find one with
        # little endian
        for fname in array.dtype.names:
            if is_little_endian(array[fname].dtype):
                data_little=True
                break

    if ( (machine_little and not data_little)
            or (not machine_little and data_little) ):

        outdata = array.byteswap(True)
        outdata.dtype = outdata.dtype.newbyteorder()


def is_little_endian(dtype):
    """
    Return True if array is little endian. Note strings are neither big
    or little endian.  The input must be a simple numpy array, not
    an array with fields.

    REVISION HISTORY:
        Created 2009, Erin Sheldon, NYU.
    """

    if numpy.little_endian:
        machine_little=True
    else:
        machine_little=False

    byteorder = dtype.base.byteorder
    return (byteorder == '<') or (machine_little and byteorder == '=')



