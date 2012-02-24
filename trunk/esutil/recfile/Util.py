import numpy
import records
import sys
import os
import math
import pprint

# These are a bunch of docs pieces that can be pieced together

_instantiate_docs="""
Instantiate a new Recfile class
    For writing:
        import recfile
        r = recfile.Open(file/fileobj, 
                         mode="w", 
                         delim=None, 
                         padnull=False, 
                         ignorenull=False, 
                         bracket_arrays=False,
                         verbose=False)

        for updating use mode="r+"
    For reading:
        import recfile
        r = recfile.Open(file/fileobj, 
                         delim=None, 
                         dtype=None, 
                         nrows=-9999, 
                         offset=0, 
                         skiplines=0,
                         verbose=False)
"""
# docs for inputs to the varius "open" functions and methods.
_open_input_docs="""
Inputs:
    file/fileobj:  A string file name or an open file object.

Keywords:
    mode: The file mode.  Default is "r" but can also be "r+","w","w+".
    delim: The delimiter used in the file.  Use "" or None for
        binary files.  Default is None. Can also be any string such 
        as ",", "\\t", etc.

    dtype:  A numpy dtype object.  REQUIRED FOR READING. 
        For example:
            [('field1', 'i4'),('field2', 'f8')]
            array.dtype
        the type must contain fields.

    nrows: 
        The number of rows in the file.  If rows is not sent, it will
        be determined from the newlines for ascii or from file data and
        dtype for binary.

    offset:  Offset into the file where reading will begin.  Used
        if opening with "r" or "r+".  For "w" or "w+" offset is
        zero by definition.

    skiplines:
        Skip the specified number of lines (rows).  Only works for
        ascii where rows are separated by '\\n'

    padnull: When writing ascii, replace nulls in strings with spaces.
        Useful for programs that don't understand nulls like sqlite
        databases.
    ignorenull:  When writing ascii, just ignore nulls when writing
        strings. Note this will not result in fixed length data so
        you cannot read it back in using recfile. Useful for 
        programs that don't understand nulls, like sqlite databases.
    bracket_arrays: 
        If True and writing ascii, arrays are written thus: 
            {el1,el2,....}
        Currently the delimiter is forced to be a comma because the
        authors were implementing postgres input files.

    verbose:
        Print information about processing.  Default False.
"""
_useful_methods_docs="""
Useful Recfile Class Methods:

    While reading, you can access rows and columns using [ ] notation.

        r=recfile.Open(...)
        data = r[rows]              # rows can be single number,slice,list
        or
        data = r[field_names][rows] # field names must come first. You
                                    # must specify something for rows
                                    # in order to read the data, otherwise
                                    # a RecfileColumnSubset object is 
                                    # returned.

        If you put field_names *after* rows, then all rows will be read
        and *then* the fields are extracted, which is inefficient. 

        If only the fields are specified, the data are not read, rather a
        RecfileColumnSubset object is returned.  
        
        You must specify the rows to actually read any data.  If you want
        them all, use the [:] slice.

        data = r[:]                   # read all rows and columns
        data = r[start:stop]          # full slices allowed
        data = r[start:stop:step]     # slice with step specified
        data = r[field_names][:]      # read all rows but subset of columns.
        data = r[field_names][rowlist] # subset of rows and columns

    read(rows=, fields=):
        Returns the data in a NumPy array.  Specific rows and fields of the
        file can be specified with the keywords.  Fields must be unique but
        can be in any order.

    write(numpy_array):
        Write the input numpy array to the file.  The array must have
        field names defined.
"""

_examples_docs="""
Examples:
    import recfile

    # Read from a binary file.  Number of rows will be determined from
    # the file data and dtype if not entered.
    file='test.bin'
    dtype=[('field1','f8'),('field2','2i4'),('field3','i8')]

    robj = recfile.Open(file, dtype=dtype)

    # read all rows and columns
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
    data = robj.read(fields=column_list)    # columns/fields are synonyms

    # In bracket notation, you must specify rows to read the data.
    data = robj['field2'][:]
    data = robj[column_list][rowlist]
    data = robj['field3'].read()


    # Read from a CSV file of the same structure, and only read a subset 
    # of the data.  Nrows can be specified to speed up reading if known,
    # otherwise they will be counted, which is slow for text.

    rows2get=[2335,122332,1550021]
    fields2get='field2'
    robj = recfile.Open('test.csv', delim=",", dtype=dtype, nrows=nrows)
    data = robj.read(rows=rows2get, fields=fields2get)
    data = robj[fields2get][rows2get]


    # Write a numpy array to a file, with ability to
    # append
    r = recfile.Open('test.tab', "r+", ",")
    r.write(my_array)

    # append more rows
    r.write(second_array)

    r.close()
"""

_tests_docs="""
Unit Tests:
    Make sure all unit tests pass.

        import recfile
        recfile.test()

    Should show no failures for reading or writing:
        Total number of write failures: 0
        Total number of read failures: 0
"""

def indent(s, n=1):
    ind = '    '*n
    return '\n'.join([ind+sp for sp in s.split('\n')])

# the main docs for this module
__doc__="""
Package:
    recfile
Classes

    Recfile 
    
    A class for reading and writing structred numpy arrays to and from
    files.  Structured arrays are also known as recarrays or arrays with
    fields.  Both binary and ascii are supported.

    This class addresses limitations of memmap class, which cannot 
    read individual columns from a file without reading the whole file.

%s
%s
%s
%s
%s
""" % (_instantiate_docs,
       _open_input_docs,
       _useful_methods_docs,
       _examples_docs,
       _tests_docs)

def Open(fileobj, mode='r', **keys):
    # doc string generated dynamically below

    # make sure it's a dtype and not just a descr
    return Recfile(fileobj, mode=mode, **keys)

Open.__doc__="""
%s
%s
%s
%s
""" % (_instantiate_docs,
       _open_input_docs,
       _useful_methods_docs,
       _examples_docs)



class Recfile(object):
    __doc__=Open.__doc__

    def __init__(self, fobj, mode='r', **keys):
        # aliases
        self.Read = self.read
        self.Write = self.write

        self.open(fobj, mode=mode, **keys)

    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def open(self, fobj, mode='r', **keys):
        """
        Class:
            Recfile
        Method:
            open
        Calling Sequence:
            r = recfile.Recfile()
            r.open(...)

            For more info, see documentation of the recfile.Open function,
            which has identical syntax to this open() method.
        """

        dtype=keys.get('dtype',None)
        nrows=keys.get('nrows',-9999)

        self.verbose=keys.get('verbose',False)

        self.bracket_arrays=keys.get('bracket_arrays',False)

        self.close()
        self.padnull=keys.get('padnull',False)
        self.ignorenull=keys.get('ignorenull',False)
        self.delim = keys.get('delim',None)
        self.skiplines=keys.get('skiplines',None)
        self.offset=keys.get('offset',None)

        if self.skiplines is None:
            self.skiplines = 0

        if self.offset is None:
            if isinstance(fobj,file):
                self.offset = fobj.tell()
            else:
                self.offset = 0

        if self.delim is not None and self.delim != "":
            self.is_ascii = True
        else:
            self.is_ascii = False

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
                # mode to w+
                if self.verbose:
                    stdout.write("Requested append on non-existent file: "
                                 "Will create a new file\n")
                mode = 'w+'

            self.fobj = open(fpath, mode)

        elif isinstance(fobj, file):
            self.fobj = fobj
        else:
            raise ValueError("Only support filenames and file objects "
                             "as input")

        if self.fobj.mode[0] == 'r':
            if dtype is None:
                raise ValueError("You must enter dtype when reading")
            self.dtype = numpy.dtype(dtype)

            if nrows is None or nrows < 0:
                self.nrows = self.get_nrows()
            else:
                self.nrows=nrows

            # we only pay attention to the offset when mode is 'r' or 'r+'
            
            if self.is_ascii:
                # for ascii we can skip lines, e.g. for a header
                # this takes precedence over offset
                if self.skiplines > 0:
                    self.nrows -= self.skiplines
                    for i in xrange(self.skiplines):
                        tmp = self.fobj.readline()

                    # now, we override any existing offset to our
                    # position after skipping lines
                    self.offset = self.fobj.tell()


            if self.offset < 0:
                self.offset=0
            # go to the offset position in the file
            if self.fobj.tell() != self.offset:
                print 'moving'
                self.fobj.seek(offset)


        
    def get_nrows(self):
        if self.delim != "" and self.delim is not None:
            # for ascii this can be slow
            nrows = 0
            for line in self.fobj:
                nrows += 1
            self.fobj.seek(self.offset)
        else:
            # For binary, try to figure out the number of rows based on
            # the number of bytes

            rowsize=self.dtype.itemsize
            # go to end
            self.fobj.seek(0,2)
            datasize = self.fobj.tell() - self.offset
            nrows = datasize/rowsize
            self.fobj.seek(self.offset)

        return nrows


    def close(self):
        """
        Close any open file object.  Make sure fobj, _hdr, and delim are None
        """
        self.offset=0
        self.delim=None
        self.nrows=0
        self.dtype=None
        if hasattr(self, 'fobj'):
            if self.fobj is not None:
                if isinstance(self.fobj, file):
                    self.fobj.close()
        self.fobj=None

        self.padnull=False
        self.ignorenull=False


    def flush(self):
        if hasattr(self,'fobj'):
            if isinstance(self.fobj, file):
                self.fobj.flush()

    def __repr__(self):
        s = []

        if isinstance(self.fobj,file):
            s += ["filename: '%s'" % self.fobj.name]
            s += ["mode: '%s'" % self.fobj.mode]
        if self.delim is not None:
            s=["delim: '%s'" % self.delim]

        s += ["nrows: %s" % self.nrows]

        if self.dtype is not None:
            drepr=pprint.pformat(self.dtype.descr)
            drepr = '  '+drepr.replace('\n','\n  ')
            s += ["dtype: \n"+drepr]

        s = "\n".join(s)
        return s


    def read(self, rows=None, fields=None, columns=None,
             view=None, split=False, **keys):
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
                          view=None,
                          split=False)

            If no arguments are given, all data are read.

        Inputs:
            rows: A scalar, sequence or array indicating a subset
                of rows to read.
            fields or columns: A scalar, sequence, or array indicating
                a subset of field to read. fields and columns mean the
                same thing.
            view: Specify an alternative view of the data.
            split: Return a tuple of results rather than a rec array. Note
                the data are still stored in one big chunk, this is just
                an alternative access method.  E.g.

                # this might return a rec array with fields accessed
                # such as data['x'] data['y'] data['index']
                data = r.read()
                # this returns a tuple with an element for each
                x,y,index = r.read(split=True)
        """
        
        if self.fobj is None:
            raise ValueError("You have not yet opened a file")

        if self.fobj.tell() != self.offset:
            self.fobj.seek(self.offset)

        rows2read = self._get_rows2read(rows)
        fields2read = self._get_fields2read(fields, columns=columns)

        if fields2read is None and rows2read is None and self.delim is None:
            # Its binary and we are reading everything.  Use fromfile.
            result = numpy.fromfile(self.fobj,dtype=self.dtype,count=self.nrows)
        else:
            robj = records.Records(
                self.fobj, mode='r', 
                nrows=self.nrows, dtype=self.dtype, 
                delim=self.delim)
            result = robj.Read(rows=rows2read, fields=fields2read)

        if view is not None:
            result = result.view(view)

        if split:
            return split_fields(result)
        else:
            return result

    def write(self, data):
        """
        Class:
            Recfile
        Method:
            write
        Purpose:
            Write data to the opened file.  The dtype of the data must match
            for successive calls to write.
        Calling Sequence:
            r=recfile.Open(.....)
            r.write(array1)
            r.write(array2)
        """
        if self.fobj is None:
            raise ValueError("You have not yet opened a file")

        if self.fobj.mode[0] != 'w' and '+' not in self.fobj.mode:
            raise ValueError("You must open with 'w*' or 'r+' to write")

        self.fobj.seek(0,2) # Seek to end of file

        dataview = data.view(numpy.ndarray) 
        if self.verbose:
            stdout.write("Writing %s: %s\n" % \
                (dataview.size,pprint.pformat(dataview.dtype.descr)))

        if (self.delim is not None):
            # let recfile deal with ascii writing
            r = records.Records(self.fobj, mode='u', delim=self.delim, 
                                bracket_arrays=self.bracket_arrays)
            # make sure the data are in native format.  This greatly 
            # simplifies the C code
            to_native_inplace(dataview)
            r.Write(dataview, padnull=self.padnull, ignorenull=self.ignorenull)
        else:
            # Write data out as a binary chunk
            dataview.tofile(self.fobj)

        # update nrows to reflect the write
        self.nrows += dataview.size


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
        data = sf[rowlist][fieldlist]

        # Note, if you send just columns, a RecfileColumnSubset object is
        # returned
        sub = sf['fieldname']
        data = sub.read(rows=)
        """

        if self.fobj is None:
            raise ValueError("You have not yet opened a file")

        res, isrows, isslice = self.process_args_as_rows_or_columns(arg)
        if isrows:
            # rows were entered: read all columns
            if isslice:
                return self.read_slice(res)
            else:
                return self.read(rows=res)

        # columns was entered.  Return a subset objects
        return RecfileColumnSubset(self, columns=res)

    def read_slice(self, arg, split=False):

        if self.fobj is None:
            raise ValueError("You have not yet opened a file")

        if self.fobj.tell() != self.offset:
            self.fobj.seek(self.offset)

        robj = records.Records(
                self.fobj, mode='r', 
                nrows=self.nrows, dtype=self.dtype, 
                delim=self.delim)
        result = robj.ReadSlice(long(arg.start), long(arg.stop), long(arg.step))


        if split:
            return split_fields(result)
        else:
            return result

        return result

    def get_memmap(self, view=None, header=False):

        if self.delim is not None:
            raise ValueError("Cannot memory map ascii files")

        if self.fobj.tell() != self.offset:
            self.fobj.seek(self.offset)


        shape = (self.nrows,)

        result = numpy.memmap(self.fobj, dtype=self.dtype, shape=shape, 
                              mode=self.fobj.mode, offset=self.fobj.tell())
        if view is not None:
            result = result.view(view)

        return result




    def __len__(self):
        return self.nrows


    def get_subset(self, rows=None, fields=None, columns=None):
        """
        sub = rf.get_subset(rows=None, fields=None, columns=None)

        Get a RecfileSubset object with the specified rows/columns. See
        the docs for RecfileSubset for more info.
        """
        return RecfileSubset(self, rows=rows, fields=fields, columns=columns)

    def process_args_as_rows_or_columns(self, arg, unpack=False):
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
            isslice=True
            if unpack:
                result = self.slice2rows(arg.start, arg.stop, arg.step)
            else:
                result = self.process_slice(arg)
        else:
            # a single object was entered.  Probably should apply some more 
            # checking on this
            isrows=True

        return result, isrows, isslice

    def process_slice(self, arg):
        start = arg.start
        stop = arg.stop
        step = arg.step

        if step is None:
            step=1
        if start is None:
            start = 0
        if stop is None:
            stop = self.nrows

        if start < 0:
            start = self.nrows + start
            if start < 0:
                raise IndexError("Index out of bounds")

        if stop < 0:
            stop = self.nrows + start + 1

        if stop < start:
            # will return an empty struct
            stop = start

        return slice(start, stop, step)

    def __getitem__old(self, arg):
        """
        sf = Recfile(....)

        # read subsets of columns and/or rows from the file.  Note, lists
        # of columns or rows cannot be tuples, due to how the __getitem__
        # call works.

        # read subsets of columns
        data = sf['fieldname']
        data = sf[ ['field1','field2',...] ]    # can also be an array

        # read subsets of rows
        data = sf[ 35 ]
        data = sf[ 35:88 ]
        data = sf[ [3,234,5551,.. ] ]           # can also be an array

        # read subset of rows *and* columns.
        data = sf['fieldname', 3:58]
        data = sf[rowlist, fieldlist]
        data = sf[fieldlist,rowlist]

        If a single argument is entered, that is set to arg.
        If more than one is entered, arg is set to a tuple.  d'oh!,  Hard
            to parse
        """
        if not isinstance(arg,tuple):
            send_arg = (arg,)
        else:
            send_arg = arg

        rows, columns = self.process_args_as_rows_and_columns(send_arg)
        return self.read(rows=rows, columns=columns)


    def process_args_as_rows_and_columns(self, args):
        """

        args must be a tuple.  Only the first one or two args are used.

        We must be able to interpret the args as as either a column name or
        row number, or sequences thereof.  Numpy arrays and slices are also
        fine.

        Examples:

            Single arguments:
                ( 'field1', )
                ( 54, )
                ( ('f1','f2'), )
                ( [3,4,5,6], )
            Two arguments:
                ( 'field36', 27 )
                ( [33,44], ['ra','dec','flux'] )
                ( 'ra', slice(5,10) )

        Returns rows,columns but one will be None.  If both entries can be
        interpreted as rows, the last is used.  Similarly for fields.

        """

        columns=None
        rows=None

        for arg in args[0:2]:

            if isinstance(arg, (tuple,list,numpy.ndarray)):
                # a sequence was entered
                if isstring(arg[0]):
                    columns = arg
                else:
                    rows = arg
            elif isstring(arg):
                # a single string was entered
                columns = arg
            elif isinstance(arg, slice):
                rows = self.slice2rows(arg.start, arg.stop, arg.step)
            else:
                # a single object was entered.  Probably should apply some more 
                # checking on this
                rows=arg

        return rows, columns

    def slice2rows(self, start, stop, step=None):
        if start is None:
            start=0
        if stop is None:
            stop=self.nrows
        if step is None:
            step=1

        tstart = self._fix_range(start)
        tstop  = self._fix_range(stop)
        if tstart == 0 and tstop == self.nrows:
            # this is faster: if all fields are also requested, then a 
            # single fread will be done
            return None
        if stop < start:
            raise ValueError("start is greater than stop in slice")
        return numpy.arange(tstart, tstop, step, dtype='intp')

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

            # no copy is made if it is an intp numpy array
            rows2read = numpy.array(rows,ndmin=1,copy=False, dtype='intp')
            if rows2read.size == 1:
                rows2read[0] = self._fix_range(rows2read[0], isslice=False)
        except:
            # single object entered
            rows2read = self._fix_range(rows, isslice=False)
            rows2read = numpy.array([rows2read], dtype='intp')


        # should we do this sort, or assume sorted?
        rows2read.sort()
        rmin = rows2read[0]
        rmax = rows2read[-1]
        if rmin < 0 or rmax >= self.nrows:
            raise ValueError("Requested rows range from %s->%s: out of "
                             "range %s->%s" % (rmin,rmax,0,self.nrows-1))
        if self.verbose:
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
        self.columns = self.recfile._get_fields2read(columns)

        # alias
        self.__call__ = self.get_subset

    def read(self, view=None, split=False):
        """
        Read the data from disk and return as a numpy array
        """

        return self.recfile.read(rows=self.rows, columns=self.columns, 
                                 view=view, split=split)


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
        self.columns = self.recfile._get_fields2read(columns)


    def read(self, rows=None, view=None, split=False):
        """
        Read the data from disk and return as a numpy array
        """

        return self.recfile.read(rows=rows, columns=self.columns, 
                                 view=view, split=split)

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
            self.recfile.process_args_as_rows_or_columns(arg, unpack=True)
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



# Testing code

# This is the data that will be output and input
# The field names match the types
_data_dtype=numpy.dtype( [('id','i8'),
                    ('i1','i1'),
                    ('u1','u1'),
                    ('i2','i2'),
                    ('2S4','2S4'),
                    ('u2','u2'),
                    ('i4','i4'),
                    ('2x2u4','u4',(2,2)),
                    ('i8','i8'),
                    ('f4','f4'),
                    ('S5','S5'),
                    ('f8','f8')] )

data = numpy.array( [(0, -10, 55, 1020, ['test','ones'], 63121, 123413, [[1,2],[3,4]],     8322,            1.25, 'hello', 21.22341),
                     (1,   5, 15, 5532, ['blah','boot'],   125,    431, [[5,6],[7,8]],     3225,           -21.7, '12345', 1.e12),
                     (2,  21, 66, -552, ['----','symb'],    32,  -3221, [[9,10],[11,12]],-10000, 5.3234212431e-7, 'heyho', -2.832393880082e15)],
                   dtype=_data_dtype)


def TestFile(delim):
    ext=extension(delim)
    fname=os.path.join('/tmp', 'test-NumpyRecords'+ext)
    return fname

def IsIterable(obj):
    try:
        it = iter(obj)
        return True
    except:
        return False

def AddPrintElement(d, elements):
    if (type(d) != numpy.string_) and (IsIterable(d)):
        for delement in d:
            AddPrintElement(delement, elements)
    else:
        elements += ['%s' % (d,) ]

def WriteTestData(fname, delim, duplicate=1):
    """
    This is until we get writing into the NumpyRecords
    """

    f = open(fname, 'w')

    tdata = data.copy()
    index = 0
    for i in range(duplicate):

        for jj in range(tdata.size):
            tdata['id'][jj] = index
            index += 1

        if delim is None or delim == "":
            tdata.tofile(f)
        else:
            for row in tdata:
                elements=[]
                AddPrintElement(row, elements)
                output = delim.join(elements)
                f.write(output)
                f.write('\n')

    f.close()


def DiffTestData(d1, d2):
    if type(d1) == numpy.string_:
        if len(d1) != len(d2):
            diff=math.fabs(len(d1)-len(d2))
        else:
            diff=0.0
            for s1,s2 in zip(d1,d2):
                if s1 != s2:
                    diff += 1.0
    elif type(d1) == numpy.ndarray:
        if d1.size != d2.size:
            diff = math.fabs(d1.size-d2.size)
        else:
            diff=0.0
            for tmp1,tmp2 in zip(d1,d2):
                diff += DiffTestData(tmp1,tmp2)
            diff = diff/len(d1)
    else:
            diff = math.fabs( d1-d2)
            if (diff > 0.0):
                if (d1 > d2):
                    bottom=d1
                else:
                    bottom=d2
                diff = diff/bottom

    return math.fabs(diff)

def CompareWriteDelim(delim):

    sys.stdout.write("\nTesting Write delimiter '"+delim+"'\n")
    sys.stdout.write('-'*79 + '\n')

    fname=TestFile(delim)

    # this is the output
    maxdiffs = {}

    # write the data
    r=records.Records(fname, "w", delim)
    r.Write(data)

    # Now read it back in and compare
    r = records.Records(fname, "r", delim, data.dtype, data.size)
    res = r.Read()

    # Set up some formatting stuff
    fwidth=20
    nhead = 4
    dashes = '-'*(fwidth*nhead + (nhead-1))+'\n'

    head_format = '%'+repr(fwidth)+'s'
    format=' '.join([head_format]*nhead) + '\n'


    ii=0
    diffs = numpy.zeros(data.size*len(data.dtype.names))
    for name in data.dtype.names:
        sys.stdout.write('Comparing Field: %s\n' % name)
        sys.stdout.write(format % ("row","orig","read","%diff"))
        sys.stdout.write(dashes)
        for i in range(data.size):
            orig = data[name][i]
            read = res[name][i]
            diff = DiffTestData(orig,read)
            diffs[ii] = diff
            ii += 1
            oprint = str(orig).replace('\n','')
            rprint = str(read).replace('\n','')
            sys.stdout.write(format % ( i,oprint,rprint,diff))


    maxdiffs['all'] = diffs.max()
    maxdiffs['delim'] = delim
    maxdiffs['type'] = 'write'
    PrintSummary(maxdiffs)
    return maxdiffs

def CompareReadDelim(delim):
    """
    Test reading a file with the given input delimiter
    The following tests are run:
        reading the whole file
        reading each row,col separately
        reading entire rows
        reading entire columns

    A dictionary is returned with the maximum differences
    between the input data to the file (a numpy array 
    defined at the top of this file) and the data read
    from the file.  The differences should be consistent
    with the precision with which the data was written
    to the file.  For delim="" the difference should be
    zero.
    """

    sys.stdout.write("\nTesting Read delimiter '"+delim+"'\n")
    sys.stdout.write('-'*79 +'\n')

    sys.stdout.write("Writing data\n")
    fname=TestFile(delim)
    WriteTestData(fname, delim)

    # Set up some formatting stuff
    fwidth=20
    nhead = 4
    dashes = '-'*(fwidth*nhead + (nhead-1)) +'\n'

    head_format = '%'+repr(fwidth)+'s'
    format=' '.join([head_format]*nhead) +'\n'

    rows=list( range(data.size) )

    # this is the output
    maxdiffs = {}

    sys.stdout.write('\n**** Reading all data ****\n')
    sys.stdout.write(dashes)

    diffs = numpy.zeros(data.size*len(data.dtype.names))

    r = records.Records(fname, "r", delim, data.dtype, data.size)
    res = r.Read()
    ii=0
    for name in res.dtype.names:
        sys.stdout.write('Comparing Field: %s\n' % name)
        sys.stdout.write(format % ("row","orig","read","%diff"))
        sys.stdout.write(dashes)
        for i in range(len(rows)):
            orig = data[name][i]
            read = res[name][rows[i]]
            diff = DiffTestData(orig,read)
            diffs[ii] = diff
            ii += 1
            oprint = str(orig).replace('\n','')
            rprint = str(read).replace('\n','')
            sys.stdout.write(format % ( rows[i],oprint,rprint,diff))


    maxdiffs['all'] = diffs.max()
    sys.stdout.write(dashes)
    sys.stdout.write('max diff reading all at once: %s\n' % diffs.max())




    # Now read each row/field separately
    sys.stdout.write('\n**** Reading each row,column separately ****\n')
    sys.stdout.write(dashes)

    diffs[:] = 0
    ii=0

    for name in data.dtype.names:
        sys.stdout.write('Comparing Field: %s\n' % name)
        sys.stdout.write(format % ("row","orig","read","%diff"))
        sys.stdout.write(dashes)
        for row in range(data.size):
            r = records.Records(fname, "r", delim, data.dtype, data.size)
            res = r.Read(rows=row, fields=name)
            r.Close()

            orig = data[name][row]
            read = res[name][0]
            diff = DiffTestData(orig,read)
            diffs[ii] = diff
            ii += 1
            oprint = str(orig).replace('\n','')
            rprint = str(read).replace('\n','')
            sys.stdout.write(format % ( row,oprint,rprint,diff))

    maxdiffs['rowcol'] = diffs.max()
    sys.stdout.write(dashes)
    sys.stdout.write('max diff reading each row,col separately: %s\n' %
                     diffs.max())




    sys.stdout.write('\n**** Reading one row at at time ****\n')
    sys.stdout.write(dashes)


    diffs[:] = 0
    ii=0

    for row in range(data.size):
        r = records.Records(fname, "r", delim, data.dtype, data.size)
        res = r.Read(rows=row)
        sys.stdout.write('Comparing Field: %s\n' % name)
        sys.stdout.write(format % ("row","orig","read","%diff"))
        sys.stdout.write(dashes)
        for name in res.dtype.names:

            orig = data[name][row]
            read = res[name][0]
            diff = DiffTestData(orig,read)
            diffs[ii] = diff
            ii += 1
            oprint = str(orig).replace('\n','')
            rprint = str(read).replace('\n','')
            sys.stdout.write(format % (name,oprint,rprint,diff))

    maxdiffs['row'] = diffs.max()
    sys.stdout.write(dashes)
    sys.stdout.write('max diff reading one row at a time: %s\n' % diffs.max())


    sys.stdout.write('\n**** Reading one field at at time ****\n')
    sys.stdout.write(dashes)


    diffs[:] = 0
    ii=0

    for name in data.dtype.names:
        sys.stdout.write('Comparing Field: %s\n' % name)
        sys.stdout.write(format % ("row","orig","read","%diff"))
        sys.stdout.write(dashes)
        r = records.Records(fname, "r", delim, data.dtype, data.size)
        res = r.Read(fields=name)
        for row in range(data.size):
            orig = data[name][row]
            read = res[name][row]
            diff = DiffTestData(orig,read)
            diffs[ii] = diff
            ii += 1
            oprint = str(orig).replace('\n','')
            rprint = str(read).replace('\n','')
            sys.stdout.write(format % (row,oprint,rprint,diff))

    maxdiffs['col'] = diffs.max()
    sys.stdout.write(dashes)
    sys.stdout.write('max diff reading one field at a time: %s\n' % diffs.max())

    maxdiffs['delim'] = delim
    maxdiffs['type'] = 'read'
    PrintSummary(maxdiffs)

    return maxdiffs






max_allowed_diff = 1.e-6
def PrintSummary(maxdiffs):
    sys.stdout.write('\n')
    sys.stdout.write("Summary of %s tests for delim='%s'\n" % \
                     (maxdiffs["type"],maxdiffs["delim"]))
    f='%15s: %20s\n'
    sys.stdout.write(f % ('type','max %diff'))
    sys.stdout.write('-'*(15+20+2)+'\n')
    nbad=0
    for n in ['all','rowcol','row','col']:
        if n in maxdiffs:
            sys.stdout.write(f % (n, maxdiffs[n]))
            if maxdiffs[n] > max_allowed_diff:
                nbad += 1
    sys.stdout.write('Number of failures: %s\n' % nbad)
    return nbad

def PrintDifflistSummary(difflist):
    nbad=0
    for d in difflist:
        nbad += PrintSummary(d)
    return nbad



def extension(delim):
    if delim == '':
        ext='.bin'
    elif delim == ',':
        ext= '.csv'
    elif delim == '\t':
        ext = '.tab'
    elif delim == ':':
        ext = '.colon'
    elif delim == ' ':
        ext = '.dat'
    else:
        raise ValueError('unsupported delim: '+delim)

    return ext

def timing(delim, duplicate):
    """
    Write the data to a file many times to get a nice big file and then
    test reading it
    """
    ext=extension(delim)
    fname = os.path.expanduser( '~/tmp/testbig'+ext )

    if not os.path.exists(fname):
        sys.stdout.write("Writing data to '"+delim+"' delimited file %s,  "+\
                         "%s times" % (fname, duplicate))
        WriteTestData(fname, delim, duplicate=duplicate)

    nrows = data.size*duplicate
    rows2read = [int(nrows*0.25), int(nrows*0.75), nrows-10]
    cols2read = ['id','u1','2S4','i4','2x2u4','f8']
    sys.stdout.write('Reading rows: %s from file %s\n' % (rows2read,fname))
    sys.stdout.write('Columns: %s\n' % cols2read)
    r = records.Records(fname, "r", delim, data.dtype, nrows)

    tmp = r.Read(rows=rows2read, fields=cols2read )

    sys.stdout.write('%s\n' % tmp)


def test():
    """
    Unit tests for the records package.

    import recfile
    recfile.test()

    This should return no failures:
        Total number of write failures: 0
        Total number of read failures: 0
    """

    difflist_read = []
    difflist_write = []

    # first test reading
    for delim in ["", ",", ":","\t", " "]:
        maxdiffs = CompareReadDelim(delim)
        difflist_read.append(maxdiffs)

    sys.stdout.write('\n\nOverall Read Test Summary:\n')
    sys.stdout.write("""
        Diffs are % except when arrays or strings were 
        found to have different number of elements
        """)
    sys.stdout.write('-'*79+'\n')

    nbad_read = PrintDifflistSummary(difflist_read)
    sys.stdout.write('\n')
    sys.stdout.write('-'*79 +'\n')
    sys.stdout.write('Total number of read failures: %s\n' % nbad_read)

    if nbad_read > 0:
        sys.stdout.write('There were errors in the reading tests; not doing'+\
                         ' write tests\n')
        sys.exit(0)


    # now writing and rading
    for delim in ["", ",", ":","\t", " "]:
        maxdiffs = CompareWriteDelim(delim)
        difflist_write.append(maxdiffs)

    sys.stdout.write('\n\nOverall Read/Write Test Summary:\n')
    sys.stdout.write("""
        Diffs are % except when arrays or strings were 
        found to have different number of elements
        """)
    sys.stdout.write('-'*79 +'\n')

    nbad_read = PrintDifflistSummary(difflist_read)
    nbad_write = PrintDifflistSummary(difflist_write)
    sys.stdout.write('\n')
    sys.stdout.write('-'*79 +'\n')
    sys.stdout.write('Total number of write failures: %s\n' % nbad_write)
    sys.stdout.write('Total number of read failures: %s\n' % nbad_read)




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
        data_little = is_little_endian(array)
    else:
        # assume all are same byte order: we only need to find one with
        # little endian
        for fname in array.dtype.names:
            if is_little_endian(array[fname]):
                data_little=True
                break

    if ( (machine_little and not data_little) 
            or (not machine_little and data_little) ):

        outdata = array.byteswap(True)
        outdata.dtype = outdata.dtype.newbyteorder()


def is_little_endian(array):
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

    byteorder = array.dtype.base.byteorder
    return (byteorder == '<') or (machine_little and byteorder == '=')



