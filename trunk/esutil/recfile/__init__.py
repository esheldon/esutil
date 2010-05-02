"""
Package:
    recfile
Classes

    Recfile 
    
    A class for reading and writing structred numpy arrays to and from
    files.  Structured arrays are also known as recarrays or arrays with
    fields.  

    This class addresses limitations of memmap class, which cannot 
    read individual columns from a file without reading the whole file.

    Instantiate a new Recfile class
        For writing:
            import recfile
            r = recfile.Open(file/fileobj, mode="r", delim=None, 
                             padnull=False, ignorenull=False, verbose=False)
        For reading:
            import recfile
            r = recfile.Open(file/fileobj, delim=None, dtype=None, 
                             nrows=-9999, offset=0, verbose=False)
            # Arguments can all be given as keywords except the file

        Inputs:
            file/fileobj:  A string file name or an open file object.

        Optional Inputs:
            mode: The file mode.  Default is "r" but can also be "r+","w","w+".

            delim: The delimiter used in the file.  Use "" or None for
                binary files.  Default is None. Can also be any string such 
                as ",", "\\t", etc.

            dtype:  A numpy dtype object.  REQUIRED FOR READING. 
                For example:
                    [('field1', 'i4'),('field2', 'f8')]
                    array.dtype
                the type must contain fields.

            nrows: The number of rows in the file.  REQUIRED FOR READING
                ASCII. If reading binary and nrows is not sent, it will be
                determined from the file data and dtype.

            offset:  Offset into the file where reading will begin.  Used
                if opening with "r" or "r+".  For "w" or "w+" offset is
                zero by definition.

            padnull: When writing ascii, replace nulls in strings with spaces.
                Useful for programs that don't understand nulls like sqlite
                databases.
            ignorenull:  When writing ascii, just ignore nulls when writing
                strings. Note this will not result in fixed length data so
                you cannot read it back in using recfile. Useful for 
                programs that don't understand nulls, like sqlite databases.

    Useful Recfile Class Methods:

        While reading, you can access rows and columns using [ ] notation.

            r=recfile.Open(...)
            data = r[rows]              # rows can be single number,slice,list
            or
            data = r[field_names][rows] # field names must come first. You
                                        # must specify something for rows
                                        # in order to read the data, otherwise
                                        # a sub-columns object is returned.

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
            Returns the data in a NumPy array.  Specific rows and fields 
            of the file can be specified with the keywords.  Rows must be
            sorted and unique.  Fields must be unique but can be in any
            order.
        write(numpy_array):
            Write the input numpy array to the file.  The array must have
            field names defined.

    Examples:
        import recfile

        # Read from a binary file.  Number of rows will be determined from
        # the file data and dtype if not entered.
        file='test.bin'
        dtype=[('field1','f8'),('field2','2i4'),('field3','i8')]
        nrows=10000000

        robj = recfile.Open(file, dtype=dtype)

        # read all rows and columns
        data = robj.read()
        data = robj[:]

        # read a subset of rows
        data = robj.read(rows=row_list)
        data = robj[ 3500:5238 ]

        # get every 3rd in a slice
        data = robj[ 10:1234:3 ]

        # send a full list
        data = robj[ rowlist ]

        # read a subset of columns.
        data = robj.read(columns=column_list)
        data = robj.read(fields=column_list)    # columns/fields are synonyms

        # In bracket notation, you must specify rows to read the data, or use
        # .read()
        data = robj['fieldname'][:]
        data = robj[ fieldlist ][ rowlist ]
        data = robj['fieldname'].read()


        # Read from a CSV file of the same structure, and only read a subset 
        # of the data.  Nrows must be specified.
        rows2get=[2335,122332,1550021]
        fields2get='field2'
        robj = recfile.Open('test.csv', delim=",", dtype=dtype, nrows=nrows)
        data = robj.read(rows=rows2get, fields=fields2get)
        data = robj[fields2get][rows2get]


        # Write a numpy array to a file, with ability to
        # append
        r = recfile.Open('test.tab', "r+", ",")
        r.Write(my_array)

        # append more rows
        r.Write(second_array)

    Todo:
        implement auto-getting of size for ascii

 


    Records - A low level C++ class for reading from a file of fixed-length
        records into numerical python arrays. The file can be binary or ASCII.
        You should use the higher level Recfile class, either instantiated
        directly or through Open.

    
Unit Tests:
    Make sure all unit tests pass.

        import recfile
        recfile.test()

    Should show no failures for reading or writing:
        Total number of write failures: 0
        Total number of read failures: 0

Modification history:
    Created: 2008-07-18, Erin Sheldon
    Wrapper class Recfile added.  This is not returned by Open.  Support
        for [ ] style reading notation. 2009-11-20, ESS, BNL
    Added simple slicing for ASCII.  2010-02-18, Erin Sheldon, BNL

"""

# This is a wrapper that returns an instance of the Records
# class.  

# this only works in 2.5
#from .records import Records
#from .Util import Open
#from .Util import test

from records import Records
import Util
from Util import Recfile
from Util import Open
from Util import test

