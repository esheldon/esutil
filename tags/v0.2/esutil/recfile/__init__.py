"""
Package:
    recfile
Classes
    Records - A class for reading from a file of fixed-length records into
        numerical python arrays. The file can be binary or ASCII.
    An new Records class is instantiated using the Open() method:
        For writing:
            import recfile
            r = recfile.Open(file/fileobj, mode='w', delim='')
        For reading:
            import recfile 
            r = recfile.Open(file/fileobj, delim="", dtype=None, nrows=-9999)
            # Arguments can all be given as keywords except the file

        Inputs:
            file/fileobj:  A string file name or an open file object.
            mode: The file mode.  Default is "r" but can be "u" or "w".
            delim: The delimiter used in the file.  Default is "" for 
                binary files, but can be any string such as ",", "\\t", etc.
            dtype:  A numpy dtype object.  REQUIRED FOR READING. For example:
                numpy.dtype([('field1', 'i4'),('field2', 'f8')])
                some_numpy_array.dtype
            nrows: The number of rows in the file.  REQUIRED FOR READING.

    Class Methods:
        Read(rows=, fields=):
            Returns the data in a NumPy array.  Specific rows and fields
            of the file can be specified with the keywords.  Rows must be
            sorted and unique.  Can be in any order.
        Write(numpy_array):
            Write the input numpy array to the file.  The array must have
            field names defined.

    Examples:
        import numpy
        import recfile

        # Read from a binary file
        file='test.bin'
        dtype=numpy.dtype([('field1','f8'),('field2','2i4'),('field3','i8')])
        nrows=10000000

        robj = recfile.Open(file, dtype=dtype, nrows=nrows)
        res=robj.Read()

        # Read from a CSV file of the same structure, and only read a subset 
        # of the data
        rows2get=[2335,122332,1550021]
        fields2get='field2'
        robj = recfile.Open('test.csv', delim=',', dtype=dtype, nrows=nrows)
        res = robj.Read(rows=rows2get, fields=fields2get)

        # Write a numpy array to a file
        r = recfile.Open('test2.csv', 'w', ',')
        r.Write(my_array)

Unit Tests:
    Make sure all unit tests pass.

        import recfile
        recfile.test()

    Should show no failures for reading or writing:
        Total number of write failures: 0
        Total number of read failures: 0

Modification history:
    Created: 2008-07-18, Erin Sheldon
"""

# This is a wrapper that returns an instance of the Records
# class.  

# this only works in 2.5
#from .records import Records
#from .Util import Open
#from .Util import test

from records import Records
import Util
from Util import Open
from Util import test

