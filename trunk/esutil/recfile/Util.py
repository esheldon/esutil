import numpy
# this only works in 2.5
#from . import records
import records
import sys
import os
import math


def Open(fileobj, mode="r", delim="", dtype=None, nrows=-9999):
    """
    Instantiate a new Records class
        For writing:
            import recfile
            r = recfile.Open(file/fileobj, mode="w", delim="")
        For reading:
            import recfile
            r = recfile.Open(file/fileobj, delim="", dtype=None, nrows=-9999)
            # Arguments can all be given as keywords except the file

        Inputs:
            file/fileobj:  A string file name or an open file object.
            mode: The file mode.  Default is "r" but can be "u" or "w".
            delim: The delimiter used in the file.  Use "" or None for
                binary files.  Default is "". Can also be any string such 
                as ",", "\\t", etc.
            dtype:  A numpy dtype object.  REQUIRED FOR READING. For example:
                [('field1', 'i4'),('field2', 'f8')]
                some_numpy_array.dtype
            nrows: The number of rows in the file.  REQUIRED FOR READING.

    Class Methods:
        Read(rows=, fields=):
            Returns the data in a NumPy array.  Specific rows and fields 
            of the file can be specified with the keywords.  Rows must be
            sorted and unique.  Fields must be unique but can be in any
            order.
        Write(numpy_array):
            Write the input numpy array to the file.  The array must have
            field names defined.

    Examples:
        import recfile

        # Read from a binary file
        file='test.bin'
        dtype=[('field1','f8'),('field2','2i4'),('field3','i8')]
        nrows=10000000

        robj = recfile.Open(file, dtype=dtype, nrows=nrows)
        res=robj.Read()

        # Read from a CSV file of the same structure, and only read a subset 
        # of the data
        rows2get=[2335,122332,1550021]
        fields2get='field2'
        robj = recfile.Open('test.csv', delim=",", dtype=dtype, nrows=nrows)
        res = robj.Read(rows=rows2get, fields=fields2get)


        # Write a numpy array to a file
        r = recfile.Open('test.tab', "w", "\\t")
        r.Write(my_array)

    Modification history:
        Created: 2008-07-18, Erin Sheldon

    """
    if delim is None:
        delim=""
    dt=numpy.dtype( dtype )
    
    return records.Records(fileobj, mode, delim, dtype=dt, nrows=nrows)



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

def WriteData(fname, delim, duplicate=1):
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


def DiffData(d1, d2):
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
                diff += DiffData(tmp1,tmp2)
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
            diff = DiffData(orig,read)
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
    WriteData(fname, delim)

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
            diff = DiffData(orig,read)
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
            diff = DiffData(orig,read)
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
            diff = DiffData(orig,read)
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
            diff = DiffData(orig,read)
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
        WriteData(fname, delim, duplicate=duplicate)

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





