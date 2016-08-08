from __future__ import with_statement, print_function
import sys, os
import tempfile
import warnings
import numpy
from numpy import arange, array

from . import Util
from .Util import Recfile

import unittest

if sys.version_info > (3,0,0):
    stype=(str,bytes)
else:
    stype=str

try:
    xrange=xrange
except:
    xrange=range

def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReadWrite)
    unittest.TextTestRunner(verbosity=2).run(suite)

class TestReadWrite(unittest.TestCase):
    def setUp(self):

        nvec = 2
        ashape=(21,21)
        Sdtype = 'S6'
        # all currently available types, scalar, 1-d and 2-d array columns
        dtype=[
            ('u1scalar','u1'),
            ('i1scalar','i1'),
            ##('b1scalar','?'),
            ('u2scalar','u2'),
            ('i2scalar','i2'),
            ('u4scalar','u4'),
            ('i4scalar','i4'),
            ('i8scalar','i8'),
            ('f4scalar','f4'),
            ('f8scalar','f8'),
            ##('c8scalar','c8'), # complex, two 32-bit
            ##('c16scalar','c16'), # complex, two 64-bit

            ('u1vec','u1',nvec),
            ('i1vec','i1',nvec),
            ##('b1vec','?',nvec),
            ('u2vec','u2',nvec),
            ('i2vec','i2',nvec),
            ('u4vec','u4',nvec),
            ('i4vec','i4',nvec),
            ('i8vec','i8',nvec),
            ('f4vec','f4',nvec),
            ('f8vec','f8',nvec),
            ##('c8vec','c8',nvec),
            ##('c16vec','c16',nvec),

            ('u1arr','u1',ashape),
            ('i1arr','i1',ashape),
            ##('b1arr','?',ashape),
            ('u2arr','u2',ashape),
            ('i2arr','i2',ashape),
            ('u4arr','u4',ashape),
            ('i4arr','i4',ashape),
            ('i8arr','i8',ashape),
            ('f4arr','f4',ashape),
            ('f8arr','f8',ashape),
            ##('c8arr','c8',ashape),
            ##('c16arr','c16',ashape),

            ('Sscalar',Sdtype),
            ('Svec',   Sdtype, nvec),
            ('Sarr',   Sdtype, ashape),
        ]


        nrows=4
        data=numpy.zeros(nrows, dtype=dtype)
        names=data.dtype.names


        dtypes=['u1','i1','u2','i2','u4','i4','i8','f4','f8','c8','c16']
        for t in dtypes:

            sname=t+'scalar'
            vname=t+'vec'
            aname=t+'arr'

            if t in ['c8','c16']:
                if sname in names:
                    data[sname] = [complex(i+1,(i+1)*2) for i in xrange(nrows)]

                if vname in names:
                    for row in xrange(nrows):
                        for i in xrange(nvec):
                            index=(row+1)*(i+1)
                            data[vname][row,i] = complex(index,index*2)

                if aname in names:
                    for row in xrange(nrows):
                        for i in xrange(ashape[0]):
                            for j in xrange(ashape[1]):
                                index=(row+1)*(i+1)*(j+1)
                                data[aname][row,i,j] = complex(index,index*2)

            else:
                if sname in names:
                    data[sname] = 1 + numpy.arange(nrows, dtype=t)

                if vname in names:
                    data[vname] = 1 + numpy.arange(nrows*nvec,dtype=t).reshape(nrows,nvec)

                if aname in names:
                    arr = 1 + numpy.arange(nrows*ashape[0]*ashape[1],dtype=t)
                    data[aname] = arr.reshape(nrows,ashape[0],ashape[1])

        for t in ['b1']:
            sname=t+'scalar'
            vname=t+'vec'
            aname=t+'arr'

            if sname in names:
                data[sname] = (numpy.arange(nrows) % 2 == 0).astype('?')

            if vname in names:
                data[vname] = (numpy.arange(nrows*nvec) % 2 == 0).astype('?').reshape(nrows,nvec)

            if aname in names:
                arr = (numpy.arange(nrows*ashape[0]*ashape[1]) % 2 == 0).astype('?')
                data[aname] = arr.reshape(nrows,ashape[0],ashape[1])


        if 'Sscalar' in data.dtype.names:
            data['Sscalar'] = ['hello','world','good','bye']
        if 'Svec' in data.dtype.names:
            data['Svec'][:,0] = 'hello'
            data['Svec'][:,1] = 'world'

        if 'Sarr' in data.dtype.names:
            s = 1 + numpy.arange(nrows*ashape[0]*ashape[1])
            s = ['%s' % el for el in s]
            data['Sarr'] = numpy.array(s).reshape(nrows,ashape[0],ashape[1])

        self.data = data

        swap_dtype=[]
        for d in dtype:
            name=d[0]
            if 'S' not in name:
                tp = '>'+d[1]
            else:
                tp = d[1]

            if len(d) == 3:
                nd = (d[0], tp, d[2])
            else:
                nd = (d[0], tp)

            swap_dtype.append(nd)

        swap_data=numpy.zeros(nrows, dtype=swap_dtype)

        for n in swap_data.dtype.names:
            swap_data[n] = data[n]
        self.swap_data = data


    def _get_prefix(self, name, delim):
        if delim is not None:
            delimstr=delim.replace(' ','-')
            delimstr=delim.replace('\t','tab')
        else:
            delimstr='binary'

        prefix='recfile-%s-%s-' % (name,delimstr)
        return prefix

    def _get_testfile(self, name, delim):
        prefix=self._get_prefix(name, delim)
        fname=tempfile.mktemp(prefix=prefix,suffix='.rec')
        return fname

    def testWriteRead(self):
        """
        Test a basic table write, data and a header, then reading back in to
        check the values
        """

        dtype=self.data.dtype
        nrows=self.data.size
        for delim in [None,",", ":","\t", " "]:
            for doswap in [False,True]:

                if doswap:
                    data = self.swap_data
                else:
                    data = self.data

                try:
                    fname=self._get_testfile("testWriteRead",delim)
                    with Recfile(fname,mode='w',delim=delim) as robj:

                        try:
                            robj.write(data)
                            write_success=True
                        except:
                            write_success=False

                        self.assertTrue(write_success,"testing write does not raise an error doswap: %s" % doswap)
                        if not write_success:
                            skipTest("cannot test result if write failed")

                    with Recfile(fname, mode='r', dtype=dtype, delim=delim) as robj:
                        d = robj.read()

                    self.compare_rec(data, d, "table read/write")

                    # see if our convenience functions are working
                    Util.write(fname, data)
                    d = Util.read(fname, dtype)
                    self.compare_rec(data, d, "test read/write with convenience functions doswap: %s" % doswap)

                finally:
                    if os.path.exists(fname):
                        os.remove(fname)


    def testSubsets(self):
        """
        Test reading subsets and slices
        """

        dtype=self.data.dtype
        nrows=self.data.size
        for delim in [None,",", ":","\t", " "]:
            for doswap in [False,True]:
                sstr = " doswap: %s" % doswap

                try:
                    fname=self._get_testfile("testSubsets",delim)
                    with Recfile(fname,mode='w',delim=delim) as robj:
                        # initial write
                        robj.write(self.data)

                    with Recfile(fname, mode='r', dtype=dtype, delim=delim) as robj:

                        # row slices
                        d=robj[:]
                        self.compare_rec(self.data, d, "row range all"+sstr)

                        d=robj[1:3]
                        self.compare_rec(self.data[1:3], d, "row range"+sstr)

                        d=robj[0:4:2]
                        self.compare_rec(self.data[0:4:2], d, "row range step 2"+sstr)

                        # test reading single columns
                        for f in self.data.dtype.names:
                            d = robj[f][:]


                            d = robj.read(columns=f)
                            self.compare_array(self.data[f], d, "test read all rows %s column subset scalar name %s" % (f,sstr))

                            d = robj.read(columns=[f])
                            self.compare_array(self.data[f], d[f], "test read all rows %s column subset %s" % (f,sstr))

                            d = robj[f][:]
                            self.compare_array(self.data[f], d, "test read all rows %s column subset slice %s" % (f,sstr))

                            rows = [1,3]
                            d = robj.read(columns=f, rows=rows)
                            self.compare_array(self.data[f][rows], d, "test read rows %s column subset scalar name %s" % (f,sstr))

                            d = robj.read(columns=[f], rows=rows)
                            self.compare_array(self.data[f][rows], d[f], "test read rows %s column subset %s" % (f,sstr))

                        cols=['u2scalar','f4vec','Sarr']

                        # column subset
                        d = robj.read(columns=cols)
                        for f in d.dtype.names:
                            self.compare_array(self.data[f], d[f], "test column list %s %s" % (f,sstr))

                        # column subset and rows subset
                        rows = [1,3]
                        d = robj[cols][rows]
                        for f in d.dtype.names:
                            self.compare_array(self.data[f][rows], d[f], "test column list %s row subset slice %s" % (f,sstr))

                        d = robj.read(rows=rows, columns=cols)
                        for f in d.dtype.names:
                            self.compare_array(self.data[f][rows], d[f], "test column list %s row subset %s" % (f,sstr))

                        # combined with row slices

                        d = robj[cols][:]
                        for f in d.dtype.names:
                            self.compare_array(self.data[f], d[f], "test column list %s slice %s" % (f,sstr))

                        d = robj[cols][1:3]
                        for f in d.dtype.names:
                            self.compare_array(self.data[f][1:3], d[f], "test column list %s row slice %s" % (f,sstr))

                        d = robj[cols][0:4:2]
                        for f in d.dtype.names:
                            self.compare_array(self.data[f][0:4:2], d[f], "test column list %s row slice step 2 %s" % (f,sstr))

                finally:
                    if os.path.exists(fname):
                        os.remove(fname)


    def testAppend(self):
        """
        Test creating a table and appending new rows.
        """

        dtype=self.data.dtype
        nrows=self.data.size
        for delim in [None,",", ":","\t", " "]:

            try:
                fname=self._get_testfile("testSubsets",delim)
                with Recfile(fname,mode='w',delim=delim) as robj:
                    # initial write
                    robj.write(self.data)

                    # appending
                    data2 = self.data.copy()
                    data2['f4scalar'] = 3
                    robj.write(data2)

                with Recfile(fname, mode='r', dtype=dtype, delim=delim) as robj:

                    d = robj.read()
                    self.assertEqual(d.size, self.data.size*2)

                    self.compare_rec(self.data, d[0:self.data.size], "Comparing initial write")
                    self.compare_rec(data2, d[self.data.size:], "Comparing appended data")

            finally:
                if os.path.exists(fname):
                    os.remove(fname)


    def compare_names(self, read_names, true_names, lower=False, upper=False):
        for nread,ntrue in zip(read_names,true_names):
            if lower:
                tname = ntrue.lower()
                mess="lower: '%s' vs '%s'" % (nread,tname)
            else:
                tname = ntrue.upper()
                mess="upper: '%s' vs '%s'" % (nread,tname)
            self.assertEqual(nread, tname, mess)


    def compare_headerlist_header(self, header_list, header):
        """
        The first is a list of dicts, second a FITSHDR
        """
        for entry in header_list:
            name=entry['name'].upper()
            value=entry['value']
            hvalue = header[name]
            if isinstance(hvalue,str):
                hvalue = hvalue.strip()
            self.assertEqual(value,hvalue,"testing header key '%s'" % name)

            if 'comment' in entry:
                self.assertEqual(entry['comment'].strip(),
                                 header.get_comment(name).strip(),
                                 "testing comment for header key '%s'" % name)

    def compare_array_tol(self, arr1, arr2, tol, name):
        self.assertEqual(arr1.shape, arr2.shape,
                         "testing arrays '%s' shapes are equal: "
                         "input %s, read: %s" % (name, arr1.shape, arr2.shape))

        adiff = numpy.abs( (arr1-arr2)/arr1 )
        maxdiff = adiff.max()
        res=numpy.where(adiff  > tol)
        for i,w in enumerate(res):
            self.assertEqual(w.size,0,
                             "testing array '%s' dim %d are "
                             "equal within tolerance %e, found "
                             "max diff %e" % (name,i,tol,maxdiff))


    def compare_array(self, arr1, arr2, name):
        #print("\narr1:",arr1)
        #print("arr2:",arr2)
        self.assertEqual(arr1.shape, arr2.shape,
                         "testing arrays '%s' shapes are equal: "
                         "input %s, read: %s" % (name, arr1.shape, arr2.shape))

        res=numpy.where(arr1 != arr2)
        for i,w in enumerate(res):
            self.assertEqual(w.size,0,"testing array '%s' dim %d are equal" % (name,i))

    def compare_rec(self, rec1, rec2, name):
        for f in rec1.dtype.names:
            self.assertEqual(rec1[f].shape, rec2[f].shape,
                             "testing '%s' field '%s' shapes are equal: "
                             "input %s, read: %s" % (name, f,rec1[f].shape, rec2[f].shape))

            res=numpy.where(rec1[f] != rec2[f])
            for w in res:
                self.assertEqual(w.size,0,"testing column %s" % f)

    def compare_rec_subrows(self, rec1, rec2, rows, name):
        for f in rec1.dtype.names:
            self.assertEqual(rec1[f][rows].shape, rec2[f].shape,
                             "testing '%s' field '%s' shapes are equal: "
                             "input %s, read: %s" % (name, f,rec1[f].shape, rec2[f].shape))

            res=numpy.where(rec1[f][rows] != rec2[f])
            for w in res:
                self.assertEqual(w.size,0,"testing column %s" % f)

            #self.assertEqual(2,3,"on purpose error")

    def compare_rec_with_var(self, rec1, rec2, name, rows=None):
        """

        First one *must* be the one with object arrays

        Second can have fixed length

        both should be same number of rows

        """

        if rows is None:
            rows = arange(rec2.size)
            self.assertEqual(rec1.size,rec2.size,
                             "testing '%s' same number of rows" % name)

        # rec2 may have fewer fields
        for f in rec2.dtype.names:

            # f1 will have the objects
            if fitsio.fitslib.is_object(rec1[f]):
                self.compare_object_array(rec1[f], rec2[f], 
                                          "testing '%s' field '%s'" % (name,f),
                                          rows=rows)
            else:                    
                self.compare_array(rec1[f][rows], rec2[f], 
                                   "testing '%s' num field '%s' equal" % (name,f))

    def compare_object_array(self, arr1, arr2, name, rows=None): 
        """
        The first must be object
        """
        if rows is None:
            rows = arange(arr1.size)

        for i,row in enumerate(rows):
            if isinstance(arr2[i],stype):
                self.assertEqual(arr1[row],arr2[i],
                                "%s str el %d equal" % (name,i))
            else:
                delement = arr2[i]
                orig = arr1[row]
                s=len(orig)
                self.compare_array(orig, delement[0:s], 
                                   "%s num el %d equal" % (name,i))

    def compare_rec_with_var_subrows(self, rec1, rec2, name, rows):
        """

        Second one must be the one with object arrays

        """
        for f in rec1.dtype.names:
            if fitsio.fitslib.is_object(rec2[f]):

                for i in xrange(rec2.size):
                    if isinstance(rec2[f][i],stype):
                        self.assertEqual(rec1[f][i],rec2[f][i],
                                        "testing '%s' str field '%s' el %d equal" % (name,f,i))
                    else:
                        delement = rec1[f][i]
                        orig = rec2[f][i]
                        s=orig.size
                        self.compare_array(orig, delement[0:s], 
                                           "testing '%s' num field '%s' el %d equal" % (name,f,i))
            else:                    
                self.compare_array(rec1[f], rec2[f], 
                                   "testing '%s' num field '%s' equal" % (name,f))





if __name__ == '__main__':
    test()

