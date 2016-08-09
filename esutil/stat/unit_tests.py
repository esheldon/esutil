from __future__ import print_function

import esutil
import numpy
from numpy import where

def test():
    data = numpy.array([0,3,1,12,7,5], dtype='i4')
    binsize=1.0
    res = esutil.stat.histogram(data, binsize=binsize, rev=True)

    print('Got result: ',res)

    res2 = esutil.stat.histogram(data, binsize=binsize, extern=False, rev=True)

    print('compare hist to python only: ')
    esutil.misc.colprint(res[0], res2[0])

    wbad, = where( numpy.abs( res[0]-res2[0] ) != 0)
    if wbad.size != 0:
        print('%s Errors found' % wbad.size)
    else:
        print('OK')


    print('\ncompare reverse indices to python only: ')
    esutil.misc.colprint(res[1], res2[1])

    wbad, = where( numpy.abs( res[0]-res2[0] ) != 0)
    if wbad.size != 0:
        print('%s Errors found' % wbad.size)
    else:
        print('OK')




if __name__=='__main__':
    test()
