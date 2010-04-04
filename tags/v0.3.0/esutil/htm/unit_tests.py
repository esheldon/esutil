import htm
from sys import stdout
import numpy

def test():
    errors = 0
    tests = 0

    depth = 10

    # see if we can create a class
    stdout.write('Creating class at depth %s....' % depth)
    h=htm.HTM(depth)
    stdout.write('OK\n')
    tests += 1


    # make sure the depth matches up
    stdout.write('Checking depth....')

    outd = h.depth()

    if outd != depth:
        stdout.write('Error\n')
        errors += 1
    else:
        stdout.write('OK\n')
    tests += 1


    # try looking up scalar ra/dec
    stdout.write('Looking up htm id of scalar ra/dec....')
    expected = 10518792
    htmid = h.lookup_id(200.0, 0.0)
    if htmid[0] != expected:
        stdout.write('Error\n')
        errors += 1
    else:
        stdout.write('OK\n')
    tests += 1

    # try looking up vector ra/dec
    stdout.write('Looking up htm id of vector ra/dec....')
    ra = numpy.array([200.0, 115.25])
    dec = numpy.array([0.0, 24.3])
    expected = numpy.array([10518792, 15368378])

    htmid = h.lookup_id(ra,dec)
    if htmid[0] != expected[0] or htmid[1] != expected[1]:
        stdout.write('Error\n')
        errors += 1
    else:
        stdout.write('OK\n')
    tests += 1


    # try the matching
    stdout.write('Matching by ra/dec, expect 10 matches ordered by distance....')
    two = 2.0/3600.
    # offset second list by fraction of 2 arcsec in dec
    # not last ones don'e match at all
    ra1 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 21.36])
    dec1 = numpy.array( [24.3,          24.3,            24.3,  -28.25, -15.32])
    ra2 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 55.25])
    dec2 = numpy.array( [24.3+0.75*two, 24.3 + 0.25*two, 24.3 - 0.33*two, -28.25 + 0.58*two, 75.22])

    m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,two,maxmatch=0)

    if m1.size != 10:
        stdout.write('Error\n')
        errors += 1
    else:
        stdout.write('OK\n')
    tests += 1

    for i in range(m1.size):
        stdout.write('%s %s %s\n' % (m1[i],m2[i],d12[i]))


    # try the matching with maxmatch=2
    stdout.write('Matching with maxmatch=2, expect 7 matches ordered by distance....')

    m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,two,maxmatch=2)

    if m1.size != 7:
        stdout.write('Error\n')
        errors += 1
    else:
        stdout.write('OK\n')
    tests += 1

    for i in range(m1.size):
        stdout.write('%s %s %s\n' % (m1[i],m2[i],d12[i]))


    # try the matching with maxmatch=1
    stdout.write('Matching with maxmatch=1, expect 4 matches ordered by distance....')

    m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,two,maxmatch=1)

    if m1.size != 4:
        stdout.write('Error\n')
        errors += 1
    else:
        stdout.write('OK\n')
    tests += 1

    for i in range(m1.size):
        stdout.write('%s %s %s\n' % (m1[i],m2[i],d12[i]))




    # try the matching
    stdout.write('Writing matched to file, expect 10 matches ordered by distance....')
    two = 2.0/3600.
    # offset second list by fraction of 2 arcsec in dec
    # not last ones don'e match at all
    ra1 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 21.36])
    dec1 = numpy.array( [24.3,          24.3,            24.3,  -28.25, -15.32])
    ra2 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 55.25])
    dec2 = numpy.array( [24.3+0.75*two, 24.3 + 0.25*two, 24.3 - 0.33*two, -28.25 + 0.58*two, 75.22])

    fname='/tmp/test.bin'
    m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,two,file=fname,maxmatch=0)

    res = h.read(fname)

    if res.size != 10:
        stdout.write('Error\n')
        errors += 1
    else:
        stdout.write('OK\n')
    tests += 1

    for i in range(res.size):
        stdout.write('%s %s %s\n' % (res['i1'][i],res['i2'][i],res['d12'][i]))


    # try counts in radial bins
    stdout.write("\nTesting bincount....\n\n")

    ten=10.0/3600.
    eighty=30.0/3600.0
    hundred=100.0/3600.0
    ra1 = numpy.array(  [200.0,         200.0,         200.0,           175.23,  21.36])
    dec1 = numpy.array( [24.3,           24.3,          24.3,           -28.25, -15.32])
    ra2 = numpy.array(  [200.0,         200.0,         200.0,           175.23,  55.25])
    dec2 = numpy.array( [24.3+two, 24.3 + ten, 24.3 - eighty, -28.25 + hundred,  75.22])

    # these must be in degrees unless scale is sent, in which case it is in
    # units of radians*scale
    rmin = 5/3600.
    rmax = 150/3600.
    nbin = 10

    lower,upper,counts = h.bincount(rmin,rmax,nbin,ra1,dec1,ra2,dec2,getbins=True)
    for i in range(nbin):
        stdout.write("%s %s %s %s\n" % (i,lower[i], upper[i], counts[i]))
    counts_truth = numpy.array([0, 0, 3, 0, 0, 3, 0, 0, 1, 0],dtype='i8')
    wbad, = numpy.where(counts != counts_truth)
    if wbad.size > 0:
        stdout.write('    ....Error\n')
        errors += 1
    else:
        stdout.write('    ....OK\n')
    tests += 1



    stdout.write('\n' + '-'*50 + '\n')
    stdout.write('Founds %s errors in %s tests\n' % (errors,tests))

if __name__ == '__main__':
    test()
