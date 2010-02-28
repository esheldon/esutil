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






    stdout.write('\n' + '-'*50 + '\n')
    stdout.write('Founds %s errors in %s tests\n' % (errors,tests))
