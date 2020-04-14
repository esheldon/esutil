from . import sfile_tests
from . import htm_tests
from . import hist_tests
from . import int_tests
from . import pickle_tests

def test():
    sfile_tests.test()
    htm_tests.test()
    hist_tests.test()
    int_tests.test()
    pickle_tests.test()
