import numpy
import sys,os
import time
from sys import stdout,stderr
from glob import glob

import distutils.sysconfig
import os

with_ups=False
try:
    ind=sys.argv.index('with_ups')
    del sys.argv[ind]
    with_ups=True
except:
    pass


def create_ups():

    # numpy distutils does not put data files under prefix, only in the package
    # directory, but we need the ups file to go directly under prefix/ups look
    # for a --prefix option in the args

    # default to system prefix
    prefix=distutils.sysconfig.PREFIX
    
    for i in xrange(1, len(sys.argv) ):
        arg = sys.argv[i]
        if arg.find('--prefix') != -1:
            if arg.find('='):
                prefix=arg.split('=')[1]
            else:
                # next arg is prefix
                prefix = sys.argv[i+1]

    prefix=os.path.expandvars(prefix)
    prefix=os.path.expanduser(prefix)


    ups_dir = os.path.join(prefix, 'ups')

    if not os.path.exists(ups_dir):
        os.mkdir(ups_dir)
    ups_file = os.path.join(ups_dir, 'esutil.table')

    # create the ups file
    # generate the lib directory
    main_libdir=distutils.sysconfig.get_python_lib()
    pylib_install_subdir = main_libdir.replace(distutils.sysconfig.PREFIX+os.sep,'')
    pylib_install_subdir = pylib_install_subdir.replace('dist-packages','site-packages')

    sys.stdout.write("writing ups file: %s\n" % ups_file)
    tablefile=open(ups_file,'w')
    tab=\
"""setupOptional("cjson")
setupOptional("numpy")
envPrepend(PYTHONPATH,${PRODUCT_DIR}/%s)
""" % pylib_install_subdir 

    tablefile.write(tab)
    tablefile.close()

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('esutil')
    config.add_data_dir('esutil/include')
    return config

def setup_package():

    from numpy.distutils.core import setup
    setup(name='esutil', configuration=configuration)

    if 'install' in sys.argv and with_ups:
        create_ups()

if __name__ == '__main__':
    setup_package()

