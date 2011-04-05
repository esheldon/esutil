import numpy
import sys,os
import time
from sys import stdout,stderr
from glob import glob

from distutils.core import setup,Extension

import distutils.sysconfig
import os

def create_ups():
    from optparse import OptionParser
    parser=OptionParser(__doc__)
    parser.add_option("--prefix",default=distutils.sysconfig.PREFIX,
                      help="the prefix")

    options, args = parser.parse_args(sys.argv[1:])
    prefix=options.prefix
    prefix=os.path.expandvars(prefix)
    prefix=os.path.expanduser(prefix)

    main_libdir=distutils.sysconfig.get_python_lib()
    pylib_install_subdir = main_libdir.replace(distutils.sysconfig.PREFIX+os.sep,'')
    ups_dir = os.path.join(prefix, 'ups')

    if not os.path.exists(ups_dir):
        os.mkdir(ups_dir)
    ups_file = os.path.join(ups_dir, 'esutil.table')

    sys.stdout.write("writing ups file: %s\n" % ups_file)
    tablefile=open(ups_file,'w')
    tab="""setupOptional("cjson")
setupOptional("numpy")
envPrepend(PYTHONPATH,${PRODUCT_DIR}/%s)\n""" % pylib_install_subdir 

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

    create_ups()

if __name__ == '__main__':
    setup_package()

