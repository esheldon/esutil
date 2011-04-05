import numpy
import sys,os
import time
from sys import stdout,stderr
from glob import glob

from distutils.core import setup,Extension

import distutils.sysconfig
import os

main_libdir=distutils.sysconfig.get_python_lib()
pylib_install_subdir = main_libdir.replace(distutils.sysconfig.PREFIX+os.sep,'')

if not os.path.exists('ups'):
    os.mkdir('ups')
tablefile=open('ups/esutil.table','w')
tab="""
setupOptional("python")
setupOptional("cjson")
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

    try:
        setup(name='esutil',
              configuration=configuration)
    finally:
        sys.stdout.write("failed")
    return



if __name__ == '__main__':
    setup_package()

