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
# The default version of this file will be overwritten on setup to include
# paths determined from the python version.  This is useful to have in place
# though so that dependencies can be checked *before* installation.  Currently
# there are no required dependencies, so this is somewhat moot.

setupOptional("python")
setupOptional("cjson")
envPrepend(PYTHONPATH,${PRODUCT_DIR}/%s)
""" % pylib_install_subdir 
tablefile.write(tab)
tablefile.close()



# can we build recfile?
packages = ['esutil']
ext_modules = []
try:
    import numpy
    include_dirs=[numpy.get_include()]
    include_dirs += ['esutil/include']
    have_numpy=True
except:
    have_numpy=False
    ext_modules=[]
    include_dirs=[]

    stdout.write('Numpy not found:  Not building C extensions\n')
    time.sleep(5)



if have_numpy:
    # recfile
    recfile_sources = ['esutil/recfile/records.cpp',
                       'esutil/recfile/records_wrap.cpp']
    recfile_module = Extension('esutil.recfile._records', 
                               sources=recfile_sources)
    ext_modules.append(recfile_module)
    packages.append('esutil.recfile')


    # HTM
    include_dirs += ['esutil/htm','esutil/htm/htm_src']
    htm_sources = glob('esutil/htm/htm_src/*.cpp')
    htm_sources += ['esutil/htm/htmc.cc','esutil/htm/htmc_wrap.cc']
    htm_module = Extension('esutil.htm._htmc',
                           sources=htm_sources)

    ext_modules.append(htm_module)
    packages.append('esutil.htm')


    # pyfits
    include_dirs += ['esutil/pyfitspatch']
    pyfits_sources = glob('esutil/pyfitspatch/*.c')
    pyfits_module = Extension("esutil.pyfitspatch.pyfitsComp",
                              sources=pyfits_sources)
    ext_modules.append(pyfits_module)
    packages.append('esutil.pyfitspatch')

    # pyfits 2.4.0
    include_dirs += ['esutil/pyfitspatch240']
    pyfits_sources = glob('esutil/pyfitspatch240/*.c')
    pyfits_module = Extension("esutil.pyfitspatch240.pyfitsComp",
                              sources=pyfits_sources)
    ext_modules.append(pyfits_module)
    packages.append('esutil.pyfitspatch240')




    # stat package
    include_dirs += ['esutil/stat']
    chist_sources = glob('esutil/stat/*.cc')
    chist_module = Extension('esutil.stat._chist', 
                             sources=chist_sources)
    ext_modules.append(chist_module)
    packages.append('esutil.stat')


    # integrate package
    include_dirs += ['esutil/integrate']
    cgauleg_sources = glob('esutil/integrate/*.cc')
    cgauleg_module = Extension('esutil.integrate._cgauleg', 
                               sources=cgauleg_sources)
    ext_modules.append(cgauleg_module)
    packages.append('esutil.integrate')





# data_files copies the ups/esutil.table into prefix/ups
setup(name='esutil',
      description='Erin Sheldons Python Utilities',
      url='http://code.google.com/p/esutil/',
      packages=packages,
      data_files=[('ups',['ups/esutil.table'])],
      ext_modules=ext_modules,
      include_dirs=include_dirs)
