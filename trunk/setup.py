import sys,os
import time
from sys import stdout,stderr
from glob import glob
import platform

from distutils.core import setup,Extension

import distutils.sysconfig
import os

main_libdir=distutils.sysconfig.get_python_lib()
pylib_install_subdir = main_libdir.replace(distutils.sysconfig.PREFIX+os.sep,'')
pylib_install_subdir = pylib_install_subdir.replace('dist-packages','site-packages')

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

if platform.system()=='Darwin':
    extra_compile_args=['-arch','i386','-arch','x86_64']
    extra_link_args=['-arch','i386','-arch','x86_64']
else:
    extra_compile_args=[]
    extra_link_args=[]


if have_numpy:
    # recfile
    recfile_sources = ['esutil/recfile/records.cpp',
                       'esutil/recfile/records_wrap.cpp']
    recfile_module = Extension('esutil.recfile._records', 
                               extra_compile_args=extra_compile_args, 
                               extra_link_args=extra_link_args,
                               sources=recfile_sources)
    ext_modules.append(recfile_module)
    packages.append('esutil.recfile')

    # cosmology package
    cosmo_sources = glob('esutil/cosmology/*.c')
    cosmo_module = Extension('esutil.cosmology._cosmolib', 
                             extra_compile_args=extra_compile_args, 
                             extra_link_args=extra_link_args,
                             sources=cosmo_sources)
    ext_modules.append(cosmo_module)
    packages.append('esutil.cosmology')



    # HTM
    include_dirs += ['esutil/htm','esutil/htm/htm_src']
    htm_sources = glob('esutil/htm/htm_src/*.cpp')
    htm_sources += ['esutil/htm/htmc.cc','esutil/htm/htmc_wrap.cc']
    htm_module = Extension('esutil.htm._htmc',
                           extra_compile_args=extra_compile_args, 
                           extra_link_args=extra_link_args,
                           sources=htm_sources)

    ext_modules.append(htm_module)
    packages.append('esutil.htm')




    # stat package
    #include_dirs += ['esutil/stat']
    chist_sources = ['chist.cc','chist_wrap.cc']
    chist_sources = ['esutil/stat/'+s for s in chist_sources]
    chist_module = Extension('esutil.stat._chist', 
                             extra_compile_args=extra_compile_args, 
                             extra_link_args=extra_link_args,
                             sources=chist_sources)
    ext_modules.append(chist_module)
    stat_util_sources = ['_stat_util.c']
    stat_util_sources = ['esutil/stat/'+s for s in stat_util_sources]
    stat_util_module = Extension('esutil.stat._stat_util', 
                                 extra_compile_args=extra_compile_args, 
                                 extra_link_args=extra_link_args,
                                 sources=stat_util_sources)
    ext_modules.append(stat_util_module)
    packages.append('esutil.stat')


    # integrate package
    #include_dirs += ['esutil/integrate']
    cgauleg_sources = glob('esutil/integrate/*.cc')
    cgauleg_module = Extension('esutil.integrate._cgauleg', 
                               extra_compile_args=extra_compile_args, 
                               extra_link_args=extra_link_args,
                               sources=cgauleg_sources)
    ext_modules.append(cgauleg_module)
    packages.append('esutil.integrate')



long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read()

classifiers = ["Development Status :: 5 - Production/Stable"
               ,"License :: OSI Approved :: GNU General Public License (GPL)"
               ,"Topic :: Scientific/Engineering :: Astronomy"
               ,"Intended Audience :: Science/Research"
              ]


# data_files copies the ups/esutil.table into prefix/ups
setup(name='esutil',
      version="0.5.2",
      classifiers=classifiers,
      description='Erin Sheldons Python Utilities',
      url='http://code.google.com/p/esutil/',
      packages=packages,
      data_files=[('ups',['ups/esutil.table'])],
      ext_modules=ext_modules,
      include_dirs=include_dirs)
