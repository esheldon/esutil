import sys,os
import time
from sys import stdout,stderr
from glob import glob
import platform
import tempfile
import subprocess
import shutil

from distutils.core import setup,Extension
from distutils.command.build_ext import build_ext

import distutils.sysconfig
import os

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

extra_compile_args=[]
extra_link_args=[]

#
# Figure out if we need to add any extra flags:
#
local_tmp = 'tmp'

def try_compile(cpp_code, compiler, cflags=[], lflags=[]):
    """Check if compiling some code with the given compiler and flags works properly.
    """
    # Put the temporary files in a local tmp directory, so that they stick around after failures.
    if not os.path.exists(local_tmp): os.makedirs(local_tmp)

    # We delete these manually if successful.  Otherwise, we leave them in the tmp directory
    # so the user can troubleshoot the problem if they were expecting it to work.
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cpp', dir=local_tmp) as cpp_file:
        cpp_file.write(cpp_code.encode())
        cpp_name = cpp_file.name

    # Just get a named temporary file to write to:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.os', dir=local_tmp) as o_file:
        o_name = o_file.name

    # Another named temporary file for the executable
    with tempfile.NamedTemporaryFile(delete=False, suffix='.exe', dir=local_tmp) as exe_file:
        exe_name = exe_file.name

    # Try compiling with the given flags
    cc = [compiler.compiler_so[0]]
    cmd = cc + compiler.compiler_so[1:] + cflags + ['-c',cpp_name,'-o',o_name]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = p.stdout.readlines()
        p.communicate()
        if p.returncode != 0:
            # Print the error if there was one, to help the user diagnose the problem.
            print('Trying compile command:')
            print(' '.join(cmd))
            print('Output was:')
            print('   ',b'   '.join(lines).decode())
        returncode = p.returncode
    except (IOError,OSError) as e:
        print('Trying compile command:')
        print(cmd)
        print('Caught error: ',repr(e))
        returncode = 1
    if returncode != 0:
        # Don't delete files in case helpful for troubleshooting.
        return False

    # Link
    cc = compiler.linker_so[0]
    cmd = [cc] + compiler.linker_so[1:] + lflags + [o_name,'-o',exe_name]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = p.stdout.readlines()
        p.communicate()
        if p.returncode != 0:
            print('Trying link command:')
            print(' '.join(cmd))
            print('Output was:')
            print('   ',b'   '.join(lines).decode())
        returncode = p.returncode
    except (IOError,OSError) as e:
        print('Trying link command:')
        print(' '.join(cmd))
        print('Caught error: ',repr(e))
        returncode = 1

    return returncode == 0

def check_flags(compiler):
    """Check if we need to adjust the standard cflags for specific systems
    """
    # Start with a canonical set of flags to use
    cflags = extra_compile_args
    lflags = extra_link_args

    if platform.system()=='Darwin':
        # Usually Macs need this, but they might not, so try it, and only add if it works.
        cflags1 = cflags + ['-stdlib=libc++']
        lflags1 = lflags + ['-stdlib=libc++']
        cpp_code = r"""
            #include <iostream>
            int main() {
                std::cout<<"Hello World\n";
                return 0;
            }
            """
        if try_compile(cpp_code, compiler, cflags1, lflags1):
            # Success
            print('Compilation succeeded with -stdlib=libc++.')
            cflags = cflags1
            lflags = lflags1
        elif try_compile(cpp_code, compiler, cflags, lflags):
            # Failed with stdlib, but success without
            # Leave cflags, lflags as they are.
            print('Compilation succeeded without -stdlib=libc++.')
        else:
            # Failed either way.  :(
            print('Error: Unable to determine correct compile flags')
            exit()

    return cflags, lflags

class my_builder(build_ext):
    def build_extensions(self):
        cflags, lflags = check_flags(self.compiler)

        # Add the appropriate extra flags for that compiler.
        for e in self.extensions:
            e.extra_compile_args = cflags
            for flag in lflags:
                e.extra_link_args.append(flag)

        # Now run the normal build function.
        build_ext.build_extensions(self)

#
# Make the extensions to be built
#

if have_numpy:
    # recfile
    include_dirs += ['esutil/recfile']
    recfile_sources = ['esutil/recfile/records.cpp',
                       'esutil/recfile/records_wrap.cpp']
    recfile_module = Extension('esutil.recfile._records',
                               extra_compile_args=extra_compile_args,
                               extra_link_args=extra_link_args,
                               sources=recfile_sources,
                               include_dirs=include_dirs)
    ext_modules.append(recfile_module)
    packages.append('esutil.recfile')

    # cosmology package
    cosmo_sources = glob('esutil/cosmology/*.c')
    cosmo_module = Extension('esutil.cosmology._cosmolib',
                             extra_compile_args=extra_compile_args,
                             extra_link_args=extra_link_args,
                             sources=cosmo_sources,
                             include_dirs=include_dirs)
    ext_modules.append(cosmo_module)
    packages.append('esutil.cosmology')



    # HTM
    include_dirs += ['esutil/htm','esutil/htm/htm_src']
    htm_sources = glob('esutil/htm/htm_src/*.cpp')
    htm_sources += ['esutil/htm/htmc.cc','esutil/htm/htmc_wrap.cc']
    htm_module = Extension('esutil.htm._htmc',
                           extra_compile_args=extra_compile_args,
                           extra_link_args=extra_link_args,
                           sources=htm_sources,
                           include_dirs=include_dirs)

    ext_modules.append(htm_module)
    packages.append('esutil.htm')




    # stat package
    #include_dirs += ['esutil/stat']
    #chist_sources = ['chist.cc','chist_wrap.cc']
    #chist_sources = ['esutil/stat/'+s for s in chist_sources]
    chist_sources = ['esutil/stat/chist_pywrap.c']
    chist_module = Extension('esutil.stat._chist',
                             extra_compile_args=extra_compile_args,
                             extra_link_args=extra_link_args,
                             sources=chist_sources,
                             include_dirs=include_dirs)
    ext_modules.append(chist_module)
    stat_util_sources = ['_stat_util.c']
    stat_util_sources = ['esutil/stat/'+s for s in stat_util_sources]
    stat_util_module = Extension('esutil.stat._stat_util',
                                 extra_compile_args=extra_compile_args,
                                 extra_link_args=extra_link_args,
                                 sources=stat_util_sources,
                                 include_dirs=include_dirs)
    ext_modules.append(stat_util_module)
    packages.append('esutil.stat')


    # integrate package

    #cgauleg_sources = glob('esutil/integrate/*.cc')
    cgauleg_sources = glob('esutil/integrate/cgauleg_pywrap.c')
    cgauleg_module = Extension('esutil.integrate._cgauleg',
                               extra_compile_args=extra_compile_args,
                               extra_link_args=extra_link_args,
                               sources=cgauleg_sources,
                               include_dirs=include_dirs)
    ext_modules.append(cgauleg_module)
    packages.append('esutil.integrate')


long_description="""
A python package including a wide variety of utilities, focused primarily on
numerical python, statistics, and file input/output.   Includes specialized
tools for astronomers.
"""

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: GNU General Public License (GPL)",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.6",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
]


setup(name='esutil',
      version="0.6.7",
      author="Erin Scott Sheldon",
      author_email="erin.sheldon@gmail.com",
      classifiers=classifiers,
      description='Erin Sheldons Python Utilities',
      long_description=long_description,
      license = "GPL",
      url='http://code.google.com/p/esutil/',
      packages=packages,
      cmdclass = {'build_ext': my_builder},
      ext_modules=ext_modules)
#, install_requires=['numpy'])

# If we get to here, then all was fine.  Go ahead and delete the files in the tmp directory.
if os.path.exists(local_tmp):
    shutil.rmtree(local_tmp)
