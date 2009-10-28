import sys,os
import time
from sys import stdout,stderr

from distutils.core import setup,Extension



# create the ups table
pyvers='%s.%s' % sys.version_info[0:2]
d1='lib/python%s/site-packages' % pyvers
d2='lib64/python%s/site-packages' % pyvers

if not os.path.exists('ups'):
    os.mkdir('ups')
tablefile=open('ups/esutil.table','w')
tab="""
# The default version of this file will be overwritten on setup to include
# paths determined from the python version.  This is useful to have in place
# though so that dependencies can be checked *before* installation.  Currently
# there are no required dependencies, so this is somewhat moot.

setupOptional("python")
envPrepend(PYTHONPATH,${PRODUCT_DIR}/%s)
envPrepend(PYTHONPATH,${PRODUCT_DIR}/%s)
""" % (d1,d2)
tablefile.write(tab)
tablefile.close()



# can we build recfile?
packages = ['esutil']
try:
    import numpy
    include_dirs=numpy.get_include()
    ext_modules = [Extension('esutil.recfile._records', 
                               sources=['esutil/recfile/records.cpp',
                                        'esutil/recfile/records_wrap.cpp'])]
    packages.append('esutil.recfile')

except:
    stdout.write('Numpy not found:  Not building recfile\n')
    time.sleep(5)

    ext_modules=[]
    include_dirs=[]




# data_files copies the ups/esutil.table into prefix/ups
setup(name='esutil',
      description='Erin Sheldons Python Utilities',
      url='http://code.google.com/p/esutil/',
      packages=packages,
      data_files=[('ups',['ups/esutil.table'])],
      ext_modules=ext_modules,
      include_dirs=include_dirs)
