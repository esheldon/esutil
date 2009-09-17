from distutils.core import setup

# data_files copies the ups/esutil.table into prefix/ups
setup(name='esutil',
      version='0.1',
      description='Erin Sheldons Utilities',
      packages=['esutil'],
      data_files=[('ups',['ups/esutil.table'])])
