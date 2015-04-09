This python packages includes a wide variety of utilities, focused primarily on numerical python, statistics, and file input/output.

## News ##

  * 2014-09-28: Added the Matcher class to htm.  the htm.match() method now uses a Matcher internally and the "reverse indices" style is deprecated.  Instead of using match\_prepare() and match() use a Matcher object

  * 2014-06-12: Tagged version 0.5.2 [Release Notes](http://code.google.com/p/esutil/source/browse/trunk/RELEASE_NOTES)  File downloads are no longer supported by google code.  See the install instructions below to get the source.


[older news](http://code.google.com/p/esutil/wiki/News).  Click [here](http://code.google.com/p/esutil/source/list) to view a full list of changes


### Modules ###

Included are the following sub-modules:


  * **[numpy\_util](http://code.google.com/p/esutil/source/browse/trunk/esutil/numpy_util.py)**:  A set of tools for manipulation of numerical python arrays, with a focus on arrays with fields, aka recarrays or structured arrays.  If numpy is not installed this module can still be imported safely.

  * **[stat](http://code.google.com/p/esutil/source/browse/trunk/esutil/stat/util.py)**:  Perform statistics, fitting,  etc.  Includes a histogram tool similar to that of IDL.

  * **[sqlite\_util](http://code.google.com/p/esutil/source/browse/trunk/esutil/sqlite_util.py)**:  Tools to work with sqlite databases.  Can stuff numpy recarrays into tables, creating the table if necessary.  Can read from tables into numpy recarrays.

  * **[random](http://code.google.com/p/esutil/source/browse/trunk/esutil/random.py)**: Contains a class to generate random points from any arbitrary probability distribution.

  * **Astronomy related tools**:
    * **[wcsutil](http://code.google.com/p/esutil/source/browse/trunk/esutil/wcsutil.py)**: Tools for working with the astronomical World Coordinate System format.
    * **[cosmology](http://code.google.com/p/esutil/source/browse/trunk/esutil/cosmology/cosmology.py)**: Calculate cosmological distances.
    * **[coords](http://code.google.com/p/esutil/source/browse/trunk/esutil/coords.py)**: Tools for working with astronomical coordinates and coordinate transformations.  Used to be called astro\_util. There is still an alias for that name available.
    * **[htm](http://code.google.com/p/esutil/source/browse/trunk/esutil/htm/__init__.py)**: This is a Class to deal with the Hierarchical Triangular Mesh, which is a method for breaking the unit sphere into a tree structure where each node in the tree is represented by a spherical triangle.  A primary advantage of the HTM over other schemes is that it deals perfectly well with the poles.   This class contains code for matching ra/dec lists, measuring two-point correlation functions (using the bincount method), and finding node information for ra/dec points.


  * **[/include/NumpyVector.h](http://code.google.com/p/esutil/source/browse/trunk/esutil/include/NumpyVector.h)**  This is a C++ template class to simplify using 1-d numpy arrays in C++ codes linked to python.
  * **[/include/NumpyVoidVector.h](http://code.google.com/p/esutil/source/browse/trunk/esutil/include/NumpyVoidVector.h)**  This is a C++ class similar to `NumpyVector.h` but the type can be determined at run time.
  * **[/include/NumpyRecords.h](http://code.google.com/p/esutil/source/browse/trunk/esutil/include/NumpyRecords.h)**  This is a C++ class similar to `NumpyVector.h` but for record arrays.

  * **[integrate](http://code.google.com/p/esutil/source/browse/trunk/esutil/integrate/util.py)**:  Integration utilities.  Currently contains a class to perform Gauss-Legendre integration.


  * **[io](http://code.google.com/p/esutil/source/browse/trunk/esutil/io.py)**:  A module containing functions that can read/write to a large variety of file formats with a single interface.  The type of the file is generally determined from the file extension for ordinary files.  Can read from lists of files.  **Status**:  Currently read() function works with FITS, JSON, YAML, XML, and REC files (REC files are read/written using sfile below) and write() works with FITS, JSON, YAML, and REC.  Need to work in more formats for write() as well as database support.

  * **[sfile](http://code.google.com/p/esutil/source/browse/trunk/esutil/sfile.py)**:  Tools to read and write numpy arrays from a simple and efficient file format (.rec).  Supports recarrays (aka structured arrays, arrays with fields) and ordinary arrays. The format is a simple ascii header followed by data.  Data can be appended for structured arrays.  Uses the recfile package when extracting subsets of the data from disk or when writing/reading ascii.

  * **[recfile](http://code.google.com/p/esutil/source/browse/trunk/esutil/recfile/__init__.py)**:  A local copy of the recfile package. This is a C++ python extension module for reading from and writing to files with fixed length records. Data are read into or written from numerical python arrays (numpy). Files can be binary or text files with fixed delimiter, such as csv, tab, or white space.  Work on recfile usually occurs within the esutil repository and is copied over to recfile for releases.  The project home page is at: http://code.google.com/p/recfile/


  * **[json\_util](http://code.google.com/p/esutil/source/browse/trunk/esutil/json_util.py)**:  Convenience functions for working with JSON files.  http://en.wikipedia.org/wiki/JSON

  * **[hdfs](http://code.google.com/p/esutil/source/browse/trunk/esutil/hdfs.py)**:  Work more transparently with the Hadoop File System (HDFS file system: http://hadoop.apache.org/hdfs/)

  * **[ostools](http://code.google.com/p/esutil/source/browse/trunk/esutil/ostools.py)**:  Tools for working with the os and files.

  * **[misc](http://code.google.com/p/esutil/source/browse/trunk/esutil/misc.py)**:  Miscellaneous tools.

  * **[xmltools](http://code.google.com/p/esutil/source/browse/trunk/esutil/xmltools.py)**:  A set of tools for reading/writing xml to and from python dictionaries.  These are modified versions of the code found here: http://code.activestate.com/recipes/573463/



  * **[oracle\_util](http://code.google.com/p/esutil/source/browse/trunk/esutil/oracle_util.py)**:  Utilities for working with oracle databases.  Requires a hacked version of cx\_Oracle that supports numpy.  If cx\_Oracle is not installed this module is not imported. **Status**: need to include new code from desdb stuff.

## Installation ##

### Using pip ###

```
pip install esutil

# update fitsio (and everything else)
pip install fitsio --upgrade

# if pip refuses to update to a newer version
pip install fitsio --upgrade --ignore-installed

# if you only want to upgrade fitsio
pip install fitsio --no-deps --upgrade --ignore-installed
```

### From Source ###

To get the stable source tree, download from pypi
```
    https://pypi.python.org/packages/source/e/esutil/esutil-v0.5.2.tar.gz
    https://pypi.python.org/pypi/esutil/0.5.2
```

To keep up to date, use svn

svn checkout http://esutil.googlecode.com/svn/trunk/ esutil

Then use "svn update" and reinstall to keep current.

With either svn or the tar file, enter the directory and type

`    python setup.py install --prefix=/path/to/install_directory`

If you specify a prefix, make sure to add
/path/to/install/lib/python2.7/site-packages
to your PYTHONPATH (replace python2.7 with your version)

### See Also ###

  * [pgnumpy](http://code.google.com/p/pgnumpy/): Interface between numerical python (NumPy?) and the Postgres database. Record arrays can be transferred to and from tables transparently.

  * [pycolumns](http://code.google.com/p/pycolumns/): A simple, efficient column-oriented, pythonic database. Data are input and output as numerical python arrays, and indexing is provided by berkeley db through the numpydb package.

  * [numpydb](http://code.google.com/p/numpydb/): A numerical python interface to the berkeley database.


The following will be included soon:


  * **imtools**:  Tools for manipulating images.