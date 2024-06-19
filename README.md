A python package including a wide variety of utilities, focused primarily on
numerical python, statistics, and file input/output.   Includes specialized
tools for astronomers.

Installation
------------
The easiest way is through conda which has pre-compiled binaries
```
conda install esutil

# or if you don't have conda forge in your channels
conda install -c conda-forge esutil
```

You can also use pip
```
pip install esutil
```

Windows
-------

The unit tests pass on windows with the exception of the recfile package (and
the sfile package that uses it).
