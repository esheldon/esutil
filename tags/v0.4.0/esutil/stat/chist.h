#ifndef _chist_h
#define _chist_h

#include <Python.h>

PyObject* chist(
        PyObject* data_pyobj,
        PyObject* datamin_pyobj,
        PyObject* sort_pyobj,
        PyObject* binsize_pyobj,
        PyObject* nbin_pyobj,
        bool dorev) throw (const char *);

#endif
