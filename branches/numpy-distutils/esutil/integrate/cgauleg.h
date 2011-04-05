#ifndef _cgauleg_h
#define _cgauleg_h

#include <Python.h>

/*
PyObject* cgauleg(
        double x1,
        double x2,
        int npts) throw (const char *);
		*/

PyObject* cgauleg(
        PyObject* x1var,
        PyObject* x2var,
        PyObject* nptsvar) throw (const char *);


#endif
