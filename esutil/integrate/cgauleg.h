#ifndef _cgauleg_h
#define _cgauleg_h

#include <Python.h>
#include "numpy/arrayobject.h"

PyObject* cgauleg(
        double x1,
        double x2,
        long npts);

#endif
