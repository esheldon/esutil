#include "htm.h"
#include <iostream>

HTM::HTM(int depth) throw (const char *) {
    mDepth = depth;
}

PyObject* HTM::lookup_id(
        PyObject* ra_array, 
        PyObject* dec_array) throw (const char* ) {
    std::cout<<"Hello\n";
    Py_INCREF(Py_None);
    return Py_None;
}
