#include <iostream>
#include "htm.h"
#include "NumpyVector.h"


HTMC::HTMC(int depth) throw (const char *) {
    init(depth);
}

void HTMC::init(int depth) throw (const char *) {
    mDepth = depth;
    mHtmInterface.init(depth);
}

PyObject* HTMC::lookup_id(
        PyObject* ra_array, 
        PyObject* dec_array) throw (const char* ) {


    // wrap the input ra,dec objects, making sure they are doubles

    NumpyVector<double> ra(ra_array);
    NumpyVector<double> dec(dec_array);

    if (ra.size() != dec.size()) {
        throw "ra/dec must be the same size";
    }

    // The output data.  int32 should be OK for depth=10
    NumpyVector<npy_int32> htmid(ra.size());

    for (npy_intp i=0; i<ra.size(); i++) {
        htmid[i] = mHtmInterface.lookupID(ra[i], dec[i]);
    }

    PyObject* htmidPyObj = htmid.getref();
    return htmidPyObj;
}

PyObject* HTMC::match_helper(
        PyObject* angle_array,
        PyObject* ra1_array, // all in degrees
        PyObject* dec1_array,
        PyObject* ra2_array, 
        PyObject* dec2_array,
        PyObject* htmrev2_array,
        int32_t minid,
        int32_t maxid) throw (const char *) {

    Py_INCREF(Py_None);
    return Py_None;
}


