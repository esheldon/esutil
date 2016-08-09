#include <Python.h>
#include "numpy/arrayobject.h"

/*

   assume hist and rev are contiguous, which we can
   guarantee in the caller
*/

static PyObject* PyCHist_chist(PyObject* self, PyObject* args) {

    PyObject* data_pyobj=NULL;
    double datamin=0;
    PyObject* sort_pyobj=NULL;
    double binsize=0;
    PyObject* hist_pyobj=NULL;
    PyObject* rev_pyobj=NULL;

    npy_int64 *hist=NULL, *rev=NULL;

    int dorev=0;
    npy_intp nbin = 0, ndata=0, nrev=0;
    npy_int64
        i=0,
        binnum_old = 0,
        offset = 0, data_index = 0, binnum=0, tbin = 0;
    double thisdata=0;

    if (!PyArg_ParseTuple(args, (char*)"OdOdiO",
                          &data_pyobj,
                          &datamin,
                          &sort_pyobj,
                          &binsize,
                          &hist_pyobj,
                          &rev_pyobj)) {
        return NULL;
    }

    if (rev_pyobj != Py_None) {
        dorev=1;
        rev = (npy_int64 *) PyArray_DATA(rev_pyobj);
        nrev = PyArray_SIZE(rev_pyobj);
    }

    ndata = PyArray_SIZE(sort_pyobj);
    nbin  = PyArray_SIZE(hist_pyobj);

    hist=(npy_int64 *) PyArray_DATA(hist_pyobj);

    // this is my reverse engineering of the IDL reverse
    // indices
    binnum_old = -1;

    for (i=0; i<ndata; i++) {

        offset = i+nbin+1;
        data_index = *(npy_int64 *) PyArray_GETPTR1(sort_pyobj, i);


        if (dorev) {
            rev[offset] = data_index;
        }

        // data might not be contiguous, so use the
        // more general getter
        thisdata = *(double *) PyArray_GETPTR1(data_pyobj, data_index);

        binnum = (npy_int64) ( (thisdata-datamin)/binsize);

        if (binnum >= 0 && binnum < nbin) {
            // Should we upate the reverse indices?
            if (dorev && (binnum > binnum_old) ) {
                tbin = binnum_old + 1;
                while (tbin <= binnum) {
                    rev[tbin] = offset;
                    tbin++;
                }
            }
            // Update the histogram
            hist[binnum] = hist[binnum] + 1;
            binnum_old = binnum;
        }
    }

    tbin = binnum_old + 1;
    while (tbin <= nbin) {
        if (dorev) {
            rev[tbin] = nrev;
        }
        tbin++;
    }

    Py_RETURN_NONE;

}

static PyMethodDef chist_methods[] = {
    {"chist",               (PyCFunction)PyCHist_chist, METH_VARARGS, "histogrammer"},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_chist",      /* m_name */
        "Define c version of histogrammer",  /* m_doc */
        -1,                  /* m_size */
        chist_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__chist(void) 
#else
init_chist(void) 
#endif
{
    PyObject* m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else

    m = Py_InitModule3("_chist", chist_methods, "Define c version of histogrammer.");

    if (m==NULL) {
        return;
    }
#endif

    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
