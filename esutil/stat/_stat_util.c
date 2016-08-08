#include <Python.h>
#include <numpy/arrayobject.h> 

static PyObject *
PyStatUtil_random_sample(PyObject *self, PyObject *args) 
{
    PyObject* randind_obj=NULL;
    npy_intp *randind=NULL;
    long int nmax=0, nrand=0, seed=0;
    npy_intp dims[1];
    long int i=0, ntoselect=0, ntocheck=0;
    double prob=0;
    if (!PyArg_ParseTuple(args, (char*)"lll", &nmax, &nrand, &seed)) {
        return NULL;
    }

    if (nmax <= 0 || nrand <= 0) {
        PyErr_Format(PyExc_ValueError,"nmax/nrand must be >= 0, got %ld/%ld", nmax, nrand);
        return NULL;
    }
    if (nrand > nmax) {
        PyErr_Format(PyExc_ValueError,"nrand must be <= nmax, got %ld/%ld", nmax, nrand);
        return NULL;
    }


    srand48(seed);

    dims[0] = nrand;
    randind_obj = PyArray_SimpleNew(1, dims, NPY_INTP);
    randind = PyArray_DATA(randind_obj);

    ntoselect=nrand;
    ntocheck=nmax;

    ntocheck=nmax;
    for (i=0; i<nmax; i++) {

        prob = ( (double) ntoselect ) / ((double) ntocheck );
        if (drand48() < prob) {
            randind[nrand-ntoselect] = i;
            ntoselect -= 1;
        }
        if (ntoselect == 0) {
            break;
        }

        ntocheck -= 1;
    }
    return randind_obj;
}


static PyMethodDef stat_util_module_methods[] = {
    {"random_sample", (PyCFunction)PyStatUtil_random_sample, METH_VARARGS,  "r=random_sample(nmax,nrand)"},
    {NULL}  /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_stat_util",      /* m_name */
        "Defines the some gmix fit methods",  /* m_doc */
        -1,                  /* m_size */
        stat_util_module_methods,    /* m_methods */
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
PyInit__stat_util(void) 
#else
init_stat_util(void) 
#endif
{
    PyObject* m;


    //PyGMixEMObjectType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    /*
    if (PyType_Ready(&PyGMixEMObjectType) < 0) {
        return NULL;
    }
    */
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    /*
    if (PyType_Ready(&PyGMixEMObjectType) < 0) {
        return;
    }
    */
    m = Py_InitModule3("_stat_util", stat_util_module_methods, 
            "This module gmix fit related routines.\n");
    if (m==NULL) {
        return;
    }
#endif

    /*
    Py_INCREF(&PyGMixEMObjectType);
    PyModule_AddObject(m, "GMixEM", (PyObject *)&PyGMixEMObjectType);
    */

    import_array();
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
