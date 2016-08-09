#include <Python.h>
#include "cgauleg.h"
#include "numpy/arrayobject.h"

PyObject* PyCGauleg_cgauleg(PyObject* self, PyObject* args) {

    double x1=0, x2=0;
    long npts_long=0;
    if (!PyArg_ParseTuple(args, (char*)"ddl", &x1, &x2, &npts_long)) {
        return NULL;
    }

	npy_intp npts = npts_long;

    PyObject* xarray = PyArray_ZEROS(
        1,
        &npts,
        NPY_FLOAT64,
        0
    );
    PyObject* warray = PyArray_ZEROS(
        1,
        &npts,
        NPY_FLOAT64,
        0
    );

    double* x = (double* ) PyArray_DATA(xarray);
    double* w = (double* ) PyArray_DATA(warray);

	int i, j, m;
	double xm, xl, z1, z, p1, p2, p3, pp=0, pi, EPS, abszdiff;

	EPS = 4.e-11;
	pi = 3.141592653589793;

	m = (npts + 1)/2;

	xm = (x1 + x2)/2.0;
	xl = (x2 - x1)/2.0;
	z1 = 0.0;

	for (i=1; i<= m; ++i) 
	{

		z=cos( pi*(i-0.25)/(npts+.5) );

		abszdiff = fabs(z-z1);

		while (abszdiff > EPS) 
		{
			p1 = 1.0;
			p2 = 0.0;
			for (j=1; j <= npts;++j)
			{
				p3 = p2;
				p2 = p1;
				p1 = ( (2.0*j - 1.0)*z*p2 - (j-1.0)*p3 )/j;
			}
			pp = npts*(z*p1 - p2)/(z*z -1.);
			z1=z;
			z=z1 - p1/pp;

			abszdiff = fabs(z-z1);

		}

		x[i-1] = xm - xl*z;
		x[npts+1-i-1] = xm + xl*z;
		w[i-1] = 2.0*xl/( (1.-z*z)*pp*pp );
		w[npts+1-i-1] = w[i-1];


	}


	PyObject* output_tuple = PyTuple_New(2);
	PyTuple_SetItem(output_tuple, 0, xarray);
	PyTuple_SetItem(output_tuple, 1, warray);
	return output_tuple;
}

static PyMethodDef cgauleg_methods[] = {
    {"cgauleg",               (PyCFunction)PyCGauleg_cgauleg, METH_VARARGS, "run gauleg"},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_cgauleg",      /* m_name */
        "Define c version of gauleg",  /* m_doc */
        -1,                  /* m_size */
        cgauleg_methods,    /* m_methods */
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
PyInit__cgauleg(void) 
#else
init_cgauleg(void) 
#endif
{
    PyObject* m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else

    m = Py_InitModule3("_cgauleg", cgauleg_methods, "Define c version of gauleg.");

    if (m==NULL) {
        return;
    }
#endif

    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
