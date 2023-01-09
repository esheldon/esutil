/*
 
 This is a python class definition, wrapping the cosmological distance
 calculations in cosmolib.c.  The "struct cosmo" is the underlying
 "class" and the functions are the methods.
 
 These wrappers are minimal.  Scalars are converted as needed, but there is no
 conversion of the input types to arrays for the "vec" vectorized versions of
 the functions.  It is the responsibility of the python wrapper in cosmology.py
 to take care of that.

 I think this is the right compromise:  it is messier to write the C versions
 of these type checks, but is fairly trivial to write the python checks and
 conversions.

 I also could have generated this with SWIG.  But I'm experienced enough in
 creating these classes that it is actually less work to write it explicitly
 than mess around with SWIG complications.  And this file is a factor of
 ten smaller than the corresponding SWIG wrapper. The size of the SWIG wrapper
 is dominated by all the type conversions, which I do in the python wrapper.

 April 2011
 Erin Sheldon, Brookhaven National Laboratory

 */

#include <Python.h>
#include "cosmolib.h"
#include <numpy/arrayobject.h> 

struct PyCosmoObject {
  PyObject_HEAD
  struct cosmo* cosmo;
};



static void
PyCosmoObject_dealloc(struct PyCosmoObject* self)
{
    free(self->cosmo);

#if PY_MAJOR_VERSION >= 3
    // introduced in python 2.6
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif


}


static int
PyCosmoObject_init(struct PyCosmoObject* self, PyObject *args, PyObject *kwds)
{
    double DH;
    int flat;
    double omega_m, omega_l, omega_k;

    free(self->cosmo);

    if (!PyArg_ParseTuple(args, 
                          (char*)"diddd", 
                          &DH, &flat, &omega_m, &omega_l, &omega_k)) {
        printf("failed to Parse init");
        return -1;
    }

    self->cosmo = cosmo_new(DH, flat, omega_m, omega_l, omega_k);
    if (self->cosmo == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate struct cosmo");
        return -1;
    }
    return 0;
}

static PyObject *
PyCosmoObject_repr(struct PyCosmoObject* self) {
#if PY_MAJOR_VERSION >= 3
    const char* code="y";
#else
    const char* code="s";
#endif

    char repr[255];
    if (self->cosmo != NULL) {
        sprintf(repr, "flat:    %d\n"
                      "DH:      %f\n"
                      "omega_m: %f\n" 
                      "omega_l: %f\n" 
                      "omega_k: %f", 
                      self->cosmo->flat, 
                      self->cosmo->DH, 
                      self->cosmo->omega_m, 
                      self->cosmo->omega_l, 
                      self->cosmo->omega_k);
        return Py_BuildValue(code, repr);
    }  else {
        return Py_BuildValue(code, "");
    }
}

static PyObject* PyCosmoObject_DH(struct PyCosmoObject* self) {
    return PyFloat_FromDouble(self->cosmo->DH);
}
static PyObject* PyCosmoObject_flat(struct PyCosmoObject* self) {
    return PyLong_FromLong((long) self->cosmo->flat);
}
static PyObject* PyCosmoObject_omega_m(struct PyCosmoObject* self) {
    return PyFloat_FromDouble(self->cosmo->omega_m);
}
static PyObject* PyCosmoObject_omega_l(struct PyCosmoObject* self) {
    return PyFloat_FromDouble(self->cosmo->omega_l);
}
static PyObject* PyCosmoObject_omega_k(struct PyCosmoObject* self) {
    return PyFloat_FromDouble(self->cosmo->omega_k);
}

/*
   The wrapper methods and vectorizations.

   For the array inputs, the caller is responsible for making sure the input is
   an array, contiguous, of the right data type.  That is much more easily
   done in the python wrapper.
*/


static PyObject*
PyCosmoObject_ez_inverse(struct PyCosmoObject* self, PyObject* args) {
    double z;
    double ezinv;

    if (!PyArg_ParseTuple(args, (char*)"d", &z)) {
        return NULL;
    }

    ezinv = ez_inverse(self->cosmo, z);
    return PyFloat_FromDouble(ezinv);
}
static PyObject*
PyCosmoObject_ez_inverse_vec(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zObj=NULL, *resObj=NULL;;
    double *z, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"O", &zObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zObj);
    z = (double* )PyArray_DATA(zObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = ez_inverse(self->cosmo, z[i]);
    }

    return resObj;

}



static PyObject*
PyCosmoObject_ez_inverse_integral(struct PyCosmoObject* self, PyObject* args) {
    double zmin, zmax;
    double ezinv_int;

    if (!PyArg_ParseTuple(args, (char*)"dd", &zmin, &zmax)) {
        return NULL;
    }

    ezinv_int = ez_inverse_integral(self->cosmo, zmin, zmax);
    return PyFloat_FromDouble(ezinv_int);
}




// comoving distance and vectorizations
static PyObject*
PyCosmoObject_Dc(struct PyCosmoObject* self, PyObject* args) {
    double zmin, zmax;
    double d;

    if (!PyArg_ParseTuple(args, (char*)"dd", &zmin, &zmax)) {
        return NULL;
    }

    d = Dc(self->cosmo, zmin, zmax);
    return PyFloat_FromDouble(d);

}


static PyObject*
PyCosmoObject_Dc_vec1(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zminObj=NULL, *resObj=NULL;;
    double *zmin, zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"Od", &zminObj, &zmax)) {
        return NULL;
    }

    n = PyArray_SIZE(zminObj);
    zmin = (double* )PyArray_DATA(zminObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = self->cosmo->DH*ez_inverse_integral(self->cosmo, zmin[i], zmax); 
    }

    return resObj;

}

static PyObject*
PyCosmoObject_Dc_vec2(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zmaxObj=NULL, *resObj=NULL;;
    double zmin, *zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"dO", &zmin, &zmaxObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zmaxObj);
    zmax = (double* )PyArray_DATA(zmaxObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = self->cosmo->DH*ez_inverse_integral(self->cosmo, zmin, zmax[i]); 
    }

    return resObj;
}

static PyObject*
PyCosmoObject_Dc_2vec(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zmaxObj, *zminObj=NULL, *resObj=NULL;
    double *zmin, *zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"OO", &zminObj, &zmaxObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zminObj);
    zmin = (double* )PyArray_DATA(zminObj);
    zmax = (double* )PyArray_DATA(zmaxObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = self->cosmo->DH*ez_inverse_integral(self->cosmo, zmin[i], zmax[i]); 
    }

    return resObj;
}

// transverse comoving distance and vectorizations
static PyObject*
PyCosmoObject_Dm(struct PyCosmoObject* self, PyObject* args) {
    double zmin, zmax;
    double d;

    if (!PyArg_ParseTuple(args, (char*)"dd", &zmin, &zmax)) {
        return NULL;
    }

    d = Dm(self->cosmo, zmin, zmax);
    return PyFloat_FromDouble(d);

}

static PyObject*
PyCosmoObject_Dm_vec1(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zminObj=NULL, *resObj=NULL;;
    double *zmin, zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"Od", &zminObj, &zmax)) {
        return NULL;
    }

    n = PyArray_SIZE(zminObj);
    zmin = (double* )PyArray_DATA(zminObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = Dm(self->cosmo, zmin[i], zmax); 
    }

    return resObj;

}

static PyObject*
PyCosmoObject_Dm_vec2(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zmaxObj=NULL, *resObj=NULL;;
    double zmin, *zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"dO", &zmin, &zmaxObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zmaxObj);
    zmax = (double* )PyArray_DATA(zmaxObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = Dm(self->cosmo, zmin, zmax[i]); 
    }

    return resObj;
}

static PyObject*
PyCosmoObject_Dm_2vec(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zmaxObj, *zminObj=NULL, *resObj=NULL;
    double *zmin, *zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"OO", &zminObj, &zmaxObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zminObj);
    zmin = (double* )PyArray_DATA(zminObj);
    zmax = (double* )PyArray_DATA(zmaxObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = Dm(self->cosmo, zmin[i], zmax[i]); 
    }

    return resObj;
}


// Angular diameter distance
static PyObject*
PyCosmoObject_Da(struct PyCosmoObject* self, PyObject* args) {
    double zmin, zmax;
    double d;

    if (!PyArg_ParseTuple(args, (char*)"dd", &zmin, &zmax)) {
        return NULL;
    }

    d = Da(self->cosmo, zmin, zmax);
    return PyFloat_FromDouble(d);

}

static PyObject*
PyCosmoObject_Da_vec1(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zminObj=NULL, *resObj=NULL;;
    double *zmin, zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"Od", &zminObj, &zmax)) {
        return NULL;
    }

    n = PyArray_SIZE(zminObj);
    zmin = (double* )PyArray_DATA(zminObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = Da(self->cosmo, zmin[i], zmax); 
    }

    return resObj;

}

static PyObject*
PyCosmoObject_Da_vec2(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zmaxObj=NULL, *resObj=NULL;;
    double zmin, *zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"dO", &zmin, &zmaxObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zmaxObj);
    zmax = (double* )PyArray_DATA(zmaxObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = Da(self->cosmo, zmin, zmax[i]); 
    }

    return resObj;
}

static PyObject*
PyCosmoObject_Da_2vec(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zmaxObj, *zminObj=NULL, *resObj=NULL;
    double *zmin, *zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"OO", &zminObj, &zmaxObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zminObj);
    zmin = (double* )PyArray_DATA(zminObj);
    zmax = (double* )PyArray_DATA(zmaxObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = Da(self->cosmo, zmin[i], zmax[i]); 
    }

    return resObj;
}


// luminosity distance
static PyObject*
PyCosmoObject_Dl(struct PyCosmoObject* self, PyObject* args) {
    double zmin, zmax;
    double d;

    if (!PyArg_ParseTuple(args, (char*)"dd", &zmin, &zmax)) {
        return NULL;
    }

    d = Dl(self->cosmo, zmin, zmax);
    return PyFloat_FromDouble(d);

}

static PyObject*
PyCosmoObject_Dl_vec1(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zminObj=NULL, *resObj=NULL;;
    double *zmin, zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"Od", &zminObj, &zmax)) {
        return NULL;
    }

    n = PyArray_SIZE(zminObj);
    zmin = (double* )PyArray_DATA(zminObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = Dl(self->cosmo, zmin[i], zmax); 
    }

    return resObj;

}

static PyObject*
PyCosmoObject_Dl_vec2(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zmaxObj=NULL, *resObj=NULL;;
    double zmin, *zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"dO", &zmin, &zmaxObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zmaxObj);
    zmax = (double* )PyArray_DATA(zmaxObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = Dl(self->cosmo, zmin, zmax[i]); 
    }

    return resObj;
}

static PyObject*
PyCosmoObject_Dl_2vec(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zmaxObj, *zminObj=NULL, *resObj=NULL;
    double *zmin, *zmax, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"OO", &zminObj, &zmaxObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zminObj);
    zmin = (double* )PyArray_DATA(zminObj);
    zmax = (double* )PyArray_DATA(zmaxObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = Dl(self->cosmo, zmin[i], zmax[i]); 
    }

    return resObj;
}

// Comoving volume element and vectorization
static PyObject*
PyCosmoObject_dV(struct PyCosmoObject* self, PyObject* args) {
    double z;
    double dv;

    if (!PyArg_ParseTuple(args, (char*)"d", &z)) {
        return NULL;
    }

    dv = dV(self->cosmo, z);
    return PyFloat_FromDouble(dv);

}

static PyObject*
PyCosmoObject_dV_vec(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zObj=NULL, *resObj=NULL;;
    double *z, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"O", &zObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zObj);
    z = (double* )PyArray_DATA(zObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = dV(self->cosmo, z[i]); 
    }

    return resObj;

}

// Comoving volume between zmin and zmax
static PyObject*
PyCosmoObject_V(struct PyCosmoObject* self, PyObject* args) {
    double zmin, zmax;
    double v;

    if (!PyArg_ParseTuple(args, (char*)"dd", &zmin, &zmax)) {
        return NULL;
    }

    v = V(self->cosmo, zmin, zmax);
    return PyFloat_FromDouble(v);

}



// Inverse critical density
static PyObject*
PyCosmoObject_scinv(struct PyCosmoObject* self, PyObject* args) {
    double zl, zs;
    double d;

    if (!PyArg_ParseTuple(args, (char*)"dd", &zl, &zs)) {
        return NULL;
    }

    d = scinv(self->cosmo, zl, zs);
    return PyFloat_FromDouble(d);

}

static PyObject*
PyCosmoObject_scinv_vec1(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zlObj=NULL, *resObj=NULL;;
    double *zl, zs, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"Od", &zlObj, &zs)) {
        return NULL;
    }

    n = PyArray_SIZE(zlObj);
    zl = (double* )PyArray_DATA(zlObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = scinv(self->cosmo, zl[i], zs); 
    }

    return resObj;

}

static PyObject*
PyCosmoObject_scinv_vec2(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zsObj=NULL, *resObj=NULL;;
    double zl, *zs, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"dO", &zl, &zsObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zsObj);
    zs = (double* )PyArray_DATA(zsObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = scinv(self->cosmo, zl, zs[i]); 
    }

    return resObj;
}

static PyObject*
PyCosmoObject_scinv_2vec(struct PyCosmoObject* self, PyObject* args) {
    PyObject* zsObj, *zlObj=NULL, *resObj=NULL;
    double *zl, *zs, *res;
    npy_intp n, i;

    if (!PyArg_ParseTuple(args, (char*)"OO", &zlObj, &zsObj)) {
        return NULL;
    }

    n = PyArray_SIZE(zlObj);
    zl = (double* )PyArray_DATA(zlObj);
    zs = (double* )PyArray_DATA(zsObj);

    resObj = PyArray_ZEROS(1, &n, NPY_FLOAT64, 0);
    res = (double* )PyArray_DATA(resObj);

    for (i=0; i<n; i++) {
        res[i] = scinv(self->cosmo, zl[i], zs[i]); 
    }

    return resObj;
}




static PyMethodDef PyCosmoObject_methods[] = {
    {"DH",          (PyCFunction)PyCosmoObject_DH,          METH_VARARGS, "DH\n\nGet the Hubble distance"},
    {"flat",          (PyCFunction)PyCosmoObject_flat,          METH_VARARGS, "flat\n\nReturn if universe if flat"},
    {"omega_m",          (PyCFunction)PyCosmoObject_omega_m,          METH_VARARGS, "omega_m\n\nGet omega matter"},
    {"omega_l",          (PyCFunction)PyCosmoObject_omega_l,          METH_VARARGS, "omega_m\n\nGet omega lambda"},
    {"omega_k",          (PyCFunction)PyCosmoObject_omega_k,          METH_VARARGS, "omega_m\n\nGet omega curvature"},
    {"ez_inverse",          (PyCFunction)PyCosmoObject_ez_inverse,          METH_VARARGS, "ez_inverse(z)\n\nGet 1/E(z)"},
    {"ez_inverse_vec",          (PyCFunction)PyCosmoObject_ez_inverse_vec,          METH_VARARGS, "ez_inverse_vec(z)\n\nGet 1/E(z) for z an array"},
    {"ez_inverse_integral", (PyCFunction)PyCosmoObject_ez_inverse_integral, METH_VARARGS, "ez_inverse_integral(zmin, zmax)\n\nGet integral of 1/E(z) from zmin to zmax"},
    {"Dc",               (PyCFunction)PyCosmoObject_Dc,               METH_VARARGS, "Dc(zmin,zmax)\n\nComoving distance between zmin and zmax"},
    {"Dc_vec1",          (PyCFunction)PyCosmoObject_Dc_vec1,          METH_VARARGS, "Dc_vec1(zmin,zmax)\n\nComoving distance between zmin(array) and zmax"},
    {"Dc_vec2",          (PyCFunction)PyCosmoObject_Dc_vec2,          METH_VARARGS, "Dc_vec2(zmin,zmax)\n\nComoving distance between zmin and zmax(array)"},
    {"Dc_2vec",          (PyCFunction)PyCosmoObject_Dc_2vec,          METH_VARARGS, "Dc_2vec(zmin,zmax)\n\nComoving distance between zmin and zmax both arrays"},
    {"Dm",              (PyCFunction)PyCosmoObject_Dm,              METH_VARARGS, "Dm(zmin,zmax)\n\nTransverse comoving distance between zmin and zmax"},
    {"Dm_vec1",         (PyCFunction)PyCosmoObject_Dm_vec1,         METH_VARARGS, "Dm_vec1(zmin,zmax)\n\nTransverse Comoving distance between zmin(array) and zmax"},
    {"Dm_vec2",         (PyCFunction)PyCosmoObject_Dm_vec2,         METH_VARARGS, "Dm_vec2(zmin,zmax)\n\nTransverse Comoving distance between zmin and zmax(array)"},
    {"Dm_2vec",         (PyCFunction)PyCosmoObject_Dm_2vec,         METH_VARARGS, "Dm_2vec(zmin,zmax)\n\nTransverse Comoving distance between zmin and zmax both arrays"},
    {"Da",             (PyCFunction)PyCosmoObject_Da,             METH_VARARGS, "Da(zmin,zmax)\n\nAngular diameter distance distance between zmin and zmax"},
    {"Da_vec1",        (PyCFunction)PyCosmoObject_Da_vec1,        METH_VARARGS, "Da_vec1(zmin,zmax)\n\nAngular diameter distance distance between zmin(array) and zmax"},
    {"Da_vec2",        (PyCFunction)PyCosmoObject_Da_vec2,        METH_VARARGS, "Da_vec2(zmin,zmax)\n\nAngular diameter distance distance between zmin and zmax(array)"},
    {"Da_2vec",        (PyCFunction)PyCosmoObject_Da_2vec,        METH_VARARGS, "Da_2vec(zmin,zmax)\n\nAngular diameter distance distance between zmin and zmax both arrays"},
    {"Dl",             (PyCFunction)PyCosmoObject_Dl,             METH_VARARGS, "Dl(zmin,zmax)\n\nLuminosity distance distance between zmin and zmax"},
    {"Dl_vec1",        (PyCFunction)PyCosmoObject_Dl_vec1,        METH_VARARGS, "Dl_vec1(zmin,zmax)\n\nLuminosity distance distance between zmin(array) and zmax"},
    {"Dl_vec2",        (PyCFunction)PyCosmoObject_Dl_vec2,        METH_VARARGS, "Dl_vec2(zmin,zmax)\n\nLuminosity distance distance between zmin and zmax(array)"},
    {"Dl_2vec",        (PyCFunction)PyCosmoObject_Dl_2vec,        METH_VARARGS, "Dl_2vec(zmin,zmax)\n\nLuminosity distance distance between zmin and zmax both arrays"},
    {"dV",                  (PyCFunction)PyCosmoObject_dV,                  METH_VARARGS, "dV(z)\n\nComoving volume element at redshift z"},
    {"dV_vec",              (PyCFunction)PyCosmoObject_dV_vec,              METH_VARARGS, "dV(z)\n\nComoving volume element at redshift z(array)"},
    {"V",                   (PyCFunction)PyCosmoObject_V,                   METH_VARARGS, "V(z)\n\nComoving volume between zmin and zmax"},
    {"scinv",               (PyCFunction)PyCosmoObject_scinv,               METH_VARARGS, "scinv(zl,zs)\n\nInverse critical density distance between zl and zs"},
    {"scinv_vec1",          (PyCFunction)PyCosmoObject_scinv_vec1,          METH_VARARGS, "scinv_vec1(zl,zs)\n\nInverse critical density distance between zl(array) and zs"},
    {"scinv_vec2",          (PyCFunction)PyCosmoObject_scinv_vec2,          METH_VARARGS, "scinv_vec2(zl,zs)\n\nInverse critical density distance between zl and zs(array)"},
    {"scinv_2vec",          (PyCFunction)PyCosmoObject_scinv_2vec,          METH_VARARGS, "scinv_2vec(zl,zs)\n\nInverse critical density distance between zl and zs both arrays"},

    {NULL}  /* Sentinel */
};






static PyTypeObject PyCosmoType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_cosmolib.cosmo",             /*tp_name*/
    sizeof(struct PyCosmoObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyCosmoObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyCosmoObject_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Cosmology Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyCosmoObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyCosmoObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyCosmoObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};


static PyMethodDef cosmotype_methods[] = {
    {NULL}  /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_cosmolib",      /* m_name */
        "Define cosmo type and methods ",  /* m_doc */
        -1,                  /* m_size */
        cosmotype_methods,    /* m_methods */
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
PyInit__cosmolib(void) 
#else
init_cosmolib(void) 
#endif
{
    PyObject* m;

    PyCosmoType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyCosmoType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else

    if (PyType_Ready(&PyCosmoType) < 0)
        return;

    m = Py_InitModule3("_cosmolib", cosmotype_methods, "Define cosmo type and methods.");

    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyCosmoType);
    PyModule_AddObject(m, "cosmo", (PyObject *)&PyCosmoType);

    import_array();


#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
