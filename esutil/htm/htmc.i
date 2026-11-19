%module htmc
%{
#include <stdexcept>
#include "htmc.h"
%}
%feature("kwargs");

// convert any C++ exception into a python RuntimeError
%exception {
    try {
        $action
    } catch (const char *e) {
        PyErr_SetString(PyExc_RuntimeError, e);
        SWIG_fail;
    } catch (const SpatialException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        SWIG_fail;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        SWIG_fail;
    }
}

%include "htmc.h"
