%module records
%{
#include "records.hpp"
%}
%feature("kwargs");


%typemap(throws) const char * %{
    PyErr_SetString(PyExc_RuntimeError, $1);
    SWIG_fail;
%}


%include "records.hpp"

