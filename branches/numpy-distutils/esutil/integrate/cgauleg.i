%module cgauleg
%{
#include "cgauleg.h"
%}

%typemap(throws) const char * %{
    PyErr_SetString(PyExc_RuntimeError, $1);
    SWIG_fail;
%}


%include "cgauleg.h"

