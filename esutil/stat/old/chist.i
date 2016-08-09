%module chist
%{
#include "chist.h"
%}

%typemap(throws) const char * %{
    PyErr_SetString(PyExc_RuntimeError, $1);
    SWIG_fail;
%}


%include "chist.h"

