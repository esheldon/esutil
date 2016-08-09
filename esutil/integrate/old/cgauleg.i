%module cgauleg
%{
#define SWIG_FILE_WITH_INIT
#include "cgauleg.h"
%}

#include "../include/numpy.i"

%init %{
import_array();
%}

%include "cgauleg.h"

