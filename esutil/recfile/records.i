%include exception.i       

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}


%module records
%{
#include "records.hpp"
%}
%feature("kwargs");



%include "records.hpp"

