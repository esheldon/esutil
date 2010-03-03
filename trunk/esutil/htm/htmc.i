%module htmc
//%include std_string.i
%{
#include "htmc.h"
%}
//%feature("kwargs");

// must you declare with throw (const char *)?
%typemap(throws) const char * %{
    PyErr_SetString(PyExc_RuntimeError, $1);
    SWIG_fail;
%}


//%include "htmc.h"

class HTMC {
    public:

        HTMC(int depth=10) throw (const char *);
        void init(int depth=10) throw (const char *);
        ~HTMC() {};

        // take in ra/dec and output the htm index for each
        PyObject* lookup_id(
                PyObject* ra_array, 
                PyObject* dec_array) throw (const char *);

        // this requires the reverse indices must already be created,
        // and other obscure inputs. The python wrapper takes care of
        // all that.
        PyObject* cmatch(
                PyObject* radius_array,
                PyObject* ra1_array, // all in degrees
                PyObject* dec1_array,
                PyObject* ra2_array, 
                PyObject* dec2_array,
                PyObject* htmrev2_array,
                PyObject* minid_obj,
                PyObject* maxid_obj, 
                PyObject* maxmatch_obj,
                PyObject* filename_obj) throw (const char *);

        int depth() {
            return mDepth;
        }

    private:

        htmInterface mHtmInterface;
        int mDepth;
};


