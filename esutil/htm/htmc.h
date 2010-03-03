#ifndef _htm_python_wrapper_h
#define _htm_python_wrapper_h

#include <Python.h>
#include "SpatialInterface.h"
#include <stdint.h>
#include <vector>
#include "numpy/arrayobject.h"

typedef struct {
	npy_intp i1;	
	npy_intp i2;	
	double d12;
} PAIR_INFO;

struct PAIR_INFO_ORDERING {
	bool operator()(PAIR_INFO const& pi1, PAIR_INFO const& pi2) {
		return pi1.d12 < pi2.d12;
	}
};

// doesn't seem to work to include it here with swig...
//#include "../NumpyVector.h"

// called HTMC because we will have another python-only class that
// inherits from this one.
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
				PyObject* maxmatch_obj) throw (const char *);


        int depth() {
            return mDepth;
        }

    private:



        htmInterface mHtmInterface;
        int mDepth;
};

#endif
