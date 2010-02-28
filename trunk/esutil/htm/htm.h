#ifndef _htm_python_wrapper_h
#define _htm_python_wrapper_h

#include <Python.h>
#include "SpatialInterface.h"
#include <stdint.h>

// doesn't seem to work to include it here with swig...
//#include "../NumpyVector.h"

class HTM {
	public:

		HTM(int depth=10) throw (const char *);
        void init(int depth=10) throw (const char *);
		~HTM() {};

        // take in ra/dec and output the htm index for each
		PyObject* lookup_id(
                PyObject* ra_array, 
                PyObject* dec_array) throw (const char *);

        // this requires the reverse indices must already be created,
        // and other obscure inputs. The python wrapper takes care of
        // all that.
        PyObject* match_helper(
                PyObject* angle_array,
                PyObject* ra1_array, // all in degrees
                PyObject* dec1_array,
                PyObject* ra2_array, 
                PyObject* dec2_array,
                PyObject* htmrev2_array,
                int32_t minid,
                int32_t maxid) throw (const char *);

        int depth() {
            return mDepth;
        }

    private:
        htmInterface mHtmInterface;
        int mDepth;
};

#endif
