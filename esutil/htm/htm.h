#ifndef _htm_python_wrapper_h
#define _htm_python_wrapper_h

#include <Python.h>

// doesn't seem to work to include it here with swig...
//#include "../NumpyVector.h"

class HTM {
	public:
		HTM(int depth) throw (const char *);
		~HTM() {};

        // take in ra/dec and output the htm index for each
		PyObject* lookup_id(
                PyObject* ra_array, 
                PyObject* dec_array) throw (const char *);

    private:
        int mDepth;
};

#endif
