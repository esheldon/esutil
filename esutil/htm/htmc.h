#ifndef _htm_python_wrapper_h
#define _htm_python_wrapper_h

#include <Python.h>
#include "SpatialInterface.h"
#include <stdint.h>
#include <vector>
#include <map>
#include "numpy/arrayobject.h"

typedef struct {
	int64_t i1;	
	int64_t i2;	
	double d12;
} PAIR_INFO;

struct PAIR_INFO_ORDERING {
	bool operator()(PAIR_INFO const& pi1, PAIR_INFO const& pi2) {
		return pi1.d12 < pi2.d12;
	}
};

// doesn't seem to work to include it here with swig...
#include "../include/NumpyVector.h"

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

        PyObject* intersect(
                            double ra, // all in degrees
                            double dec,
                            double radius, // degrees
                            int inclusive
                           ) throw (const char *);


        // this requires the reverse indices must already be created,
        // and other obscure inputs. The python wrapper takes care of
        // all that.
        PyObject* cmatch(
                PyObject* radius_array, // degrees
                PyObject* ra1_array, // all in degrees
                PyObject* dec1_array,
                PyObject* ra2_array, 
                PyObject* dec2_array,
                PyObject* htmrev2_array,
                PyObject* minid_obj,
                PyObject* maxid_obj,
				PyObject* maxmatch_obj,
				const char *filename) throw (const char *);

        PyObject* cbincount(
                PyObject* rmin_object, // units of scale*angle in radians
                PyObject* rmax_object, // units of scale*angle in radians
				PyObject* nbin_object, 
                PyObject* ra1_array, // all in degrees
                PyObject* dec1_array,
                PyObject* ra2_array, 
                PyObject* dec2_array,
                PyObject* htmrev2_array,
                PyObject* minid_obj,
                PyObject* maxid_obj,
				PyObject* scale_object=NULL,
                int verbose=0) // will bin in radians*scale.  
                                            // Same length as ra1.
                              throw (const char *);




        int depth() {
            return mDepth;
        }

    private:



        htmInterface mHtmInterface;
        int mDepth;
};

class Matcher {
	public:

        Matcher(int depth,
                PyObject* ra,
                PyObject* dec) throw (const char *);
        ~Matcher() {};

        int get_depth() {
            return depth;
        }

        PyObject* match(PyObject* radius_array, // degrees
                        PyObject* ra_array, // degrees
                        PyObject* dec_array,
                        PyObject* maxmatch_obj,
                        const char* filename) throw (const char *);


    private:

        void init_hmap(void);

        int depth;
        htmInterface htm_interface;

        NumpyVector<double> ra;
        NumpyVector<double> dec;

        std::map<int64_t, std::vector<int64_t> > hmap;

};


#endif
