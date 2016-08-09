%module htmc
%{
#include "htmc.h"
%}
%feature("kwargs");

%typemap(throws) const char * %{
    PyErr_SetString(PyExc_RuntimeError, $1);
    SWIG_fail;
%}

%include "htmc.h"

/*

class HTMC {
    public:

        HTMC(int depth=10) throw (const char *);
        void init(int depth=10) throw (const char *);
        ~HTMC() {};

        // take in ra/dec and output the htm index for each
        void lookup_id(
                PyObject* ra_array, 
                PyObject* dec_array,
                PyObject* htm_ids_array) throw (const char *);

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
                PyObject* radius_array,
                PyObject* ra1_array, // all in degrees
                PyObject* dec1_array,
                PyObject* ra2_array, 
                PyObject* dec2_array,
                PyObject* htmrev2_array,
                PyObject* minid_obj,
                PyObject* maxid_obj, 
                PyObject* maxmatch_obj,
                const char* filename) throw (const char *);

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



        int get_depth() {
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


};

*/
