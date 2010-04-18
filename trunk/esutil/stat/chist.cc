#include "chist.h"
#include "numpy/arrayobject.h"
#include "NumpyVector.h"

PyObject* chist(
        PyObject* data_pyobj,
        PyObject* datamin_pyobj,
        PyObject* sort_pyobj,
        PyObject* binsize_pyobj,
        PyObject* nbin_pyobj,
        bool dorev) throw (const char *) {

    // For now, to keep things simple, just convert to double if it isn't
    // already.  This will cost us a factor of two in memory, so some day
    // might want to do something better.

    NumpyVector<double> data(data_pyobj);

    // this should end up as a 1-element array
    NumpyVector<double> datamin_array(datamin_pyobj);

    // sort vector converted to 64-bit integer
    NumpyVector<npy_int64> sort(sort_pyobj);
    
    NumpyVector<double> binsize_array(binsize_pyobj);

    NumpyVector<npy_int64> nbin_array(nbin_pyobj);

    
    // copy out scalars
    double datamin = datamin_array[0];
    double binsize = binsize_array[0];
    npy_int64 nbin = nbin_array[0];

    // Make the histogram
    //std::cout<<"Creating histogram with nbin="<<nbin<<"\n";
    NumpyVector<npy_int64> hist(nbin);

    // Check if rev is sent, if so we'll fill it in
    NumpyVector<npy_int64> rev;
    if (dorev) {
        npy_int64 revsize = sort.size() + nbin + 1;
        //std::cout<<"Will do reverse indices, size: "<<revsize<<"\n";
        rev.init(revsize);
    } else {
        //std::cout<<"Not doing reverse indices\n";
    }



    // this is my reverse engineering of the IDL reverse
    // indices
    npy_int64 binnum_old = -1;

    for (npy_int64 i=0; i<sort.size(); i++) {

        npy_int64 offset = i+nbin+1;
        npy_int64 data_index = sort[i];


        if (dorev) {
            rev[offset] = data_index;
        }

        npy_int64 binnum = (npy_int64) ( (data[data_index]-datamin)/binsize);

        if (binnum >= 0 && binnum < nbin) {
            // Should we upate the reverse indices?
            if (dorev && (binnum > binnum_old) ) {
                npy_int64 tbin = binnum_old + 1;
                while (tbin <= binnum) {
                    rev[tbin] = offset;
                    tbin++;
                }
            }
            // Update the histogram
            hist[binnum] = hist[binnum] + 1;
            binnum_old = binnum;
        }
    }

    npy_int64 tbin = binnum_old + 1;
    while (tbin <= nbin) {
        if (dorev) {
            rev[tbin] = rev.size();
        }
        tbin++;
    }





    if (dorev) {
        PyObject* output_tuple = PyTuple_New(2);
		PyTuple_SetItem(output_tuple, 0, hist.getref());
		PyTuple_SetItem(output_tuple, 1, rev.getref());
        return output_tuple;
    } else {
        PyObject* output = hist.getref();
        return output;
    }
}


