#include <iostream>
#include <vector>
#include <math.h>
#include "htmc.h"
#include "NumpyVector.h"
#include <algorithm> // for transform


// A couple of utility functions
// raturn great circle distance in degrees
double gcirc(
        double ra1, double dec1, 
        double ra2, double dec2)

{

    double sindec1, cosdec1, sindec2, cosdec2, 
           radiff, cosradiff, cosdis; 

    static const double
        D2R=0.0174532925199433;

    sindec1 = sin(dec1*D2R);
    cosdec1 = cos(dec1*D2R);

    sindec2 = sin(dec2*D2R);
    cosdec2 = cos(dec2*D2R);

    radiff = (ra1-ra2)*D2R;
    cosradiff = cos(radiff);

    cosdis = sindec1*sindec2 + cosdec1*cosdec2*cosradiff;

    if (cosdis < -1.0) cosdis=-1.0;
    if (cosdis >  1.0) cosdis= 1.0;

    return( acos(cosdis)/D2R );

}







HTMC::HTMC(int depth) throw (const char *) {
    init(depth);
}

void HTMC::init(int depth) throw (const char *) {
    mDepth = depth;
    mHtmInterface.init(depth);
}

PyObject* HTMC::lookup_id(
        PyObject* ra_array, 
        PyObject* dec_array) throw (const char* ) {


    // wrap the input ra,dec objects, making sure they are doubles
    // no copy is made if the are already double arrays
    NumpyVector<double> ra(ra_array);
    NumpyVector<double> dec(dec_array);

    if (ra.size() != dec.size()) {
        throw "ra/dec must be the same size";
    }

    // The output data.  int32 should be OK for depth=10
    NumpyVector<npy_int32> htmid(ra.size());

    for (npy_intp i=0; i<ra.size(); i++) {
        htmid[i] = mHtmInterface.lookupID(ra[i], dec[i]);
    }

    PyObject* htmidPyObj = htmid.getref();
    return htmidPyObj;
}



/*
 * Essentially no error checking is done here, it should be done
 * in the python wrapper match().  e.g. ra1/dec1 and ra2/dec2 must
 * be equal length, radius must be either lenght of ra1/dec1 or a 
 * single number.
 */

PyObject* HTMC::cmatch(
        PyObject* radius_array, // degrees
        PyObject* ra1_array, // all in degrees
        PyObject* dec1_array,
        PyObject* ra2_array, 
        PyObject* dec2_array,
        PyObject* htmrev2_array,
        PyObject* minid_obj,
        PyObject* maxid_obj,
        PyObject* maxmatch_obj) throw (const char *) {

    // no copies made if already double vectors
    NumpyVector<double> radius(radius_array);
    npy_intp nrad = radius.size();

    NumpyVector<double> ra1(ra1_array);
    NumpyVector<double> dec1(dec1_array);
    NumpyVector<double> ra2(ra2_array);
    NumpyVector<double> dec2(dec2_array);

    NumpyVector<int32_t> htmrev2(htmrev2_array);

    // get these as numpyvectors even though they are only length 1
    // because it does a good job with conversions
    NumpyVector<int32_t> minidVec(minid_obj);
    NumpyVector<int32_t> maxidVec(maxid_obj);
    NumpyVector<int32_t> maxmatchVec(maxmatch_obj);

    int32_t minid = minidVec[0];
    int32_t maxid = maxidVec[0];
    int32_t maxmatch = maxmatchVec[0];

    // These will temporarily hold the results
    std::vector<int32_t> m1;
    std::vector<int32_t> m2;
    std::vector<double> d12;

    static const double
        D2R=0.0174532925199433;

    // This is used in the basic calculations
    const SpatialIndex &index = mHtmInterface.index();


    double rad=0, d=0;
    if (nrad == 1) {
        rad = radius[0];
        d = cos( rad*D2R );
    }



    npy_intp n1 = ra1.size();
    for (npy_intp i1=0; i1<n1; i1++) {
        // Declare the domain and the lists
        SpatialDomain domain;    // initialize empty domain
        ValVec<uint64> plist, flist;	// List results

        if (nrad > 1) {
            rad = radius[i1];
            d = cos( rad*D2R );
        }

        // Find the triangles around this point
        domain.setRaDecD(ra1[i1],dec1[i1],d); //put in ra,dec,d E.S.S.
        domain.intersect(&index,plist,flist);	  // intersect with list


        // number of triangles found
        npy_intp nfound = flist.length() + plist.length();
        std::vector<int32_t> idlist(nfound);
        npy_intp idcount=0;

        // We could speed this up when no distance is needed by
        // just keeping everything in the full nodes without
        // doing a distance calculation
        
        // ----------- FULL NODES -------------
        for(size_t i = 0; i < flist.length(); i++)
        {  
            idlist[idcount] = (uint32 )flist(i);
            idcount++;
        }
        // ----------- Partial Nodes ----------
        for(size_t i = 0; i < plist.length(); i++)
        {  
            idlist[idcount] = (uint32 )plist(i);
            idcount++;
        }


        // these are temporary vectors to hold matches to this point

        std::vector<int32_t> m1tmp;
        std::vector<int32_t> m2tmp;
        std::vector<double> d12tmp;

        std::vector<PAIR_INFO> pair_info;

        for (npy_intp j=0; j<nfound; j++) {

            int32_t leafid = idlist[j];

            // Make sure leaf is in list for ra2,dec2
            if ( leafid >= minid && leafid <= maxid) {

                int32_t leafbin = idlist[j] - minid;

                // Any found in this leaf?
                if ( htmrev2[leafbin] != htmrev2[leafbin+1] ) {

                    // Now loop over the sources
                    int32_t nLeafBin = htmrev2[leafbin+1] - htmrev2[leafbin];

                    for (int32_t ileaf=0; ileaf<nLeafBin;ileaf++) {

                        npy_intp i2 = htmrev2[ htmrev2[leafbin] + ileaf ];

                        // Returns distance in degrees
                        double dis = gcirc(ra1[i1], dec1[i1], ra2[i2], dec2[i2]);

                        // Turns out, this pushing is not a bottleneck!
                        // Time is negligible compared to the leaf finding
                        // and the gcirc.
                        if (dis <= rad) {
                            PAIR_INFO pi;
                            pi.i1 = i1;
                            pi.i2 = i2;
                            pi.d12 = dis;
                            pair_info.push_back(pi);
                        } // Within max distance 

                    } // loop over objects in leaf 

                } // any in leaf?

            } // leaf id in list 2?
        } // loop over leaves

        npy_intp nkeep = pair_info.size();
        if ( nkeep > 0 ) {

            // Sort the result by distance
            std::sort( pair_info.begin(), pair_info.end(), PAIR_INFO_ORDERING());

            if ((maxmatch > 0) ) {
                // setting maxmatch to zero is same as "keep all matches"
                if (nkeep > maxmatch) {
                    nkeep=maxmatch;
                }
            }
            for (npy_intp ci=0; ci<nkeep; ci++) {
                m1.push_back(pair_info[ci].i1);
                m2.push_back(pair_info[ci].i2);
                d12.push_back(pair_info[ci].d12);
            }
        }

    } // loop over list 1



    // This will hold the tuple of match1 and match2 and possibly
    // d12

    PyObject* output_tuple = PyTuple_New(3);


    npy_intp ntotal = m1.size();
    if (ntotal > 0) {
        // copy out data
        NumpyVector<npy_intp> m1out(ntotal);
        NumpyVector<npy_intp> m2out(ntotal);
        NumpyVector<double> d12out(ntotal);

        for (npy_intp i=0; i<ntotal; i++) {
            m1out[i] = m1[i];
            m2out[i] = m2[i];
            d12out[i] = d12[i];
        }

        PyTuple_SetItem(output_tuple, 0, m1out.getref());
        PyTuple_SetItem(output_tuple, 1, m2out.getref());
        PyTuple_SetItem(output_tuple, 2, d12out.getref());

    } else {
        Py_INCREF(Py_None);
        PyTuple_SetItem(output_tuple, 0, Py_None);
        Py_INCREF(Py_None);
        PyTuple_SetItem(output_tuple, 1, Py_None);
        Py_INCREF(Py_None);
        PyTuple_SetItem(output_tuple, 2, Py_None);
    }



    return output_tuple;


}



