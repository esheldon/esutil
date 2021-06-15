#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <math.h>
#include "htmc.h"
#include <algorithm> // for transform

#define NPY_PI 3.141592653589793238462643383279502884L
#define R2D (180.0/NPY_PI)
#define D2R (NPY_PI/180.0)

#if PY_MAJOR_VERSION >= 3
static int *init_numpy(void) {
    import_array();
    return NULL;
}
#else
static void init_numpy(void) {
    import_array();
}
#endif


/*
// alternative sphdist, might be a bit more accurate, a bit slower
void eq2xyz(long double ra, long double dec, long double* x, long double* y, long double* z) {
    // static const long double D2R = 0.0174532925199433;
    static const long double node = 1.6580627893946132;

    long double theta = ra * D2R - node;
    long double phi = dec * D2R;

    *x = cosl(theta)*cosl(phi);
    *y = sinl(theta)*cosl(phi);
    *z = sinl(phi);
}

// A couple of utility functions
// raturn great circle distance in degrees
long double sphdist(long double ra1, long double dec1, 
               long double ra2, long double dec2,
               bool degrees) {

    // static const long double D2R=0.0174532925199433;
    // static const long double R2D=57.29577951308232;

    if (ra1 == ra2 && dec1 == dec2) {
        return 0.0;
    }
    long double x1, y1, z1,
           x2, y2, z2;
    long double cosdis, dis;

    eq2xyz(ra1, dec1, &x1, &y1, &z1);
    eq2xyz(ra2, dec2, &x2, &y2, &z2);

    cosdis = x1*x2 + y1*y2 + z1*z2;

    if (cosdis < -1.0) {
        cosdis = -1.0;
    }
    if (cosdis > 1.0) {
        cosdis = 1.0;
    }

    dis = acosl(cosdis);

    if (degrees) {
        dis *= R2D;
    }

    return dis;
}
*/

// A couple of utility functions
// raturn great circle distance in degrees

double gcirc(double ra1, double dec1, 
             double ra2, double dec2,
             bool degrees)
{

    double sindec1, cosdec1, sindec2, cosdec2, 
           radiff, cosradiff, dis, cosdis; 

    if (ra1 == ra2 && dec1 == dec2) {
        return 0.0;
    }

    sindec1 = sin(dec1*D2R);
    cosdec1 = cos(dec1*D2R);

    sindec2 = sin(dec2*D2R);
    cosdec2 = cos(dec2*D2R);

    radiff = (ra1-ra2)*D2R;
    cosradiff = cos(radiff);

    cosdis = sindec1*sindec2 + cosdec1*cosdec2*cosradiff;

    if (cosdis < -1.0) cosdis=-1.0;
    if (cosdis >  1.0) cosdis= 1.0;

    dis = acos(cosdis);
    if (degrees) {
        dis *= R2D;
    }
    return( dis );

}


HTMC::HTMC(int depth) throw (const char *) {
    init(depth);
}

void HTMC::init(int depth) throw (const char *) {
    mDepth = depth;
    mHtmInterface.init(depth);

    init_numpy();
}

void HTMC::lookup_id(PyObject* ra_array, 
                     PyObject* dec_array,
                     PyObject* htm_ids_array) throw (const char* ) {

    npy_intp num = PyArray_SIZE(ra_array);

    for (npy_intp i=0; i<num; i++) {
        double    *raptr    = (double *)    PyArray_GETPTR1(ra_array, i);
        double    *decptr   = (double *)    PyArray_GETPTR1(dec_array, i);
        npy_int64 *idptr    = (npy_int64 *) PyArray_GETPTR1(htm_ids_array, i);

        npy_int64 id = (npy_int64) mHtmInterface.lookupID(*raptr, *decptr);

        *idptr = id;
    }

}

PyObject* HTMC::intersect(double ra, // all in degrees
                          double dec,
                          double radius, // degrees
                          int inclusive) throw (const char *) {

    // static const double D2R=0.0174532925199433;
    npy_intp nfound=0;

    // This is used in the basic calculations
    const SpatialIndex &index = mHtmInterface.index();

    double d = cos( radius*D2R );

    // Declare the domain and the lists
    SpatialDomain domain;    // initialize empty domain
    ValVec<uint64> plist, flist;	// List results

    // Find the triangles around this point
    domain.setRaDecD(ra,dec,d);
    domain.intersect(&index,plist,flist);

    // number of triangles found
    if (inclusive) {
        nfound = flist.length() + plist.length();
    } else {
        nfound = flist.length();
    }

    PyObject* idlist=PyArray_ZEROS(
                                   1,
                                   &nfound,
                                   NPY_INT64,
                                   0);

    npy_intp *idptr=NULL, id_index=0;

    // ----------- FULL NODES -------------
    for(size_t i = 0; i < flist.length(); i++)
    {  
        idptr = (npy_intp* ) PyArray_GETPTR1(idlist, id_index);
        *idptr = flist(i);

        id_index++;
    }
    if (inclusive) {
        // ----------- Partial Nodes ----------
        for(size_t i = 0; i < plist.length(); i++)
        {  
            idptr = (npy_intp* ) PyArray_GETPTR1(idlist, id_index);
            *idptr = plist(i);

            id_index++;
        }
    }

    return idlist;

}




PyObject* HTMC::cbincount(double rmin, // units of scale*angle in radians
                          double rmax, // units of scale*angle in radians
                          long nbin, 
                          PyObject* ra1_array, // all in degrees
                          PyObject* dec1_array,
                          PyObject* ra2_array, 
                          PyObject* dec2_array,
                          PyObject* htmrev2_array,
                          PyObject* minmax_ids_array,
                          PyObject* scale_array,
                          int verbose) throw (const char *) {

    double scale=1, logscale=0;

    double logrmin = log10(rmin);
    double logrmax = log10(rmax);

    npy_int64 minid = *(npy_int64* ) PyArray_GETPTR1(minmax_ids_array, 0);
    npy_int64 maxid = *(npy_int64* ) PyArray_GETPTR1(minmax_ids_array, 1);

    npy_intp n1 = PyArray_SIZE(ra1_array);

    npy_intp nscale=0;
    bool degrees = true;
    if (scale_array != Py_None) {
        degrees = false;
        nscale = PyArray_SIZE(scale_array);

        // we can just do this once
        if (nscale==1) {
            scale = *(double *) PyArray_GETPTR1(scale_array, 0);
            logscale = log10(scale);
        }
    }

    double log_binsize = (logrmax-logrmin)/nbin;
    if (log_binsize < 0) {
        throw("found log_binsize < 0");
    }


    // Output counts in bins
    npy_intp npnbin = nbin;
    PyObject* counts_array = PyArray_ZEROS(
        1,
        &npnbin,
        NPY_INT64,
        0
    );


    // This is used in the basic calculations
    const SpatialIndex &index = mHtmInterface.index();

    // static const double D2R=0.0174532925199433;
    int step=500;
    int linelen=70*step;
    npy_intp totcount=0;

    if (verbose) {
        std::cout<<"rmin: "<<rmin<<"\n";
        std::cout<<"rmax: "<<rmax<<"\n";
        std::cout<<"degrees?: "<<(degrees ? "True" : "False")<<"\n";
        std::cout<<"nbin: "<<nbin<<"\n";
        std::cout<<"logrmin: "<<logrmin<<"\n";
        std::cout<<"logrmax: "<<logrmax<<"\n";

        std::cout<<"log binsize: "<<log_binsize<<"\n";

        std::cout << "\n" <<
            "Each dot is " << step << " points" << std::endl;
    }

    for (npy_intp i1=0; i1<n1; i1++) {
        // Declare the domain and the lists
        SpatialDomain domain;    // initialize empty domain
        ValVec<uint64> plist, flist;	// List results

        // one for each point
        if (nscale > 1) {
            scale = *(double *) PyArray_GETPTR1(scale_array, i1);
            logscale = log10(scale);
        }

        // get actual max search radius in radians for this point
        double d=0;
        double maxangle = rmax/scale;
        if (degrees) { 
            d = cos( maxangle*D2R );
        } else {
            d = cos( maxangle );
        }

        // Find the triangles around this point
        double ra1  = *(double *) PyArray_GETPTR1(ra1_array,  i1);
        double dec1 = *(double *) PyArray_GETPTR1(dec1_array, i1);

        domain.setRaDecD(ra1,dec1,d);
        domain.intersect(&index,plist,flist);	  // intersect with list

        // number of triangles found
        npy_intp nfound = flist.length() + plist.length();
        std::vector<int64_t> idlist(nfound);
        npy_intp idcount=0;

        // ----------- FULL NODES -------------
        for(size_t i = 0; i < flist.length(); i++)
        {  
            idlist[idcount] = flist(i);
            idcount++;
        }
        // ----------- Partial Nodes ----------
        for(size_t i = 0; i < plist.length(); i++)
        {  
            idlist[idcount] = plist(i);
            idcount++;
        }

        for (npy_intp j=0; j<nfound; j++) {
            int64_t leafid = idlist[j];

            // Make sure leaf is in list for ra2,dec2
            if ( leafid >= minid && leafid <= maxid) {
                int64_t leafbin = idlist[j] - minid;

                // Any found in this leaf?
                npy_int64 hlo = *(npy_int64* ) PyArray_GETPTR1(htmrev2_array, leafbin);
                npy_int64 hhi = *(npy_int64* ) PyArray_GETPTR1(htmrev2_array, leafbin+1);

                if ( hlo != hhi) {

                    // Now loop over the sources in this leaf node
                    int64_t nLeafBin = hhi - hlo;

                    for (int64_t ileaf=0; ileaf<nLeafBin;ileaf++) {

                        npy_int64 index = hlo + ileaf;
                        npy_int64 i2 = *(npy_int64* ) PyArray_GETPTR1(htmrev2_array, index);

                        double ra2  = *(double *) PyArray_GETPTR1(ra2_array,  i2);
                        double dec2 = *(double *) PyArray_GETPTR1(dec2_array, i2);

                        // double dis = sphdist(ra1, dec1, ra2, dec2, degrees);
                        double dis = gcirc(ra1, dec1, ra2, dec2, degrees);
                        if (dis <= maxangle) {
                            double logr = logscale + log10(dis);

                            int radbin = (int) ( (logr-logrmin)/log_binsize );
                            if (radbin >=0 && radbin < nbin) {
                                npy_int64 *cptr = (npy_int64 *) PyArray_GETPTR1(counts_array, radbin);
                                *cptr += 1;
                                totcount+=1;
                            } // in one of our radial bins

                        } // Within max angle

                    } // loop over objects in leaf 
                } // points exist in this leafbin
            } // leafid in range of list 2
        } // loop over HTM leaves


        if (verbose) {
            if ( ( ((i1+1) % step) == 0 && (i1 > 0) ) 
                 || (i1 == (n1-1)) ) {
                std::cout<<".";
                if ( ((i1+1) % linelen) == 0 || (i1 == (n1-1)) ) {
                    std::cout<<"\n"<<(i1+1)<<"/"<<n1<<"  pair count: "<<totcount<<"\n";
                }
                fflush(stdout);
            }
        }

    } // loop over list 1

    if (verbose) {
        std::cout<<"\n";
        fflush(stdout);
    }

    return counts_array;
}

Matcher::Matcher(int depth,
                 PyObject* ra_input,
                 PyObject* dec_input) throw (const char *)
{
    init_numpy();

    this->depth = depth;
    this->htm_interface.init(depth);

    this->ra = ra_input;
    this->dec = dec_input;

    Py_INCREF(ra_input);
    Py_INCREF(dec_input);

    this->npoints = PyArray_SIZE(this->ra);

    init_hmap();
}
void Matcher::init_hmap(void)
{
    std::map<int64_t,std::vector<int64_t> >::iterator iter;
    int64_t htmid=0;
    for (npy_intp i=0; i<this->npoints; i++) {

        double ra = *(double *) PyArray_GETPTR1(this->ra, i);
        double dec = *(double *) PyArray_GETPTR1(this->dec, i);

        htmid = htm_interface.lookupID(ra, dec);

        iter=hmap.find(htmid);

        if (iter==hmap.end()) {
            std::vector<int64_t> v;
            v.push_back(i);
            hmap[htmid] = v;
        } else {
            iter->second.push_back(i);
        }
    }
}

PyObject* Matcher::match(PyObject* ra_array, // all in degrees
                         PyObject* dec_array,
                         PyObject* radius_array, // degrees
                         long maxmatch,
                         const char* filename) throw (const char *) {

    std::map<int64_t,std::vector<int64_t> >::iterator iter;

    // no copies made if already double vectors

    npy_intp nrad = PyArray_SIZE(radius_array);

    // These will temporarily hold the results
    std::vector<int64_t> m1;
    std::vector<int64_t> m2;
    std::vector<double> d12;

    // total number of pairs
    npy_intp ntotal = 0;

    FILE* fptr=NULL;

    std::string fname=filename;

    if (fname!= "") {
        fptr = fopen(fname.c_str(), "w");
        if (fptr==NULL) 
        {
            std::stringstream err;
            err<<"Cannot open file: "<<fname<<" : "<<strerror(errno);
            throw err.str().c_str();
        }
    }

    // static const double D2R=0.0174532925199433;

    // This is used in the basic calculations
    const SpatialIndex &index = this->htm_interface.index();


    double rad=0, d=0;
    if (nrad == 1) {
        rad = *(double *) PyArray_GETPTR1(radius_array, 0);
        d = cos( rad*D2R );
    }

    npy_intp ninput = PyArray_SIZE(ra_array);

    for (npy_intp i_input=0; i_input<ninput; i_input++) {
        // Declare the domain and the lists
        SpatialDomain domain;    // initialize empty domain
        ValVec<uint64> plist, flist;	// List results

        if (nrad > 1) {
            rad = *(double *) PyArray_GETPTR1(radius_array, i_input);
            d = cos( rad*D2R );
        }

        // Find the triangles around this point
        double ra  = *(double *) PyArray_GETPTR1(ra_array,  i_input);
        double dec = *(double *) PyArray_GETPTR1(dec_array, i_input);

        domain.setRaDecD(ra,dec,d);
        domain.intersect(&index,plist,flist);	  // intersect with list


        // number of triangles found
        npy_intp nfound = flist.length() + plist.length();
        std::vector<int64_t> idlist(nfound);
        npy_intp idcount=0;

        // We could speed this up when no distance is needed by
        // just keeping everything in the full nodes without
        // doing a distance calculation

        // ----------- FULL NODES -------------
        for(size_t i = 0; i < flist.length(); i++)
        {  
            idlist[idcount] = flist(i);
            idcount++;
        }
        // ----------- Partial Nodes ----------
        for(size_t i = 0; i < plist.length(); i++)
        {  
            idlist[idcount] = plist(i);
            idcount++;
        }


        // these are temporary vectors to hold matches to this point

        std::vector<PAIR_INFO> pair_info;

        for (npy_intp j=0; j<nfound; j++) {

            int64_t htmid = idlist[j];

            iter=this->hmap.find(htmid);
            if (iter != this->hmap.end()) {

                int64_t nleaf =iter->second.size();
                for (int64_t ileaf=0; ileaf<nleaf; ileaf++) {
                    int64_t i_this = iter->second[ileaf];

                    // Returns distance in degrees
                    double tra  = *(double *) PyArray_GETPTR1(this->ra, i_this);
                    double tdec = *(double *) PyArray_GETPTR1(this->dec, i_this);

                    // double dis = sphdist(ra, dec, tra, tdec, true);
                    double dis = gcirc(ra, dec, tra, tdec, true);

                    // Turns out, this pushing is not a bottleneck!
                    // Time is negligible compared to the leaf finding
                    // and the distance calculations
                    if (dis <= rad) {
                        PAIR_INFO pi;
                        pi.i1 = i_input;
                        pi.i2 = i_this;
                        pi.d12 = dis;
                        pair_info.push_back(pi);
                    } // Within max distance 

                } // loop over objects in leaf 

            } // any in leaf?

        } // loop over input ra,dec

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
                if (fptr) {
                    fprintf(fptr, "%ld %ld %.16g\n", 
                            pair_info[ci].i1,
                            pair_info[ci].i2,
                            pair_info[ci].d12);
                } else {
                    m1.push_back(pair_info[ci].i1);
                    m2.push_back(pair_info[ci].i2);
                    d12.push_back(pair_info[ci].d12);
                }
                // keep track of the total number actually saved or written
                ntotal += 1;
            }
        }

    } // loop over list 1


    // This will hold the tuple of match1 and match2 and possibly
    // d12


    if (fptr == NULL) {

        // If we are not writing to a file, we *always* return arrays, even if
        // they are zero size

        PyObject* output_tuple = PyTuple_New(3);

        PyObject* m1out=PyArray_ZEROS(1, &ntotal, NPY_INT64, 0);
        PyObject* m2out=PyArray_ZEROS(1, &ntotal, NPY_INT64, 0);
        PyObject* d12out=PyArray_ZEROS(1, &ntotal, NPY_FLOAT64, 0);

        for (npy_intp i=0; i<ntotal; i++) {
            npy_int64 *m1ptr  = (npy_int64* ) PyArray_GETPTR1(m1out, i);
            npy_int64 *m2ptr  = (npy_int64* ) PyArray_GETPTR1(m2out, i);
            double    *d12ptr = (npy_float64* ) PyArray_GETPTR1(d12out, i);

            *m1ptr = m1[i];
            *m2ptr = m2[i];
            *d12ptr = d12[i];
        }

        PyTuple_SetItem(output_tuple, 0, m1out);
        PyTuple_SetItem(output_tuple, 1, m2out);
        PyTuple_SetItem(output_tuple, 2, d12out);

        return output_tuple;

    } else {
        fflush(fptr);
        fclose(fptr);
        return PyLong_FromLongLong((long long) ntotal);
    }




} // Matcher::match

