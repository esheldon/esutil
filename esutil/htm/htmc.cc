#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <math.h>
#include "htmc.h"
#include "NumpyVector.h"
#include <algorithm> // for transform


/*
#if PY_MAJOR_VERSION >= 3
static int *init_numpy(void) {
#else
static void init_numpy(void) {
#endif
	import_array();
}
*/


// A couple of utility functions
// raturn great circle distance in degrees
double gcirc(
        double ra1, double dec1, 
        double ra2, double dec2,
        bool degrees)

{

	double sindec1, cosdec1, sindec2, cosdec2, 
	radiff, cosradiff, dis, cosdis; 

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

	dis = acos(cosdis);
	if (degrees) {
		dis /= D2R;
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

void HTMC::lookup_id(
		PyObject* ra_array, 
		PyObject* dec_array,
        PyObject* htm_ids_array
) throw (const char* ) {

    npy_intp num = PyArray_SIZE(ra_array);

	for (npy_intp i=0; i<num; i++) {
        double    *raptr    = (double *)    PyArray_GETPTR1(ra_array, i);
        double    *decptr   = (double *)    PyArray_GETPTR1(dec_array, i);
        npy_int64 *idptr    = (npy_int64 *) PyArray_GETPTR1(htm_ids_array, i);

		npy_int64 id = (npy_int64) mHtmInterface.lookupID(*raptr, *decptr);

        *idptr = id;
	}

}

PyObject* HTMC::intersect(
		double ra, // all in degrees
        double dec,
		double radius, // degrees
        int inclusive
        ) throw (const char *) {

	static const double D2R=0.0174532925199433;
    npy_intp nfound;

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

	//NumpyVector<int64_t> idlist(nfound);
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

    /*
	PyObject* idlist_pyobj = idlist.getref();
	return idlist_pyobj;
    */
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
        PyObject* maxmatch_obj,
        const char* filename) throw (const char *) {

	// no copies made if already double vectors
	NumpyVector<double> radius(radius_array);
	npy_intp nrad = radius.size();

	NumpyVector<double> ra1(ra1_array);
	NumpyVector<double> dec1(dec1_array);
	NumpyVector<double> ra2(ra2_array);
	NumpyVector<double> dec2(dec2_array);

	NumpyVector<int64_t> htmrev2(htmrev2_array);

	// get these as numpyvectors even though they are only length 1
	// because it does a good job with conversions
	NumpyVector<int64_t> minidVec(minid_obj);
	NumpyVector<int64_t> maxidVec(maxid_obj);
	NumpyVector<int64_t> maxmatchVec(maxmatch_obj);

	int64_t minid = minidVec[0];
	int64_t maxid = maxidVec[0];
	int64_t maxmatch = maxmatchVec[0];


	// These will temporarily hold the results
	std::vector<int64_t> m1;
	std::vector<int64_t> m2;
	std::vector<double> d12;

	// total number of pairs
	int64_t ntotal = 0;

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

			int64_t leafid = idlist[j];

			// Make sure leaf is in list for ra2,dec2
			if ( leafid >= minid && leafid <= maxid) {

				int64_t leafbin = idlist[j] - minid;

				// Any found in this leaf?
				if ( htmrev2[leafbin] != htmrev2[leafbin+1] ) {

					// Now loop over the sources
					int64_t nLeafBin = htmrev2[leafbin+1] - htmrev2[leafbin];

					for (int64_t ileaf=0; ileaf<nLeafBin;ileaf++) {

						npy_intp i2 = htmrev2[ htmrev2[leafbin] + ileaf ];

						// Returns distance in degrees
						double dis = gcirc(ra1[i1], dec1[i1], ra2[i2], dec2[i2],true);

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

		NumpyVector<int64_t> m1out(ntotal);
		NumpyVector<int64_t> m2out(ntotal);
		NumpyVector<double> d12out(ntotal);

		for (npy_intp i=0; i<ntotal; i++) {
			m1out[i] = m1[i];
			m2out[i] = m2[i];
			d12out[i] = d12[i];
		}

		PyTuple_SetItem(output_tuple, 0, m1out.getref());
		PyTuple_SetItem(output_tuple, 1, m2out.getref());
		PyTuple_SetItem(output_tuple, 2, d12out.getref());

        return output_tuple;

	} else {
        fflush(fptr);
        fclose(fptr);
        return PyLong_FromLongLong((long long) ntotal);
	}




}




PyObject* HTMC::cbincount(
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
		PyObject* scale_object,
        int verbose) throw (const char *) {


	double scale=1, logscale=0;

	// get these as numpyvectors even though they are only length 1
	// because it does a good job with conversions
	NumpyVector<double> rminvec(rmin_object);
	NumpyVector<double> rmaxvec(rmax_object);
	NumpyVector<int64_t> nbinvec(nbin_object);

	double rmin=rminvec[0];
	double rmax=rmaxvec[0];
	int64_t nbin=nbinvec[0];


	double logrmin = log10(rmin);
	double logrmax = log10(rmax);


	NumpyVector<double> ra1(ra1_array);
	NumpyVector<double> dec1(dec1_array);
	NumpyVector<double> ra2(ra2_array);
	NumpyVector<double> dec2(dec2_array);
	NumpyVector<int64_t> htmrev2(htmrev2_array);

	NumpyVector<int64_t> minid_array(minid_obj);
	NumpyVector<int64_t> maxid_array(maxid_obj);
	int64_t minid = minid_array[0];
	int64_t maxid = maxid_array[0];

	// ensure scale is an array if input
	NumpyVector<double> scale_array;
	npy_intp nscale=0;
	bool degrees = true;
	if (scale_object != NULL && scale_object != Py_None) {
		scale_array.init(scale_object);
		nscale = scale_array.size();
		degrees = false;
		if (nscale > 1) {
			if (ra1.size()!=scale_array.size() 
					|| dec1.size() !=scale_array.size()) {
				throw("scale must be scalar or same size as ra1/dec1");
			}
		} else {
			scale = scale_array[0];
			logscale = log10(scale);
		}
	}


	double log_binsize = (logrmax-logrmin)/nbin;
	if (log_binsize < 0) {
		throw("found log_binsize < 0");
	}


	// Output counts in bins
	NumpyVector<int64_t> counts(nbin);

	// This is used in the basic calculations
	const SpatialIndex &index = mHtmInterface.index();

	static const double D2R=0.0174532925199433;
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
        std::cout<<"len(scale_array) = "<<scale_array.size()<<"\n";

        std::cout << "\n" <<
            "Each dot is " << step << " points" << std::endl;
    }

	npy_intp n1 = ra1.size();
	for (npy_intp i1=0; i1<n1; i1++) {
		// Declare the domain and the lists
		SpatialDomain domain;    // initialize empty domain
		ValVec<uint64> plist, flist;	// List results

		if (nscale > 1) {
			scale = scale_array[i1];
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
		domain.setRaDecD(ra1[i1],dec1[i1],d); //put in ra,dec,d E.S.S.
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
				if ( htmrev2[leafbin] != htmrev2[leafbin+1] ) {

					// Now loop over the sources in this leaf node
					int64_t nLeafBin = htmrev2[leafbin+1] - htmrev2[leafbin];

					for (int64_t ileaf=0; ileaf<nLeafBin;ileaf++) {

						npy_intp i2 = htmrev2[ htmrev2[leafbin] + ileaf ];

						double dis = gcirc(ra1[i1], dec1[i1], ra2[i2], dec2[i2],degrees);
						if (dis <= maxangle) {
							double logr = logscale + log10(dis);

							int radbin = (int) ( (logr-logrmin)/log_binsize );
							if (radbin >=0 && radbin < nbin) {
								counts[radbin] += 1;
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

	PyObject* countsPyObject= counts.getref();
	return countsPyObject;
}

Matcher::Matcher(int depth,
                 PyObject* ra_input,
                 PyObject* dec_input) throw (const char *)
{
    this->depth = depth;
    this->htm_interface.init(depth);

	// wrap the input ra,dec objects, making sure they are doubles
	// no copy is made if the are already double arrays
	this->ra.init(ra_input);
	this->dec.init(dec_input);
    
    init_hmap();
    init_numpy();
}
void Matcher::init_hmap(void)
{
    std::map<int64_t,std::vector<int64_t> >::iterator iter;
    int64_t htmid=0;
    int64_t num=ra.size();
    for (int64_t i=0; i<num; i++) {
        htmid = htm_interface.lookupID(ra[i], dec[i]);

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

PyObject* Matcher::match(
		PyObject* ra_array, // all in degrees
        PyObject* dec_array,
		PyObject* radius_array, // degrees
        PyObject* maxmatch_obj,
        const char* filename) throw (const char *) {

    std::map<int64_t,std::vector<int64_t> >::iterator iter;

	// no copies made if already double vectors

	NumpyVector<double> ra(ra_array);
	NumpyVector<double> dec(dec_array);

	NumpyVector<double> radius(radius_array);
	npy_intp nrad = radius.size();

	// get as NumpyVectors even though they are only length 1
	// because it does a good job with conversions
	NumpyVector<int64_t> maxmatchVec(maxmatch_obj);
	int64_t maxmatch = maxmatchVec[0];

	// These will temporarily hold the results
	std::vector<int64_t> m1;
	std::vector<int64_t> m2;
	std::vector<double> d12;

	// total number of pairs
	int64_t ntotal = 0;

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

	static const double
		D2R=0.0174532925199433;

	// This is used in the basic calculations
	const SpatialIndex &index = this->htm_interface.index();


	double rad=0, d=0;
	if (nrad == 1) {
		rad = radius[0];
		d = cos( rad*D2R );
	}

	npy_intp ninput = ra.size();
	for (npy_intp i_input=0; i_input<ninput; i_input++) {
		// Declare the domain and the lists
		SpatialDomain domain;    // initialize empty domain
		ValVec<uint64> plist, flist;	// List results

		if (nrad > 1) {
			rad = radius[i_input];
			d = cos( rad*D2R );
		}

		// Find the triangles around this point
		domain.setRaDecD(ra[i_input],dec[i_input],d); //put in ra,dec,d E.S.S.
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
                    double dis = gcirc(ra[i_input],
                                       dec[i_input],
                                       this->ra[i_this],
                                       this->dec[i_this],true);

                    // Turns out, this pushing is not a bottleneck!
                    // Time is negligible compared to the leaf finding
                    // and the gcirc.
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

		NumpyVector<int64_t> m1out(ntotal);
		NumpyVector<int64_t> m2out(ntotal);
		NumpyVector<double> d12out(ntotal);

		for (npy_intp i=0; i<ntotal; i++) {
			m1out[i] = m1[i];
			m2out[i] = m2[i];
			d12out[i] = d12[i];
		}

		PyTuple_SetItem(output_tuple, 0, m1out.getref());
		PyTuple_SetItem(output_tuple, 1, m2out.getref());
		PyTuple_SetItem(output_tuple, 2, d12out.getref());

        return output_tuple;

	} else {
        fflush(fptr);
        fclose(fptr);
        return PyLong_FromLongLong((long long) ntotal);
	}




} // Matcher::match

