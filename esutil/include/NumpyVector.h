/*
 * NumpyVector.h
 *
 * This is simple wrapper class for 1-d and scalar numpy arrays.  Only
 * numerical data are supported at this time.  It should be straightforward
 * to expand this to higher dimensions.  It is *not* easy to support 
 * std::string, but char* type strings may be easier..
 *
 * This is a header-only template class.  Simply include it and use.
 *
 * Examples:
 * #include "NumpyVector.h"
 *
 *    // creating a new int vector of size 100.  Internally this is a
 *    // PyArrayObject
 *    NumpyVector<int> vec(100);
 *
 *    // Get some info
 *
 *    // the string representation of the type, from <typeinfo>
 *    cout<<"type name is: "<<vec.type_name()<<"\n";
 *    // The numby type number
 *    int type_num = vec.type_num();
 *    // the number of elements in the vector
 *    npy_intp nel = vec.size();
 *    // the stride of the array
 *    npy_inpt stride = vec.stride();
 *
 *
 *
 *    // double vector from an input python object (PyObject*).  
 *    // Could be array, python  sequence, or scalar.  The result will only
 *    // be copied if the obj is not already a double vector in native byte
 *    // ordering.
 *
 *    NumpyVector<double> vec(obj);
 *
 *
 *    // Access data in a way that is aware of strides and is type-safe.  
 *    // No bounds checking is done
 *    vec[35] = 22.2;
 *    double val = vec[35];
 *
 *    // Loop over data in a simple, type-safe, and stride-aware way.
 *    for (npy_intp i=0; i<vec.size(); i++) {
 *        val = vec[i];
 *        vec[i] = val*26;
 *    }
 *
 *    // get pointer to the data.  Careful when using if data are strided
 *    double* p = vec.ptr();
 *
 *    // if you *know* the data are contiguous, this is an easy and fast way
 *    // to access the data.
 *    for (npy_intp i=0; i<vec.size(); i++) {
 *        val = *p;
 *        ++p;
 *    }
 *
 *    // This is the fastest way to loop using strides.
 *
 *    double* p = vec.ptr();
 *    npy_intp stride = vec.stride();  // zero for scalars
 *    for (npy_intp i=0; i<vec.size(); i++) {
 *        val = *p;
 *        p = p + stride;
 *    }
 *
 *
 *    // get pointer to particular location.  This is stride-aware.
 *    double* p = vec.ptr(22);
 *
 *    // get a reference for returning to python.  Reference counting is
 *    // done correctly
 *
 *    PyObject* output = vec.getref();
 *    return output;
 *
 *   
 */


#ifndef _numpy_vector_h
#define _numpy_vector_h

#include <Python.h>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <map>
//#include "TypeInfo.h"
#include "numpy/arrayobject.h"



/* A helper class to convert C++ types to NumPy type numbers */
class NumpyTypeInfo {
    public:
        NumpyTypeInfo();
        ~NumpyTypeInfo() {};
        void init();		

		int getid(const char* name) {

			if (numpy_id_map.count(name) > 0) {
				return numpy_id_map[name];
			} else {
				std::stringstream err;
				err<<"unsupported type: "<<name<<"\n";
				throw err.str().c_str();
			}
		}
        // public, static member data.  Static members can be
        // created once and shared by all instances.
		static std::map<const char*,int> numpy_id_map;
};
// this static data must be re-declared here
std::map<const char*,int> NumpyTypeInfo::numpy_id_map;

NumpyTypeInfo::NumpyTypeInfo() {
	// DONT FORGET THIS!!!!
    import_array();
    init();
}

void NumpyTypeInfo::init() {
    // static class members, only create once and shared by all
    // instances
    if (numpy_id_map.empty()) {

        const char* tname;

		// the following also cover the standard types as they are just a
		// re-naming of them to platform independent, length specified type
		// names

		// in other words, these are also covered by the npy_* types
		//
        //   char
        //   unsigned char
        //   short
        //   unsigned short
        //   int
        //   unsigned int
        //   long
        //   unsigned long
        //   long long
        //   unsigned long long
        //   float
        //   double
		//
		//   This means you can do this and it will work
		//
		//       name = typeid(int).name();
		//       id = info.getid(name);

		//   even though we didn't explicitly define that below.  It may not be
		//   the same as npy_int32 because the size of int is platform
		//   dependent.
        
        tname = typeid(npy_int8).name();
        numpy_id_map[tname] = NPY_INT8;
        tname = typeid(npy_uint8).name();
        numpy_id_map[tname] = NPY_UINT8;


        tname = typeid(npy_int16).name();
        numpy_id_map[tname] = NPY_INT16;
        tname = typeid(npy_uint16).name();
        numpy_id_map[tname] = NPY_UINT16;


        tname = typeid(npy_int32).name();
        numpy_id_map[tname] = NPY_INT32;
        tname = typeid(npy_uint32).name();
        numpy_id_map[tname] = NPY_UINT32;


        tname = typeid(npy_int64).name();
        numpy_id_map[tname] = NPY_INT64;
        tname = typeid(npy_uint64).name();
        numpy_id_map[tname] = NPY_UINT64;


        tname = typeid(npy_float32).name();
        numpy_id_map[tname] = NPY_FLOAT32;

        tname = typeid(npy_float64).name();
        numpy_id_map[tname] = NPY_FLOAT64;

    }
}



template <class T> class NumpyVector {
	public:

        // simple constructor
		NumpyVector() throw (const char *);

        // destructor.  Always decref the array
		~NumpyVector() {
			Py_XDECREF(mArray);
		};

		// Construct from existing python object.  See init(PyObject* obj)
		NumpyVector(PyObject* obj) throw (const char *);

		// Construct a new array with the given length
		NumpyVector(npy_intp size) throw (const char *);



        // Initialize from an input python object, converting to an array of
        // the right type and native byte order if necessary. If already the
        // right type, etc. then no copy is made.
        //
        // If the data are already an array, must be zero or 1-dimensional.
        //
        // This can be called at *any time* and any existing data will
        // be released
        
        void init(PyObject* obj)  throw (const char *);

		// Initialize from scratch based on size and typenum
        //
        // This can be called at *any time* and any existing data will
        // be released
		
        void init(npy_intp size)  throw (const char *);


        // Random access to the underlying data at the specified location.
        // Returns a writable/readable reference. No bounds checking are 
        // applied.
        //   TODO:  deal properly with const correctness...ugh
        
        T& operator[] (npy_intp index) {
            if (mArray == NULL) {
             throw "Error: attempt to get pointer from an uninitialized array";
            }

            T& ref= *(T* ) PyArray_GetPtr((PyArrayObject*) mArray, &index);
            return ref;
        };


        // Get a pointer to the data.
		T* ptr() throw (const char *);

        // Get a pointer to the data at the indicated location
        // Strides are properly accounted for.  No bounds checking
        // are performed.
		T* ptr(npy_intp index) throw (const char *);



        // Get a reference the underlying python object and incref the object.
        // This is useful if you want to get a PyObject* that will be returned
        // to the outside world. The internal version will be decrefed when the
        // object is destructed or goes out of scope, so reference counting
        // is correct..

		PyObject* getref() throw (const char *);




        // get the type name.  Equivalent to typeid(T).name()
		const char* type_name() {
			return mTypeName;
		}
        // Return the numpy type number
		int type_num() {
			return mTypeNum;
		}
        // Return the number of elements in the data.
		npy_intp size() {
			return mSize;
		}
        // stride of the data.
        npy_intp stride() {
            return mStride;
        }

	
	private:

        // private method to initialize type number and name
        void SetType() throw (const char* );

		const char* mTypeName;
		int mTypeNum;
		npy_intp mSize;
        npy_intp mNdim; // should be 1 or 0
        npy_intp mStride;

		PyObject* mArray;

		NumpyTypeInfo mNumpyTypeInfo;
};

template <class T>
NumpyVector<T>::NumpyVector()  throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

	mNumpyTypeInfo.init();

	// don't forget to initialize
	mArray = NULL;
	mSize=0;
    mNdim=0;
    mStride=0;

    // Initialize internal type info
	SetType();
}


template <class T>
NumpyVector<T>::NumpyVector(PyObject* obj)  throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

	mNumpyTypeInfo.init();

	// don't forget to initialize
	mArray = NULL;
	mSize=0;
    mNdim=0;
    mStride=0;

    // Initialize internal type info
	SetType();
    
	// Get the data.  This may or may not make a copy.
	init(obj);
}


// Create given the length and typenum
template <class T>
NumpyVector<T>::NumpyVector(npy_intp size) throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

	mNumpyTypeInfo.init();

	// don't forget to initialize
	mArray = NULL;
	mSize=0;
    mNdim=0;
    mStride=0;

    // Initialize internal type info
    SetType();

    // create a new array from the size and type info
	init(size);
}

template <class T>
void NumpyVector<T>::init(PyObject* obj)  throw (const char *) {

	// clear any existing array
	Py_XDECREF(mArray);
	mSize=0;

	if (obj == NULL || obj == Py_None) {
		throw "cannot convert the input object to an array: is NULL or None";
	}


    // Is the input object already an array?
    if (PyArray_Check(obj)) {

        // If it is the right type, then just check it is not byteswapped
        // we will have to decref tmp if a copy is made.

        if (1 < PyArray_NDIM(obj)) {
            throw "Input array dimensions must be <= 1";
        }

        PyArray_Descr* descr = PyArray_DESCR(obj);

        if (descr->type_num == mTypeNum) {
            // it is the right type, just make sure it isn't byteswapped.
            if (PyArray_ISNOTSWAPPED(obj)) {
                // We are set!  Just copy the reference.
                mArray = obj;
                Py_INCREF(obj);
            }
        } else {
            // Either it is not the right type or it is byteswapped.  So we
            // need to make a copy.
            mArray = PyArray_Cast((PyArrayObject* ) obj, mTypeNum);

            if (mArray == NULL) {
                // this causes a segfault, don't do it
                //Py_XDECREF(descr);
                std::stringstream err;
                err<<"Cold not cast from type "<<descr->type_num
                    <<" to type "<<mTypeNum;
                throw err.str().c_str();
            }
        }
    
    } else {
        // This is not a PyArray, we need to do a more complex conversion

        // can be scalar, but not higher dimensional than 1
        int min_depth=0, max_depth=1;

        // require the array is in native byte order
        int requirements = NPY_NOTSWAPPED;


        PyArray_Descr* descr=NULL;
        descr = PyArray_DescrNewFromType(mTypeNum);

        if (descr == NULL) {
            throw "could not create array descriptor";
        }
        // This will steal a reference to descr, so we don't need to decref
        // descr as long as we decref the array!
        mArray = PyArray_CheckFromAny(
                obj, descr, min_depth, max_depth, requirements, NULL);

        /*
        // We MUST decref this no matter what
        mArray = PyArray_FROM_OTF(obj, mTypeNum, requirements);
        */
        if (mArray == NULL) {
            // this causes a segfault, don't do it
            //Py_XDECREF(descr);
            throw "Could not get input as array";
        }
    }

    // set the size
	mSize = PyArray_SIZE(mArray);

    // dimensions and stride
    mNdim = PyArray_NDIM(mArray);
    if (mNdim == 0) {
        mStride = 0;
    } else {
        mStride = PyArray_STRIDE(mArray, 0);
    }

}

template <class T>
void NumpyVector<T>::init(npy_intp size)  throw (const char *) {

	// clear any existing array
	Py_XDECREF(mArray);
	mSize=0;

	if (size < 1)  {
		throw "size must be >= 1";
	}

	// Create output flags array
	int ndim=1;
	mArray = PyArray_ZEROS(
			ndim, 
			&size,
            mTypeNum,
			NPY_FALSE);

	if (mArray ==NULL) {
		throw "Could not allocate array";
	}

	mSize = PyArray_SIZE(mArray);
    // dimensions and stride
    mNdim = ndim;
    mStride = PyArray_STRIDE(mArray, 0);

}


// Get a reference the object.  incref the object.
// This is useful if you want to get a PyObject* that will be returned
// to the outside world
template <class T>
PyObject* NumpyVector<T>::getref() throw (const char *) {
	Py_XINCREF(mArray);
	return mArray;
}


template <class T> 
T* NumpyVector<T>::ptr() throw (const char *) {
	if (mArray == NULL) {
		throw "Error: attempt to get pointer from an uninitialized array";
	}

	npy_intp index=0;
	return (T* ) PyArray_GetPtr((PyArrayObject*) mArray, &index);
}

template <class T>
T* NumpyVector<T>::ptr(npy_intp index) throw (const char *) {
	if (mArray == NULL) {
		throw "Error: attempt to get pointer from an uninitialized array";
	}

	return PyArray_GetPtr((PyArrayObject*) mArray, &index);
}


template <class T>
void NumpyVector<T>::SetType() throw (const char *) {

    const char *tname = typeid(T).name();

	mTypeNum = mNumpyTypeInfo.getid(tname);
	mTypeName = tname;

	/*
	if (mNumpyTypeInfo.id.count(tname) > 0) {
        mTypeName = tname;
        mTypeNum = mNumpyTypeInfo.id[tname];
    } else {
        std::stringstream err;
        err<<"unsupported type: "<<mTypeName<<"\n";
        throw err.str().c_str();
    }
	*/
}





#endif
