/*
   NumpyVector.h

   This is simple wrapper class for 1-d and scalar numpy arrays.  Only
   numerical data are supported at this time.  It should be straightforward
   to expand this to higher dimensions.  It is *not* easy to support 
   std::string, but char* type strings may be easier..

   This is a header-only template class.  Simply include it and use.

   Examples:
      #include "NumpyVector.h"

      // creating a new int vector of size 100.  Internally this is a
      // PyArrayObject
      NumpyVector<int> vec(100);



      // Get some info about the array

      // the string representation of the type, from <typeinfo>
      cout<<"type name is: "<<vec.type_name()<<"\n";

      // The numpy type number
      int type_num = vec.type_num();

      // the number of elements in the vector
      npy_intp nel = vec.size();

      // the stride of the array
      npy_inpt stride = vec.stride();



      // double vector from an input python object (PyObject*).  
      // Could be array, python  sequence, or scalar.  The result will only
      // be copied if the obj is not already a double vector in native byte
      // ordering.

      NumpyVector<double> vec(obj);


      // Access data in a way that is aware of strides and is type-safe.  
      // No bounds checking is done
      vec[35] = 22.2;
      double val = vec[35];

      // Loop over data in a simple, type-safe, and stride-aware way.
      for (npy_intp i=0; i<vec.size(); i++) {
          val = vec[i];
          vec[i] = val*26;
      }

      // using an iterator.  This is not faster than the above.
      for (NumpyVector<npy_double>::iterator it=dvec.begin(); 
              it != dvec.end(); it++) {
          double val=*it;
      }

      // This is the fastest way to loop using strides.  use char* to avoid
      // warnings from g++

      char* p = (char*) vec.void_ptr();
      npy_intp stride = vec.stride();  // zero for scalars
      for (npy_intp i=0; i<vec.size(); i++) {
          val = *(double *) *p;
          p = p + stride;
      }


      // get pointer to particular location.  This is stride-aware.  Do *NOT*
      // perform pointer arithmetic with this pointer unless you know the
      // stride is equal to the element size (e.g. 8 for double).  See example
      // above for proper way to do pointer arithmetic

      double* p = vec.ptr();
      double* p = vec.ptr(22);


      // get a reference for returning to python.  Reference counting is
      // done correctly

      PyObject* output = vec.getref();
      return output;


 */


#ifndef _numpy_vector_h
#define _numpy_vector_h

#include <Python.h>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <map>
#include <stdint.h>
#include <iterator>
#include "numpy/arrayobject.h"



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

        void* void_ptr() throw (const char*) {
            if (mArray == NULL) {
                throw "Error: attempt to get pointer from an uninitialized array";
            }

            npy_intp index=0;
            return PyArray_GetPtr((PyArrayObject*) mArray, &index);
        }


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


        // Interestingly, the iterator is actually slower than just
        // subscripting using brackets []
        class iterator : public std::iterator<std::forward_iterator_tag, T> {
            // use char* to avoid warnings from g++ about void* arithmetic
            char* _ptr;
            int _stride;

            public:
                iterator() : _ptr(NULL), _stride(0) {}
                iterator(char* x, int stride) :_ptr(x), _stride(stride) {}

                iterator(const iterator& mit) : _ptr(mit._ptr), _stride(mit._stride) {}

                iterator& operator++() {
                    _ptr += _stride;
                    return *this;
                }
                iterator operator++(int) {
                    iterator tmp(*this); 
                    operator++(); 
                    return tmp;
                }

                bool operator==(const iterator& rhs) {return _ptr==rhs._ptr;}
                bool operator!=(const iterator& rhs) {return _ptr!=rhs._ptr;}
                T& operator*() {return *(T*) _ptr;}
        };

        iterator begin() {
            char* tptr = (char*) this->void_ptr();
            iterator tmp(tptr, mStride);
            return tmp;
        }
        iterator end() {
            // point just past last element, accounting for stride
            char* tptr = (char*) this->void_ptr();
            iterator tmp(tptr + mSize*mStride, mStride);
            return tmp;
        }


	
	private:

        // This fills in the static type map if it doesn't exist
        void init_type_info();

        // private method to initialize type number and name
        void set_type() throw (const char* );

		const char* mTypeName;
		int mTypeNum;
		npy_intp mSize;
        npy_intp mNdim; // should be 1 or 0
        npy_intp mStride;

		PyObject* mArray;

		static std::map<const char*,int> mNumpyIdMap;
};

// this static data must be re-declared here
template <class T>
std::map<const char*,int> NumpyVector<T>::mNumpyIdMap;

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


template <class T>
NumpyVector<T>::NumpyVector()  throw (const char *) {
    init_numpy();

    init_type_info();

	// don't forget to initialize
	mArray = NULL;
	mSize=0;
    mNdim=0;
    mStride=0;

    // Initialize internal type info
	set_type();
}


template <class T>
NumpyVector<T>::NumpyVector(PyObject* obj)  throw (const char *) {
    init_numpy();

    init_type_info();

	// don't forget to initialize
	mArray = NULL;
	mSize=0;
    mNdim=0;
    mStride=0;

    // Initialize internal type info
	set_type();
    
	// Get the data.  This may or may not make a copy.
	init(obj);
}


// Create given the length and typenum
template <class T>
NumpyVector<T>::NumpyVector(npy_intp size) throw (const char *) {
    init_numpy();

    init_type_info();

	// don't forget to initialize
	mArray = NULL;
	mSize=0;
    mNdim=0;
    mStride=0;

    // Initialize internal type info
    set_type();

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

        if (descr->type_num == mTypeNum && PyArray_ISNOTSWAPPED(obj)) {
			// We are set!  Just copy the reference.
			mArray = obj;
			Py_INCREF(obj);
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
        int requirements = NPY_NOTSWAPPED | NPY_ENSUREARRAY;


        PyArray_Descr* descr=NULL;
        descr = PyArray_DescrNewFromType(mTypeNum);

        if (descr == NULL) {
            throw "could not create array descriptor";
        }

        // This will steal a reference to descr, and always returns a new
        // reference for array.  We don't need to decref descr as long as we
        // decref the array
        mArray = PyArray_CheckFromAny(
                obj, descr, min_depth, max_depth, requirements, NULL);

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

	if (size < 0)  {
		throw "size must be >= 0";
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

	return (T*) PyArray_GetPtr((PyArrayObject*) mArray, &index);
}



template <class T>
void NumpyVector<T>::set_type() throw (const char *) {

    const char *name = typeid(T).name();
    if (mNumpyIdMap.count(name) > 0) {
        mTypeNum = mNumpyIdMap[name];
    } else {
        std::stringstream err;
        err<<"NumpyArray: unsupported type: '"<<name<<"'\n";
        throw err.str().c_str();
    }

	mTypeName = name;

}


template <class T>
void NumpyVector<T>::init_type_info() {
    // static class members, only create once and shared by all
    // instances
    if (mNumpyIdMap.empty()) {

        const char* tname;

        tname = typeid(npy_int8).name();
        mNumpyIdMap[tname] = NPY_INT8;
        tname = typeid(npy_uint8).name();
        mNumpyIdMap[tname] = NPY_UINT8;


        tname = typeid(npy_int16).name();
        mNumpyIdMap[tname] = NPY_INT16;
        tname = typeid(npy_uint16).name();
        mNumpyIdMap[tname] = NPY_UINT16;


        tname = typeid(npy_int32).name();
        mNumpyIdMap[tname] = NPY_INT32;
        tname = typeid(npy_uint32).name();
        mNumpyIdMap[tname] = NPY_UINT32;


        tname = typeid(npy_int64).name();
        mNumpyIdMap[tname] = NPY_INT64;
        tname = typeid(npy_uint64).name();
        mNumpyIdMap[tname] = NPY_UINT64;


        tname = typeid(npy_float32).name();
        mNumpyIdMap[tname] = NPY_FLOAT32;

        tname = typeid(npy_float64).name();
        mNumpyIdMap[tname] = NPY_FLOAT64;


		// On OS X 10.6 these can have different names than above
        tname = typeid(short).name();
        mNumpyIdMap[tname] = NPY_SHORT;
        tname = typeid(unsigned short).name();
        mNumpyIdMap[tname] = NPY_USHORT;

        tname = typeid(int).name();
        mNumpyIdMap[tname] = NPY_INT;
        tname = typeid(unsigned int).name();
        mNumpyIdMap[tname] = NPY_UINT;

        tname = typeid(long).name();
        mNumpyIdMap[tname] = NPY_LONG;
        tname = typeid(unsigned long).name();
        mNumpyIdMap[tname] = NPY_ULONG;

        tname = typeid(long long).name();
        mNumpyIdMap[tname] = NPY_LONGLONG;
        tname = typeid(unsigned long long).name();
        mNumpyIdMap[tname] = NPY_ULONGLONG;


        tname = typeid(float).name();
        mNumpyIdMap[tname] = NPY_FLOAT;
        tname = typeid(double).name();
        mNumpyIdMap[tname] = NPY_DOUBLE;


        // And these can *also* have different names
        tname = typeid(int8_t).name();
        mNumpyIdMap[tname] = NPY_INT8;
        tname = typeid(uint8_t).name();
        mNumpyIdMap[tname] = NPY_UINT8;


        tname = typeid(int16_t).name();
        mNumpyIdMap[tname] = NPY_INT16;
        tname = typeid(uint16_t).name();
        mNumpyIdMap[tname] = NPY_UINT16;


        tname = typeid(int32_t).name();
        mNumpyIdMap[tname] = NPY_INT32;
        tname = typeid(uint32_t).name();
        mNumpyIdMap[tname] = NPY_UINT32;


        tname = typeid(int64_t).name();
        mNumpyIdMap[tname] = NPY_INT64;
        tname = typeid(uint64_t).name();
        mNumpyIdMap[tname] = NPY_UINT64;


    }
}





#endif
