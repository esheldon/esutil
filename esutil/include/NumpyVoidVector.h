/*
 * NumpyVoidVector.h
 *
 * This is simple wrapper class for 1-d and scalar numpy arrays.  
 * The purpose of this class is to handle the reference counting
 * and to simplify creation of arrays from input descriptors.
 *
 * For explicitly types vectors, use the NumpyVector class.
 *
 * This is a header-only class.  Simply include it and use.
 *
 * Examples:
 * #include "NumpyVoidVector.h"
 *
 *    // creating a new vector of any type from a PyObject*
 *    NumpyVoidVector vec(obj);
 *
 *    //
 *    // Create from a type string
 *    //
 *
 *    // specifying length
 *    NumpyVoidVector vec("i4", 25);
 *    // converting the input object
 *    NumpyVoidVector vec("f8", [1.2,3.5,725.2]);
 *
 *    //
 *    // Create from a full numpy PyArray_Descr
 *    //
 *
 *    NumpyVoidVector vec(descr, obj);
 *    NumpyVoidVector vec(descr, 35);
 *
 *
 *
 *    // get a reference for returning to python.  Reference counting is
 *    // done correctly
 *
 *    PyObject* output = vec.getref();
 *    return output;
 *
 *
 *    // Get some info about the array
 *
 *    // The numpy type number
 *    int type_num = vec.type_num();
 *
 *    // the number of elements in the vector
 *    npy_intp nel = vec.size();
 *
 *    // the stride of the array
 *    npy_intp stride = vec.stride();
 *
 *    // Size of each element
 *    npy_intp itemsize = vec.item_size();
 *
 *
 *    // Access data in a way that is aware of strides
 *    void* p = vec.ptr();
 *
 *    // if we know the data are double...
 *    if (vec.type_num() == NPY_FLOAT64) {
 *        double* p = (double* ) vec.ptr();
 *    }
 *
 *    // get pointer to particular location.  This is stride-aware.
 *    void* p = vec.ptr(22);
 *
 *    // If we knew the data were float32
 *    for (npy_intp i=0; i<vec.size(); i++) {
 *        npy_float32* p = (npy_float32* ) vec.ptr(i);
 *        // do something interesting
 *    }
 *
 *
 *    // if you *know* the data are contiguous and of a given type, this 
 *    // is an easy and fast way to access the data.
 *    npy_float32* p = (npy_float32* )vec.ptr(); 
 *    for (npy_intp i=0; i<vec.size(); i++) {
 *        float32 val = *p;
 *        // do something
 *        ++p;
 *    }
 *
 *    // This is the fastest way to loop using strides.
 *    // in this example, we know the data type is int32
 *
 *    npy_int32* p = (npy_int32* ) vec.ptr();
 *    npy_intp stride = vec.stride();  // zero for scalars
 *    for (npy_intp i=0; i<vec.size(); i++) {
 *        npy_int32 val = *p;
 *        p = p + stride;
 *    }
 *
 *
 *
 *
 *   
 */


#ifndef _numpy_void_vector_h
#define _numpy_void_vector_h

#include <Python.h>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <map>
#include <stdint.h>
#include "numpy/arrayobject.h"



class NumpyVoidVector {
	public:

        // simple constructor
		NumpyVoidVector() throw (const char *);

        // destructor.  Always decref the array
		~NumpyVoidVector() {
			Py_XDECREF(mArray);
		};

        //
        // Construct from existing python object, allowing it to be any type.
        // See init(PyObject* obj)
        //

        NumpyVoidVector(PyObject* obj) throw (const char *);

        //
        // construct from a string dtype
        //

        // from a python object
		NumpyVoidVector(const char* dtype, PyObject* obj) throw (const char *);
        // with specified length
		NumpyVoidVector(const char* dtype, npy_intp size) throw (const char *);

        // these just call the above
        // from a python object
		NumpyVoidVector(
                const string& dtype, PyObject* obj) throw (const char *);
        // with specified length
		NumpyVoidVector(
                const string& dtype, npy_intp size) throw (const char *);


        //
        // Constructing with a specified PyArray_Descr
        //

        // From a python object
		NumpyVoidVector(
                PyArray_Descr* descr, PyObject* obj) throw (const char *);
		// with specified length
		NumpyVoidVector(
                PyArray_Descr* descr, npy_intp size) throw (const char *);




        // Initialize from an input python object, converting to an array of
        // the right type and native byte order if necessary. If already the
        // right type, etc. then no copy is made.
        //
        // If the data are already an array, must be zero or 1-dimensional.
        //
        // This can be called at *any time* and any existing data will
        // be released
        
        void init(PyObject* obj)  throw (const char *);

        // init from a string name for dtype
        void init(const char* dtype, PyObject* obj)  throw (const char *);
        void init(const char* dtype, npy_intp size)  throw (const char *);
        void init(const string& dtype, PyObject* obj)  throw (const char *);
        void init(const string& dtype, npy_intp size)  throw (const char *);

        // same but forcing given type
        void init(PyArray_Descr* descr, PyObject* obj) throw (const char *);
        void init(PyArray_Descr* descr, npy_intp size)  throw (const char *);



        // Get a pointer to the data.
		void* ptr() throw (const char *);

        // Get a pointer to the data at the indicated location
        // Strides are properly accounted for.  No bounds checking
        // are performed.
		void* ptr(npy_intp index) throw (const char *);


        // Get a reference the underlying python object and incref the object.
        // This is useful if you want to get a PyObject* that will be returned
        // to the outside world. The internal version will be decrefed when the
        // object is destructed or goes out of scope, so reference counting
        // is correct..

		PyObject* getref() throw (const char *);




        // Return the numpy type number
		int type_num() {
			return mTypeNum;
		}
        // the number of dimensions.  Should be zero or 1
		npy_intp ndim() {
			return mNdim;
		}
        // Return the number of elements in the data.
		npy_intp size() {
			return mSize;
		}
        // stride of the data.
        npy_intp stride() {
            return mStride;
        }
        // size of each row
        npy_intp item_size() {
            return mItemSize;
        }

        // convert strings to descr
        PyArray_Descr* descr_from_string(
                const char* dtype) throw (const char*);
	
	private:

        // run this after we've created the array
        void set_type_info();

		int mTypeNum;
		npy_intp mSize;
        npy_intp mNdim; // should be 1 or 0
        npy_intp mItemSize;
        npy_intp mStride;

		PyObject* mArray;

};

void
NumpyVoidVector::set_type_info() {
    if (mArray != NULL) {
        mTypeNum = PyArray_TYPE(mArray);
        mSize = PyArray_SIZE(mArray);
        mNdim = PyArray_NDIM(mArray);
        // Will segfault if we try to
        // get strides of zero dim array
        if (mNdim == 0) {
            mStride = 0;
        } else {
            mStride = PyArray_STRIDE(mArray, 0);
        }

        mItemSize = PyArray_ITEMSIZE(mArray);
    } else {
        mTypeNum = -1;
        mSize = 0;
        mNdim = 0;
        mStride = 0;
        mItemSize = 0;
    }
}


NumpyVoidVector::NumpyVoidVector()  throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

    // This will zero everything since array is not
    // created
    mArray=NULL;
    set_type_info();
}


NumpyVoidVector::NumpyVoidVector(PyObject* obj)  throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

	// Get the data.  This may or may not make a copy.
    mArray=NULL;
	init(obj);
}


NumpyVoidVector::NumpyVoidVector(PyArray_Descr* descr, PyObject* obj)  throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

	// Get the data.  This may or may not make a copy.
    mArray=NULL;
	init(descr, obj);

}



// Create given the length and typenum
NumpyVoidVector::NumpyVoidVector(PyArray_Descr* descr, npy_intp size) throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

    // create a new array from the size and type info
    mArray=NULL;
	init(descr,size);
}


NumpyVoidVector::NumpyVoidVector(
        const char* dtype, PyObject* obj)  throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

	// Get the data.  This may or may not make a copy.
    mArray=NULL;
	init(dtype, obj);

}
NumpyVoidVector::NumpyVoidVector(
        const char* dtype, npy_intp size)  throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

	// Get the data.  This may or may not make a copy.
    mArray=NULL;
	init(dtype, size);

}

NumpyVoidVector::NumpyVoidVector(
        const string& dtype, PyObject* obj)  throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

	// Get the data.  This may or may not make a copy.
    mArray=NULL;
	init(dtype.c_str(), obj);
}
NumpyVoidVector::NumpyVoidVector(
        const string& dtype, npy_intp size)  throw (const char *) {
	// DONT FORGET THIS!!!!
	import_array();

	// Get the data.  This may or may not make a copy.
    mArray=NULL;
	init(dtype.c_str(), size);
}



void NumpyVoidVector::init(PyObject* obj)  throw (const char *) {

	// clear any existing array
	Py_XDECREF(mArray);

	if (obj == NULL || obj == Py_None) {
		throw "cannot convert the input object to an array: is NULL or None";
	}


    // can be scalar, but not higher dimensional than 1
    int min_depth=0, max_depth=1;

    // require the array is in native byte order
    int requirements = NPY_NOTSWAPPED;
    mArray = PyArray_CheckFromAny(
            obj, NULL, min_depth, max_depth, requirements, NULL);

    if (mArray == NULL) {
        throw "Could not get input as array";
    }

    set_type_info();

}

void NumpyVoidVector::init(
        PyArray_Descr* descr, PyObject* obj)  throw (const char *) {

	// clear any existing array
	Py_XDECREF(mArray);

	if (obj == NULL || obj == Py_None) {
		throw "cannot convert the input object to an array: is NULL or None";
	}


    // can be scalar, but not higher dimensional than 1
    int min_depth=0, max_depth=1;

    // require the array is in native byte order
    int requirements = NPY_NOTSWAPPED;
    mArray = PyArray_CheckFromAny(
            obj, descr, min_depth, max_depth, requirements, NULL);

    if (mArray == NULL) {
        stringstream err;
        err<<"Could not get input as array of type: '"<<descr->kind<<"'";
        throw err.str().c_str();
    }
    // this is important
    Py_INCREF(descr);

    set_type_info();

}


void NumpyVoidVector::init(
        PyArray_Descr* descr, npy_intp size)  throw (const char *) {

	// clear any existing array
	Py_XDECREF(mArray);

	if (size < 1)  {
		throw "size must be >= 1";
	}

	// Create output flags array
	int ndim=1;
    mArray = PyArray_Zeros(
            ndim, 
            &size, 
            descr, 
            NPY_FALSE);

	if (mArray ==NULL) {
        stringstream err;
        err<<"Could not allocate array of type: '"<<descr->kind<<"'";
        throw err.str().c_str();
	}

    // this is important
    Py_INCREF(descr);
    set_type_info();
}


void NumpyVoidVector::init(
        const char* dtype, PyObject* obj)  throw (const char *) {

	// clear any existing array
	Py_XDECREF(mArray);

	if (obj == NULL || obj == Py_None) {
		throw "cannot convert the input object to an array: is NULL or None";
	}

    // We need to generate a PyArray_Descr* from this
    // string.
    PyArray_Descr* descr = descr_from_string(dtype);

    // can be scalar, but not higher dimensional than 1
    int min_depth=0, max_depth=1;

    // require the array is in native byte order
    int requirements = NPY_NOTSWAPPED;
    mArray = PyArray_CheckFromAny(
            obj, descr, min_depth, max_depth, requirements, NULL);

    if (mArray == NULL) {
        stringstream err;
        err<<"Could not get input as array of type: '"<<dtype<<"'";
        throw err.str().c_str();
    }

    set_type_info();

}
void NumpyVoidVector::init(
        const string& dtype, PyObject* obj)  throw (const char *) {
    init(dtype.c_str(), obj);
}

void NumpyVoidVector::init(
        const char* dtype, npy_intp size)  throw (const char *) {

    stringstream err;
	// clear any existing array
	Py_XDECREF(mArray);

	if (size < 1)  {
		throw "size must be >= 1";
	}

    // We need to generate a PyArray_Descr* from this
    // string.
    PyArray_Descr* descr = descr_from_string(dtype);

    // Create output flags array.  Will steal a referene to descr
    // so we can forget about it
	int ndim=1;
    mArray = PyArray_Zeros(
            ndim, 
            &size, 
            descr, 
            NPY_FALSE);

	if (mArray ==NULL) {
        stringstream err;
        err<<"Could not allocate array of type: '"<<dtype<<"'";
        throw err.str().c_str();
	}

    // this is important
    set_type_info();
}
void NumpyVoidVector::init(
        const string& dtype, npy_intp size)  throw (const char *) {
    init(dtype.c_str(), size);
}
PyArray_Descr* NumpyVoidVector::descr_from_string(
        const char* dtype) throw (const char*) {

    stringstream err;

    // We need to generate a PyArray_Descr* from this
    // string.  Don't forget to decref this
    PyObject* pyobj_dtype= PyString_FromString(dtype);

    PyArray_Descr* descr;
    if (!PyArray_DescrConverter(pyobj_dtype, &descr)) {
        Py_XDECREF(pyobj_dtype);
        err<<"could not convert dtype to PyArray_Descr: '"<<dtype<<"'";
        throw err.str().c_str();
    }
    Py_XDECREF(pyobj_dtype);

    return descr;

}


// Get a reference the object.  incref the object.
// This is useful if you want to get a PyObject* that will be returned
// to the outside world
PyObject* NumpyVoidVector::getref() throw (const char *) {
	Py_XINCREF(mArray);
	return mArray;
}


void* NumpyVoidVector::ptr() throw (const char *) {
	if (mArray == NULL) {
		throw "Error: attempt to get pointer from an uninitialized array";
	}

	npy_intp index=0;
	return PyArray_GetPtr((PyArrayObject*) mArray, &index);
}

void* NumpyVoidVector::ptr(npy_intp index) throw (const char *) {
	if (mArray == NULL) {
		throw "Error: attempt to get pointer from an uninitialized array";
	}

	return PyArray_GetPtr((PyArrayObject*) mArray, &index);
}




#endif
