/*
  NumpyRecords.h
 
  This is a header-only C++ wrapper class for numpy arrays with fields, A.K.A.
  recarray.  It provides access to the underlying data by name and row
  number.  This is a header-only class, just include it.
 
  RESTRICTIONS: sub-arrays are not yet supported.  Numerical data must be in
  native byte order.
 
  Examples: 
 
      // This is a header-only class, just include it
      #include "NumpyRecords.h"
 
      // initialize from an array with fields.  let's assume this array has a
      // 4-byte integer field, an 8-byte float field, or double, and a 5 byte
      // string field "str".  The dtype would look like:

      dtype = [('id','i4'),('x','f8'),('str','S5')]
 
      NumpyRecords rec(obj);
 
      //
      // Info about the array as a whole
      //

      // the number of rows is returned by size()
      npy_intp nrows = rec.size();

      // size of each row, aka the stride
      // these are synonymous
      npy_intp rowsize = rec.rowsize();
      rowsize = rec.stride();

      // total number of bytes in the entire array
      npy_intp nbytes = rec.nbytes();

 
      //
      // Info about the fields
      //
      
      // number of fields
      npy_intp nfields  = rec.nfields();

      // vector of field names
      vector<string> names = rec.names();
 
      // the name of the 4th field, index 3
      string name3 = rec.name(3);
 
      // number of bytes in a single element of the requested field
      npy_intp idsize = rec.elsize("id");

      // the offset of the field "x" can be accessed either by
      // name or index.  This is how many bytes you need to offset
      // from the beginning of a row to access this field
      
      npy_intp xoffset = rec.offset("x");
 
      // The numpy type code of the field
      int typecode = rec.typecode("id");


      //
      //
      // accessing the underlying data
      //
      //
 
      //
      // Elements can be accessed in a type safe way using template methods.
      // Conversions are done internally.  This can be done either using a
      // function get() or procedurally using copy().
      //

      // copy id as int32, it's declared type
      int32_t i  = rec.get<int32_t>("id", 1);

      // can also convert fields to other types
      int64_t xi = rec.get<int64_t>("x", 5);

      // This is the procedural copy.  The type is simply determined by the
      // type of the input
      rec.copy("x", 5, xi);
      string s;
      rec.copy("str", 3, s);
 
      // You can also copy entire fields to std::vector.
      // convert "id" to an int64 vector
      vector<int64_t> v;
      rec.copy("id", v);

      vector<string> svec;
      rec.copy("str", svec);
      rec.copy("f4field", svec);

      // You can also get a NumpyVector or NumpyVoidVector view of a field Copy
      // is performed if the data types differ or data is not the native byte
      // order.   NumpyVector provides an iterator.
      NumpyVector<double> dvec;
      rec.get("x", dvec);

      NumpyVoidVector dvec;
      rec.get("x", dvec);

      // You can even access the underlying python object for the field
      PyObject* dvec_obj;
      dvec_pyobj = rec.get("x");

      //
      // Unsafe access:  these methods provide pointer access to the underlying
      // data, so can be faster but you need to know the data type.  Also, for
      // string data you need to know the element size, which you can get from
      // the elsize() method. 
      //
      
      // get a pointer to field "id" in the 8th row: Do not perform ++ on int32
      // pointer, rather see below for an example using the stride
      npy_int32* i = (npy_int32*) rec.ptr("id", 8);
 
 
      // The following is a faster way to iterate over all rows: get
      // a pointer to a field in the first row and use the row size to move to the
      // next row.
 
      // this points to "x" in row 0
      char* xptr = rec.ptr("x");
      npy_intp stride = rec.stride();
 
      for (npy_intp i=0; i<rec.size(); i++) {
          double x = *xptr;
 
          // work with this data...
 
          // now point to the next one
          xptr += stride;
      }
 
      // you can also just get a pointer to the beginning of the entire array and
      // then use the stride and row size to access various fields
 
      char* ptr = rec.ptr();
      npy_intp stride = rec.stride();
      npy_intp ioffset = rec.offset("id");
      npy_intp xoffset = rec.offset("x");
 
      for (npy_intp row=0; row<rec.size(); row++) {
          npy_int32* id = (npy_int32*) (ptr+ioffset);
          double*    x  = (double*)    (ptr+xoffset);
 
          // work with this data....
 
          // now move to the next row
          ptr += stride;
      }
 
    // Accessing string fields via ptr().  Because numpy strings are not null
    // terminated, you should use std::string or your own constructed null
    // terminated string as a container for them

    npy_intp strsize = rec.elsize("str");
    string s(strsize, ' ');
    ptr = rec.ptr("str", 1);
    for (npy_intp i=0;i<strsize;i++) {
        s[i] = sptr[i];
    }
 

    Copyright (C) 2010  Erin Sheldon, BNL.  erin.sheldon at gmail.com

    This program is part of esutil.  http://code.google.com/p/esutil/

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


 */

#ifndef _numpy_records_h
#define _numpy_records_h


#include <Python.h>
#include "numpy/arrayobject.h"

#include <vector>
#include <string>
#include <map>
#include <stdint.h>
#include <sstream>

// this is only used for error messages
#include <typeinfo>

#include "NumpyVector.h"
#include "NumpyVoidVector.h"

class NumpyRecords {
    public:
        NumpyRecords() throw (const char*) {
            import_array();
            set_defaults();
        }

        NumpyRecords(PyObject* obj) throw (const char*) {
            import_array();
            set_defaults();
            init(obj);
        }

		~NumpyRecords() {
            clear();
		}

        // always call this in the constructor
        void set_defaults() {
            this->array_=NULL;
            this->data_=NULL;
            this->nbytes_=0;
            this->size_=0;
            this->stride_=0;
        }
        void clear() {
            this->typecodes_.clear();
            this->names_.clear();
            this->offsets_.clear();
            this->elsizes_.clear();

            this->field_descr_.clear();

            this->nmap_.clear();

            Py_XDECREF(this->array_);
            set_defaults();
        }


        void init(PyObject* obj) throw (const char*) {
            // clear any existing array and info
            clear();

            if (!PyArray_Check(obj)) {
                throw "input object must be an array";
            }

            this->array_ = obj;
            this->data_ = PyArray_DATA(obj);
            Py_INCREF(obj);

            set_sizes();
            set_field_info();
        }


        //
        // accessors
        //

        // get a pointer at the beginning of all the data
        char* ptr() {
            return (char*) this->data_;
        }

        // return a pointer to the field in the first row.  Make sure you use
        // the offsets and stride to properly access other rows!
        char* ptr(const char* cname) throw (const char*) {
            std::string name=cname;
            return ptr(name);
        }
        char* ptr(std::string& name) {
            check_name(name);

            npy_intp fi = this->nmap_[name];
            char* data = (char*) this->data_;
            return data + this->offsets_[fi];
        }


        // return a pointer to the field and row
        char* ptr(const char* cname, npy_intp row) throw (const char*) {
            std::string name=cname;
            return ptr(name,row);
        }
        char* ptr(std::string& name, npy_intp row) throw (const char*) {
            check_name(name);
            check_row(row);
            npy_intp fi = this->nmap_[name];

            char* data = (char*) this->data_;

            return data + row*this->stride_ + this->offsets_[fi];
        }

        // get the an element of the indicated column and row
        // with the specified type. This one is type safe
        template <typename T> T get(std::string name, npy_intp row) {
            T tmp;
            copy(name, row, tmp);
            return tmp;
        }
        template <typename T> T get(npy_intp field_index, npy_intp row) {
            T tmp;
            copy(field_index, row, tmp);
            return tmp;
        }


        void get(std::string name, NumpyVoidVector& vec) {
            check_name(name);
            npy_intp fi = this->nmap_[name];
            get(fi, vec);
        }
        void get(npy_intp field_index, NumpyVoidVector& vec) {
            PyObject* f = get(field_index);
            vec.init(f);
            // clean up the reference for this field
            Py_XDECREF(f);
        }


        template <typename T>
        void get(std::string name, NumpyVector<T>& vec) {
            check_name(name);
            npy_intp fi = this->nmap_[name];
            get(fi, vec);
        }
        template <typename T>
        void get(npy_intp field_index, NumpyVector<T>& vec) {

            PyObject* f = get(field_index);

            vec.init(f);

            // clean up the reference for this field
            Py_XDECREF(f);
        }


        // Get the field as a C numpy array
        PyObject* get(std::string name) {
            check_name(name);
            npy_intp fi = this->nmap_[name];
            return get(fi);
        }
        // This one is more dangerous since you have to do the
        // ref counting yourself!
        PyObject* get(npy_intp field_index) {

            check_field_index_bound(field_index);

            PyObject *ret=NULL;

            PyArrayObject* array = (PyArrayObject* ) this->array_;
            PyArray_Descr* fdescr = this->field_descr_[field_index];
            npy_intp offset = this->offsets_[field_index];

            ret = PyArray_GetField(array, fdescr, offset);

            if (ret == NULL) {
                std::stringstream err;
                err<<"Got NULL from PyArray_NewFromDescr";
                throw err.str().c_str();
            }

            return ret;

        }

        // total number of bytes
        npy_intp nbytes() {
            return this->nbytes_;
        }

        // number of bytes in requested field
        npy_intp elsize(std::string name) throw (const char*) {
            check_name(name);
            npy_intp fi=this->nmap_[name];
            return elsize(fi);
        }
        npy_intp elsize(npy_intp i) throw (const char*) {
            check_field_index_bound(i);
            return this->elsizes_[i];
        }



        // number of rows
        npy_intp size() {
            return this->size_;
        }

        // The itemsize/rowsize/stride are all synonymous for size of an
        // entire row
        npy_intp strides() {
            return this->stride_;
        }
        npy_intp stride() {
            return this->stride_;
        }
        npy_intp itemsize() {
            return this->stride_;
        }
        npy_intp rowsize() {
            return this->stride_;
        }

        // the offset for a particular field
        npy_intp offset(std::string name) throw (const char*) {
            check_name(name);
            npy_intp fi=this->nmap_[name];
            return offset(fi);
        }
        npy_intp offset(npy_intp i) throw (const char*) {
            check_field_index_bound(i);
            return this->offsets_[i];
        }

        // number of fields in this array
        npy_intp nfields() {
            return this->names_.size();
        }

        // access to the name of a particular field
        std::string name(npy_intp i) throw (const char*) {
            check_field_index_bound(i);
            return this->names_[i];
        }

        // copy of the entire names vector
        std::vector<std::string> names() {
            return this->names_;
        }

        int typecode(npy_intp i) throw (const char*) {
            check_field_index_bound(i);
            return this->typecodes_[i];
        }


        //
        // convert the requested field/row element to the requested type
        //

        // first special methods for strings
        void copy(std::string name, npy_intp row, std::string& str) {
            check_name(name);
            npy_intp fi=this->nmap_[name];
            copy(fi, row, str);
        }
        void copy(npy_intp field_index, npy_intp row, std::string& str) {
            check_field_index_bound(field_index);
            check_row(row);
            int typecode = this->typecodes_[field_index];

            // for conversions
            std::stringstream ss;
            char* ptr = (char*) this->data_;

            // point data to the requested position
            ptr += row*this->stride_ + this->offsets_[field_index];

            switch (typecode) {
                case NPY_STRING:
                    {
                        // copy character by character into the string
                        npy_intp strsize = this->elsizes_[field_index];
                        str.resize(strsize);
                        for (npy_intp i=0; i<strsize; i++) {
                            str[i] = ptr[i];
                        }
                    }
                    break;
				case NPY_INT8: 
					{
                        npy_int8 tmp = *(npy_int8*) ptr; 

                        // convert to integer rep, otherwise it will write it
                        // as a char
                        short ts = (short) tmp;
                        ss<<ts;
                        str = ss.str();
					}
                    break;
				case NPY_UINT8: 
					{
                        npy_uint8 tmp = *(npy_uint8*) ptr; 

                        // convert to integer rep, otherwise it will write it
                        // as a char
                        short ts = (short) tmp;
                        ss<<ts;
                        str = ss.str();

					}
                    break;

				case NPY_INT16: 
					{
                        npy_int16 tmp = *(npy_int16*) ptr; 
                        ss<<tmp;
                        str = ss.str();
					}
                    break;
				case NPY_UINT16: 
					{
                        npy_uint16 tmp = *(npy_uint16*) ptr; 
                        ss<<tmp;
                        str = ss.str();
					}
                    break;

				case NPY_INT32: 
					{
                        npy_int32 tmp = *(npy_int32*) ptr; 
                        ss<<tmp;
                        str = ss.str();
					}
                    break;
				case NPY_UINT32: 
					{
                        npy_uint32 tmp = *(npy_uint32*) ptr; 
                        ss<<tmp;
                        str = ss.str();
					}
                    break;

				case NPY_INT64: 
					{
                        npy_int64 tmp = *(npy_int64*) ptr; 
                        ss<<tmp;
                        str = ss.str();
					}
                    break;
				case NPY_UINT64: 
					{
                        npy_uint64 tmp = *(npy_uint64*) ptr; 
                        ss<<tmp;
                        str = ss.str();
					}
                    break;

				case NPY_FLOAT32: 
					{
                        npy_float32 tmp = *(npy_float32*) ptr; 
                        ss<<tmp;
                        str = ss.str();
					}
                    break;
				case NPY_FLOAT64: 
					{
                        npy_float64 tmp = *(npy_float64*) ptr; 
                        ss<<tmp;
                        str = ss.str();
					}
                    break;

				default:

                    std::stringstream err;
					err<<"Conversion from type code "<<typecode<<" to string is not supported";
					throw err.str().c_str();

            }

        } // copy scalar to string




        template <typename T> void copy(std::string name, npy_intp row, T& var) {
            check_name(name);
            npy_intp fi=this->nmap_[name];
            copy(fi, row, var);
        }
        template <typename T> void copy(npy_intp field_index, npy_intp row, T& var) {
            check_field_index_bound(field_index);
            check_row(row);
            int typecode = this->typecodes_[field_index];

            char* ptr = (char*) this->data_;

            switch (typecode) {
				case NPY_INT8: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_int8 tmp = *(npy_int8*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;
				case NPY_UINT8: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_uint8 tmp = *(npy_uint8*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;

				case NPY_INT16: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_int16 tmp = *(npy_int16*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;
				case NPY_UINT16: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_uint16 tmp = *(npy_uint16*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;


				case NPY_INT32: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_int32 tmp = *(npy_int32*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;
				case NPY_UINT32: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_uint32 tmp = *(npy_uint32*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;

				case NPY_INT64: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_int64 tmp = *(npy_int64*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;
				case NPY_UINT64: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_uint64 tmp = *(npy_uint64*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;


				case NPY_FLOAT32: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_float32 tmp = *(npy_float32*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;

				case NPY_FLOAT64: 
					{
                        // point data to the requested position
                        ptr += row*this->stride_ + this->offsets_[field_index];
                        npy_float64 tmp = *(npy_float64*) ptr; 
                        // now do the conversion
                        var = (T) tmp;
					}
                    break;



				default:

                    std::stringstream err;

                    std::string tname = typeid(T).name();
					err<<"Conversion from type code "<<typecode<<" to '"<<tname<<"' is not supported";
					throw err.str().c_str();

            }

        } // copy scalar

        // copy out to vectors

        // first special string methods
        void copy(std::string name, std::vector<std::string>& svec) {
            check_name(name);
            npy_intp fi=this->nmap_[name];
            copy(fi, svec);
        }
        void copy(npy_intp field_index, std::vector<std::string>& svec) {
            check_field_index_bound(field_index);
            int typecode = this->typecodes_[field_index];

            char* ptr = (char*) this->data_;

            // point data to the requested position
            ptr += this->offsets_[field_index];

            switch (typecode) {
                case NPY_STRING:
                    {
                        svec.resize(this->size_);
                        npy_intp strsize = this->elsizes_[field_index];
                        for (npy_intp row=0;row<this->size_;row++) {
                            // copy character by character into the string
                            svec[row].resize(strsize);
                            for (npy_intp i=0; i<strsize; i++) {
                                svec[row][i] = ptr[i];
                            }
                        }
                    }
                    break;
				case NPY_INT8: 
                    copy2stringvector<npy_int8>(ptr, svec);
                    break;
				case NPY_UINT8: 
                    copy2stringvector<npy_uint8>(ptr, svec);
                    break;

				case NPY_INT16: 
                    copy2stringvector<npy_int16>(ptr, svec);
                    break;
				case NPY_UINT16: 
                    copy2stringvector<npy_uint16>(ptr, svec);
                    break;

				case NPY_INT32: 
                    copy2stringvector<npy_int32>(ptr, svec);
                    break;
				case NPY_UINT32: 
                    copy2stringvector<npy_uint32>(ptr, svec);
                    break;

				case NPY_INT64: 
                    copy2stringvector<npy_int64>(ptr, svec);
                    break;
				case NPY_UINT64: 
                    copy2stringvector<npy_uint64>(ptr, svec);
                    break;

				case NPY_FLOAT32: 
                    copy2stringvector<npy_float32>(ptr, svec);
                    break;
				case NPY_FLOAT64: 
                    copy2stringvector<npy_float64>(ptr, svec);
                    break;

				default:

                    std::stringstream err;
					err<<"Conversion from type code "<<typecode<<" to string is not supported";
					throw err.str().c_str();

            }

        } // copy scalar to string




        template <typename T> void copy(std::string name, std::vector<T>& vec) {
            check_name(name);
            npy_intp fi=this->nmap_[name];
            copy(fi, vec);
        }
        template <typename T> void copy(npy_intp field_index, std::vector<T>& vec) {
            check_field_index_bound(field_index);
            int typecode = this->typecodes_[field_index];

            char* ptr = (char*) this->data_;
            // point data to the requested position in first row
            ptr += this->offsets_[field_index];

            switch (typecode) {
				case NPY_INT8: 
                    copy2vector<npy_int8>(ptr, vec);
                    break;
				case NPY_UINT8: 
                    copy2vector<npy_uint8>(ptr, vec);
                    break;

				case NPY_UINT16: 
                    copy2vector<npy_uint16>(ptr, vec);
                    break;
				case NPY_INT16: 
                    copy2vector<npy_int16>(ptr, vec);
                    break;

				case NPY_UINT32: 
                    copy2vector<npy_uint32>(ptr, vec);
                    break;
				case NPY_INT32: 
                    copy2vector<npy_int32>(ptr, vec);
                    break;


				case NPY_UINT64: 
                    copy2vector<npy_uint64>(ptr, vec);
                    break;
				case NPY_INT64: 
                    copy2vector<npy_int64>(ptr, vec);
                    break;

				case NPY_FLOAT32: 
                    copy2vector<npy_float32>(ptr, vec);
                    break;
				case NPY_FLOAT64: 
                    copy2vector<npy_float64>(ptr, vec);
                    break;


				default:
                    std::stringstream err;
                    std::string tname = typeid(T).name();
					err<<"Conversion from type code "<<typecode<<" to '"<<tname<<"' is not supported";
					throw err.str().c_str();

            }
        } // copy vectors


    private:

        // private methods

        // get info about the record size and offsets of each field
        // into the row
        void set_sizes() {
            this->stride_ = PyArray_ITEMSIZE(this->array_);
            this->nbytes_   = PyArray_NBYTES(this->array_);
            this->size_     = PyArray_SIZE(this->array_);
        }

        // get information about all fields
        void set_field_info() throw (const char*) {

            std::stringstream err;
            PyArray_Descr* descr = PyArray_DESCR(this->array_);

            if (!PyTuple_Check(descr->names)) {
                throw "input object must have fields";
            }

            npy_intp nfields = PyTuple_Size(descr->names);
            if (nfields == 0) {
                throw "input object must have > 0 fields";
            }

            this->names_.resize(nfields);
            this->offsets_.resize(nfields);
            this->elsizes_.resize(nfields);
            this->typecodes_.resize(nfields);
            this->field_descr_.resize(nfields);

            // hold descriptor information
            PyArray_Descr *fdescr, *title;

            for (npy_intp i=0; i<nfields; i++) {

                // get the name of this field
                PyObject* name_obj = PyTuple_GET_ITEM(descr->names, i);
                std::string name = PyString_AS_STRING(name_obj);
                //name = name+'\0';
                this->names_[i] = name;

                // now type info for this field
                PyObject* item = PyDict_GetItemString(
                        descr->fields,
                        this->names_[i].c_str());

                npy_intp toffset;
                if (PyArg_ParseTuple(item, "Ol|O", &fdescr, &toffset, &title)) {

                    this->offsets_[i] = toffset;
                    this->typecodes_[i] = fdescr->type_num;
                    this->elsizes_[i] = fdescr->elsize;
                    this->field_descr_[i] = fdescr;

                } else {
                    err<<"could not parse field "<<this->names_[i];
                    throw err.str().c_str();
                }

                this->nmap_[name] = i;
                fflush(stdout);

            }
        }


        void check_field_index_bound(npy_intp i) throw (const char*) {
            if (i < 0 || i >= (npy_intp)this->names_.size()) {
                std::stringstream err;
                err<<"requested field "<<i
                    <<" is out of bounds [0, "<<this->names_.size()-1<<"]";
                throw err.str().c_str();
            }
        }

        //void check_name(const char* name) throw (const char*) {
        void check_name(const char* cname)  throw (const char*) {
            std::string name=cname;
            check_name(name);
        }
        void check_name(std::string& name) throw (const char*) {
            if (this->nmap_.count(name) == 0) {
                std::stringstream err;
                err<<"field name '"<<name<<"' not found";
                throw err.str().c_str();
            }
        }

        void check_row(npy_intp row) throw (const char*) {
            if (row < 0 || row >= this->size_) {
                std::stringstream err;
                err<<"row "<<row<<" is out of bounds: [0, "<<(this->size_-1)<<"]";
                throw err.str().c_str();
            }
        }

        template <typename T> void copy2stringvector(char* ptr, std::vector<std::string>& svec) {
            svec.resize(this->size_);
            for (npy_intp i=0;i<this->size_;i++) {
                std::stringstream ss;
                T tmp = *(T*) ptr; 
                ss<<tmp;
                svec[i] = ss.str();
                ptr += this->stride_;
            }
        }
        template <typename T1, typename T2> void copy2vector(char* ptr, std::vector<T2>& vec) {
            vec.resize(this->size_);
            for (npy_intp i=0;i<this->size_;i++) {
                T1* tptr = (T1*) ptr; 
                vec[i] = (T2) ( *tptr );
                ptr += this->stride_;
            }
        }



        // private data
        PyObject* array_;
        void* data_;

        npy_intp size_;
        npy_intp nbytes_;
        npy_intp stride_;

        std::vector<int> typecodes_;
        std::vector<std::string> names_;
        std::vector<npy_intp> offsets_;
        std::vector<npy_intp> elsizes_;
        // these are just the pointers, stolen ref, no copy
        std::vector<PyArray_Descr *> field_descr_;

        std::map<std::string, npy_intp> nmap_;
};

#endif // _numpy_records_h
