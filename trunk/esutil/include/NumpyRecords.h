/*
 * NumpyRecords.h
 *
 * This is a header-only C++ wrapper class for numpy arrays with fields, A.K.A.
 * recarray.  It provides access to the underlying data by name and row
 * number.  This is a header-only class, just include it.
 *
 * NOTE: sub-arrays are not yet supported.
 *
 * Examples: 
 *
 *     // This is a header-only class, just include it
 *     #include "NumpyRecords.h"
 *
 *     // initialize from an array with fields.
 *     // let's assume this array has a 4-byte integer field and
 *     // an 8-byte float field, or double.  The dtype would look like:
 *     //       [('id','i4'),('x','f8')]
 *
 *     NumpyRecords rec(obj);
 *
 *     // the number of rows is returned by size()
 *     npy_intp nrows = rec.size();
 *     // size of each row, total number of bytes in array
 *     npy_intp itemsize = rec.itemsize();
 *     npy_intp nbytes = rec.nbytes();
 *
 *     //
 *     // get info about the fields
 *     //
 *     npy_intp nfields  = rec.nfields();
 *     std::string names = rec.names();
 *
 *     // the name of the 4th field, index 3
 *     std::string name3 = rec.name(3);
 *
 *     // the strides of the field "x" can be accessed either by
 *     // name or index
 *     npy_intp dstride = rec.strides("x");
 *
 *     //
 *     // accessing the underlying data
 *     //
 *
 *     // get a pointer to field "id" in the 8th row:
 *     npy_int32* i = (npy_int32*) rec.ptr("id", 8);
 *
 *
 *     // because of bounds checking, the above can be slow.  This is a faster
 *     // way to iterate over all rows: get a pointer to a field 
 *     // in the first row and iterate over all rows.
 *
 *     // this points to "x" in row 0
 *     double* xptr = (double *) rec.ptr("x");
 *     npy_intp itemsize = rec.itemsize();
 *
 *     for (npy_intp i=0; i<rec.size(); i++) {
 *         double x = *xptr;
 *
 *         // work with this data...
 *
 *         // now get the next one
 *         xptr += itemsize;
 *     }
 *
 *     // you can also just get a pointer to the beginning of the entire array and
 *     // then use the strides and itemsize to access various fields
 *
 *     void* ptr = rec.ptr();
 *     npy_intp itemsize = rec.itemsize();
 *     npy_intp istrides = rec.strides("id");
 *     npy_intp xstrides = rec.strides("x");
 *
 *     for (npy_intp row=0; row<rec.size(); row++) {
 *         npy_int32* id = (npy_int32*) (ptr+istrides);
 *         double*    x  = (double*)    (ptr+xstrides);
 *
 *         // work with this data....
 *
 *         // now move to the next row
 *         ptr += itemsize;
 *     }
 *
 *
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
            this->_array=NULL;
            this->_data=NULL;
            this->_nbytes=0;
            this->_size=0;
            this->_itemsize=0;
        }
        void clear() {
            this->_names.clear();
            this->_nmap.clear();
            Py_XDECREF(this->_array);
            set_defaults();
        }


        void init(PyObject* obj) throw (const char*) {
            // clear any existing array and info
            clear();

            if (!PyArray_Check(obj)) {
                throw "input object must be an array";
            }

            this->_array = obj;
            this->_data = PyArray_DATA(obj);
            Py_INCREF(obj);

            set_sizes();
            set_field_info();
        }


        //
        // accessors
        //

        // get a pointer
        void* ptr() {
            return this->_data;
        }

        // return a pointer to the field in the first row.  Make sure you use
        // the strides and itemsize to properly access other rows!
        void* ptr(const char* cname) throw (const char*) {
            std::string name=cname;
            return ptr(name);
        }
        void* ptr(std::string& name) {
            check_name(name);

            npy_intp fi = this->_nmap[name];
            return NULL;

            return this->_data + this->_strides[fi];
        }


        // return a pointer to the field and row
        void* ptr(const char* cname, npy_intp row) throw (const char*) {
            std::string name=cname;
            return ptr(name,row);
        }
        void* ptr(std::string& name, npy_intp row) throw (const char*) {
            check_name(name);
            check_row(row);
            npy_intp fi = this->_nmap[name];

            return this->_data + row*this->_itemsize + this->_strides[fi];
        }


        // total number of bytes
        npy_intp nbytes() {
            return this->_nbytes;
        }

        // number of rows
        npy_intp size() {
            return this->_size;
        }

        // The size of an entire row
        npy_intp itemsize() {
            return this->_itemsize;
        }

        // strides
        // If no argument, return strides of the entire record
        npy_intp strides() {
            return this->_itemsize;
        }
        // the strides for a particular field

        npy_intp strides(std::string name) throw (const char*) {
            check_name(name);
            npy_intp fi=this->_nmap[name];
            return strides(fi);
        }
        npy_intp strides(npy_intp i) throw (const char*) {
            check_field_index_bound(i);
            return this->_strides[i];
        }

        // number of fields in this array
        npy_intp nfields() {
            return this->_names.size();
        }

        // access to the name of a particular field
        std::string name(npy_intp i) throw (const char*) {
            check_field_index_bound(i);
            return this->_names[i];
        }

        // copy of the entire names vector
        std::vector<std::string> names() {
            return this->_names;
        }

        int typecode(npy_intp i) throw (const char*) {
            check_field_index_bound(i);
            return this->_type_codes[i];
        }

    private:

        // private methods

        // get info about the record size and offsets of each field
        // into the row
        void set_sizes() {
            this->_itemsize = PyArray_ITEMSIZE(this->_array);
            this->_nbytes   = PyArray_NBYTES(this->_array);
            this->_size     = PyArray_SIZE(this->_array);
        }

        // get information about all fields
        void set_field_info() throw (const char*) {

            std::stringstream err;
            PyArray_Descr* descr = PyArray_DESCR(this->_array);

            if (!PyTuple_Check(descr->names)) {
                throw "input object must have fields";
            }

            npy_intp nfields = PyTuple_Size(descr->names);
            if (nfields == 0) {
                throw "input object must have > 0 fields";
            }

            this->_names.resize(nfields);
            this->_strides.resize(nfields);
            this->_type_codes.resize(nfields);

            // hold descriptor information
            PyArray_Descr *fdescr, *title;

            for (npy_intp i=0; i<nfields; i++) {

                // get the name of this field
                PyObject* name_obj = PyTuple_GET_ITEM(descr->names, i);
                std::string name = PyString_AS_STRING(name_obj);
                //name = name+'\0';
                this->_names[i] = name;

                // now type info for this field
                PyObject* item = PyDict_GetItemString(
                        descr->fields,
                        this->_names[i].c_str());

                npy_intp tstrides;
                if (PyArg_ParseTuple(item, "Ol|O", &fdescr, &tstrides, &title)) {

                    this->_strides[i] = tstrides;
                    this->_type_codes[i] = fdescr->type_num;

                } else {
                    err<<"could not parse field "<<this->_names[i];
                    throw err.str().c_str();
                }

                this->_nmap[name] = i;
                fflush(stdout);

            }
        }


        void check_field_index_bound(npy_intp i) throw (const char*) {
            if (i < 0 && i >= (npy_intp)this->_names.size()) {
                std::stringstream err;
                err<<"requested field "<<i
                    <<" is out of bounds [0"<<this->_names.size()-1<<"]";
                throw err.str().c_str();
            }
        }

        //void check_name(const char* name) throw (const char*) {
        void check_name(const char* cname)  throw (const char*) {
            std::string name=cname;
            check_name(name);
        }
        void check_name(std::string& name) throw (const char*) {
            if (this->_nmap.count(name) == 0) {
                std::stringstream err;
                err<<"field name '"<<name<<"' not found";
                throw err.str().c_str();
            }
        }

        void check_row(npy_intp row) throw (const char*) {
            if (row < 0 && row >= this->_size) {
                std::stringstream err;
                err<<"row is out of bounds: "<<row;
                throw err.str().c_str();
            }
        }



        // private data
        PyObject* _array;
        void* _data;

        npy_intp _size;
        npy_intp _nbytes;
        npy_intp _itemsize;

        std::vector<int> _type_codes;
        std::vector<std::string> _names;
        std::vector<npy_intp> _strides;

        //std::map<const char*, npy_intp> _nmap;
        std::map<std::string, npy_intp> _nmap;
};

#endif // _numpy_records_h
