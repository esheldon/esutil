#include <Python.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>
#include "numpy/arrayobject.h"

#ifndef _readfields_2_h
#define _readfields_2_h

using namespace std;


class Records {
    public:
	/*
		Records() throw (const char*);

        Records(PyObject* fileobj, 
				const char* mode,
				PyObject* delim) throw (const char *);

		PyObject* Open(PyObject* fileobj, 
				const char* mode,
				PyObject* delim) throw (const char *);
				*/

#ifdef SWIG
%feature("docstring",
"
Class
    Records - A class for reading from a file of fixed-length records into
        numerical python arrays. The file can be binary or ASCII.
    An new Records class is instantiated using the Open() method:
        For writing:
            import records
            r = records.Open(file/fileobj, mode='w', delim='')
        For reading:
            import records
            r = records.Open(file/fileobj, delim='', dtype=None, nrows=-9999)
            # Arguments can all be given as keywords except the file

        Inputs:
            file/fileobj:  A string file name or an open file object.
            mode: The file mode.  Default is 'r' but can be 'u' or 'w'.
            delim: The delimiter used in the file.  Default is "" for 
                binary files, but can be any string such as ',', '\\t', etc.
            dtype:  A numpy dtype object.  REQUIRED FOR READING. For example:
                numpy.dtype([('field1', 'i4'),('field2', 'f8')])
                some_numpy_array.dtype
            nrows: The number of rows in the file.  REQUIRED FOR READING.

    Class Methods:
        Read(rows=, fields=):
            Returns the data in a NumPy array.  Specific rows and fields 
            of the file can be specified with the keywords.  Rows must be
            sorted and unique.  Can be in any order.
        Write(numpy_array):
            Write the input numpy array to the file.  The array must have
            field names defined.

    Examples:
        import numpy
        import records

        # Read from a binary file
        file='test.bin'
        dtype=numpy.dtype([('field1','f8'),('field2','2i4'),('field3','i8')])
        nrows=10000000

        robj = records.Open(file, dtype=dtype, nrows=nrows)
        res=robj.Read()

        # Read from a CSV file of the same structure, and only read a subset 
        # of the data
        rows2get=[2335,122332,1550021]
        fields2get='field2'
        robj = records.Open('test.csv', delim=',', dtype=dtype, nrows=nrows)
        res = robj.Read(rows=rows2get, fields=fields2get)

        # Write a numpy array to a file
        r = records.Open('test.csv', 'w', ',')
        r.Write(my_array)

Modification history:
    Created: 2008-07-18, Erin Sheldon
");
#endif
		Records(PyObject* fileobj, 
				const char* mode, // mode won't be used if file is object
				PyObject* delim, 
				PyObject* dtype=NULL,
				long long nrows=-9999) throw (const char *);

        ~Records();

		// Some documentation.  SWIG can use this to make a python doc
		// string
#ifdef SWIG
%feature("docstring",
		"
		Read(rows=None, fields=None)

		A class method for the Records class.  Reads the specified rows
		and fields from the open file and returns the data in a NumPy array.

		Inputs:
		    rows:  A sorted unique set of rows.  May be a scala/rlist/array.
		      Default is all rows.
		    fields: The fields to read.  May be a single string or a list
		      of strings.  Can be in any order.  Default is all fields.
		Examples:
		    import numpy
		    import records
		    # Read from a binary file
		    file='test.bin'
		    dtype=numpy.dtype([('field1','f8'),('field2','2i4'),('field3','i8')])
		    nrows=10000000

		    robj = records.Open(file, dtype=dtype, nrows=nrows)
		    res=robj.Read()

		    # Read from a CSV file of the same structure, and only read a subset 
		    # of the data
		    rows2get=[2335,122332,1550021]
		    fields2get='field2'
		    robj = records.Open('test.csv', delim=',', dtype=dtype, nrows=nrows)
		    res = robj.Read(rows=rows2get, fields=fields2get)");
#endif
		PyObject* Read(
				PyObject* rows=NULL,
				PyObject* fields=NULL) throw (const char*);

#ifdef SWIG
%feature("docstring",
		"
		Write(numpy_array)

		A class method for the Records class.  Writes the input numpy array
		to the opened file.

		Inputs:
		    array: A NumPy array with fields defined for the records.
		Examples:
		    import numpy
		    import records
		    r = records.Open('test.csv', 'w', ',')
		    r.Write(my_array)
		");
#endif
		PyObject* Write(PyObject* obj) throw (const char *);

#ifdef SWIG
%feature("docstring",
		"
		Close()

		If the file was opened locally, close the file pointer.
		");
#endif
		void Close() throw (const char*);


    private:
		// Move this to public when needed for testing
        PyObject* Test();

		// Initialize member variables
		void InitializeVariables();

		// Create an output array.  Data are copied here when reading
		void CreateOutputArray();

		void ReadPrepare();

		void ReadFromFile();
		void ReadAllAsBinary();
		void ReadRows();
		void ReadRow();
		void ReadFields();
		void ReadField(long long fnum);
		void ReadFieldAsBinary(long long fnum);
		void ReadFieldAsAscii(long long fnum);
		void ReadAsciiBytes(long long fnum);
		void ScanVal(long long fnum);
		void SkipField(long long fnum);
		void SkipFieldAsBinary(long long fnum);
		void SkipFieldAsAscii(long long fnum);
		void ReadWholeRowBinary();
		void SkipRows(long long current_row, long long row2read);
		void SkipAsciiRows(long long nskip);
		void SkipBinaryRows(long long nskip);

		void MakeScanFormats(bool add_delim);
		void MakePrintFormats();

		void SubDtype(
				PyObject* descr, 
				PyObject* subnames,
				PyObject** newdescr,
				vector<long long>& matchids);

		PyObject* ExtractSubDescr(
				PyArray_Descr* descr, 
				vector<string>& names);

		void WriteAllAsBinary();
		void WriteRows();
		void WriteField(long long fnum);
		void WriteNumberAsAscii(char* buffer, long long type);
		void WriteStringAsAscii(long long fnum);


		void ListStringMatch(
				vector<string> snames,
				PyObject* list, 
				vector<long long>& matchids);

		// Copy some info from a fields["fname"].descr into a tuple This will
		// become part of a list of tuples dtype send to the converter
		PyObject* FieldDescriptorAsTuple(
				PyArray_Descr* fdescr, const char* name);

		long long SequenceCheck(PyObject* obj);

		void CopyFieldInfo(
				PyArray_Descr* descr, 
				vector<string>& names, 
				vector<long long>& offsets,
				vector<long long>& sizes,
				vector<long long>& nel,
				vector<long long>& typenums,
				long long& rowsize);


		void CopyDescrOrderedNames(
				PyArray_Descr* descr, 
				vector<string>& names);

		void CopyDescrOrderedOffsets(
				PyArray_Descr* descr, 
				vector<string>& names,
				vector<long long>& offsets,
				vector<long long>& sizes,
				vector<long long>& nel,
				vector<long long>& typenums);


		// Must decref this arr no matter what. Use Py_XDECREF in case it
		// is NULL
		PyObject* Object2IntpArray(PyObject* obj);

		// Get the file pointer or open the file if it is a string.  
		void GetFptr(PyObject* file_obj, const char* mode);


		// Set the file type based on the delimeter
		void SetFileType();
		// Check the input and if good copy into mDelim string
		void ProcessDelim(PyObject* delim_obj);
		// Check the input descr and get a new reference to it in mTypeDescr
		void ProcessDescr(PyObject* descr);
		// Check the input nrows and copy to mNrows
		void ProcessNrows(long long nrows); 

		// Process the rows keyword and get a version that is an array
		void ProcessRowsToRead(PyObject* rows);
		// Process the fields keyword and extract a sub descr if necessary
		void ProcessFieldsToRead(PyObject* fields);

		void DebugOut(const char* mess);
		void PyDictPrintKeys(PyObject* dict);



		// Data


		// --- means we will initialize 
		// +++ possibly need to decref

		// File name or object
		PyObject* mFileObj;                                    //---
		string mMode;

		// The input type descriptor for each row of the file
		PyObject* mTypeDescr;                                  //--- +++
		// Optional rows to read, default to all.   We will decref
		PyObject* mRowsToRead;                                  //--- +++

		// The return object
		PyArrayObject* mReturnObject;                          //---
		// points to data area
		char* mData;                                           //---

		// A buffer for when skipping ascii
		string mBuffer;

		// Will hold scan and print formats for each data type
		vector<string> mScanFormats;
		vector<string> mPrintFormats;


		FILE* mFptr;                                           //---
		bool mFptrIsLocal;                                     //---

		// Delimiter for ascii files
		string mDelim;

		// Reading as binary or ascii?
		bool mReadAsWhitespace;                                //---

		// Read whole file with big fread?
		bool mReadWholeFileBinary;
		// Can read whole rows in binary?
		bool mReadWholeRowBinary;                              //---





        // Info about each row of file
        vector<string> mNames;        // Names of all fields in file
        vector<long long> mOffsets;   // offsets of each field in each row
        vector<long long> mSizes;     // size of each field in each row
		vector<long long> mNel;       // number of elements in this field
		vector<long long> mTypeNums;  // type numbers for each field
        long long mRowSize;           // total size of each row
        vector<long long> mKeep; // boolean, tells if we are keeping each field
		long long mNfields;           // number of fields

        // Info about the fields we are keeping
		PyObject* mKeepTypeDescr;    // descr for the kept fields +++
        vector<string> mKeepNames;   // Names of fields we will retrieve
		// offsets within the kept data structure
        vector<long long> mKeepOffsets;  
        vector<long long> mKeepSizes;      // size of kept fields
		vector<long long> mKeepNel;        // number of elements each field
		vector<long long> mKeepTypeNums;   // type numbers
        long long mKeepRowsize;            // size of kept data structure
        vector<long long> mKeepId;         // index back into above info
		long long mKeepNfields; // number of fields kept

        npy_intp mNrows;             // Total number of rows in file
        npy_intp mNrowsToRead;       // Number of rows we are actually reading.

		int mFileType;
		int mAction;

		// Action codes
		static const int READ = 0;
		static const int WRITE = 1;

		// File types
		static const int BINARY_FILE = 0;
		static const int ASCII_FILE = 1;


		static const bool mDebug=false;
		//static const bool mDebug=true;
};

// Should only be executed once
//import_array();

#endif
