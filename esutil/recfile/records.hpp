#include <Python.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include "numpy/arrayobject.h"

#ifndef _readfields_2_h
#define _readfields_2_h

using namespace std;


class Records {
    public:

		Records(const char* filename, 
				const char* mode,
				PyObject* delim=NULL, 
				PyObject* dtype=NULL,
				long long nrows=-9999,
                long offset=0,
                int bracket_arrays=0,
                bool padnull=false,
                bool ignorenull=false
        ) throw (const char *);

        ~Records();

		void close() throw (const char*);

        // TODO convert
		PyObject* Write(PyObject* obj) throw (const char *);


        PyObject* write_string(PyObject* obj) throw (const char* );
        PyObject* update_row_count(long nrows) throw (const char* );


        // new style
        PyObject* read_columns(PyObject* arrayobj,
                               PyObject* colnums,
                               PyObject* rows) throw (const char* );

        PyObject* read_binary_slice(PyObject* arrayobj,
                                    long long row1,
                                    long long row2,
                                    long long step) throw (const char* );

    private:

		// Check the input nrows and copy to mNrows
		void process_nrows(long long nrows); 

		void do_seek(npy_intp seek_distance);
        void goto_offset(void);

        // new style
        npy_intp get_nrows_to_read(PyObject* rows);
        npy_intp get_ncols_to_read(PyObject* rows);

        void scan_column_values(long long fnum, char* buff);
        void read_ascii_bytes(long long colnum, char* buff);
        void read_from_text_column(long long colnum, char* buff);
        void read_from_binary_column(long long colnum, char* buff);

        void read_binary_columns(PyObject* arrayobj,
                                 PyObject* colnums,
                                 PyObject* rows) throw (const char* );
        void skip_ascii_col_range(npy_intp start, npy_intp stop);
        void read_text_columns(PyObject* arrayobj,
                               PyObject* colnums,
                               PyObject* rows) throw (const char* );


		// Initialize member variables
		void init_variables();


        void ensure_writable(void) throw (const char* );
        void ensure_readable(void) throw (const char* );
        void ensure_binary(void) throw (const char* );
        void ensure_text(void) throw (const char* );

        npy_intp process_slice(npy_intp row1, npy_intp row2, npy_intp step);
		void skip_rows(long long current_row, long long row2read);
		void skip_text_rows(long long nskip);
		void skip_binary_rows(long long nskip);

		void make_scan_formats(vector<string> &formats, bool add_delim);
		void make_print_formats(vector<string> &formats);

        // TODO still need to be converted
		void WriteAllAsBinary();
		void WriteRows();
		void WriteField(long long fnum);
        void WriteArrayFieldWithBrackets(long long fnum);
        void _WriteArrayWithBrackets(long long fnum, long long dim);
		void WriteNumberAsAscii(char* buffer, long long type);
		void WriteStringAsAscii(long long fnum);


		void copy_field_info(PyArray_Descr* descr);
		void copy_descr_ordered_names(PyArray_Descr* descr);
		void copy_descr_ordered_offsets(PyArray_Descr* descr);

		// Get the file pointer or open the file if it is a string.  
		void set_fptr(const char* filename, const char* mode);


		// Set the file type based on the delimeter
		void set_file_type();
		// Check the input and if good copy into mDelim string
		void process_delim(PyObject* delim_obj);
		// Check the input descr and get a new reference to it in mTypeDescr
		void process_descriptor(PyObject* descr);

		void debugout(const char* mess);



		// Data


		// --- means we will initialize 
		// +++ possibly need to decref

        // mode opening file
		string mMode;

        long mFileOffset;

		int mFileType;
		int mAction;

        npy_intp mNrows;             // Total number of rows in file

		// The input type descriptor for each row of the file
		PyObject* mTypeDescr;                                  //--- +++

		// Will hold scan and print formats for each data type
		vector<string> mScanFormats;
		vector<string> mPrintFormats;

		FILE* mFptr;                                           //---

		// Delimiter for ascii files
		string mDelim;
        // this can be different when bracket_arrays is sent
        // since we demand commas there
        string mArrayDelim;

		// Reading as binary or ascii?
		bool mReadAsWhitespace;                                //---

        // when writing text, padd out nulls in strings with spaces
		bool mPadNull;

        // when writing text, do not write beyond the null
        // for reading back in this may cause problems for some delimiters
		bool mIgnoreNull;

        // for postgres
        int mBracketArrays;

        // Info about each row of file
        vector<string> mNames;        // Names of all fields in file
        vector<long long> mOffsets;   // offsets of each field in each row
        vector<long long> mSizes;     // size of each field in each row
		vector<long long> mNel;       // number of elements in this field
        vector<long long> mNdim;      // ndim for each field
        vector<vector<long long> > mDims;      // a dims array
		vector<long long> mTypeNums;  // type numbers for each field
        long long mRowSize;           // total size of each row
        vector<long long> mKeep; // boolean, tells if we are keeping each field
		long long mNfields;           // number of fields

        // TODO temporarily point to data being written; need to convert
        char *mData;

        // constants

		// Action bits
		static const int READ = 1;
		static const int WRITE = 2;

		// File types
		static const int BINARY_FILE = 0;
		static const int ASCII_FILE = 1;


		static const bool mDebug=false;
		//static const bool mDebug=true;
};


#endif
