#include "records.hpp"

/*
   we need this because someone made the import_array macro return a value
   in python 3

   first lesson of C macros:  do not make them use the return statment
*/

#if PY_MAJOR_VERSION >= 3
static int *init_numpy(void) {
#else
static void init_numpy(void) {
#endif
	import_array();
}

// check unicode for python3, string for python2
static int is_python_string(const PyObject* obj)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_Check(obj) || PyBytes_Check(obj);
#else
    return PyUnicode_Check(obj) || PyString_Check(obj);
#endif
}

// unicode is common to python 2 and 3
static char* get_unicode_as_string(PyObject* obj)
{
    PyObject* tmp=NULL;
    char* strdata=NULL;
    tmp = PyObject_CallMethod(obj,(char*)"encode",NULL);

    strdata = strdup( PyBytes_AsString(tmp) );
    Py_XDECREF(tmp);

    return strdata;
}


static string get_object_as_string(PyObject* obj)
{
    PyObject* format=NULL;
    PyObject* args=NULL;
    string strdata;
    PyObject* tmpobj1=NULL;

    if (PyUnicode_Check(obj)) {

        strdata=get_unicode_as_string(obj);

    } else {

#if PY_MAJOR_VERSION >= 3

        if (PyBytes_Check(obj)) {
            strdata = PyBytes_AsString(obj);
        } else {
            PyObject* tmpobj2=NULL;
            format = Py_BuildValue("s","%s");
            // this is not a string object
            args=PyTuple_New(1);

            PyTuple_SetItem(args,0,obj);
            tmpobj2 = PyUnicode_Format(format, args);
            tmpobj1 = PyObject_CallMethod(tmpobj2,"encode",NULL);

            Py_XDECREF(args);
            Py_XDECREF(tmpobj2);

            strdata = PyBytes_AsString(tmpobj1);
            Py_XDECREF(tmpobj1);
            Py_XDECREF(format);
        }

#else
        // convert to a string as needed
        if (PyString_Check(obj)) {
            strdata = PyString_AsString(obj);
        } else {
            format = Py_BuildValue("s","%s");
            args=PyTuple_New(1);

            PyTuple_SetItem(args,0,obj);
            tmpobj1= PyString_Format(format, args);

            strdata = PyString_AsString(tmpobj1);
            Py_XDECREF(args);
            Py_XDECREF(tmpobj1);
            Py_XDECREF(format);
        }
#endif
    }

    return strdata;
}



Records::Records(
        const char *filename,
		const char* mode,
		PyObject* delimobj, 
		PyObject* dtype,
		long long nrows,
        long offset,
        int bracket_arrays,
        bool padnull,
        bool ignorenull)
{
    init_numpy();

	init_variables();

    mBracketArrays = bracket_arrays;
    mPadNull=padnull;
    mIgnoreNull=ignorenull;

	mMode=mode;

	set_fptr(filename, mMode.c_str());
	process_delim(delimobj);
	set_file_type();

    mFileOffset = offset;

	if (mMode[0] == 'r' || mMode == "w+") {
		if ( (dtype == NULL) || (nrows==-9999) ) {
			throw std::runtime_error("You must send the datatype and number of rows when reading");
		}
		// Open for reading
		mAction=READ;
        if (mMode.size() > 1 && mMode=="r+") {
            mAction |= WRITE;
        } else if (mMode[0]=='w') {
            // both write and read
            mAction |= WRITE;
        }

        goto_offset();

		process_descriptor(dtype);
		process_nrows(nrows);

	} else {
		// Only opened for writing
		mAction=WRITE;
	}

    make_scan_formats(mScanFormats,true);
    make_print_formats(mPrintFormats);

}


Records::~Records() 
{

	// always decref; can be NULL but otherwise points to an input
	// type descriptor and we did an INCREF
	Py_XDECREF(mTypeDescr);
	this->close();

}

void Records::close() 
{
	if (mFptr != NULL) {
		if (mDebug) debugout("Closing file");
		fclose(mFptr);
		mFptr=NULL;
	}
}


void Records::init_variables()
{


    mData=NULL;

	// The type descriptor for each row of the file.  Will decref since we
	// increfed as we made a copy
	mTypeDescr=NULL;

	mFptr=NULL;

	mDelim="";
    mArrayDelim="";

	// must be set later!!
	mAction=READ;
	mFileType = BINARY_FILE;

	mReadAsWhitespace=false;

	mNrows=0;

    mIgnoreNull=false;
    mPadNull=false;
    mBracketArrays=0;
	return;

}

void Records::process_nrows(long long nrows)  
{
	if (mDebug) {cerr<<"nrows = "<<nrows<<endl;fflush(stdout);}
	if (nrows < 1) {
		throw std::runtime_error("Input nrows must be >= 1");
	}
	mNrows = nrows;
}


void Records::ensure_writable(void) 
{
	if (mFptr == NULL) {
		throw std::runtime_error("File is not open");
	}
	if ( (mAction & WRITE) == 0) {
		throw std::runtime_error("File is not open for writing");
	}
}
void Records::ensure_readable(void) 
{
	if (mFptr == NULL) {
		throw std::runtime_error("File is not open");
	}
	if ( (mAction & READ) == 0) {
		throw std::runtime_error("File is not open for reading");
	}
}

void Records::ensure_binary(void) 
{
    if (mFileType != BINARY_FILE) {
		throw std::runtime_error("attempt to read ascii data as binary");
    }
}
void Records::ensure_text(void) 
{
    if (mFileType != ASCII_FILE) {
		throw std::runtime_error("attempt to read binary data as text");
    }
}

void Records::goto_offset(void)
{
    fseek(mFptr, mFileOffset, SEEK_SET);
}

void Records::do_seek(npy_intp seek_distance)  {
	if (seek_distance > 0) {
		if(fseeko(mFptr, seek_distance, SEEK_CUR) != 0) {
			string err="Error skipping fields";
			throw std::runtime_error(err);
		}
	}
}

void Records::skip_rows(long long current_row, long long row2read) 
{
	long long rows2skip=0;
	if (mFileType == BINARY_FILE) {
		rows2skip = row2read-current_row;
		skip_binary_rows(rows2skip);
	} else {
		if (mReadAsWhitespace) {
			rows2skip = row2read - current_row;// + 1;
		} else {
			rows2skip = row2read - current_row;
		}
		skip_text_rows(rows2skip);
	}
}



void Records::skip_text_rows(long long nskip) 
{
	if (nskip > 0) {
		long long nlines = 0;
		char c;
		while (nlines < nskip) {
			c = fgetc(mFptr);
			if (c == EOF) {
				throw std::runtime_error("Reached EOF prematurely");
			}
			if (c == '\n') {
				nlines++;
			}
		}
	}
}

void Records::skip_binary_rows(long long nskip) 
{
	if (nskip > 0) {
		if (fseeko(mFptr, mRowSize*nskip, SEEK_CUR) != 0) {
			throw std::runtime_error("Failed to fseek");
		}
	}
}





// read all the elements of a field
void Records::scan_column_values(long long fnum, char* input_buff) 
{

    int skipping=false;
    string tmp;

    char *buff=NULL;

    if (input_buff) {
        skipping=false;
        buff=input_buff;

    } else {
        skipping=true;

        // a buffer big enough for one scan

        long buffsize = mSizes[fnum]/mNel[fnum] + 1;
        tmp.resize(buffsize,'\0');
        buff = &tmp[0];

    }

	int type_num = mTypeNums[fnum];

	for (long long el=0; el<mNel[fnum]; el++) {
		int ret = fscanf(mFptr, mScanFormats[type_num].c_str(), buff);
		if (ret != 1) {
			string err="ScanVal: Error reading field: "+mNames[fnum];
			if (feof(mFptr)) {
				err += ": EOF reached unexpectedly";
			}
			else {
				err += ": Read error";
			}
			throw std::runtime_error(err);
		}
        if (!skipping) {
            buff += mSizes[fnum]/mNel[fnum] ;
        }
	}
}


void Records::read_ascii_bytes(long long colnum, char* buff)  
{

    int skipping=false;
	char c;

    if (!buff) {
        skipping=true;
    }

	// Read the expected number of bytes *per element* as opposed to binary
	int size_per_el = mSizes[colnum]/mNel[colnum];

	// Loop over each element for ascii. Must do this because
	// of the delimters
	for (long long el=0; el<mNel[colnum]; el++) {

		for (long long i=0; i<size_per_el; i++) {
			c=fgetc(mFptr);
			if (c==EOF) {
				string err=
					"EOF reached unexpectedly reading field: "+
					mNames[colnum];
				throw std::runtime_error(err);
			}

            // if NULL, we are skipping this data
            if (!skipping) {
                *buff = c;
                buff++;
            }
		}

		// Read the delimiter or EOL
		c=fgetc(mFptr);

	}
}


// read single entry
void Records::read_from_text_column(long long colnum, char* buff) 
{

	if (mTypeNums[colnum] == NPY_STRING) {
		read_ascii_bytes(colnum, buff);
	} else {
		scan_column_values(colnum, buff);
		// For whitespace we haven't read the delimiter yet
		if (mReadAsWhitespace) {
			fgetc(mFptr);
		}
	}
}

// read single entry
void Records::read_from_binary_column(long long colnum, char* buff) 
{
    int nread = fread(buff, mSizes[colnum], 1, mFptr);
    if (nread != 1) {
        string err="Error reading field: "+mNames[colnum];
        throw std::runtime_error(err);
    }
}

npy_intp Records::get_nrows_to_read(PyObject* rows)
{

    npy_intp nrows = mNrows;
    if (rows != Py_None) {
        nrows = PyArray_SIZE(rows);
    }

    return nrows;
}

npy_intp Records::get_ncols_to_read(PyObject* colnums)
{

    npy_intp ncols = mNfields;
    if (colnums != Py_None) {
        ncols = PyArray_SIZE(colnums);
    }

    return ncols;
}






/*

   subsets of rows and columns

   colnums and rows must be of type npy_int64, and must be unique
   and sorted

*/

PyObject* Records::read_columns(PyObject* arrayobj,
                                PyObject* colnums,
                                PyObject* rows) 

{

    if (mFileType == BINARY_FILE) {
        read_binary_columns(arrayobj, colnums, rows);
    } else {
        read_text_columns(arrayobj, colnums, rows);
    }
    Py_RETURN_NONE;
}


// stop is exclusive
void Records::skip_ascii_col_range(npy_intp start, npy_intp stop) 
{
    for (npy_intp col=start; col<stop; col++) {
        read_from_text_column(col, NULL);
    }
}

void Records::read_text_columns(PyObject* arrayobj,
                                     PyObject* colnums,
                                     PyObject* rows) 
{
    bool doall_rows=false, doall_cols=false;
	npy_intp
        current_row=0, current_col=0,
        row2read=0, col2read=0;

    ensure_readable();
    ensure_text();

    npy_intp ncols2read = get_ncols_to_read(colnums);
    npy_intp nrows2read = get_nrows_to_read(rows);

	if (nrows2read != mNrows) {
        doall_rows=false;
    } else {
        doall_rows=true;
    }
	if (ncols2read != mNfields) {
        doall_cols=false;
    } else {
        doall_cols=true;
    }

    //cerr<<"doall rows: "<<doall_rows<<" doall cols: "<<doall_cols<<"\n";

    // always begin at the user's requested file offset
    goto_offset();

    for (npy_intp irow=0; irow<nrows2read; irow++) {
        char *ptr= (char *) PyArray_GETPTR1(arrayobj, irow);

        if (doall_rows) {
            row2read = irow;
        } else {
            row2read = *(npy_int64 *) PyArray_GETPTR1(rows, irow);
            if (row2read > current_row) {
                skip_rows(current_row, row2read);
                current_row=row2read;
            } 
        }

        current_col=0;

        for (npy_intp icol=0; icol<ncols2read; icol++) {
            if (doall_cols) {
                col2read=icol;
            } else {
                col2read = *(npy_int64 *) PyArray_GETPTR1(colnums, icol);
            }

            if (col2read > current_col) {
                skip_ascii_col_range(current_col, col2read);
                current_col=col2read;
            } 

            read_from_text_column(col2read, ptr);

            // move the data pointer. Assumes C contiguous within a column

            ptr += mSizes[col2read];

            // move the current col to the next one to indicate we have moved
            // in the file passed the requested column
            current_col++;

        }

        // skip the rest of the row if needed
        if (current_col < mNfields) {
            skip_ascii_col_range(current_col, mNfields);
        }

        current_row++ ;
    }

}



void Records::read_binary_columns(PyObject* arrayobj,
                                  PyObject* colnums,
                                  PyObject* rows) 
{
    bool doall_rows=false;
	npy_intp
        current_row=0, current_col=0,
        current_offset=0,
        row2read=0, col2read=0,
        seek_distance=0;
    long long colsize=0;

    ensure_readable();
    ensure_binary();

    npy_intp ncols2read = get_ncols_to_read(colnums);
    npy_intp nrows2read = get_nrows_to_read(rows);

	if (nrows2read != mNrows) {
        doall_rows=false;
    } else {
        doall_rows=true;
    }

    // always begin at the user's requested file offset
    goto_offset();

    for (npy_intp irow=0; irow<nrows2read; irow++) {
        char *ptr= (char *) PyArray_GETPTR1(arrayobj, irow);

        if (doall_rows) {
            row2read = irow;
        } else {
            row2read = *(npy_int64 *) PyArray_GETPTR1(rows, irow);
        }

        //fprintf(stderr,"current_row: %ld row2read: %ld\n", current_row, row2read);

		if (row2read > current_row) {
			skip_rows(current_row, row2read);
			current_row=row2read;
		} 

        current_col=0;
        current_offset=0; // offset into this row

        for (npy_intp icol=0; icol<ncols2read; icol++) {
            //fprintf(stderr,"col2read: %ld\n", col2read);
            col2read = *(npy_int64 *) PyArray_GETPTR1(colnums, icol);
            colsize=mSizes[col2read];

            if (col2read > current_col) {
                seek_distance = mOffsets[col2read] - current_offset;
                do_seek(seek_distance);

                current_col=col2read;
                current_offset += seek_distance;
            } 

            read_from_binary_column(col2read, ptr);

            // account for offset after read
            current_offset += colsize;

            // move the data pointer also. Assumes C contiguous within
            // a columns
            ptr += colsize;

            current_col++;
        }

        // skip the rest of the row if needed
        if (current_offset < mRowSize) {
            seek_distance = mRowSize - current_offset;
            do_seek(seek_distance);
        }

        current_row++ ;
    }

}

npy_intp Records::process_slice(npy_intp row1, npy_intp row2, npy_intp step) 
{
	// Just do some error checking on the requested rows
	stringstream serr;
	if (row1 < 0) {
		serr<<"Requested first row < 0";
		throw std::runtime_error(serr.str());
	}
	if (row2 > mNrows) {
		serr<<"Requested slice beyond delcared size "<<mNrows;
		throw std::runtime_error(serr.str());
	}

	if (step <= 0) {
		serr<<"Requested step must be > 0";
		throw std::runtime_error(serr.str());
	}

	// we use python slicing rules:  [n1:n2:step] really means  from n1 to n2-1
	// so the number of rows to read is simply (n2-n1)/step + (n2-21) % step
	npy_intp rdiff = (row2-row1);

	npy_intp extra = 0;
	if ((rdiff % step) != 0) {
		extra = 1;
	}
	npy_intp nrows = rdiff/step + extra;


	if (mDebug) cerr<<"slice: ("<<row1<<", "<<row2<<", "<<step<<") nrows: "<<nrows<<"/"<<mNrows<<"\n";
	return nrows;
}



/*

   read all columns in a row slice

   binary only; there is only one text reader for
   all reading types

   the input array must have the right size
*/


PyObject* Records::read_binary_slice(PyObject* arrayobj,
                                     long long row1,
                                     long long row2,
                                     long long step) 
{

    ensure_readable();
    ensure_binary();

	npy_intp nrows2read = process_slice(row1, row2, step);

    // always begin at the user's requested file offset
    goto_offset();

    if (row1 > 0) {
		skip_binary_rows(row1);
    }

    if (step==1) {
        // we can use a single fread
        void *ptr = PyArray_GETPTR1(arrayobj, 0);

        npy_intp nread = (npy_intp) fread(ptr, mRowSize, nrows2read, mFptr);
        if (nread != nrows2read) {
            throw std::runtime_error("Error reading slice");
        } 

    } else {

        for (npy_intp irow=0; irow<nrows2read; irow++) {

            void *ptr = PyArray_GETPTR1(arrayobj, irow);

            size_t nread = fread(ptr, mRowSize, 1, mFptr);
            if (nread != 1) {
                throw std::runtime_error("Failed to read row data");
            }

            skip_binary_rows(step-1);

        }
    }


    Py_RETURN_NONE;
}





/*


PyObject* Records::ReadSlice(long long row1, long long row2, long long step)  
{
    ensure_readable();

    // always start at the users requested offset
    goto_offset();

	// juse some error checking and return implied length
	mNrowsToRead = ProcessSlice(row1, row2, step);

	// slice we read all fields, so send Py_None
	ProcessFieldsToRead(Py_None);
	CreateOutputArray();

	ReadPrepare();


	if (mReadWholeFileBinary) {
		ReadAllAsBinary();
	} else {
		ReadRowsSlice(row1, step);
	}

	return (PyObject* ) mReturnObject;
}

PyObject* Records::Read(
		PyObject* rows,
		PyObject* fields) 
{
    ensure_readable();

    goto_offset();

	ProcessRowsToRead(rows);
	ProcessFieldsToRead(fields);
	CreateOutputArray();
	ReadPrepare();

	ReadFromFile();

	return (PyObject* ) mReturnObject;
}

void Records::ReadPrepare()
{
    mReadWholeFileBinary = false;
    mReadWholeRowBinary = false;

	if (mFileType == BINARY_FILE 
			&& mNrowsToRead == mNrows 
			&& mKeepNfields == mNfields) {

		mReadWholeFileBinary = true;
	} else if (mFileType == BINARY_FILE
			&& mKeepNfields == mNfields ) {

		mReadWholeRowBinary = true;
	} else if (mFileType == ASCII_FILE) {
		//make_scan_formats(true);
	}
}

void Records::ReadFromFile()
{
	if (mReadWholeFileBinary) {
		ReadAllAsBinary();
	} else {
		ReadRows();
	}
}

void Records::ReadAllAsBinary()
{
	if (mDebug) debugout("Reading all in one big fread()");
	int nread = fread(mData, mRowSize, mNrows, mFptr);
	if (nread != mNrows) {
		throw std::runtime_error("Error reading entire file as binary");
	} 
}


// need to use long long
void Records::ReadRows()
{

	// Will hold row data if we are skipping rows (stored as array)
	npy_intp* rows=NULL;
	npy_intp current_row=0;
	npy_intp row2read=0;

	if (mNrowsToRead != mNrows) {
		// No data created or copied here
		rows = (npy_intp*) PyArray_DATA(mRowsToRead);
	}
	if (mDebug) debugout("Reading rows");

	// Loop over the rows to read, which could be a subset of the 
	// total number of rows in the file.
	for (npy_intp irow=0;  irow<mNrowsToRead; irow++) {
		if (mNrowsToRead != mNrows) {
			row2read=rows[irow];
		} else {
			row2read=irow;
		}

		// Skip rows?
		if (row2read > current_row) {
			skip_rows(current_row, row2read);
			current_row=row2read;
		} 

		ReadRow();
		current_row++;
	}


}

void Records::ReadRowsSlice(npy_intp row1, npy_intp step) 
{

	if (mDebug) debugout("Reading rows by slice");

	if (step == 1 && mFileType == BINARY_FILE) {

		// We can just read a big chunk
		if (row1 > 0) {
			skip_rows(0, row1);
		}

		npy_intp nread = fread(mData, mRowSize, mNrowsToRead, mFptr);
		if (nread != mNrowsToRead) {
			throw std::runtime_error("Error reading slice");
		} 

	} else {

		npy_intp row2read = row1;
		npy_intp current_row = 0;

		for (npy_intp irow=0;  irow<mNrowsToRead; irow++) {

			// Skip rows
			if (row2read > current_row) {
				skip_rows(current_row, row2read);
				current_row=row2read;
			} 

			ReadRow();

			current_row++;
			row2read += step;
		}
	}

}




void Records::ReadRow()
{
	if (mReadWholeRowBinary) {
		// We can read a whole line if reading all fields
		ReadWholeRowBinary();
		
	} else if (mFileType == BINARY_FILE) {
		// Reading particular fields of a binary file.
		ReadBinaryFields();
	
	} else {
		// Reading particular fields
		ReadAsciiFields();
	}
}

void Records::ReadBinaryFields()
{
	// use messy code here for a significant speedup
	npy_intp 
		last_offset=0, last_fsize=0, seek_distance=0, offset=0;

	for (npy_intp fnum=0; fnum<mNfields; fnum++) {
		if (mKeep[fnum]) {
			// How far to we move before we read this data?
			offset  = mOffsets[fnum];
			seek_distance = offset-(last_offset + last_fsize);

			// This could be zero if we didn't skip any fields
			DoSeek(seek_distance);

			// Read the data
			ReadFieldAsBinary(fnum);

			last_offset=offset;
			last_fsize=mSizes[fnum];
		}
	}

	// Do we need to move past any remaining fields?
	seek_distance = mRowSize - (last_offset+last_fsize);
	DoSeek(seek_distance);

}

void Records::DoSeek(npy_intp seek_distance) {
	if (seek_distance > 0) {
		if(fseeko(mFptr, seek_distance, SEEK_CUR) != 0) {
			string err="Error skipping fields";
			throw std::runtime_error(err);
		}
	}
}

void Records::ReadAsciiFields()
{
	for (npy_intp fnum=0; fnum<mNfields; fnum++) {
		// This program understands when a field is skipped
		ReadFieldAsAscii(fnum);
	}
}




void Records::ReadFieldAsBinary(long long fnum)
{
	// Read the requested number of bytes
	int nread = fread(mData, mSizes[fnum], 1, mFptr);
	if (nread != 1) {
		string err="Error reading field: "+mNames[fnum];
		throw std::runtime_error(err);
	}
	// Move the data pointer
	mData = mData+mSizes[fnum];
}

void Records::ReadFieldAsAscii(long long fnum)
{

	if (mTypeNums[fnum] == NPY_STRING) {
		ReadAsciiBytes(fnum);
	} else {
		ScanVal(fnum);
		// For whitespace we haven't read the delimiter yet
		if (mReadAsWhitespace) {
			//char c = fgetc(mFptr);
			fgetc(mFptr);
		}
	}

	// Move the data pointer if we actually read this to the buffer
	if (mKeep[fnum]) {
		mData = mData+mSizes[fnum];
	}
}

void Records::ReadAsciiBytes(long long fnum)
{
	char c;
	char* buff;
	// If we are skipping this field just read into a different buffer
	if (mKeep[fnum]) {
		buff = mData;
	} else {
		//buff = &mBuffer[0];
		buff = (char *) mBuffer.c_str();
	}

	// Read the expected number of bytes *per element* as opposed to binary
	int size_per_el = mSizes[fnum]/mNel[fnum];

	// Loop over each element for ascii. Must do this because
	// of the delimters
	for (long long el=0; el<mNel[fnum]; el++) {

		for (long long i=0; i<size_per_el; i++) {
			c=fgetc(mFptr);
			if (c==EOF) {
				string err=
					"EOF reached unexpectedly reading field: "+
					mNames[fnum];
				throw std::runtime_error(err);
			}
			*buff = c;
			buff++;
		}

		// Read the delimiter or EOL
		c=fgetc(mFptr);

	}
}

void Records::ScanVal(long long fnum)
{

	char* buff;
	// If we are skipping this field just read into a different buffer
	if (mKeep[fnum]) {
		buff = mData;
	} else {
		//buff = &mBuffer[0];
		buff = (char *) mBuffer.c_str();
	}


	int type_num = mTypeNums[fnum];

	//{cerr<<"  ScanVal with format: "<<mScanFormats[type_num].c_str()<<endl;
	//		fflush(stdout);}
	for (long long el=0; el<mNel[fnum]; el++) {
		int ret = fscanf(mFptr, mScanFormats[type_num].c_str(), buff);
		if (ret != 1) {
			string err="ScanVal: Error reading field: "+mNames[fnum];
			if (feof(mFptr)) {
				err += ": EOF reached unexpectedly";
			}
			else {
				err = + ": Read error";
			}
			throw std::runtime_error(err);
		}
		buff += mSizes[fnum]/mNel[fnum] ;
	}
}

void Records::ReadWholeRowBinary()
{
	int nread = fread(mData, mRowSize, 1, mFptr);
	if (nread != 1) {
		throw std::runtime_error("Failed to read row data");
	}
	mData+=mRowSize;
}




void Records::CreateOutputArray()
{

	// this way we don't worry about freeing
	npy_intp d[1];
	PyArray_Dims shape;

	shape.ptr = d;
	shape.len = 1;


	if (mDebug) debugout("Creating output array");

	shape.ptr[0] = mNrowsToRead;

	if (mDebug) debugout("  Allocating");
	mReturnObject = (PyArrayObject* ) 
		PyArray_Zeros(
				1, 
				shape.ptr, 
				(PyArray_Descr *) mKeepTypeDescr, 
				NPY_FALSE);

	if (mReturnObject==NULL) {
		throw std::runtime_error("Could not allocate array");
	}

	
	// Now the array has been created, and will not be XDECREFEd under
	// any circumstances.  This is so the user can see the state of the
	// array should any errors occur.  Thus we must also keep around the
	// descr.  Se we will add an extra reference so we can decref it later

	Py_INCREF(mKeepTypeDescr);

	// Make a pointer to the data area
	mData = mReturnObject->data;
}


// given a numpy  PyArray_Descr* and a list of field names return a new
// type descriptor containing only the subset

void Records::SubDtype(
		PyObject* indescr, 
		PyObject* subnamesobj,
		PyObject** newdescr,
		vector<long long>& matchids) {

	PyArray_Descr* descr=(PyArray_Descr* ) indescr;
	//vector<string> names;

	// make string vector
	//copy_descr_ordered_names(descr);

	// This makes sure they end up in the original order: important
	// for skipping fields and such
	
	// First deal with a scalar string or list input
	if (PyList_Check(subnamesobj)) {
		ListStringMatch(mNames, subnamesobj, matchids);
	} else if (is_python_string(subnamesobj)) {
		// Must decref
		PyObject* tmplist = PyList_New(0);
		// Makes a copy on append.
		PyList_Append(tmplist, subnamesobj);
		ListStringMatch(mNames, tmplist, matchids);
		Py_XDECREF(tmplist);
	} else {
		throw std::runtime_error("fields keyword must be string or list");
	}
	vector<string> matchnames;
	matchnames.resize(matchids.size());
	for (unsigned long long i=0; i<matchids.size(); i++) {
		//matchnames[i] = names[matchids[i]];
		matchnames[i] = mNames[matchids[i]];
	}

	// Now based on the matches create a new dtype
	*newdescr = ExtractSubDescr(descr, matchnames);

}




// Extract a subset of the fields from a PyArray_Descr and return a new
// descr with that info
PyObject* Records::ExtractSubDescr(
		PyArray_Descr* descr, 
		vector<string>& names)
{

	PyArray_Descr *fdescr=NULL;
	char* title=NULL;
	long long offset;

	PyObject* dlist=PyList_New(0);
	PyArray_Descr* newdescr=NULL;

	if (mDebug) {cerr<<"Extracting sub descr"<<endl;fflush(stdout);}
	for (unsigned long long i=0; i<names.size(); i++) {
		PyObject* item =
			PyDict_GetItemString(descr->fields, names[i].c_str());

		if (item!=NULL) {
			if (!PyArg_ParseTuple(item, "Oi|O", &fdescr, &offset, &title)) {
				if (mDebug) 
				{cerr<<"Field: "<<names[i]<<" not right format"<<endl;}
			} else {

				PyObject* tup = 
					FieldDescriptorAsTuple(fdescr, names[i].c_str());

				// copy is made of tuple
				if (PyList_Append(dlist, tup) != 0) {
					throw std::runtime_error("Could not append to list");
				}
				Py_XDECREF(tup);

			}
		} else {
			if (mDebug) 
			{cerr<<"field: "<<names[i]<<" does not exist. offset->-1"<<endl;}
		}
	}

	// Now convert this list to a descr
	if (mDebug) {cerr<<"Converting list to descr"<<endl;fflush(stdout);}
	if (!PyArray_DescrConverter(dlist, &newdescr)) {
		throw std::runtime_error("data type not understood");
	}
	if (mDebug) {cerr<<"  Done"<<endl;fflush(stdout);};

	return( (PyObject* )newdescr);
}







// Copy some info from a fields["fname"].descr into a tuple
// This will become part of a list of tuples dtype send to the converter
PyObject* Records::FieldDescriptorAsTuple(PyArray_Descr* fdescr, const char* name)
{
	// Use a string stream to convert all the char and possible int
	// elements of a type string
	stringstream typestream (stringstream::in | stringstream::out);
	string typestring;

	long long nel=0, tupsize=0;
	PyObject* shape=NULL;
	if (fdescr->subarray != NULL) {
		// This is a sub-array and requires the tuple to have a
		// length specified Here we are implicitly only allowing
		// subarrays of basic numbers or strings

		typestream << fdescr->subarray->base->byteorder;
		typestream << fdescr->subarray->base->type;
		if (fdescr->subarray->base->type_num == NPY_STRING) {
			typestream << fdescr->subarray->base->elsize;
		}
		nel = fdescr->elsize/fdescr->subarray->base->elsize;

		// Need to incref this because the PyTuple_SetItem will
		// steal a reference
		shape = fdescr->subarray->shape;
		tupsize=3;
	} else {
		typestream << fdescr->byteorder;
		typestream << fdescr->type;
		if (fdescr->type_num == NPY_STRING) {
			typestream << fdescr->elsize;
		}
		nel = 1;
		tupsize=2;
	}

	typestream >> typestring;

	// A copy is made when inserting into the list 
	// so we need to decref this
	PyObject* tup=PyTuple_New(tupsize);

	// In setitems references are stolen, so better to just
	// put the expressions in there than possibly worry later
	// about references
	PyTuple_SetItem(
			tup,
			0,
#if PY_MAJOR_VERSION >= 3
			PyBytes_FromString(name)
#else
			PyString_FromString(name)
#endif
    );
	PyTuple_SetItem(
			tup,
			1,
#if PY_MAJOR_VERSION >= 3
			PyBytes_FromString(typestring.c_str())
#else
			PyString_FromString(typestring.c_str())
#endif
    );

	if (tupsize == 3) {
		PyTuple_SetItem(
				tup,
				2,
				shape);
		Py_XINCREF(shape);

	}

	if (mDebug) {
		cerr<<"("
			<<"'"
			<<get_object_as_string(PyTuple_GetItem(tup,0))<<"'"
			<<", '"
			<<get_object_as_string(PyTuple_GetItem(tup,1))<<"'";
		if (nel > 1) {
			cerr <<", "<<nel;
		}
		cerr <<")"<<endl;
	}



	return(tup);

}


// Must decref this arr no matter what. Use Py_XDECREF in case it
// is NULL
// AHHHHH!!!!  On my macbook core 2 duo, which is 64-bit, intp is 32-bit!!! Can't 
// figure out how to make it use 64-bit
PyObject* Records::Object2IntpArray(PyObject* obj)
{

	// NPY_DEFAULT is currently NPY_CARRAY
	int min_depth=0, max_depth=0, flags=NPY_DEFAULT;
	PyObject* arr=NULL;

	if (obj == NULL || obj == Py_None) {
		return NULL;
	}

	PyArray_Descr* descr=NULL;
	descr = PyArray_DescrNewFromType(NPY_INTP);

	if (descr == NULL) {
		throw std::runtime_error("could not create NPY_INPT descriptor");
	}
	// This will steal a reference to descr, so we don't need to decref
	// descr as long as we decref the array!
	arr = PyArray_FromAny(obj, descr, min_depth, max_depth, flags, NULL);
	if (arr == NULL) {
		throw std::runtime_error("Could not convert rows keyword to an array of type NPY_INTP");
	}
	return arr;
}



void Records::ListStringMatch(
		vector<string> snames,
		PyObject* list, 
		vector<long long>& matchids)
{

	if (mDebug) {cerr<<"Matching fields to subfields"<<endl;fflush(stdout);}
	long long len=SequenceCheck(list);

	matchids.clear();
	if (len <= 0) {
		// Just return all
		matchids.resize(snames.size());
		for (unsigned long long i=0; i<matchids.size(); i++)
		{
			matchids[i] = i;
		}
	} else {
		// Get strings from list.
		vector<string> goodones;
		for (long long i=0; i<len; i++) {
			PyObject* item = PySequence_GetItem(list, i);
			if (!is_python_string(item)) {
				cerr<<"fields["<<i<<"] is not a string; skipping"<<endl;
				fflush(stdout);
			} else {
				string ts = get_object_as_string(item);
				goodones.push_back(ts);
			}
		}
		if (goodones.size() == 0) {
			throw std::runtime_error("None of the requested fields are in string form");
		} else {
			// loop over snames and see which ones match the input list
			// this preserves order, which is important.
			for (unsigned long long i=0; i<snames.size(); i++) {
				string name=snames[i];
				// See if there is a match
				vector<string>::iterator matchel;
				matchel = find(goodones.begin(),goodones.end(),name);
				if (matchel != goodones.end()) {
					matchids.push_back(i);
				}
			}
		}
	}
	if (matchids.size() == 0) {
		throw std::runtime_error("None of the requested field names matched");
	}

}


long long Records::SequenceCheck(PyObject* obj)
{
	if (obj == NULL) {
		return -1;
	}
	long long len=0;
	// The docs claim this check always succeeds, but not on NULL
	if (PySequence_Check(obj)) {
		len=PySequence_Size(obj);
	} else {
		len=-1;
	}
	return len;

}
*/



/*
   For writing a header.  the new offset comes from the position after writing the header
*/

PyObject* Records::write_header_and_update_offset(PyObject* obj) 
{
    ensure_writable();

    // should not be necessary, since file should be empty
    rewind(mFptr);

    string header = get_object_as_string(obj);
    fprintf(mFptr, "%s", header.c_str());

    mFileOffset = ftell(mFptr);

    Py_RETURN_NONE;
}

/*

   special function to help SFile to update the header row count

   Rewind the file, write a new SIZE = line, then move back to
   the end of the file

*/

PyObject* Records::update_row_count(long nrows) 
{
    ensure_writable();

    // go back to the beginning
    rewind(mFptr);

    // write the fixed-size SIZE entry
    fprintf(mFptr, "SIZE = %20ld\n", nrows);

    // seek back to the end of the file
    fseek(mFptr, 0, SEEK_END);

    Py_RETURN_NONE;
}

PyObject* Records::read_sfile_header(void) 
{

    ensure_readable();

    // go back to the beginning
    rewind(mFptr);

	char endbuff[4]={0};
    size_t count=0;

	while (1) {
        char c = fgetc(mFptr);

        if (EOF==c) {
            throw std::runtime_error("EOF reached before reading header end");
        }

        count++;

        endbuff[0] = endbuff[1];
        endbuff[1] = endbuff[2];

        endbuff[2] = c;

        if (0==strncmp(endbuff,"END",3)) {
            break;
        }
    }

    // we need to add
    // 1 for the newline character
    // 1 for the empty line

    count += 2;

    string hdr;
    hdr.resize(count);
    rewind(mFptr);
    size_t nread = fread(&hdr[0], 1, count, mFptr);
    if (nread != count) {
        throw std::runtime_error("Error reading header");
    }

    return Py_BuildValue("sl", hdr.c_str(), ftell(mFptr));

}




PyObject* Records::Write(PyObject* obj) 
{
    ensure_writable();

    // always write from the end
    fseek(mFptr, 0, SEEK_END);

	PyObject* ret=Py_None;
	Py_INCREF(Py_None);


	if (!PyArray_Check(obj)) {
		throw std::runtime_error("Input must be a NumPy array object");
	}
	mNrows = PyArray_Size(obj);

	PyArray_Descr* descr = PyArray_DESCR(obj);

	copy_field_info(descr);

	mNfields = mNames.size();

	mData = (char* ) PyArray_DATA(obj);

	if (mDebug) debugout("Writing data");
	if (mFileType == BINARY_FILE) {
		WriteAllAsBinary();
	} else{
		WriteRows();
	}

	if (mDebug) debugout("Finished writing");
	return(ret);
}

void Records::WriteAllAsBinary() 
{
	// This is easy!
	if (mDebug) debugout("Writing in one big fwrite");
	npy_intp nwrite = fwrite(mData, mRowSize, mNrows, mFptr);
	if (nwrite < mNrows) {
		stringstream serr;
		string err;
		serr<<"Error occured writing binary data: Expected "
			<<mNrows<<" but only wrote "<<nwrite;

		err=serr.str();
		throw std::runtime_error(err);
	}

}

void Records::WriteRows() 
{
	if (mDebug) {
		cerr<<"Writing "<<mNrows<<" rows as ASCII"<<endl;
		fflush(stdout);
	}
	if (mDebug) debugout("Writing rows");
	for (long long row=0; row< mNrows; row++) {
		for (long long fnum=0; fnum< mNfields; fnum++) {

            if (mBracketArrays && mNdim[fnum] > 0) {
                WriteArrayFieldWithBrackets(fnum);
            } else {
                WriteField(fnum);
            }
		} // fields
		// Write the newline character
		fputc('\n', mFptr);
	} // rows
}

void Records::WriteField(long long fnum)  
{

	long long nel=mNel[fnum];
	long long elsize = mSizes[fnum]/nel;
	long long type_num = mTypeNums[fnum];

	for (long long el=0; el<nel; el++) {

		if (type_num == NPY_STRING) {
			WriteStringAsAscii(fnum);
		} else {
			WriteNumberAsAscii(mData, type_num);
		}

		// Add a delimiter between elements
		if (el < (nel-1) ) {
            fprintf(mFptr, "%s", mDelim.c_str());
		}

		mData += elsize;

	}

	// Also will add a delim after the field
	if ( fnum < (mNfields-1) ) {
		fprintf(mFptr, "%s", mDelim.c_str());
	}

}

void Records::WriteArrayFieldWithBrackets(long long fnum)  
{

    // [3,2] looks like this:
    //   {{0.332407,0.864918},{0.777847,0.915038},{0.969121,0.866417}}
    // [3,2,4]
    // {{{0.976173,0.220988,0.207728,0.150891},{0.77637,0.405874,0.817494,0.0382292}},{{0.295267,0.0950662,0.629128,0.584864},{0.331606,0.749993,0.848343,0.430986}},{{0.379886,0.483621,0.280487,0.732344},{0.975598,0.518987,0.75701,0.274867}}}

	//long long nel=mNel[fnum];
	//long long elsize = mSizes[fnum]/nel;
	//long long type_num = mTypeNums[fnum];

    // Begin with the first dimension
    _WriteArrayWithBrackets(fnum, 0);

	// Also will add a regular delim after the field
	if ( fnum < (mNfields-1) ) {
		fprintf(mFptr, "%s", mDelim.c_str());
	}

}


void Records::_WriteArrayWithBrackets(long long fnum, long long dim)  {

	long long nel=mNel[fnum];
	long long elsize = mSizes[fnum]/nel;
	long long type_num = mTypeNums[fnum];

    // size of this dimension
    long long thisdim = mDims[fnum][dim];

    fprintf(mFptr,"{");
    for (int i=0; i<thisdim; i++) {

        if (dim < (mNdim[fnum]-1)) {
            // If we arent' on the last dimension, don't write anything yet
            // just call recursively
            _WriteArrayWithBrackets(fnum, dim+1);
        } else {

            if (type_num == NPY_STRING) {
                WriteStringAsAscii(fnum);
            } else {
                WriteNumberAsAscii(mData, type_num);
            }

            //WriteNumberAsAscii(mData, type_num);
            mData += elsize;
        }

        // Add an array delimiter between elements
        if (i < (thisdim-1) ) {
            fprintf(mFptr, "%s", mArrayDelim.c_str());
        }
    }
    fprintf(mFptr,"}");
}



void Records::WriteStringAsAscii(long long fnum) 
{
	char* buffer=NULL;

	buffer = mData;

	long long slen = mSizes[fnum]/mNel[fnum];
	for (long long i=0; i<slen; i++) {
		char c=buffer[0];
		if (c == '\0') {
			if (mIgnoreNull) {
				// we assume the user cares about nothing beyond the null
				// this will break out of writing this the rest of this field.
				break;
			}
			if ( mPadNull ) {
				c=' ';
			}
		}
		int res = fputc( (int) c, mFptr);
		if (res == EOF) {
			throw std::runtime_error("Error occured writing string field");
		}
		buffer++;
	}
}

void Records::WriteNumberAsAscii(char* buffer, long long type) 
{
	int res;

	switch (type) {
		case NPY_INT8:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_int8* )buffer ); 	
			break;
		case NPY_UINT8:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_uint8* )buffer ); 	
			break;

		case NPY_INT16:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_int16* )buffer ); 	
			break;
		case NPY_UINT16:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_uint16* )buffer ); 	
			break;

		case NPY_INT32:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_int32* )buffer ); 	
			break;
		case NPY_UINT32:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_uint32* )buffer ); 	
			break;

		case NPY_INT64:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_int64* )buffer ); 	
			break;
		case NPY_UINT64:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_uint64* )buffer ); 	
			break;

#ifdef NPY_INT128
		case NPY_INT128:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_int128* )buffer ); 	
			break;
		case NPY_UINT128:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_uint128* )buffer ); 	
			break;
#endif
#ifdef NPY_INT256
		case NPY_INT256:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_int256* )buffer ); 	
			break;
		case NPY_UINT256:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_uint256* )buffer ); 	
			break;
#endif

		case NPY_FLOAT32:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_float32* )buffer ); 	
			break;
		case NPY_FLOAT64:
			res= fprintf( mFptr, 
					mPrintFormats[type].c_str(), *(npy_float64* )buffer ); 	
			break;
#ifdef NPY_FLOAT128
		case NPY_FLOAT128:
			res= fprintf( mFptr,
					mPrintFormats[type].c_str(),*(npy_float128* )buffer ); 	
			break;
#endif

		default:
			stringstream serr;
			string err;
			serr << "Unsupported type "<<type;
			err=serr.str();
			throw std::runtime_error(err);
			break;
	}

	if (res < 0) {
		throw std::runtime_error("Error writing data");
	}
}






/*
void Records::ProcessFieldsToRead(PyObject* fields)
{

	if (mDebug) debugout("Processing requested fields");
	mKeep.resize(mNfields, 0);
	if (fields == NULL || fields == Py_None) {
		mKeepNfields = mNfields;
		mKeepId.resize(mNfields);
		for (long long i=0; i<mNfields; i++) {
			mKeepId[i] = i;
		}
		mKeepTypeDescr = mTypeDescr;
		Py_INCREF(mTypeDescr);
	} else {
		SubDtype(mTypeDescr, fields, &mKeepTypeDescr, mKeepId);
		mKeepNfields = mKeepId.size();
	}

	// This tells us if we keep a given field
	if (mDebug) debugout("Setting mKeep vector");
	for (long long i=0; i<mKeepNfields; i++) {
		mKeep[ mKeepId[i] ] = 1;
	}

	if (mDebug) {
		cerr<<"Will read "<<mKeepNfields<<"/"<<mNfields<<" fields"<<endl;
		fflush(stdout);
	}

}



void Records::ProcessRowsToRead(PyObject* rows)
{
	// Convert to an array of the desired type.  We will xdecref this 
	mRowsToRead = Object2IntpArray(rows);
	if (mRowsToRead == NULL) {
		// If returns NULL and no excepton thrown, means we will read all
		mNrowsToRead = mNrows;
	} else {
		// How many to read
		mNrowsToRead = PyArray_SIZE(mRowsToRead);
	}

	if (mNrowsToRead > mNrows) {
		stringstream serr;
		serr<<"You said the file has "<<mNrows<<" rows but requested to read "
			<<mNrowsToRead<<" rows";
		throw std::runtime_error(serr.str());
	}

	if (mDebug) {
		cerr<<"Will read "<<mNrowsToRead<<"/"<<mNrows<<" rows"<<endl;
		fflush(stdout);
	}
}

*/

void Records::process_descriptor(PyObject* descr)
{
	if (descr == NULL) {
		throw std::runtime_error("Input descr is NULL");
	}

	if (!PyArray_DescrCheck(descr)) {
		throw
			std::runtime_error("Input descr must be a NumPy type descriptor. e.g. "
			"arr.dtype, or numpy.dtype(typelist)");
	}

	// Get a new reference to this descr and make sure to decref later
	// on destruction
	mTypeDescr = descr;
	Py_XINCREF(descr);

	// Copy info for each field into a simpler form
	copy_field_info( (PyArray_Descr* ) mTypeDescr );

	// Each vector should now be number of fields long
	mNfields = mNames.size();
}


void Records::set_fptr(const char *filename, const char* mode)
{
	if (mDebug) debugout("Getting fptr");

    string fstr=filename;
    mFptr = fopen(fstr.c_str(), mode);
    if (mFptr==NULL) {
        string err="Could not open file: "+fstr;
        throw std::runtime_error(err);
    }
    return;

}

void Records::process_delim(PyObject* delim_obj)
{
	if (delim_obj == NULL || delim_obj == Py_None) {
		mDelim="";
        mArrayDelim="";
	} else {
		if (is_python_string(delim_obj)) {
			mDelim = get_object_as_string(delim_obj);

            if (mBracketArrays) {
                mArrayDelim = ",";
            } else {
                mArrayDelim = mDelim;
            }
		} else {
			throw std::runtime_error("delim keyword must be a string or None");
		}
	}

	if (mDelim[0] == ' ') {
		mReadAsWhitespace=true;
	} else {
		mReadAsWhitespace=false;
	}

	if (mDebug) {cerr<<"Using delim = \""<<mDelim<<"\""<<endl; fflush(stdout);}
}

void Records::set_file_type()
{
	if (mDelim == "") {
		mFileType = BINARY_FILE;
		if (mDebug) debugout("File type set to BINARY_FILE");
	} else {
		mFileType = ASCII_FILE;
		if (mDebug) debugout("File type set to ASCII_FILE");
	}

}





void Records::debugout(const char* mess)
{
	cerr<<mess<<endl;
	fflush(stdout);
}


// These get functions do not rely on internal data
void Records::copy_field_info(PyArray_Descr* descr)
{
	if (mDebug) debugout("Copying field info");
	if (mDebug) debugout("Copying ordered names");
	copy_descr_ordered_names(descr);
	if (mDebug) debugout("Copying offsets");
	copy_descr_ordered_offsets(descr);
	mRowSize= descr->elsize;
}

void Records::copy_descr_ordered_names(PyArray_Descr* descr)
{
	// Get the ordered names
	mNames.clear();

	for (long long i=0; i<PyTuple_Size(descr->names); i++) {

		PyObject* tmp = PyTuple_GET_ITEM(descr->names, i);
		string tname=get_object_as_string(tmp);
		if (mDebug) {cerr<<"  "<<tname<<endl;}
		mNames.push_back(tname);
	}

}

void Records::copy_descr_ordered_offsets(PyArray_Descr* descr)
{

	mOffsets.assign(mNames.size(), -1);
	mSizes.assign(mNames.size(), -1);
	mTypeNums.assign(mNames.size(), -1);
	mNel.assign(mNames.size(), -1);
    mNdim.assign(mNames.size(),-1);
    mDims.resize(mNames.size());

	// Get the offsets, ordered with names
	PyArray_Descr *fdescr, *title;

	// WARNING:  this is long int and being copied to long long
	long int offset;

	if (mDebug) {cerr<<"Copying ordered descr info:"<<endl;fflush(stdout);}
	for (unsigned long long i=0; i<mNames.size(); i++) {
		PyObject* item=
			PyDict_GetItemString(descr->fields, mNames[i].c_str());


        // default 0 dimensions
        mNdim[i] = 0;
        mDims[i].resize(0);

		if (item!=NULL) {
			if (!PyArg_ParseTuple(item, "Ol|O", &fdescr, &offset, &title)) {
				if (mDebug) 
				{cerr<<"Field: "<<mNames[i]<<" not right format"<<endl;}
			} else {
				mOffsets[i] = offset;
				mSizes[i] = fdescr->elsize;
				mTypeNums[i] = fdescr->type_num;
				if (fdescr->subarray != NULL) {
                    //cerr<<"subarray is not NULL for '"<<mNames[i]<<"'\n";
					// Here we are implicitly only allowing subarrays
					// if basic numbers or strings
					mNel[i] = mSizes[i]/fdescr->subarray->base->elsize;
					mTypeNums[i] = fdescr->subarray->base->type_num;


                    PyObject* shape = fdescr->subarray->shape;
#if PY_MAJOR_VERSION >= 3
                    if (PyLong_Check(shape) ) {
#else
                    if (PyInt_Check(shape) ) {
#endif
                        // this happens when a single dim array shows up
                        // with just the nel
                        mNdim[i] = 1;
                        mDims[i].assign(1,mNel[i]);
                    } else if (PyTuple_Check(shape) ) {
                        mNdim[i] = PyTuple_Size(shape);
                        mDims[i].resize(mNdim[i]);
                        for (int ii=0; ii<mNdim[i]; ii++) {
                            PyObject* tmp = PyTuple_GetItem(shape, ii);
#if PY_MAJOR_VERSION >= 3
                            mDims[i][ii] = PyLong_AsLong(tmp);
#else
                            mDims[i][ii] = PyInt_AsLong(tmp);
#endif
                        }
                    }


				} else {
					mNel[i] = 1;
				}
				if (mDebug) {
					cerr<<"  Offset("<<mNames[i]<<"): "<<mOffsets[i]<<endl;
					cerr<<"  Size("<<mNames[i]<<"): "<<mSizes[i]<<endl;
					cerr<<"  nel("<<mNames[i]<<"): "<<mNel[i]<<endl;
					cerr<<"  ndim("<<mNames[i]<<"): "<<mNdim[i]<<endl;
                    if (mNdim[i] > 0) {
                        cerr<<"    dims: [";
                        for (int ii=0;ii<mNdim[i];ii++) {
                            cerr<<mDims[i][ii];
                            if (ii < mNdim[i]-1){
                                cerr<<",";
                            }
                        }
                        cerr<<"]\n";
                    }
					cerr<<"  type_num("<<mNames[i]<<"): "<<mTypeNums[i]<<endl;
					cerr<<"  type("<<mNames[i]<<"): "<<fdescr->type<<endl;
					cerr<<endl;
				}
			}
		} else {
			if (mDebug) 
			{cerr<<"field: "<<mNames[i]<<" does not exist. offset->-1"<<endl;}
		}
	}

	if (mDebug) debugout("  Done");
}


void Records::make_scan_formats(vector<string> &formats, bool add_delim)
{

	formats.clear();
	int nf=24;
	formats.resize(nf, "%");

	formats[NPY_INT8] += NPY_INT8_FMT;
	formats[NPY_UINT8] += NPY_UINT8_FMT;
	
	formats[NPY_INT16] += NPY_INT16_FMT;
	formats[NPY_UINT16] += NPY_UINT16_FMT;

	formats[NPY_INT32] += NPY_INT32_FMT;
	formats[NPY_UINT32] += NPY_UINT32_FMT;

	formats[NPY_INT64] += NPY_INT64_FMT;
	formats[NPY_UINT64] += NPY_UINT64_FMT;

#ifdef NPY_INT128
	formats[NPY_INT128] += NPY_INT128_FMT;
	formats[NPY_UINT128] += NPY_UINT128_FMT;
#endif
#ifdef NPY_INT256
	formats[NPY_INT256] += NPY_INT256_FMT;
	formats[NPY_UINT256] += NPY_UINT256_FMT;
#endif

	// They put %g for these..!!??
	formats[NPY_FLOAT] += "f";
	formats[NPY_DOUBLE] += "lf";

#ifdef NPY_LONGDOUBLE
	formats[NPY_LONGDOUBLE] += "Lf";
#endif

	// The types for long long integers are incorrect in the
	// ndarrayobject.h header.  Uses Ld instead of lld.  
	// We need to loop over and fix this since we don't know ahead
	// of time on this platform which is the lld type
	for (int i=0; i<nf; i++) {
		if (formats[i] == "%Ld") {
			formats[i] = "%lld";
		}
		if (formats[i] == "%Lu") {
			formats[i] = "%llu";
		}
	}

	// Only add in the 
	if ((!mReadAsWhitespace) && (add_delim) ) {
		for (int i=0; i<nf; i++) {
			if (formats[i] != "%") {
				formats[i] += ' '+mDelim;
			}
		}
	}
}

void Records::make_print_formats(vector<string> &formats)
{

	make_scan_formats(formats,false);
	
	formats[NPY_FLOAT] = "%.7g";
    // for g the .16 means 16 total, 15 mantissa which is what we want for double
	formats[NPY_DOUBLE] = "%.16g";

	formats[NPY_STRING] = "%s";

}


