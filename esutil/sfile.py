"""
Module:
    sfile

Purpose:

    Read and write numpy arrays to a simple file format.  The format is a
    simple ascii header followed by data, either in binary form or ascii.  For
    details of the ascii header, see the documentation for the read_header()
    function.

    The power of this code comes from two features of the `recfile`
    package:

        1) The recfile package understands the structure of recarrays and can
        efficientlly read subsets of rows and columns. Unlike the memmap class
        distributed with numpy, individual fields from these types of arrays
        can be read without reading the whole file into memory.  This is
        accomplished using a C++ class linked to python.

        2) Structured arrays can be written to and read from delimited ascii
        files.  This is done efficiently using the same C++ class linked to
        python.

    Another nice feature not supported by the format module of numpy is the
    ability to append data to the file.


Convenience functions:
    read_header():  A convenience function to read the header from the simple
        file format.  Uses an SFile instance internally.
    read():  A convenience function to read data and optionally the header
        from the simple file format. Uses an SFile instance internally.
    write(): A convenience function to write data to the simple file format,
        including an ascii header. Uses an SFile instance internally.

    For more docs, check the docs for the individual functions.


Classes

    SFile.  See docs for sfile.SFile for details.
"""
# vim: set filetype=python :
import sys
import os
import pprint
import copy

import numpy as np

# importing array will allow array types in the header to be properly
# eval()uated.  Not sure if this works for all dtypes.
from numpy import array  # noqa

from . import recfile

SFILE_VERSION = "1.0"


class SFile(object):
    """
    Class SFile

    This class implements a simple file format for holding numerical python
    arrays.  The format is a simple ascii header followed by data, either in
    binary form or ascii.  For details of the ascii header, see the
    documentation for the read_header() function.


    Examples with record arrays
        from esutil.sfile import SFile

        # Open and read from a .rec file
        with SFile(filename) as sf:

            # see what's in the file
            print(sf)
            filename: 'test.rec'
            mode: 'r'
            size: 64526192
            dtype:
              [('x', '<f4'),
               ('y', '<f4'),
               ('flux', '<f8'),
               ('name', '|S20')]
            hdr:
              {'dataset': 'dc4',
               'pyvers':'v2.6.2'}


            # get the header
            hdr = sf.read_header()

            #
            # A few ways to read all the data
            #
            data = sf[:]
            data = sf.read()

            # reading subsets of the data

            # read an entire column name 'x' from the file.  Only data for that
            # column are read
            data = sf['x'][:]
            data = sf.read(columns='x')

            # read a list of columns.  Any list, tuple or numpy array of
            # strings can be sent.
            data = sf[ ['x','y'] ][:]
            data = sf.read(columns=column_list)

            # read a subset of rows.  Can use slices, single numbers for rows,
            # or list/array of row numbers.

            data = sf[35]
            data = sf[35:100]
            data = sf[ [35,66,22,23432] ]
            data = sf[ row_list ]
            data =sf.read(rows=row_list)

            # read a subset of rows and columns.  Only those rows/columns are
            # read from the file.

            data = sf[columns][rows]
            data = sf[ 35:5012 ][ ['x','y'] ]
            data = sf.read(colums=columns, rows=rows)


        # writing data to a CSV file
        data[0:5]
            array([(0, 0.13933435082435608),
                   (1, 0.78211665153503418),
                   (2, 0.13652685284614563),
                   (3, 0.54205995798110962),
                   (4, 0.2745462954044342)],
                      dtype=[('ind', '<i4'), ('rnd', '<f4')])

        sf = Open('test.rec', 'w+', delim=',')
        with SFile('test.rec','w+',delim=',') as sf:
            sf.write(data)
            sf.write(more_data)

            # check the first row of the data we have written
            sf[0]
                array([(0, 0.13933435082435608)],
                         dtype=[('ind', '<i4'), ('rnd', '<f4')])

            sf.write(more_data)

    """

    def __init__(
        self,
        filename=None,
        mode="r",
        delim=None,
        padnull=False,
        ignorenull=False,
        **keys
    ):

        self.open(
            filename,
            mode=mode,
            delim=delim,
            padnull=padnull,
            ignorenull=ignorenull,
            **keys
        )

    def open(
        self, filename, mode="r", delim=None, padnull=False,
        ignorenull=False, **keys
    ):
        """
        Open the file.  If the file already exists and the mode is 'r*' then
        a read of the header is attempted.  If this succeeds, delim is gotten
        from the header and the delim= keyword is ignored.
        """

        self.close()

        self._padnull = padnull
        self._ignorenull = ignorenull
        self._mode = mode

        self._filename = filename

        if filename is None:
            return

        # expand shortcut variables
        fpath = os.path.expanduser(filename)
        fpath = os.path.expandvars(fpath)
        self._filename = fpath

        if mode == "r+" and not os.path.exists(self._filename):
            # path doesn't exist but we want to append.  Change the
            # mode to write
            mode = "w+"

        if self._mode[0] == "r":
            # if reading:
            # For 'r' and 'r+' try to read the header
            self._hdr = self.read_header()

            self._delim = _match_key(self._hdr, "_delim")
            self._size = _match_key(self._hdr, "_size", require=True)
            self._descr = _match_key(self._hdr, "_dtype", require=True)
            self._dtype = np.dtype(self._descr)

            self._robj = recfile.Recfile(
                self._filename,
                mode=self._mode,
                delim=self._delim,
                dtype=self._dtype,
                nrows=self.get_nrows(),
                offset=self._data_start,
                padnull=self._padnull,  # these get sent in case mode is r+
                ignorenull=self._ignorenull,
            )

        else:
            # we are starting from scratch, so we can't read a header
            #
            # get delim from the keyword.  This will be used for writing later
            self._delim = delim

            self._robj = recfile.Recfile(
                self._filename,
                mode=self._mode,
                delim=self._delim,
                padnull=self._padnull,
                ignorenull=self._ignorenull,
            )

    def close(self):
        """
        Close any open file object.  Make sure fobj, _hdr, and delim are None
        """

        if hasattr(self, "_robj"):
            if self._robj is not None:
                self._robj.close()

        self._filename = None
        self._mode = None
        self._robj = None
        self._hdr = None
        self._data_start = None
        self._delim = None
        self._size = 0
        self._descr = None
        self._dtype = None

        self._padnull = False
        self._ignorenull = False

    def get_nrows(self):
        """
        get the number of rows in the file
        """
        if self._hdr is None:
            raise RuntimeError("no file has been opened for reading")

        return self._hdr["_SIZE"]

    @property
    def nrows(self):
        """
        get the number of rows in the file
        """
        return self.get_nrows()

    @property
    def dtype(self):
        """
        get the number of rows in the file
        """
        return self._dtype

    def get_header(self):
        """
        get a copy of the header
        """
        return self._hdr

    def get_mode(self):
        """
        get the file open mode
        """
        return self._mode

    def get_filename(self):
        """
        get the file name
        """
        return self._filename

    def _ensure_open(self):
        """
        check if a file is open, if not raise a RuntimeError
        """
        if self._robj is None:
            raise RuntimeError("no file is open")

    def _ensure_open_for_writing(self):
        """
        check if a file is open for writing, if not raise a RuntimeError
        """
        self._ensure_open()
        if self._robj.mode[0] != "w" and "+" not in self._mode:
            raise ValueError("You must open with 'w*' or 'r+' to write")

    def _ensure_open_for_reading(self):
        """
        check if a file is open for reading, if not raise a RuntimeError
        """
        self._ensure_open()

        if self._robj.mode[0] != "r" and "+" not in self._mode:
            raise ValueError("You must open with 'w+' or 'r*' to read")

    def _ensure_structured(self, data):
        """
        check if the data has fields, if not raise a ValueError
        """
        if data.dtype.names is None:
            raise ValueError("data must be a structured array with fields")

    def _ensure_compatible_dtype(self, data):
        """
        if we are writing binary we demand exact match.

        For text we just make sure everything matches if the
        byte order is ignored
        """
        self._ensure_structured(data)

        # if self._dtype is not None, there was data in the file
        if self._dtype is not None:

            bad = False
            if self._delim is None:
                if self._dtype != data.dtype:
                    bad = True
            else:
                names = self._dtype.names
                nnames = len(names)
                input_names = data.dtype.names
                ninput = len(input_names)

                if ninput != nnames:
                    mess = "got %d fields instead of %d" % (ninput, nnames)
                    bad = True
                else:
                    descr = self._dtype.descr
                    idescr = data.dtype.descr
                    for d1, d2 in zip(descr, idescr):
                        l1 = len(d1)
                        l2 = len(d2)
                        if l1 != l2:
                            mess = "field dim mismatch: %s vs %s" % (d1, d2)
                            bad = True
                            break

                        if d1[0] != d2[0]:
                            mess = "field name mismatch: %s vs %s" % (d1[0], d2[0])  # noqa
                            bad = True
                            break

                        # skip byte order
                        if d1[1][1:] != d2[1][1:]:
                            mess = "field type mismatch: %s vs %s" % (
                                d1[1][1:],
                                d2[1][1:],
                            )
                            bad = True
                            break

                        if l1 == 3:
                            if d1[2] != d2[2]:
                                mess = "field shape mismatch: %s vs %s" % (d1[2], d2[2])  # noqa
                                bad = True
                                break

                if bad:
                    raise ValueError(
                        "attempt to write an incompatible "
                        "data type: " + mess
                    )

    def write(self, data, header=None):
        """
        write data to the file, appending if the file is not empty

        paramters
        ---------
        data: array
            A structured numerical python array.  If data already
            exists in the file, this data must have compatible
            data type.  For binary files this includes the byte
            order.
        header: dict, optional
            Optional dictionary to write into the header.  This
            can only be written the first time.
        """

        self._ensure_open_for_writing()

        # check compatible, in case there is already data in the file
        self._ensure_compatible_dtype(data)

        # this will make self._dtype if it is the first write
        self._write_header(data, header=header)

        self._robj.write(data)

    def read(
        self,
        rows=None,
        fields=None,
        columns=None,
        header=False,
        view=None,  # ignored
        split=False,
        reduce=False,
    ):
        """
        Read data from the file.

        parameters
        -----------
        rows: sequence or scalar, optional
            A scalar, array, list or tuple of row numbers to read.
            Default is None, meaning read all rows.

        columns: sequence or scalar
            A scalar, list or tuple of strings naming columns to read
            from the file.  Default is None or read all columns.
        fields:  Same as sending columns=.

        header: bool, optional
            If True, return both the array and the header dict in
            a tuple.
        split: bool, optional
            If True, return a list of arrays for each column rather
            than a structured array.
        reduce: bool, optional
            If True, and there is only one field requested, reduce
            it to a plain array. This is equivalent to sending
            columns=(scalar column name)

        returns
        -------

        A structured array with fields.

        If the columns= is a scalar column name (rather than list of names or
        None), then the data is a plain array holding the column data
        """

        self._ensure_open_for_reading()

        result = self._do_read(rows=rows, fields=fields, columns=columns)

        if split:
            result = split_fields(result)
        elif reduce:
            result = reduce_array(result)

        if header:
            return result, copy.deepcopy(self._hdr)
        else:
            return result

    def __getitem__(self, arg):
        """

        # read subsets of columns and/or rows from the file.  This only works
        # for record types
        sf = SFile(....)


        # read subsets of rows
        data = sf[ 35 ]
        data = sf[ 35:88 ]
        data = sf[ [3,234,5551,.. ] ]

        # read subsets of columns
        data = sf['fieldname'][:]
        data = sf[ ['field1','field2',...] ][:]

        # read subset of rows *and* columns.
        data = sf['fieldname'][3:58]
        data = sf[fieldlist][rowlist]
        """

        return self._robj[arg]

    def _do_read(self, rows=None, fields=None, columns=None):
        """
        use the recfile object to read the data
        """

        if columns is None:
            columns = fields

        return self._robj.read(rows=rows, columns=columns)

    """
    def get_subset(self, rows=None, fields=None, columns=None):
        robj = recfile.Open(self.fobj, nrows=self._size, mode='r',
                            offset=self.fobj.tell(),
                            dtype=self._dtype, delim=self._delim)
        return robj.get_subset(rows=rows, fields=fields, columns=columns)
    """

    def _make_header(self, data, header=None):
        if header is None:
            head = {}
        else:
            head = copy.deepcopy(header)

        for key in ["_size", "_nrows", "_delim", "_shape", "_has_fields"]:
            if key in head:
                del head[key]
            if key.upper() in head:
                del head[key.upper()]

        descr = data.dtype.descr

        if self._delim is not None:
            head["_DELIM"] = self._delim

            # Text file. Remove the byte order specification.
            descr = self._remove_byteorder(descr)

        head["_DTYPE"] = descr
        head["_VERSION"] = SFILE_VERSION

        return head

    def _write_header(self, data, header=None):

        if self._hdr is not None:
            # we are appending data.
            # Just update the nrows and move to the end

            self._update_size(data.size)
        else:

            # this is a dict of variable size
            self._hdr = self._make_header(data, header=header)

            # store some of the info
            self._descr = self._hdr["_DTYPE"]
            self._dtype = np.dtype(self._descr)

            size_string = self._get_size_string(data.size)
            self._size = data.size

            # As long as the dict contains types that can be represented as
            # constants, this pretty printing can be eval()d.

            hdr_dict_string = pprint.pformat(self._hdr)

            lines = [
                size_string,
                hdr_dict_string,
                "END",
                "",  # to add a new line
                "",  # to add a blank line
            ]

            total_str = "\n".join(lines)

            self._robj.robj.write_header_and_update_offset(total_str)

    def _remove_byteorder(self, descr):
        if isstring(descr):
            return descr[1:]

        new_descr = []
        for d in descr:
            # d is a tuple, make it a list
            newd = list(copy.copy(d))

            tdef = newd[1]
            tdef = tdef[1:]
            newd[1] = tdef
            newd = tuple(newd)
            new_descr.append(newd)

        return new_descr

    def _update_size(self, size_add):
        """
        update the size in the file

        TODO: update for new recfile where file object
        is maintained within the C++ code
        """
        size_current = self._size
        if size_current is None:
            raise RuntimeError(
                "Attempting to update size but not found in header"
            )

        size_new = size_current + size_add

        self._robj.robj.update_row_count(size_new)
        self._size = size_new
        self._hdr["_SIZE"] = size_new

    def _get_size_string(self, size):
        # Specially formatted fixed-length for updating later
        s = "SIZE = %20d" % size
        return s

    def _extract_size_from_string(self, line):
        lsplit = line.split("=")
        if len(lsplit) != 2:
            raise ValueError("First line of header must be SIZE = %20d")
        fname = lsplit[0].strip()

        # also allow old NROWS word for compatibility
        if fname.upper() != "SIZE" and fname.upper() != "NROWS":
            raise ValueError("First line of header must be SIZE = %20d")

        if fname.upper() == "NROWS":
            self.cannot_append_oldheader = True
        else:
            self.cannot_append_oldheader = False

        size = eval(lsplit[1])

        return size

    def read_header(self):
        """
        Name:
            read_header()

        Calling Sequence:
            sf = sfile.Open(file)
            hdr = sf.read_header()

        Read the header from a simple self-describing file format with an
        ascii header.  See the write() function for information about reading
        this file format, and read() for reading.


        The file format:
          First line:
              SIZE = --------------number

        where if possible the number should be formatted as %20d.  This is
        large enough to hold a 64-bit number.  This exact formatting is
        required so SIZE can be updated *in place* when appending rows to a
        file.  Note the file can always be read as long as the first line reads
        'SIZE = some_number' but appending requires the exact format.

        Last two lines of the header region must be:
                END
                blank line
        case does not matter.


        In between the SIZE and END lines is the header data.  This is a
        string that must eval() to a dictionary.  It must contain the
        following entry:

              _DTYPE = array data type description in list of tuples or
                string form (case does not matter, can also be called _dtype).

                    [('field1', 'f8'), ('f2','2i4')]


        There should also be a _VERSION tag.

              '_VERSION': '1.0'

        If '_VERSION' is not present, it is assumed that the version is 1.0,
        but you should always set this.  If you use this module to write data,
        it will always be set.


        If the file holds a simple array, and the dtype field is a simple
        string, then the following keyword, if present, will be used to
        reshape the array:

              '_SHAPE'

        If the total elements in the _shape field matches the size then it
        will be used to reshape the array before returning or when using
        memory maps.

        If the data are ascii then delimiter must be given in the keyword

              _DELIM

        This can be for example ',', ' ', or a tab character.  Again, case does
        not matter.

        The rest of the keywords can by any variable can be used as long as it
        can be eval()d.

        An example header:
            SIZE =                   10
            {'_VERSION': '1.0',
             '_DELIM': ',',
             '_DTYPE': [('x', 'f4'),
                        ('y', 'f4'),
                        ('ra', 'f8'),
                        ('dec', 'f8'),
                        ('exposurename', 'S20'),
                        ('ccd', 'i1'),
                        ('size_flags', 'i4'),
                        ('magi', 'f4'),
                        ('sigma0', 'f4'),
                        ('star_flag', 'i4'),
                        ('shear_flags', 'i4'),
                        ('shapelet_sigma', 'f4'),
                        ('shear1', 'f4'),
                        ('shear2', 'f4'),
                        ('shear_cov00', 'f4'),
                        ('shear_cov01', 'f4'),
                        ('shear_cov11', 'f4')],
             'listvar': [1, 2, 3],
             'subd': {'subd1': 'subfield', 'sublist': [8.5, 6.6]},
             'svar': 'hello',
             'test1': 35}
            END

            -- data begins --
        """

        if self._filename is None:
            raise ValueError("you opened with filename None")

        # read first line, which should be
        # SIZE = .....
        # or
        # NROWS = ...

        dummy_dtype = [("ra", "f8")]
        with recfile.Recfile(self._filename, dtype=dummy_dtype) as robj:
            hdrstring, offset = robj.robj.read_sfile_header()

        self._data_start = offset
        lines = hdrstring.split("\n")
        size = self._extract_size_from_string(lines[0])

        hdrdict_string_lines = lines[1: len(lines) - 3]
        hdrdict_string = " ".join(hdrdict_string_lines)
        hdr = eval(hdrdict_string)

        hdr["_SIZE"] = size

        # this will leave open the possibility of changing the header or other
        # details later
        if "_version" in hdr or "_VERSION" in hdr:
            pass
        else:
            hdr["_VERSION"] = "1.0"

        return hdr

    def _h2string(self, strip=True):
        if self._hdr is None:
            return ""
        newd = {}

        skipkeys = [
            "_VERSION",
            "_DTYPE",
            "_SIZE",
            "_NROWS",
            "_HAS_FIELDS",
            "_DELIM",
            "_SHAPE",
        ]
        for key in self._hdr:
            if strip:
                if key not in skipkeys:
                    newd[key] = self._hdr[key]
            else:
                newd[key] = self._hdr[key]

        if len(newd) == 0:
            return ""
        return pprint.pformat(newd)

    def __repr__(self):

        top = ["filename: '%s'" % self._filename]

        s = ["mode: '%s'" % self._mode]
        if self._delim is not None:
            s += ["delim: '%s'" % self._delim]

        s += ["size: %s" % self._size]

        if self._descr is not None:
            drepr = pprint.pformat(self._descr).split("\n")
            drepr = ["  " + d for d in drepr]
            # drepr = '  '+drepr.replace('\n','\n  ')
            s += ["dtype:"]
            s += drepr

        if self._hdr is not None:
            hs = self._h2string()
            if hs != "":
                hs = "  " + hs.replace("\n", "\n  ")
                if hs != "":
                    s += ["hdr: \n" + hs]

        slist = []
        for tmp in s:
            slist.append("    " + tmp)

        slist = top + slist
        rep = "\n".join(slist)
        return rep

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


def write(outfile, data, **keys):
    """
    Name:
        sfile.write()

    Calling Sequence:
        sfile.write(data, outfile, header=None, delim=None,
                    padnull=False, ignorenull=False, append=False)

    Write a numpy array into a simple self-describing file format with an ascii
    header.  See the read() function for information about reading this file
    format.  See the docs for the SFile class for an idea of the full
    functionality wrapped by this covenience function.


    Inputs:
        outfile: string
            The filename to write
        data: array
            Numerical python array, a structured array with fields

    Optional Inputs:
        header=: A dictionary containing keyword-value pairs to be added to
            the header.

        delim=None: Delimiter between fields.  Default is None for binary.  For
            ascii can be any string, e.g. ',', ' ', or a tab character.

        padnull=False:
            When writing ascii, replace Null characters with spaces.  This is
            useful when writing files to be read in by programs that do not
            recognize null characters, e.g. sqlite databases.  But note, if
            read back in these fields will not compare equal with the original
            data!

        ignorenull=False:
            When writing ascii, ignore Null characters.  This is useful when
            writing files to be read in by programs that do not recognize null
            characters, e.g. sqlite databases.  But note you will not be
            able to read the data back in with sfile.read() becuase the fields
            are no longer the correct length!

        append=False: Append to the file. Default is False. If set to True,
            then what happens is situation dependent:
                1) if the input is a file object then it is assumed there is
                    an existing header.  The header is updated to reflect the
                    new appended rows after writing.
                2) if the input is a string and the file exists, then the file
                    is opened with mode "r+", it is assumed the header
                    exists, and the header is updated to reflext the new
                    appended rows after writing.
                3) if the input is a string and the file does *not* exist,
                    then the file is opened with mode "w" and the request
                    to append is ignored.

    Examples:
        import sfile
        hdr={'date': '2007-05-12','age': 33}
        sfile.write(data1, 'test.rec', header=hdr)

        sfile.write(data2, 'test.rec', append=True)

        If this is part of the esutil package, use
            import esutil
            esutil.sfile.write(...)

    File Format:
        The file format is an ascii header followed by data in binary or rows
        of ascii.  The data columns must be fixed length in order to map onto
        numpy arrays.  See the documentation for read_header() for details
        about the header format.

    Modification History:
        Created: 2007-05-25, Erin Sheldon, NYU
        Ignore append=True when file does not yet exist.
        Allow continuation characters "\\" to continue header keywords
        onto the next line.  2009-05-05

        Moved to object-oriented approach using the SFile class.
            2009-11-16, ESS, BNL

    """

    if isinstance(outfile, np.ndarray):
        outfile, data = data, outfile

    header = keys.get("header", None)
    delim = keys.get("delim", None)
    padnull = keys.get("padnull", False)
    ignorenull = keys.get("ignorenull", False)
    append = keys.get("append", False)

    if append:
        # if file doesn't yet exist, this will be changed to 'w+' internally.
        mode = "r+"
    else:
        mode = "w"

    with SFile(
        outfile, mode=mode, delim=delim, padnull=padnull, ignorenull=ignorenull
    ) as sf:
        sf.write(data, header=header)


def read(filename, **keys):
    """
    sfile.read()

    Read a numpy array from a simple self-describing file format with an ascii
    header.  See the write() function for information about this file format.

    parameters
    -----------
    filename: string
        Filename from which to read

    rows: sequence or scalar, optional
        A scalar, array, list or tuple of row numbers to read.
        Default is None, meaning read all rows.

    columns: sequence or scalar
        A scalar, list or tuple of strings naming columns to read
        from the file.  Default is None or read all columns.
    fields:  Same as sending columns=.

    header: bool
        If True, return both the array and the header dict in
        a tuple.

    split: bool, optional
        If True, return a list of arrays for each column rather
        than a structured array.
    reduce: bool, optional
        If True, and there is only one field requested, reduce
        it to a plain array. This is equivalent to sending
        columns=(scalar column name)

    returns
    -------

    A structured array with fields.

    If the columns= is a scalar column name (rather than list of names or
    None), then the data is a plain array holding the column data
    """

    with SFile(filename) as sf:
        data = sf.read(**keys)

    return data


def read_header(filename):
    """
    read the header from the indicated sfile

    parameters
    -----------
    filename: string
        Filename from which to read

    returns
    -------
    The header as a dictionary
    """

    with SFile(filename) as sf:
        hdr = sf.get_header()

    return hdr


def split_fields(data, fields=None, getnames=False):
    """
    Name:
        split_fields

    Calling Sequence:
        The standard calling sequence is:
            field_tuple = split_fields(data, fields=)
            f1,f2,f3,.. = split_fields(data, fields=)

        You can also return a list of the extracted names
            field_tuple, names = split_fields(data, fields=, getnames=True)

    Purpose:
        Get a tuple of references to the individual fields in a structured
        array (aka recarray).  If fields= is sent, just return those
        fields.  If getnames=True, return a tuple of the names extracted
        also.

        If you want to extract a set of fields into a new structured array
        by copying the data, see esutil.numpy_util.extract_fields

    Inputs:
        data: An array with fields.  Can be a normal numpy array with fields
            or the recarray or another subclass.
    Optional Inputs:
        fields: A list of fields to extract. Default is to extract all.
        getnames:  If True, return a tuple of (field_tuple, names)

    """

    outlist = []
    allfields = data.dtype.fields

    if allfields is None:
        if fields is not None:
            raise ValueError("Could not extract fields: data has " "no fields")
        return (data,)

    if fields is None:
        fields = allfields
    else:
        if isinstance(fields, str):
            fields = [fields]

    for field in fields:
        if field not in allfields:
            raise ValueError("Field not found: '%s'" % field)
        outlist.append(data[field])

    output = tuple(outlist)
    if getnames:
        return output, fields
    else:
        return output


def reduce_array(data):
    # if this is a structured array with fields, and only has a single
    # field, return a simple array view of that field, e.g. data[fieldname]
    if hasattr(data, "dtype"):
        if data.dtype.names is not None:
            if len(data.dtype.names) == 1:
                # get a simpler view
                return data[data.dtype.names[0]]
    else:
        return data


def _match_key(d, key, require=False):
    """
    Match the key in a case-insensitive way and return the value. Return None
    if not found or raise an error if require=True
    """
    if not isinstance(d, dict):
        raise RuntimeError("Input object must be a dict, got %s" % d)

    keys = list(d.keys())

    keyslow = [k.lower() for k in keys]
    keylow = key.lower()

    if keylow in keyslow:
        ind = keyslow.index(keylow)
        return d[keys[ind]]
    else:
        if not require:
            return None
        else:
            raise RuntimeError("Could not find required key: '%s'" % key)


_major_pyvers = int(sys.version_info[0])


if np.lib.NumpyVersion(numpy.__version__) < "1.28.0":
    np_vers = 1
else:
    np_vers = 2


def isstring(obj):
    if _major_pyvers >= 3:
        if np_vers == 2:
            string_types = (str, np.str_, np.bytes_)
        else:
            string_types = (str, np.str_, np.string_)
    else:
        string_types = (str, np.string_)

    if isinstance(obj, string_types):
        return True
    else:
        return False


def _fix_range(i, maxval):
    if i < 0:
        i = maxval - i
    if i > maxval:
        i = maxval
    return i


# deprecated
def Open(filename, mode="r", **keys):
    sf = SFile(filename, mode=mode, **keys)
    return sf
