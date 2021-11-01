"""
Utilities for using and manipulating numerical python arrays (NumPy).

    where1(logical):
        A wrapper for where for 1-d arrays.  It is the equivalent of
            w, = where(logical expression)
        E.g.
            w=where1( (x > 0.1) & (x < 1.5) )

    ahelp(array, recurse=False, pretty=True)
        Print out a formatted description of the input array.   If the array
        has fields, individual descriptions are printed for each field.  This
        is designed to be similar to help, struct, /str in IDL.


    aprint(array, type='table', page=False, nlines=ALL, fields=ALL, file=None)
        Print fields from the array in columns, optionally send to a pager or
        file.  If type='fancy', more keywords are available.

    arrscl(arr, minval, maxval, arrmin=None, arrmax=None)
        Rescale the range of an array to be between minval and maxval.

    make_xy_grid(npoints, xrange, yrange)
        Create a grid of x-y points, returning x and y as numpy arrays.

    combine_arrlist(list_of_arrays, keep=False)
        Combine the list of arrays into one big array.  Arrays must all have
        the same datatype.

    combine_fields(arrlist)
        Combine the field names and data from multiple arrays.  The arrays must
        be the same size and have disjoint sets of fields

    copy_fields(array1, array2)
        Copy common fields from one numpy array to another.  The name
        matching is case senitive.

    extract_fields(array, names, strict=True)

        Extract a set of fields from a numpy array.  A new array is returned
        with the requested fields and data copied in.  The name matching is
        case sensitive.

    remove_fields(array, names)
        Remove a set of fields from the array.  A new array is returned
        with the leftover fields and data copied in.  The name matching
        is case sensitive.

    add_fields(arr, dtype_or_descr, defaults=None)
        Create a new array with fields from the input array and new
        fields as indicated by the input numpy type descriptor.
        The data are copied from the original array.


    reorder_fields(arr, ordered_names, strict=True)
        Re-order the fields according the the listed names.  Names not in the
        list are put at the end.


    copy_fields_by_name(arr, names, values)
        Copy values into a numpy array by field name.


    split_fields(array, fields=None, getnames=False)

         Get a tuple of references to the individual fields in a structured
         array (aka recarray).  If fields= is sent, just return those fields.
         If getnames=True, return a tuple of the names extracted also.



    compare_arrays(array1, array2, ignore_missing=True, verbose=False)
        Compare the values field-by-field in two sets of numpy arrays or
        recarrays.  Return true if the data match.

    replicate(value, num, dtype=None)
        Create an array with every value set to the input value.

    is_big_endian(array)
        Return True if array is big endian.  Note strings are neither big
        or little endian.  The input must be a simple numpy array, not
        an array with fields.

    is_little_endian(array)
        Return True if array is little endian. Note strings are neither big
        or little endian.  The input must be a simple numpy array, not
        an array with fields.



    to_big_endian(array, inplace=False, keep_dtype=False)
        Convert an array to big endian byte order, updating the dtype to
        reflect this.  The array can have fields.
    to_little_endian(array, inplace=False, keep_dtype=False)
        Convert an array to little endian byte order, updating the dtype to
        reflect this.  The array can have fields.
    to_native(array, inplace=False, keep_dtype=False)
        Convert an array to native byteorder, updating the dtype to
        reflect this.  The array can have fields.

    byteswap(array, inplace=False, keep_dtype=False)
        Chance the byte order of an array, updating the dtype to reflect this.
        The array can have fields.   This is a wrapper for the .byteswap()
        method which does not update the dtype to reflect the new byte
        ordering.

    unique(arr, values=False)
        Return indices of unique elements of a numpy array, or optionally
        the unique values.  This is not order preserving.  This is currently
        implemented in a slow fashion, should be updated.

    rem_dup(arr, flag, values=False)
        Return indices of unique values of an array, selecting the one (when
        duplicates exist) with the largest value of flag.  Optionally returns
        the values in the array as well as their indices.

    match(arr1, arr2)
        match two numpy arrays.  Return the indices of the matches or [-1] if
        no matches are found.  This means arr1[ind1] == arr2[ind2] is true for
        all corresponding pairs. Arrays must contain only unique elements

    strmatch(arr, regex)
        Match the string array to the input regular expression.  Returns
        a boolean array.

    match_multi(arr1, arr2)
        Match two numpy integer arrays, one of which may be non-unique

    dict2array(dict, sort=False, keys=None)
        Convert a dictionary to a numpy array.  Works for simple typs such as
        strings, integers, floating.

    dictlist2array(dict, sort=False, keys=None)
        Convert a list of dictionaries to a structured numpy array.  Works
        for simple typs such as strings, integers, floats.


    splitarray(nper, array)
        Split up an array into chunks of at least a given size.  Return a
        list of these subarrays.  The ordering is perserved.

    between(arr, lowval, highval, type='[)')
        test values of an array are between the specified values

    outside(arr, lowval, highval, type=')(')
        test values of an array are outside the specified values

    select_percentile(x, perc, get_ranges=False, **keys)
        select data in the given percentile(s)

"""
from __future__ import print_function

import os
from sys import stdout
import copy
import pydoc
import stat
import numpy as np

from . import misc as eu_misc


def where1(conditional_expression):
    """
    Name:
        where1

    Calling Sequence:
        w = where1(conditional_expression)

    Purpose:
        A wrapper for np.where() for 1-d arrays.  It is the equivalent of
            w, = where(logical expression)

        E.g.
            w=where1( (x > 0.1) & (x < 1.5) )
            print x[w]
    """
    (w,) = np.where(conditional_expression)
    return w


def ahelp(array_in, recurse=False, pretty=True, index=0, page=False):
    """
    Name:
      ahelp()

    Purpose:
        Print out a formatted description of the input array.   If the array
        has fields, individual descriptions are printed for each field.  This
        is designed to be similar to help, struct, /str in IDL.

    Calling Sequence:
        ahelp(array, recurse=False, pretty=True, page=False)

    Inputs:
        array: A numpy array.

    Optional Inputs:
        recurse: for sub-arrays with fields, print out a full description.
            default is False.
        pretty:  If True, split field descriptions onto multiple lines if
            the name is longer than 15 characters.  Nicer for the eye, but
            harder for a machine to parse.  Also, strings are surrounded
            by quotes 'string'.  Default is True.
        page: If True, run the output through a pager.

    Example:
        ahelp(a)
        size: 1147506  nfields: 27  type: records
          run                >i4  1933
          rerun              |S3  '157'
          camcol             >i2  1
          field              >i4  11
          mjd                >i4  51886
          tai                >f8  array[5]
          ra                 >f8  102.905870701
          dec                >f8  -1.05070432844

    Revision History:
        Created: 2010-04-05, Erin Sheldon, BNL

    """

    # make sure the data can be viewed as a
    # numpy ndarray.  pyfits in particular is
    # a problem case that we must get a view of
    # as ndarray.
    if not hasattr(array_in, "view"):
        raise ValueError("data must be an array or have the .view method")

    array = array_in.view(np.ndarray)

    names = array.dtype.names
    descr = array.dtype.descr

    topformat = "size: %s  nfields: %s  type: %s\n"

    lines = []
    if names is None:
        type = descr[0][1]
        nfields = 0
        line = topformat % (array.size, nfields, type)
        lines.append(line)

    else:
        line = topformat % (array.size, len(names), "records")
        lines.append(line)
        flines = _get_field_info(
            array, recurse=recurse, pretty=pretty, index=index
        )
        lines += flines

    lines = "\n".join(lines)

    if not page:
        stdout.write(lines)
        stdout.write("\n")
    else:
        import pydoc

        pydoc.pager(lines)


def _get_field_info(array, nspace=2, recurse=False, pretty=True, index=0):
    names = array.dtype.names
    if names is None:
        raise ValueError("array has no fields")

    if len(array.shape) == 0:
        is_scalar = True
    else:
        is_scalar = False

    lines = []
    spacing = " " * nspace

    nname = 15
    ntype = 6

    # this format makes something machine readable
    format = spacing + "%-" + str(nname) + "s %" + str(ntype) + "s  %s"
    # this one is prettier since lines wrap after long names
    pformat = (
        spacing + "%-" + str(nname) + "s\n %" + str(nspace + nname + ntype) + "s  %s"  # noqa
    )

    max_pretty_slen = 25

    for i in range(len(names)):

        hasfields = False

        n = names[i]

        type = array.dtype.descr[i][1]

        if is_scalar:
            fdata = array[n]
        else:
            fdata = array[n][index]

        if np.isscalar(fdata):
            if isinstance(fdata, np.string_):
                d = fdata

                # if pretty printing, reduce string lengths
                if pretty and len(d) > max_pretty_slen:
                    d = fdata[0:max_pretty_slen]
                    # d = "'" + d +"'"
                    d = "'%s'..." % d
                    # d = d+'...'
                else:
                    if pretty:
                        d = "'%s'" % d
            else:
                d = fdata
        else:
            shape_str = ",".join(str(s) for s in fdata.shape)
            if fdata.dtype.names is not None:
                type = "rec[%s]" % shape_str
                d = ""
                hasfields = True
            else:
                d = "array[%s]" % shape_str

        if pretty and len(n) > 15:
            tline = pformat % (n, type, d)
        else:
            tline = format % (n, type, d)
        lines.append(tline)

        if hasfields and recurse:
            # new_nspace = nspace + nname + 1 + ntype + 2
            new_nspace = nspace + 4
            morelines = _get_field_info(
                array[n], nspace=new_nspace, recurse=recurse
            )
            lines += morelines

    return lines


def aprint(array, **keys):
    """
    Name:
      aprint

    Purpose:

        Print out the rows and columns of the array with fields, aka structure
        or records.  The focus is on visualizing the data rather than speed or
        efficient file output.

        Optionally results can be sent to a pager or file.

        By default, the columns are printed in a simple, machine readable
        format, but using the type= keyword you can print in other styles.

        Subsets of the fields can be chosen.  Also, printing can be
        restricted to the top N lines.

        If not type='fancy', the user has more control over the format.

    Calling Sequence:
        aprint(array, **keywords)

    Inputs:
        array: A numpy array with fields.

    Keywords:
        type:
            Default: 'table'.  Print simple columns.
            If 'fancy' print with a visually appealing format.
                The delim keyword is ignored and arrays are always bracketed.

            If 'latex' print a latex table such that the
                delimiter is '&' and the lines end in latex
                continuations.  Paging is turned off.

                Currently this just prints the data part of the
                table; in the future, the full header and footer
                will be added with control.
            If 'latex-deluxe' this is currently a synonym for 'latex'

        file:
            Send results to this file rather than standard output.
        nlines:
            Print only the top N lines.  Default is to print all.
        fields or columns:
            Only print a subset of the fields.


        header:
                Write a header.  If the input is a string, it is written as the
                header followed by a new line.  If it is boolean True, a header
                is generated with the column names.  For fancy printing there
                is always a header.

        trailer:
            Text to print after the array data.

        format:
            A format string to apply to every argument.  E.g. format='%15s'
            Since every arg gets the same format, only %s type formats should
            be used unless the types are homogeneous.
        delim or sep:
            The delimiter between fields.
        array_delim:
            The delimiter between sub-array elements.
        bracket_arrays:

            Put brackets in place to delineate dimensional boundies.  e.g.
            {{a,b,c},{d,e,f}}

            Notes: if type='fancy', brackets are always used.
                   If type='fancy', the default array_delim is ',' instead
                   of ' '


        altnames:
            An alternative list of names for each field when printing a header
            of field names.  There must be an entry for each field to be
            printed.

        nformat:
            A Format to apply to the names.  By default, the same format used
            for the arguments is tried.  If formatting fails, a simple '%s' is
            used for the names.

        title:
            A title to place above the printout when using fancy printing.

    Examples:

        # simple column printing as CSV
        >>> aprint(arr, delim=',')
        1383.91540527,200.237106323,0.266301675406
        802.613586426,249.544662476,0.921706936925
        968.170288086,206.072280884,0.702349236707
        ...

        # Add some simple formatting and header with field
        # names
        >>> aprint(arr, header=True,format='%15s')
                      x               y          sigma0
          1383.91540527   200.237106323  0.266301675406
          802.613586426   249.544662476  0.921706936925
          968.170288086   206.072280884  0.702349236707
          1392.78076172   203.387145996  0.140207546039
          286.160888672   203.858230591  0.662831780399
          1399.84436035   205.773635864  0.131057799416
           730.80657959   214.152862549  0.872058593857
          379.738677979   207.252319336  0.626150666221
          1408.07873535   208.487594604  0.135600258469
          1729.27612305   209.312911987  0.626632451812

        # fancy printing with a title
        >>> aprint(arr, title='My Data',type='fancy')
                            My Data
               x       |       y       |     sigma0
        ---------------+---------------+---------------
         1383.91540527 | 200.237106323 | 0.266301675406
         802.613586426 | 249.544662476 | 0.921706936925
         968.170288086 | 206.072280884 | 0.702349236707
         ...


    """

    if "sep" in keys:
        if "delim" not in keys:
            keys["delim"] = keys["sep"]

    aw = ArrayWriter(**keys)
    aw.write(array, **keys)

    return


def arrscl(arr, minval, maxval, arrmin=None, arrmax=None, dtype="f8"):
    """
    NAME:
      arrscl()

    CALLING SEQUENCE:
      newarr = arrscl(arr, minval, maxval, arrmin=None, arrmax=None,
                      dtype='f8')

    PURPOSE:
      Rescale the range of an array to be between minval and maxval.

    INPUTS:
      arr: An array
      minval: The minimum value for the output array
      maxval: The maximum value for the output array
    OPTIONAL INPUTS:
        dtype: Default is double, 'f8'

    OPTIONAL OUTPUTS:
      arrmin=None: An number to use for the min range of the input array. By
        default it is taken from the input array.
      arrmax=None: An number to use for the max range of the input array. By
        default it is taken from the input array.

      * arrmin,arrmax are useful if you know the array is a sample of a
        particular range, for example of they are random numbers drawn
        from [0,1] you would send arrmin=0., arrmax=1.

    OUTPUTS:
      The new array.

    REVISION HISTORY:
      Converted from IDL: 2006-10-23. Erin Sheldon, NYU

    """

    output = np.array(arr, dtype=dtype, copy=True)

    if arrmin is None:
        arrmin = output.min()
    if arrmax is None:
        arrmax = output.max()

    if output.size == 1:
        return output

    if arrmin == arrmax:
        raise ValueError("arrmin must not equal arrmax")

    a = (maxval - minval) / (arrmax - arrmin)
    b = (arrmax * minval - arrmin * maxval) / (arrmax - arrmin)

    # in place
    np.multiply(output, a, output)
    np.add(output, b, output)

    return output


def make_xy_grid(n, xrang, yrang):
    """
    NAME:
        make_xy_grid()

    CALLING SEQUENCE:
        x,y = make_xy_grid(npoints, xrange, yrange)

    PURPOSE
        Create a grid of x-y points, returning x and y as numpy arrays.

    REVISION HISTORY:
        Created: mid 2009, Erin Sheldon, BNL
    """

    rng = np.arange(n, dtype="f8")
    ones = np.ones(n, dtype="f8")

    x = arrscl(rng, xrang[0], xrang[1])
    y = arrscl(rng, yrang[0], yrang[1])

    x = np.outer(x, ones)
    y = np.outer(ones, y)
    x = x.flatten(1)
    y = y.flatten(1)

    return x, y


def combine_arrlist(arrlist, keep=False):
    """
    NAME:
        combine_arrlist

    CALLING SEQUENCE:
        arr = combine_arrlist(list_of_arrays, keep=False)

    PURPOSE:
        Combined the list of arrays into one big array.  The arrays must all
        be the same data type.

    KEYWORDS:
        keep:  By default the elements are deleted as they are added to the
            big array.  Turn this off with keep=True

    REVISION HISTORY:
        Inspired by combine_ptrlist from SDSSIDL.  2007.  Erin Sheldon, BNL
    """
    if not isinstance(arrlist, list):
        raise RuntimeError("Input must be a list of arrays")

    if len(arrlist) == 0:
        return np.zeros(0, dtype="i8")

    if len(arrlist) == 1:
        return arrlist[0]

    isarray = isinstance(arrlist[0], np.ndarray)
    isrec = isinstance(arrlist[0], np.recarray)

    if not isarray:
        mess = "Input must be a list of arrays or recarrays. Found %s" % type(
            arrlist[0]
        )
        raise RuntimeError(mess)

    # loop and get total number of entries
    counts = 0
    for data in arrlist:
        counts = counts + data.size

    output = np.zeros(counts, dtype=arrlist[0].dtype)
    if isrec:
        output = output.view(np.recarray)

    beg = 0
    if keep:
        for data in arrlist:
            num = data.size
            output[beg: beg + num] = data
            beg = beg + num
    else:
        while len(arrlist) > 0:
            data = arrlist.pop(0)
            num = data.size
            output[beg: beg + num] = data
            del data
            beg = beg + num

    return output


def combine_fields(arrlist):
    """
    Combine the field names and data from multiple arrays.  The arrays must be
    the same size and have disjoint sets of fields

    Parameters
    ----------
    arr1: ndarray
        An array with fields
    arr2: ndarray
        An array with fields

    Returns
    -------
    combined array
    """
    if len(arrlist) == 0:
        raise ValueError('send at least one array')

    if len(arrlist) == 1:
        return arrlist[0]

    num = arrlist[0].size
    descr = []
    for arr in arrlist:
        if arr.size != num:
            raise ValueError('not all arrays are the same size')
        descr += arr.dtype.descr

    new_array = np.zeros(num, dtype=descr)

    for arr in arrlist:
        copy_fields(arr, new_array)

    return new_array


def copy_fields(arr1, arr2):
    """
    NAME:
        copy_fields

    CALLING SEQUENCE:
        copy_fields(array1, array2)

    PURPOSE:
        Copy common fields from one array1 to array2.  The name
        matching is case senitive.

    REVISION HISTORY:
        Inspired by struct_assign in IDL.  2007 Erin Sheldon, BNL.

    """
    if arr1.size != arr2.size:
        raise ValueError("arr1 and arr2 must be the same size")

    names1 = arr1.dtype.names
    names2 = arr2.dtype.names

    for name in names1:
        if name in names2:
            arr2[name] = arr1[name]


def extract_fields(arr, keepnames, strict=True):
    """
    NAME:
        extract_fields

    CALLING SEQUENCE:
        newarr = extract_fields(arr, names, strict=True)

    PURPOSE:
        Extract a set of fields from a numpy array.  A new array is returned
        with the requested fields and data copied in.  The name matching is
        case sensitive.

        The order of the fields is the order in the original array.

    Inputs:
        arr: A numpy structure, or array with fields.
        names: The subset of names to extract.

    Optional Inputs:
        strict:
            If True, requested names that are not found in the input array will
            raise a ValueError.  Default is True.

    REVISION HISTORY:
        Created 2007, Erin Sheldon, NYU.
        Added strict keyword, 2010-04-07, Erin Sheldon, BNL
    """
    if not isinstance(keepnames, (tuple, list, np.ndarray)):
        keepnames = [keepnames]

    arrnames = list(arr.dtype.names)

    if strict:
        for name in keepnames:
            if name not in arrnames:
                raise ValueError("field not found: %s" % name)

    new_descr = []
    for d in arr.dtype.descr:
        name = d[0]
        if name in keepnames:
            new_descr.append(d)

    if len(new_descr) == 0:
        raise ValueError("No fields kept")

    shape = arr.shape
    new_arr = np.zeros(shape, dtype=new_descr)
    copy_fields(arr, new_arr)
    return new_arr


def remove_fields(arr, rmnames):
    """
    NAME:
        remove_fields

    CALLING SEQUENCE:
        newarr = remove_fields(arr, names)

    PURPOSE:
        Remove a set of fields from the array.  A new array is returned
        with the leftover fields and data copied in.  The name matching
        is case sensitive.

    REVISION HISTORY:
        Created 2007, Erin Sheldon, NYU.
    """
    if type(rmnames) != list:
        rmnames = [rmnames]
    descr = arr.dtype.descr
    new_descr = []
    for d in descr:
        name = d[0]
        if name not in rmnames:
            new_descr.append(d)

    if len(new_descr) == 0:
        raise ValueError("Error: All fields would be removed")

    shape = arr.shape
    new_arr = np.zeros(shape, dtype=new_descr)
    copy_fields(arr, new_arr)
    return new_arr


def add_fields(arr, add_dtype_or_descr, defaults=None):
    """
    NAME:
        add_fields

    CALLING SEQUENCE:
        newarr = add_fields(arr, dtype_or_descr, defaults=None)

    PURPOSE:
        Create a new array with fields from the input array and new
        fields as indicated by the input numpy dtype or descr object.
        Return a new array with the data copied from the original array.

    KEYWORDS:
        defaults:  By default the new fields are zeroed.  Send this keyword
            to add default values to the fields.  Must be the same length
            as the input type descriptor.

    REVISION HISTORY:
        Created 2007, Erin Sheldon, NYU.


    """
    # the descr is a list of tuples
    old_descr = arr.dtype.descr
    add_dtype = np.dtype(add_dtype_or_descr)
    add_descr = add_dtype.descr

    new_descr = copy.deepcopy(old_descr)

    old_names = list(arr.dtype.names)
    for d in add_descr:
        name = d[0]
        if old_names.count(name) == 0:
            new_descr.append(d)
        else:
            raise ValueError("field " + str(name) + " already exists")

    shape = arr.shape
    new_arr = np.zeros(shape, dtype=new_descr)

    copy_fields(arr, new_arr)

    # See if the user has indicated default values for the new fields
    if defaults is not None:
        if type(defaults) != list:
            defaults = [defaults]
        if len(defaults) != len(add_descr):
            raise ValueError("defaults must be same length as new dtype")
        copy_fields_by_name(new_arr, list(add_dtype.names), defaults)

    return new_arr


def reorder_fields(arr, ordered_names, strict=True):
    """
    NAME:
        reorder_fields

    CALLING SEQUENCE:
        newarr = reorder_fields(arr, ordered_names, strict=True)

    PURPOSE:
        Re-order the fields according the the listed names.  Names
        not in the list are put at the end.


    Inputs:
        arr: A numpy structure, or array with fields.
        ordered_names: The ordered subset of names.  These are placed in order
            at the front.  Non-matching names are placed at the back.

    Optional Inputs:
        strict:
            If True, requested names that are not found in the input array will
            raise a ValueError.  Default is True.

    REVISION HISTORY:
        Created 2007, Erin Sheldon, NYU.
        Added strict keyword, 2010-04-07, Erin Sheldon, BNL
    """

    if not isinstance(ordered_names, (tuple, list, np.ndarray)):
        ordered_names = [ordered_names]

    # this is so we can get indices
    original_names = np.array(arr.dtype.names)
    original_descr = arr.dtype.descr

    new_names = []
    new_descr = []

    for name in ordered_names:
        (w,) = np.where(original_names == name)
        if w.size != 0:
            new_names.append(name)
            new_descr.append(original_descr[w[0]])
        else:
            if strict:
                raise ValueError("field not found: '%s'" % name)

    # now put in the remaining names in original order at the back
    for i in range(original_names.size):
        name = original_names[i]
        if name not in new_names:
            new_names.append(name)
            new_descr.append(original_descr[i])

    shape = arr.shape
    new_arr = np.zeros(shape, dtype=new_descr)
    copy_fields(arr, new_arr)
    return new_arr


def copy_fields_by_name(arr, names, vals):
    """
    NAME:
        copy_fields_by_name

    CALLING SEQUENCE:
        copy_fields_by_name(arr, names, values)

    PURPOSE:
        Copy values into a numpy array by field name.

    INPUTS:
        names:  Field names to be copied, scalar or sequence.
        values: The values to be copied into each field.  These values
            can be in a sequence of the same length as names.   They
            must either be scalars or their shape must match the underlying
            structure of the field.

    EXAMPLES:
        names=['x','flux', 'source']
        values=[x_array, flux_array, name_scalar]
        copy_fields_by_name(arr, names, values)

    REVISION HISTORY:
        Created 2007, Erin Sheldon, NYU.

    """
    if type(names) != list and type(names) != np.ndarray:
        names = [names]
    if type(vals) != list and type(vals) != np.ndarray:
        vals = [vals]
    if len(names) != len(vals):
        raise ValueError("Length of names and values must be the same")

    arrnames = list(arr.dtype.names)
    for name, val in zip(names, vals):
        if name in arrnames:
            arr[name] = val


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


def compare_arrays(arr1, arr2, verbose=False, ignore_missing=True):
    """
    Name:
        compare_arrays

    Calling Sequence:
        boolval=compare_arrays(array1, array2, ignore_missing=True,
                               verbose=False)

    Purpose:
        Compare the values field-by-field in two sets of numpy arrays or
        recarrays.  Return true if the data match.

    Inputs:
        array1, array2: Two arrays with fields.

    Keywords:
        ignore_missing: Default True.  Ignore fields not found in both
            arrays.
        verbose:  By default the program is silent.  set verbose=True to
            print info about each field.

    Outputs:
        True if the matching criteria are met, False if not.

    Revision History:
        Created 2007, Erin Sheldon, NYU.
        Added ignore_missing keyword.  2009-11-02, Erin Sheldon, BNL

    """

    nfail = 0

    # If requested, check the arrays have exactly the same names.
    if not ignore_missing:
        # make sure the name lists match
        if verbose:
            stdout.write("    Matching names........")

        for n in arr1.dtype.names:
            if n not in arr2.dtype.names:
                nfail += 1
                if verbose:
                    stdout.write(
                        "\n        Field '%s' found only in " "array1" % n
                    )
        for n in arr2.dtype.names:
            if n not in arr1.dtype.names:
                nfail += 1
                if verbose:
                    stdout.write(
                        "\n        Field '%s' found only in " "array2" % n
                    )

        if verbose:
            if nfail == 0:
                stdout.write("OK")
            stdout.write("\n")

    else:
        if verbose:
            stdout.write("    Not checking that all fields names match\n")

    # Compare the data for matchine names
    for n in arr1.dtype.names:
        if n in arr2.dtype.names:
            # the field was found, let's see if the data match
            if verbose:
                stdout.write("    testing field: '%s'\n" % n)
                stdout.write("        shape...........")
            if arr2[n].shape != arr1[n].shape:
                nfail += 1
                if verbose:
                    stdout.write("shapes differ\n")
            else:
                if verbose:
                    stdout.write("OK\n")
                    stdout.write("        elements........")
                (w,) = np.where(arr1[n].ravel() != arr2[n].ravel())
                if w.size > 0:
                    nfail += 1
                    if verbose:
                        stdout.write(
                            "\n        "
                            + "%s elements in field '%s' differ\n" % (w.size, n)  # noqa
                        )
                else:
                    if verbose:
                        stdout.write("OK\n")

    if nfail == 0:
        if verbose:
            stdout.write("All tests passed\n")
        return True
    else:
        if verbose:
            stdout.write("%d differences found\n" % nfail)
        return False


def replicate(value, shape, dtype=None):
    """
    Create an array filled with the input value

    Parameters
    ----------
    value: Scalar.
        The value to be replicated
    shape: Scalar or sequence.
        The shape of the resulting array.
    dtype: data-type, optional
        The data type of the result. If None, the value is determined from
        the input value.

    Returns
    -------
    array: ndarray
        A new numerical python array with every element set to the input
        value

    Examples
    --------
    >>> import esutil
    >>> from esutil.numpy_util import replicate
    >>> replicate('hello world', 3)
    array(['hello world', 'hello world', 'hello world'], dtype='|S11')
    >>> replicate(-9999.0, (2,2))
    array([[-9999., -9999.],
           [-9999., -9999.]])
    """
    if dtype is None:
        tmp = np.array([value])
        data = np.empty(shape, dtype=tmp.dtype)
    else:
        data = np.empty(shape, dtype=dtype)
    data.fill(value)
    return data


def is_big_endian(array):
    """
    returns True if array is big endian, False otherwise.

    Parameters
    ----------
    array: numpy array
        A numerical python array.

    Returns
    -------
    Truth value:
        True for big-endian

    Notes
    -----
    Strings are neither big or little endian.  The input must be a simple numpy
    array, not an array with fields.

    """

    if np.little_endian:
        machine_big = False
    else:
        machine_big = True

    byteorder = array.dtype.base.byteorder
    return (byteorder == ">") or (machine_big and byteorder == "=")


def is_little_endian(array):
    """
    returns True if array is little endian, False otherwise.

    Parameters
    ----------
    array: numpy array
        A numerical python array.

    Returns
    -------
    Truth value:
        True for little-endian

    Notes
    -----
    Strings are neither big or little endian.  The input must be a simple numpy
    array, not an array with fields.

    """

    if np.little_endian:
        machine_little = True
    else:
        machine_little = False

    byteorder = array.dtype.base.byteorder
    return (byteorder == "<") or (machine_little and byteorder == "=")


def to_native(array, inplace=False, keep_dtype=False):
    """
    NAME:
        to_native

    CALLING SEQUENCE:
        res=to_native(array, inplace=False, keep_dtype=False)

    PURPOSE:
        Convert an array to native byte order, updating the dtype to
        reflect this.  The array can have fields.

    KEYWORDS:
        inplace:  Default False.  If True the data are byteswapped
            in place and a reference to the original array is returned.
            If False a copy is always retured, even if no data were
            swapped.
        keep_dtype: Default False.  Setting to True prevents the dtype from
            being updated to reflect the new byte order.

    REVISION HISTORY:
        Created 2009, Erin Sheldon, NYU.
    """

    if np.little_endian:
        machine_little = True
    else:
        machine_little = False

    data_little = False
    if array.dtype.names is None:
        data_little = is_little_endian(array)
    else:
        # assume all are same byte order: we only need to find one with
        # little endian
        for fname in array.dtype.names:
            if is_little_endian(array[fname]):
                data_little = True
                break

    if (machine_little and not data_little) or (not machine_little and data_little):  # noqa
        doswap = True
    else:
        doswap = False

    if doswap:
        outdata = byteswap(array, inplace, keep_dtype=keep_dtype)
    else:
        if inplace:
            outdata = array
        else:
            outdata = array.copy()

    return outdata


def descr_to_native(descr):
    """
    Remove byte order information from the input numpy dtype
    descriptor.

    parameters
    ----------
    descr:
        Numpy type descriptor.  Note a dtype object.
    """
    newd = []
    for d in descr:
        nd = list(copy.deepcopy(d))
        # remove any byte order info from front of type
        nd[1] = nd[1][1:]
        nd = tuple(nd)
        newd.append(nd)
    return newd


def to_big_endian(array, inplace=False, keep_dtype=False):
    """
    NAME:
        to_big_endian

    CALLING SEQUENCE:
        res=to_big_endian(array, inplace=False, keep_dtype=False)

    PURPOSE:
        Convert an array to big endian byte order, updating the dtype to
        reflect this.  The array can have fields.

    KEYWORDS:
        inplace:  Default False.  If True the data are byteswapped
            in place and a reference to the original array is returned.
            If False a copy is always retured, even if no data were
            swapped.
        keep_dtype: Default False.  Setting to True prevents the dtype from
            being updated to reflect the new byte order.

    REVISION HISTORY:
        Created 2009, Erin Sheldon, NYU.
    """

    doswap = False
    if array.dtype.names is None:
        if not is_big_endian(array):
            doswap = True
    else:
        # assume all are same byte order: we only need to find one with
        # little endian
        for fname in array.dtype.names:
            if not is_big_endian(array[fname]):
                doswap = True
                break

    if doswap:
        outdata = byteswap(array, inplace, keep_dtype=keep_dtype)
    else:
        if inplace:
            outdata = array
        else:
            outdata = array.copy()

    return outdata


def to_little_endian(array, inplace=False, keep_dtype=False):
    """
    NAME:
        to_little_endian

    CALLING SEQUENCE:
        res=to_little_endian(array, inplace=False, keep_dtype=False)

    PURPOSE:
        Convert an array to big endian byte order, updating the dtype to
        reflect this.  The array can have fields.

    KEYWORDS:
        inplace:  Default False.  If True the data are byteswapped
            in place and a reference to the original array is returned.
            If False a copy is always retured, even if no data were
            swapped.
        keep_dtype: Default False.  Setting to True prevents the dtype from
            being updated to reflect the new byte order.

    REVISION HISTORY:
        Created 2009, Erin Sheldon, NYU.
    """

    doswap = False
    if array.dtype.names is None:
        if not is_little_endian(array):
            doswap = True
    else:
        # assume all are same byte order: we only need to find one with
        # little endian
        for fname in array.dtype.names:
            if not is_little_endian(array[fname]):
                doswap = True
                break

    if doswap:
        outdata = byteswap(array, inplace, keep_dtype=keep_dtype)
    else:
        if inplace:
            outdata = array
        else:
            outdata = array.copy()

    return outdata


def byteswap(array, inplace=False, keep_dtype=False):
    """
    NAME:
        byteswap

    CALLING SEQUENCE:
        res=byteswap(array, inplace=False, keep_dtype=False)

    PURPOSE:
        Chance the byte order of an array, updating the dtype to reflect this.
        The array can have fields.   This is a wrapper for the .byteswap()
        method which does not update the dtype to reflect the new byte
        ordering.

    KEYWORDS:
        inplace:  Default False.  If True the data are byteswapped
            in place and a reference to the original array is returned.
            If False a copy is always retured, even if no data were
            swapped.
        keep_dtype: Default False.  Setting to True prevents the dtype from
            being updated to reflect the new byte order.

    REVISION HISTORY:
        Created 2009, Erin Sheldon, NYU.
    """

    outdata = array.byteswap(inplace)
    if not keep_dtype:
        outdata.dtype = outdata.dtype.newbyteorder()

    return outdata


def unique(arr, values=False):
    """
    NAME:
        unique

    CALLING SEQUENCE:
        un = unique(arr, values=False)

    PURPOSE:
        Return indices of unique elements of a numpy array, or optionally
        the unique values.  This is not order preserving. This is currently
        implemented in a slow fashion, should be updated.

    KEYWORDS:
        values:  Default False.  If True, return the unique values as
            opposed to just the indices which is the default.

    REVISION HISTORY:
        Created 2009, Erin Sheldon, NYU.
    """
    n = arr.size
    keep = np.zeros(n, dtype="i8")

    s = arr.argsort()

    val = arr[0]
    i = 1
    nkeep = 0
    while i < n:
        ind = s[i]
        if arr[ind] != val:
            val = arr[ind]
            nkeep += 1
            keep[nkeep] = ind
        i += 1

    keep = keep[0: nkeep + 1]
    if values:
        return arr[keep]
    else:
        return keep


def rem_dup(arr, flag, values=False):
    """
    NAME:
        rem_dup

    CALLING SEQUENCE:
        indices = rem_dup(arr, flag, values=False)
        indices, values = rem_dup(arr, flag, values=True)

    PURPOSE:
        Return unique values of an array, and optionally their
        indices in the array.  Keep the duplicate with the
        largest value of flag.
        (If flag is not needed, use np.unique() instead.)

    REVISION HISTORY:
        Created 2013, Amy Kimball, CASS.
    """

    n = arr.size
    if n == 1:
        if values:
            return 0, arr
        else:
            return 0

    s = arr.argsort()  # sort indices
    sarr = arr[s]  # sorted array

    keep = np.zeros(n, dtype="i8")  # indices of values to keep
    nkeep = 0
    sflag = flag[s]  # flags to match sorted array

    val = sarr[0]  # first value to process
    f = sflag[0]  # flag for first value

    for i in range(1, n):
        if sarr[i] != val:
            val = sarr[i]
            f = sflag[i]
            nkeep += 1
            keep[nkeep] = i
        else:
            if sflag[i] > f:
                f = sflag[i]
                keep[nkeep] = i

    keep = keep[0: nkeep + 1]
    s = s[keep]
    s.sort()
    if values:
        return s, arr[s]
    else:
        return s


def match(arr1input, arr2input, presorted=False):
    """
    NAME:
        match

    CALLING SEQUENCE:
        ind1,ind2 = match(arr1, arr2, presorted=False)

    PURPOSE:
        Match two numpy arrays.  Return the indices of the matches or empty
        arrays if no matches are found.  This means arr1[ind1] == arr2[ind2] is
        true for all corresponding pairs.  arr1 must contain only unique
        inputs, but arr2 may be non-unique.
        If you know arr1 is sorted, set presorted=True and it will run
        even faster

    METHOD:
        uses searchsorted with some sugar.  Much faster than old version
        based on IDL code.

    REVISION HISTORY:
        Created 2015, Eli Rykoff, SLAC.

    """

    # make sure 1D
    arr1 = np.array(arr1input, ndmin=1, copy=False)
    arr2 = np.array(arr2input, ndmin=1, copy=False)

    # check for integer data...
    if not issubclass(arr1.dtype.type, np.integer) or not issubclass(
        arr2.dtype.type, np.integer
    ):
        mess = "Error: only works with integer types, got %s %s"
        mess = mess % (arr1.dtype.type, arr2.dtype.type)
        raise ValueError(mess)

    if (arr1.size == 0) or (arr2.size == 0):
        mess = "Error: arr1 and arr2 must each be non-zero length"
        raise ValueError(mess)

    # make sure that arr1 has unique values...
    test = np.unique(arr1)
    if test.size != arr1.size:
        raise ValueError("Error: the arr1input must be unique")

    # sort arr1 if not presorted
    if not presorted:
        st1 = np.argsort(arr1)
    else:
        st1 = None

    # search the sorted array
    sub1 = np.searchsorted(arr1, arr2, sorter=st1)

    # check for out-of-bounds at the high end if necessary
    if arr2.max() > arr1.max():
        (bad,) = np.where(sub1 == arr1.size)
        sub1[bad] = arr1.size - 1

    if not presorted:
        (sub2,) = np.where(arr1[st1[sub1]] == arr2)
        sub1 = st1[sub1[sub2]]
    else:
        (sub2,) = np.where(arr1[sub1] == arr2)
        sub1 = sub1[sub2]

    return sub1, sub2


def match_multi(arr1input, arr2input, presorted=False):
    """
    See numpy_util.match()

    """

    return match(arr1input, arr2input, presorted=False)


def strmatch(arr, regex):
    """
    Match the input string array to the regular expression

    parameters
    ----------
    arr: numpy array
        A numpy array of strings
    regex: string
        The regular expression

    examples
    --------
    regex='.*hello.*'
    logic=strmatch(arr, regex)
    keep=where(logic)
    print arr[keep]
    """
    import re

    r = re.compile(regex)
    vmatch = np.vectorize(lambda x: bool(r.match(x)))
    return vmatch(arr)


def dict2array(d, sort=False, keys=None):
    """
    Name:
      dict2array()

    Calling Sequence:
      arr = dict2array(dict, sort=False, keys=None)

    Purpose:
      Convert a dictionary to an array with fields (recarray, structured
      array).  This works for simple types e.g.  strings, integers, floating
      points.

    Keywords:
        keys: provide a sequence of keys to copy.  This can be used to order
            the fields (standard dictionary keys are unordered) or copy only a
            subset of keys.
        sort: Sort the keys.

    Comments:
        In python >= 3.1 dictionaries can be ordered.

    Revision History:
        late 2009 created.  Erin Sheldon, BNL

    """
    desc = []

    if keys is None:
        if sort:
            keys = sorted(d)
        else:
            keys = list(d.keys())

    for key in keys:
        # check key existence in case a set of keys was sent
        if key not in d:
            raise KeyError("Requested key %s not in dictionary" % key)

        if not isinstance(d[key], (int, float, str)):
            try:
                strval = "%s" % d[key]
                val = eval(strval)
            except Exception:
                val = str(d[key])
        else:
            val = d[key]

        if isinstance(val, int):
            dt = int
        elif isinstance(val, float):
            dt = float
        elif isinstance(val, str):
            dt = "S%s" % len(val)
        else:
            raise ValueError(
                "Only support int, float, string currently, "
                "found %s" % type(d[key])
            )

        desc.append((key, dt))

    a = np.zeros(1, dtype=desc)

    for key in keys:
        a[key] = d[key]

    return a


def dictlist2array(dlist, keys=None, sort=False):
    """
    Convert a list of dictionaries to an array.  Only works for basic types
    such as scalar numbers and strings

    parameters
    ----------
    dlist: list of dicts
        A list of dictionaries.  All dicts should have the same
        entries.

    keys: list, optional
        A sequence of keys to copy.  This can be used to order the fields
        (standard dictionary keys are unordered) or copy only a subset of keys.
    sort: bool, optional
        If True, sort the keys.  default False
    """
    if len(dlist) == 0:
        return np.array([])

    if keys is None:
        keys = dlist[0].keys()

        if sort:
            keys = sorted(keys)

        keys = list(keys)

    types = {}
    # for ordering
    names = []
    for key in keys:

        names.append(key)

        if key not in dlist[0]:
            raise KeyError("Requested key %s not in dictionary" % key)

        for d in dlist:

            val = d[key]

            if isinstance(val, str):
                slen = len(val)
                if key in types:
                    if types[key]["basetype"] != "S":
                        raise ValueError("type mismatch for field '%s'" % key)

                    types[key]["len"] = max(slen, types[key]["len"])
                else:
                    types[key] = {}
                    types[key]["basetype"] = "S"
                    types[key]["len"] = slen
            else:
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        "only basic types currently supported, "
                        "got '%s'" % type(val)
                    )

                if isinstance(val, int):
                    dt = "i8"
                else:
                    dt = "f8"

                if key in types:
                    if types[key]["basetype"] != dt:
                        raise ValueError("type mismatch for field '%s'" % key)
                else:
                    types[key] = {}
                    types[key]["basetype"] = dt

    dtype = []
    for name in names:

        tinfo = types[name]
        if tinfo["basetype"] == "S":
            t = "S%d" % tinfo["len"]
        else:
            t = tinfo["basetype"]

        dtype.append((name, t))

    arr = np.zeros(len(dlist), dtype=dtype)

    for i, d in enumerate(dlist):
        for key in d:
            arr[key][i] = d[key]

    return arr


def splitarray(nper, var_input):
    """
    Name:
        splitarray()

    Purpose:
        Split up an array into chunks of at least a given size.  Return a
        list of these subarrays.  The ordering is perserved.

    Calling Sequence:
        split_list = splitarray(nper, array)

    Inputs:
        nper: Number obj elements in each sub-array.  Note, the last one
            may have fewer if len(array) % nper != 0
        array: A numpy array or object that can be converted to an array.

    Output:
        A list with all the sub-arrays.

    Example:
        In [1]: l=np.arange(25)
        In [2]: nper = 3
        In [3]: split_list = eu.numpy_util.splitarray(nper, l)
        In [4]: split_list
        Out[4]:
        [array([0, 1, 2]),
         array([3, 4, 5]),
         array([6, 7, 8]),
         array([ 9, 10, 11]),
         array([14, 12, 13]),
         array([15, 16, 17]),
         array([18, 19, 20]),
         array([23, 21, 22]),
         array([24])]


    Revision History:
        Created: 2010-04-05, Erin Sheldon, BNL

    """

    var = np.array(var_input, ndmin=0, copy=False)

    ind = np.arange(var.size)

    # this will tell us which bin the object belongs to
    bin_nums = ind // int(nper)

    h, rev = stat.histogram(bin_nums, binsize=1, min=0, rev=True)

    split_list = []
    for i in range(len(h)):
        if rev[i] != rev[i + 1]:
            w = rev[rev[i]: rev[i + 1]]

            split_list.append(var[w])

    return split_list


def between(arr, lowval, highval, type="[)"):
    """
    test values of an array are between the specified values

    parameters
    ----------
    arr: array
        numpy array
    lowval: scalar
        lower value
    highval: scalar
        high value
    type: string, optional
        Interval type, one of [] () [) (]

        default [) mimicking slices for integers.  The distinction is often
        less meaningful for floating points

    returns
    -------
    bool array with True for values in the range and False otherwise

    example
    -------

    # select elements that equal 3 or are between 10 and 100 with slice
    # symantics [), e.g. [10,100)

    a=np.arange(200)
    w,=np.where( (a==3) | between(a,10,100) )

    # select elements that equal 3 or are between 10 and 100, inclusive, e.g.
    # [10,100]

    a=np.arange(200)
    w,=np.where( (a==3) | between(a,10,100,'[]') )
    """

    if type == "[)":
        logic = (arr >= lowval) & (arr < highval)
    elif type == "[]":
        logic = (arr >= lowval) & (arr <= highval)
    elif type == "()":
        logic = (arr > lowval) & (arr < highval)
    elif type == "(]":
        logic = (arr > lowval) & (arr <= highval)
    else:
        raise ValueError("bad range type: '%s'" % type)

    return logic


def outside(arr, lowval, highval, type=")("):
    """
    test values of an array are outside the specified values

    parameters
    ----------
    arr: array
        numpy array
    lowval: scalar
        lower value
    highval: scalar
        high value
    type: string, optional
        Interval type, one of )(  ][  ](  )[

        default is )( meaning total exclusion for integers. The
        distinction is often less meaningful for floating points

    returns
    -------
    bool array with True for values outside the range and False otherwise

    example
    -------

    # select elements that equal 25 or are outside 10 and 100, exclusive
    a=np.arange(200)
    w,=np.where( (a==5) | outside(a,10,100) )

    # select elements that are outside 10 and 100, inclusive
    a=np.arange(200)
    w,=np.where( outside(a,10,100,'][') )


    """

    if type == ")(":
        logic = (arr < lowval) | (arr > highval)
    elif type == "][":
        logic = (arr <= lowval) | (arr >= highval)
    elif type == "](":
        logic = (arr <= lowval) | (arr > highval)
    elif type == ")[":
        logic = (arr < lowval) | (arr >= highval)
    else:
        raise ValueError("bad range type: '%s'" % type)

    return logic


def select_percentile(x, perc, get_ranges=False, **keys):
    """
    select data in the given percentile(s)

    parameters
    ----------
    x: array-like
        The data.
    percentile: scalar or sequence
        The percentile(s) to select on.  E.g. [25,50,75] would
        select the quartiles from the array.
    get_ranges: bool, optional
        If true, output the ranges as well as the selections
    **keys:
        Extra keywords for np.percentile  See docs for that
        function for more details.

    returns
    -------
    index_list: list
        A list containing indices for data that falls in each percentile.  For
        example, if the percentiles were [25,50,75] the list would contain
        indices that satisfy

            [x < x25, x25 < x < x50, x50 < x < x75, x > x75]

        where x25 x value at the 25th percentile.

    If get_ranges==True the return is a tuple

        (index_list, range_list)

    Where range_list for percentiles [25,50,75] would be

        [ [x.min(),x25], [x25,x50], [x50,x75], [x75,x.max()] ]

    examples
    --------
    >>> x=np.random.random(10)

    >>> x
        array([ 0.34704303,  0.56085122,  0.90532323,  0.59691811,  0.8905648 ,
            0.86714466,  0.09320939,  0.0274661 ,  0.2320517 ,  0.18905247])

    >>> select_percentiles(x, [25,50,75])
        [array([6, 7, 9]), array([0, 8]), array([1, 3]), array([2, 4, 5])]

    >>> ilist,ranges=select_percentiles(x, [25,50,75], get_ranges=True)
    >>> print(ranges)
     [[0.027466097055253047, 0.19980227889572683],
      [0.19980227889572683, 0.45394712761396222],
      [0.45394712761396222, 0.79958802092575598],
      [0.79958802092575598, 0.90532323138237181]]
    """

    x = np.asanyarray(x)

    if np.isscalar(perc):
        perc = [perc]

    nperc = len(perc)

    pcuts = np.percentile(x, perc, **keys)

    wlist = []
    ranges = []
    for i in range(nperc + 1):
        if i == 0:
            (w,) = np.where(x < pcuts[i])

            ranges.append([x.min(), pcuts[i]])
        elif i == nperc:
            (w,) = np.where(x > pcuts[i - 1])

            ranges.append([pcuts[i - 1], x.max()])
        else:
            (w,) = np.where((x > pcuts[i - 1]) & (x < pcuts[i]))
            ranges.append([pcuts[i - 1], pcuts[i]])

        wlist.append(w)

    if get_ranges:
        return wlist, ranges
    else:
        return wlist


class ArrayWriter:
    """
    Class:
        ArrayWriter
    Purpose:

        A python class to write numpy arrays as ascii data.  recarrays are
        written in columns.  Can also do a "fancy" print of the array which is
        easy on the eyes but not good for machine reading.

        This is much slower than using the recfile package, but as it
        is python only it is more flexible.

    Constructor:
        aw = ArrayWriter(file=None,
                         type='table',
                         delim=' ',
                         array_delim=' ',
                         bracket_arrays=False,
                         page=False)

    Inputs:
        file:
            File name or file object to use for printing.
        type:
            Default: 'table'.  Print simple columns.
            If 'fancy' print with a visually appealing format.
                The delim keyword is ignored and arrays are always bracketed.

            If 'latex' print a latex table such that the
                delimiter is '&' and the lines end in latex
                continuations.  Paging is turned off.

                Currently this just prints the data part of the
                table; in the future, the full header and footer
                will be added with control.
            If 'latex-deluxe' this is currently a synonym for 'latex'

        delim:
            The delimiter between fields.
        array_delim:
            The delimiter between sub-array elements.
        bracket_arrays:

            Put brackets in place to delineate dimensional boundies.  e.g.
            {{a,b,c},{d,e,f}}

            Notes: if type='fancy', brackets are always used.
                   If type='fancy', the default array_delim is ',' instead
                   of ' '

        page:
            If True, send the output to a pager.


    Examples:

        # simple column printing as CSV
        >>> aw = ArrayWriter(delim=',')
        >>> aw.write(arr)
        1383.91540527,200.237106323,0.266301675406
        802.613586426,249.544662476,0.921706936925
        968.170288086,206.072280884,0.702349236707
        ...

        # Add some simple formatting and header with field
        # names (see the write() method for possible keywords)
        >>> aw = ArrayWriter()
        >>> aw.write(arr, header=True, format='%15s')
                      x               y          sigma0
          1383.91540527   200.237106323  0.266301675406
          802.613586426   249.544662476  0.921706936925
          968.170288086   206.072280884  0.702349236707
          1392.78076172   203.387145996  0.140207546039
          286.160888672   203.858230591  0.662831780399
          1399.84436035   205.773635864  0.131057799416
           730.80657959   214.152862549  0.872058593857
          379.738677979   207.252319336  0.626150666221
          1408.07873535   208.487594604  0.135600258469
          1729.27612305   209.312911987  0.626632451812

        # fancy printing with a title.  fancy can be
        # specified on construction or write()
        >>> aw = ArrayWriter(fancy=True)
        >>> aw.write(arr1, title='My Data')
                            My Data
               x       |       y       |     sigma0
        ---------------+---------------+---------------
         1383.91540527 | 200.237106323 | 0.266301675406
         802.613586426 | 249.544662476 | 0.921706936925
         968.170288086 | 206.072280884 | 0.702349236707
         ...



    """

    def __init__(self, **keys):
        self.set_defaults()
        self.open(**keys)

    def set_defaults(self):
        self._delim = " "
        self._array_delim = " "
        self._bracket_arrays = False

        self._fobj = stdout
        self._close_the_fobj = False

        self._page = False
        self._fancy = False
        self._type = "table"

    def set_keywords(self, **keys):
        self._delim = keys.get("delim", self._delim)
        self._page = keys.get("page", self._page)
        self._type = keys.get("type", self._type)

        # deal with deprecated fancy= keyword, superceded
        # by type=
        self._fancy = keys.get("fancy", self._fancy)
        if self._fancy:
            self._type = "fancy"

        if self._type == "fancy":
            self._bracket_arrays = True
        else:
            self._bracket_arrays = keys.get(
                "bracket_arrays", self._bracket_arrays
            )

        if self._type in ["latex", "latex-deluxe"]:
            self._delim = " & "
            self._array_delim = " "
        else:
            if "array_delim" not in keys:
                if self._bracket_arrays:
                    # default to commas in arrays when we are bracketing
                    self._array_delim = ","
                else:
                    # otherwise use the same as delim
                    self._array_delim = self._delim
            else:
                self._array_delim = keys["array_delim"]

    def open(self, **keys):

        self.set_keywords(**keys)

        self._close_the_fobj = False

        # Only load a file object if page is False
        # which in turn can only be true if fancy
        # is also True
        if not self._page:
            fobj = keys.get("file", stdout)

            # if isinstance(fobj,file):
            if hasattr(fobj, "read"):
                self._fobj = fobj
            else:
                self._close_the_fobj = True
                fname = os.path.expanduser(fobj)
                fname = os.path.expandvars(fname)
                self._fobj = open(fname, "w")

    def write(self, arr, **keys):
        """
        Class:
            ArrayWriter
        Name:
            write
        Calling Sequence:
            aw=ArrayWriter(**keywords)
            aw.write(array, **keywords)
        Purpose:
            Write an array.

        Inputs:
            array:
                The array to write
        Keywords:

            NOTE: All the keywords for the constructor can also be sent to the
            write() method, but note that constructor keywords will "stick".

            nlines:
                The number of lines to write
            fields or columns:
                Only print a subset of the fields.

            header:
                Write a header.  If the input is a string, it is written as the
                header followed by a new line.  If it is boolean True, a header
                is generated with the column names.  For fancy printing there
                is always a header.


            trailer:
                Text to print after the array data.

            altnames:
                A list of names for each argument.  There must be an entry for
                each argument. The names are printed above each column when
                doing fancy printing.

            format:
                A format string to apply to every argument.  E.g. format='%15s'
                Since every arg gets the same format, only %s type formats
                should be used unless the types are homogeneous.

            title:
                A title to place above the printout when using fancy printing.

        """

        self.set_keywords(**keys)

        if self._type == "fancy":
            self.fancy_write(arr, **keys)
        elif self._type in ["latex", "latex-deluxe"]:
            self.latex_write(arr, **keys)
        else:
            self.simple_write(arr, **keys)
            return

    def simple_write(self, arrin, **keys):

        # if we are paging, we will store the lines, otherwise this won't be
        # used
        lines = []

        arr = arrin.view(np.ndarray)
        allnames = arr.dtype.names

        if allnames is None:
            # simple arrays are easy
            if self._fobj is stdout:
                for val in arr:
                    print(val)
            else:
                arr.tofile(self._fobj, sep="\n")
            return

        nall = len(allnames)

        nlines = keys.get("nlines", arr.size)
        if "fields" in keys:
            names_in = keys["fields"]
        elif "columns" in keys:
            names_in = keys["columns"]
        else:
            names_in = allnames
        header = keys.get("header", False)
        trailer = keys.get("trailer", None)

        altnames = keys.get("altnames", None)

        format = keys.get("format", None)

        if eu_misc.isstring(names_in[0]):
            names = names_in
        else:
            names = []
            for ni in names_in:
                if ni > nall:
                    raise ValueError("Field index out of range: %s" % ni)
                names.append(allnames[ni])

        nnames = len(names)

        if header is True or eu_misc.isstring(header):
            if header is True:
                # create a header from the names
                if altnames is not None:
                    if len(altnames) != nnames:
                        raise ValueError(
                            "altnames must be same length as fields "
                            "to print"
                        )
                    header = copy.copy(altnames)
                else:
                    header = copy.copy(names)
                if format is not None:
                    header = [format % n for n in header]
                header = self._delim.join(header)

            if self._page:
                lines.append(header)
            else:
                self._fobj.write(header)
                self._fobj.write("\n")

        # we have fields
        astr = ArrayStringifier(
            delim=self._array_delim, brackets=self._bracket_arrays
        )

        for i in range(nlines):
            line = ""
            iname = 0
            for n in names:

                data = arr[n][i]
                if data.ndim > 0:
                    strval = astr.stringify(data)
                    if format is not None:
                        strval = format % strval
                else:
                    if format is not None:
                        strval = format % data
                    else:
                        strval = str(data)

                line += strval

                if iname < (nnames - 1):
                    line += self._delim
                iname += 1
            if self._page:
                lines.append(line)
            else:
                self._fobj.write(line)
                self._fobj.write("\n")

            if i == (nlines - 1):
                break

        if trailer is not None:
            if self._page:
                lines.append(trailer)
            else:
                self._fobj.write(trailer)
                self._fobj.write("\n")

        if self._page:
            lines = "\n".join(lines)
            pydoc.pager(lines)
        else:
            self._fobj.flush()

    def latex_write(self, arrin, **keys):

        arr = arrin.view(np.ndarray)
        allnames = arr.dtype.names

        if allnames is None:
            # simple arrays are easy
            if self._fobj is stdout:
                for val in arr:
                    print(val)
            else:
                arr.tofile(self._fobj, sep="\n")
            return

        nall = len(allnames)

        if "fields" in keys:
            names_in = keys["fields"]
        elif "columns" in keys:
            names_in = keys["columns"]
        else:
            names_in = allnames

        format = keys.get("format", None)

        if eu_misc.isstring(names_in[0]):
            names = names_in
        else:
            names = []
            for ni in names_in:
                if ni > nall:
                    raise ValueError("Field index out of range: %s" % ni)
                names.append(allnames[ni])

        nnames = len(names)

        # we have fields
        astr = ArrayStringifier(
            delim=self._array_delim, brackets=self._bracket_arrays
        )

        nlines = arr.size
        for i in range(nlines):
            line = ""
            iname = 0
            for n in names:

                data = arr[n][i]
                if data.ndim > 0:
                    strval = astr.stringify(data)
                    if format is not None:
                        strval = format % strval
                else:
                    if format is not None:
                        strval = format % data
                    else:
                        strval = str(data)

                line += strval

                if iname < (nnames - 1):
                    line += self._delim
                iname += 1
            self._fobj.write(line)
            if i < (nlines - 1):
                self._fobj.write(r" \\")

            self._fobj.write("\n")

        self._fobj.flush()

    def fancy_write(self, arrin, **keys):
        array = arrin.view(np.ndarray)

        title = keys.get("title", None)

        # if we are paging, we will store the lines, otherwise this won't be
        # used
        lines = []

        if "fields" in keys:
            fields = keys["fields"]
        elif "columns" in keys:
            fields = keys["columns"]
        else:
            fields = array.dtype.names

        printnames = keys.get("altnames", fields)

        if len(fields) != len(printnames):
            raise ValueError(
                "altnames must correspond directly to the fields "
                "being printed"
            )

        nlines = keys.get("nlines", array.size)

        max_lens = {}
        for name in fields:
            max_lens[name] = len(name)

        # first pass through data to get lengths

        # for array fields
        astr = ArrayStringifier(
            delim=self._array_delim, brackets=self._bracket_arrays
        )
        for i in range(nlines):
            for name in fields:
                if array[name][i].ndim > 0:
                    strval = astr.stringify(array[name][i])
                    max_lens[name] = max(max_lens[name], len(strval))
                else:
                    max_lens[name] = max(
                        max_lens[name], len(str(array[name][i]))
                    )

        # now create the forms for writing each field
        forms = {}
        separator = ""
        i = 0
        ntot = len(fields)
        for name in fields:
            if isinstance(array[name][0], np.string_) or (array[name][0].ndim > 0):  # noqa
                forms[name] = " %-" + str(max_lens[name]) + "s "
            else:
                forms[name] = " %" + str(max_lens[name]) + "s "

            pad = 2
            if i == (ntot - 1):
                pad = 1
            this_sep = "%s" % "-" * (max_lens[name] + pad)

            if i > 0:
                forms[name] = "|" + forms[name]
                this_sep = "+" + this_sep
            separator += this_sep
            i += 1

        # possible header and title
        header = ""
        for i in range(len(fields)):
            n = fields[i]
            pname = printnames[i]
            header += forms[n] % eu_misc.center_text(pname, max_lens[n])

        if title is not None:
            title = eu_misc.center_text(title, len(header))

        if self._page:
            if title is not None:
                lines.append(title)
            lines.append(header)
            lines.append(separator)

        else:
            if title is not None:
                self._fobj.write(title)
                self._fobj.write("\n")

            self._fobj.write(header)
            self._fobj.write("\n")

            self._fobj.write(separator)
            self._fobj.write("\n")

        for i in range(nlines):
            line = ""
            for name in fields:
                val = array[name][i]

                if val.ndim > 0:
                    val = astr.stringify(val)

                if self._page:
                    line += forms[name] % val
                else:
                    self._fobj.write(forms[name] % val)

            if self._page:
                lines.append(line)
            else:
                self._fobj.write("\n")

        trailer = keys.get("trailer", None)
        if trailer is not None:
            if self._page:
                lines.append(trailer)
            else:
                self._fobj.write(trailer)
                self._fobj.write("\n")

        if self._page:
            lines = "\n".join(lines)
            pydoc.pager(lines)
        else:
            self._fobj.flush()

    def write_array(self, arr):
        """
        Write a simple array, possibly with brackets indicating the dimensions.
        """
        if self._bracket_arrays:
            self._fobj.write("{")

        i = 0

        dimsize = arr.shape[0]

        for a in arr:
            if a.ndim > 0:
                self.write_array(a)
            else:
                self._fobj.write(str(a))

            if i < (dimsize - 1):
                self._fobj.write(",")
            i += 1

        if self._bracket_arrays:
            self._fobj.write("}")

    def close(self):
        if self._close_the_fobj:
            self._fobj.close()

    def __del__(self):
        if self._close_the_fobj:
            self._fobj.close()


def arr2str(arr, delim=",", brackets=False):
    astr = ArrayStringifier(delim=delim, brackets=brackets)
    return astr.stringify(arr)


class ArrayStringifier:
    """
    Stringify a simple array using a delimiter and
    possibly brackets
    """

    def __init__(self, delim=",", brackets=False):
        self._delim = delim
        self._brackets = brackets
        self._values = []

    def stringify(self, arr):
        self._values = []
        if arr.dtype.names is not None:
            raise ValueError("array must be simple, not structured")
        self._process(arr)
        return "".join(self._values)

    def _process(self, arr):

        if self._brackets:
            self._values.append("{")

        i = 0

        dimsize = arr.shape[0]

        for a in arr:
            if a.ndim > 0:
                self._process(a)
            else:
                self._values.append(str(a))

            if i < (dimsize - 1):
                self._values.append(self._delim)
            i += 1

        if self._brackets:
            self._values.append("}")
