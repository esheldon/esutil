import numpy as numpy
import sys
from sys import stdout, stderr
import copy

def dict2array(d, sort=False, keys=None):
    """

    Convert a dictionary to an array with fields (recarray, structured array).
    This works for simple types e.g.  strings, integers, floating points.

    You can send the keys= keyword to provide a sequence of keys to copy.
    This can be used to order the fields (standard dictionary keys are
    unordered) or copy only a subset of keys.

    You can also send sort=True to sort the keys.

    In python >= 3.1 you can used ordered dictionaries.

    """
    desc=[]

    if keys is None:
        if sort:
            keys=sorted(d)
        else:
            keys=d.keys()

    for key in keys:
        # check key existence in case a set of keys was sent
        if key not in d:
            raise KeyError("Requested key %s not in dictionary" % key)

        if isinstance(d[key], int):
            dt=int
        elif isinstance(d[key], float):
            dt=float
        elif isinstance(d[key], str):
            dt='S%s' % len(d[key])
        else:
            raise ValueError("Only support int, float, string currently")

        desc.append( (key, dt) )

    a=numpy.zeros(1, dtype=desc)

    for key in keys:
        a[key] = d[key]

    return a

def arrscl(arr, minval, maxval, arrmin=None, arrmax=None):
    """
    NAME:
      arrscl()

    PURPOSE:
      Rescale the range of an array.

    CALLING SEQUENCE:
      newarr = arrscl(arr, minval, maxval, arrmin=None, arrmax=None)
    
    INPUTS:
      arr: An array
      minval: The minimum value for the output array
      maxval: The maximum value for the output array

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

    # makes a copy either way (asarray would not if it was an array already)
    output = numpy.array(arr)
    
    if arrmin == None: arrmin = output.min()
    if arrmax == None: arrmax = output.max()
    
    if output.size == 1:
        return output
    
    if (arrmin == arrmax):
        raise ValueError('arrmin must not equal arrmax')

    #try:
    a = (maxval - minval)/(arrmax - arrmin)
    b = (arrmax*minval - arrmin*maxval)/(arrmax - arrmin)
    #except:
    #print "Error calculating a,b: ", \
    #      sys.exc_info()[0], sys.exc_info()[1]
    #return None

    # in place
    numpy.multiply(output, a, output)
    numpy.add(output, b, output)
    
    return output

def make_xy_grid(n, xrang, yrang):
    # Create a grid on input ranges
    rng = numpy.arange(n, dtype='f8')
    ones = numpy.ones(n, dtype='f8')

    x = arrscl(rng, xrang[0], xrang[1])
    y = arrscl(rng, yrang[0], yrang[1])

    x= numpy.outer(x, ones)
    y= numpy.outer(ones, y)
    x = x.flatten(1)
    y = y.flatten(1)

    return x,y

def combine_arrlist(arrlist, keep=False):
    """
    Combined the list of arrays into one big array.  The arrays must all
    be the same data type.

    By default the elements are deleted as they are added to the big array.
    Turn this off with keep=True
    """
    if not isinstance(arrlist,list):
        raise RuntimeError('Input must be a list of arrays')

    isarray = isinstance(arrlist[0], numpy.ndarray)
    isrec = isinstance(arrlist[0], numpy.recarray)
        
    if not isarray:
        mess = 'Input must be a list of arrays or recarrays. Found %s' % \
                type(arrlist[0])
        raise RuntimeError(mess)

    # loop and get total number of entries
    counts=0
    for data in arrlist:
        counts = counts+data.size

    output = numpy.zeros(counts, dtype=arrlist[0].dtype)
    if isrec:
        output = output.view(numpy.recarray)

    beg=0
    if keep:
        for data in arrlist:
            num = data.size
            output[beg:beg+num] = data
            beg=beg+num
    else:
        while len(arrlist) > 0:
            data = arrlist.pop(0)
            num = data.size
            output[beg:beg+num] = data
            beg=beg+num

    return output


def copy_fields(arr1, arr2):
    """
    Copy common fields from one numpy array or recarray to another.
    """
    if arr1.size != arr2.size:
        raise ValueError('arr1 and arr2 must be the same size')

    names1=arr1.dtype.names
    names2=arr2.dtype.names
    for name in names1:
        if name in names2:
            arr2[name] = arr1[name]

def extract_fields(arr, keepnames):
    """
    Extract a set of fields from a numpy array or recarray.
    """
    if type(keepnames) != list and type(keepnames) != numpy.ndarray:
        keepnames=[keepnames]
    arrnames = list( arr.dtype.names )
    new_descr = []
    for d in arr.dtype.descr:
        name=d[0]
        if name in keepnames:
            new_descr.append(d)
    if len(new_descr) == 0:
        raise ValueError('No field names matched')

    shape = arr.shape
    new_arr = numpy.zeros(shape,dtype=new_descr)
    copy_fields(arr, new_arr)
    return new_arr



def remove_fields(arr, rmnames):
    """
    Remove a set of fields from a numpy array or recarray
    """
    if type(rmnames) != list:
        rmnames=[rmnames]
    descr = arr.dtype.descr
    new_descr = []
    for d in descr:
        name=d[0]
        if name not in rmnames:
            new_descr.append(d)

    if len(new_descr) == 0:
        raise ValueError('Error: All fields would be removed')

    shape = arr.shape
    new_arr = numpy.zeros(shape, dtype=new_descr)
    copy_fields(arr, new_arr)
    return new_arr

def copy_fields_by_name(arr, names, vals):
    """
    Copy values into an array with fields, or recarray, by name.
    """
    if type(names) != list and type(names) != numpy.ndarray:
        names=[names]
    if type(vals) != list and type(vals) != numpy.ndarray:
        vals=[vals]
    if len(names) != len(vals):
        raise ValueError('Length of names and values must be the same')

    arrnames = list(arr.dtype.names)
    for name,val in zip(names,vals):
        if name in arrnames:
            arr[name] = val

def add_fields(arr, add_dtype_or_descr, defaults=[]):
    """
    Add new fields to a numpy array or recarray, with an optional
    set of default values
    """
    # the descr is a list of tuples
    old_descr = arr.dtype.descr
    add_dtype = numpy.dtype(add_dtype_or_descr)
    add_descr = add_dtype.descr

    new_descr = copy.deepcopy(old_descr)

    old_names = list(arr.dtype.names)
    new_names = list(add_dtype.names)
    for d in add_descr:
        name=d[0]
        if old_names.count(name) ==0:
            new_descr.append(d)
        else:
            raise ValueError( 'field '+str(name)+' already exists')

    shape = arr.shape
    new_arr = numpy.zeros(shape, dtype=new_descr)
    
    copy_fields(arr, new_arr)
    
    # See if the user has indicated default values for the new fields
    if type(defaults) != list:
        defaults=[defaults]
    ldef=len(defaults)
    if ldef > 0:
        if ldef != len(add_descr):
            raise ValueError('defaults must be same length as new dtype')
        copy_fields_by_name(new_arr, list(add_dtype.names), defaults)

    return new_arr


def compare_arrays(arr1, arr2, verbose=False):
    """
    Compare the values field-by-field in two sets of numpy arrays or
    recarrays.
    """

    nfail = 0
    for n2 in arr2.dtype.names:
        n1 = n2
        if n1 not in arr1.dtype.names:
            n1 = n1.lower()
            if n1 not in arr1.dtype.names:
                n1 = n1.upper()
                if n1 not in arr1.dtype.names:
                    raise ValueError('field name %s not found in array 1' % n2)
            
        if verbose:
            stdout.write("    testing field: '%s'\n" % n2)
            stdout.write('        shape...........')
        if arr2[n2].shape != arr1[n1].shape:
            nfail += 1
            if verbose:
                stdout.write('shapes differ\n')
        else:
            if verbose:
                stdout.write('OK\n')
                stdout.write('        elements........')
            w,=numpy.where(arr1[n1].ravel() != arr2[n2].ravel())
            if w.size > 0:
                nfail += 1
                if verbose:
                    stdout.write('\n        '+\
                        '%s elements in field %s differ\n' % (w.size,n2))
            else:
                if verbose:
                    stdout.write('OK\n')

    if nfail == 0:
        if verbose:
            stdout.write('All tests passed\n')
        return True
    else:
        if verbose:
            stdout.write('%d differences found\n' % nfail)
        return False


def is_big_endian(array):
    """
    Return True if array is big endian.  Note strings are neither big
    or little endian
    """

    if numpy.little_endian:
        machine_big=False
    else:
        machine_big=True

    byteorder = array.dtype.base.byteorder
    return (byteorder == '>') or (machine_big and byteorder == '=')

def is_little_endian(array):
    """
    Return True if array is little endian. Note strings are neither big
    or little endian
    """

    if numpy.little_endian:
        machine_little=True
    else:
        machine_little=False

    byteorder = array.dtype.base.byteorder
    return (byteorder == '<') or (machine_little and byteorder == '=')



def to_big_endian(array, inplace=False, keep_dtype=False):
    """
    Convert an array to big endian byte order, updating the dtype
    to reflect this.  Send keep_dtype=True to prevent the dtype
    from being updated.  If inplace=False, a copy is always returned.
    If inplace=True a reference to the input array is returned.
    """

    doswap=False
    if array.dtype.names is None:
        if not is_big_endian(array):
            doswap=True
    else:
        # assume all are same byte order: we only need to find one with
        # little endian
        for fname in array.dtype.names:
            if not is_big_endian(array[fname]):
                doswap=True
                break

    if doswap:
        outdata = byteswap(outdata, inplace, keep_dtype=keep_dtype)
    else:
        if inplace:
            outdata=array
        else:
            outdata=array.copy()

    return outdata

def to_little_endian(array, inplace=False, keep_dtype=False):
    """
    Convert an array to little endian byte order, updating the dtype
    to reflect this.  Send keep_dtype=True to prevent the dtype
    from being updated. If inplace=False, a copy is always returned.
    If inplace=True a reference to the input array is returned.
    """

    doswap=False
    if array.dtype.names is None:
        if not is_little_endian(array):
            doswap=True
    else:
        # assume all are same byte order: we only need to find one with
        # little endian
        for fname in array.dtype.names:
            if not is_little_endian(array[fname]):
                doswap=True
                break

    if doswap:
        outdata = byteswap(outdata, inplace, keep_dtype=keep_dtype)
    else:
        if inplace:
            outdata=array
        else:
            outdata=array.copy()


    return outdata



def byteswap(array, inplace=False, keep_dtype=False):
    """
    byteswap an array, updating the dtype to reflect this

    If you *don't* want the dtype changed, simply use the
    built-in array method.  E.g.  array.byteswap()
    """

    outdata = array.byteswap(inplace)
    if not keep_dtype:
        outdata.dtype = outdata.dtype.newbyteorder()

    return outdata
      

def match(arr1, arr2):
    """
    ind1,ind2 = match(arr1, arr2)
    match two numpy arrays.  Return the indices of the matches or [-1] if no
    matches are found.  This means
        arr1[ind1] == arr2[ind2]
    is true for all corresponding pairs

    Arrays must contain only unique elements
    """
    dtype = 'i8'
    n1 = len(arr1)
    n2 = len(arr2)

    if (n1 == 1) or (n2 == 1):
        # one of the arrays is length one
        if n2 > 1:
            sub2, = numpy.where(arr2 == arr1[0])
            if sub2.size > 0:
                sub1 = numpy.array([0], dtype=dtype)
            else:
                sub1 = numpy.array([-1], dtype=dtype)
        else:
            sub1, = numpy.where(arr1 == arr2[0])
            if sub1.size > 0:
                sub2 = numpy.array([0], dtype=dtype)
            else:
                sub2 = numpy.array([-1], dtype=dtype)

        return sub1, sub2


    # make a combined set
    tmp = numpy.zeros(n1+n2, dtype=arr1.dtype)
    tmp[0:n1] = arr1[:]
    tmp[n1:] = arr2[:]

    ind = numpy.zeros(n1+n2, dtype=dtype)
    ind[0:n1] = numpy.arange(n1)
    ind[n1:] = numpy.arange(n2)

    vec = numpy.zeros(n1+n2, dtype='b1')
    vec[n1:] = 1

    # sort combined list
    sortind = tmp.argsort()
    tmp = tmp[sortind]
    ind = ind[sortind]
    vec = vec[sortind]

    # this finds adjacent dups but only if they are not from the
    # same array.  Since we demand unique arrays I'm not sure why
    # the second check is needed
    firstdup, = numpy.where((tmp == numpy.roll(tmp,-1)) &
                            (vec != numpy.roll(vec,-1)) )
    if firstdup.size == 0:
        sub1 = numpy.array([-1], dtype=dtype)
        sub2 = numpy.array([-1], dtype=dtype)
        return sub1, sub2

    # both duplicate values...?
    dup = numpy.zeros(firstdup.size*2, dtype=dtype)

    even = numpy.arange( firstdup.size, dtype=dtype)*2
    dup[even] = firstdup
    dup[even+1] = firstdup+1

    # indices of duplicates
    ind = ind[dup]
    # vector id of duplicates
    vec = vec[dup]

    # now subscripts
    sub1 = ind[ numpy.where( vec == 0 ) ]
    sub2 = ind[ numpy.where( vec != 0 ) ]
    return sub1, sub2


def _weave_rev_hist(data, s, binsize, hist, rev):
    """
    Weave version of histogram with reverse_indices
    """
    code = """

    int64_t nbin = hist.size();
    int64_t binnum_old = -1;

    // index of minimum value
    int64_t imin = s(0);
    for (int64_t i=0; i<s.size(); i++) {

        int64_t offset = i+nbin+1;
        int64_t data_index = s(i);


        rev(offset) = data_index;

        int64_t binnum = (int64_t) ( (data(data_index)-data(imin))/binsize);

        if (binnum >= 0 && binnum < nbin) {
            if (binnum > binnum_old) {
                int64_t tbin = binnum_old + 1;
                while (tbin <= binnum) {
                    rev(tbin) = offset;
                    tbin++;
                }
            }
            hist(binnum) = hist(binnum) + 1;
            binnum_old = binnum;
        }
    }

    int64_t tbin = binnum_old + 1;
    while (tbin <= nbin) {
        rev(tbin) = rev.size();
        tbin++;
    }

    """

    weave.inline(code, ['data','s','binsize','hist','rev'],
                 type_converters = weave.converters.blitz)
    return


def histogram(data, binsize=1., min=None, max=None, rev=False, use_weave=False):
    """
    Similar to IDL histogram.
    """
     
    dowhere=False
    s = data.argsort()
    if min is not None:
        dmin = min
        dowhere=True
    else:
        dmin = data[s[0]]

    if max is not None:
        dmax = max
        dowhere=True
    else:
        dmax = data[s[-1]]

    bsize = float(binsize)

    if dowhere:
        # where will preserve order, so subscript with s
        w,=numpy.where( (data[s] >= dmin) & (data[s] <= dmax) )
        if w.size == 0:
            raise ValueError("No data in specified min/max range\n")
        s = s[w]

    nbin = long( (dmax-dmin)/bsize ) + 1
    revsize = s.size + nbin+1

    if rev:
        revind = numpy.zeros(revsize, dtype='i8')
    hist = numpy.zeros(nbin, dtype='i8')

    # populate the array from nbin+1:nbin+1+s.size
    # with the sort indices.  Simultaneosly record bin
    # edges at the beginning of reverse indices

    if use_weave:
        _weave_rev_hist(data, s, bsize, hist, revind)
        return hist, revind

    offset = nbin+1
    i=0
    binnum_old = -1
    while i < s.size:
        data_index = s[i]
        if rev:
            revind[offset] = data_index

        val = data[data_index]

        binnum = long( (val-dmin)/bsize )
        #print 'binnum:',binnum,' binnum old:',binnum_old, 'val:',val
        if binnum >= 0 and binnum < nbin:
        #if binnum >= 0:
            if binnum > binnum_old:
                tbin = binnum_old + 1
                while tbin <= binnum:
                    if rev:
                        revind[tbin] = offset
                        #print '\t\trevind[%d] = %d' % (tbin,offset)
                    tbin += 1

            hist[binnum] += 1
            binnum_old = binnum
        #print 'rev:',revind[binnum]

        i += 1
        offset += 1

    if rev:
        #pass
        # Fill in the last ones
        tbin = binnum_old + 1
        while tbin <= nbin:
            revind[tbin] = revsize
            tbin += 1
        #revind[nbin] = revsize

    if rev:
        return hist, revind
    else:
        return hist


