try:
    import numpy
    have_numpy=True
except:
    have_numpy=False

try:
    import scipy
    from scipy.interpolate import interp1d
    import scipy.weave
    have_scipy=True
except:
    have_scipy=False



def _weave_dohist(data, s, binsize, hist, rev, dorev=False):
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


        if (dorev) {
            rev(offset) = data_index;
        }

        int64_t binnum = (int64_t) ( (data(data_index)-data(imin))/binsize);

        if (binnum >= 0 && binnum < nbin) {
            if (dorev && (binnum > binnum_old) ) {
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
        if (dorev) {
            rev(tbin) = rev.size();
        }
        tbin++;
    }

    """

    scipy.weave.inline(code, ['data','s','binsize','hist','rev','dorev'],
                       type_converters = weave.converters.blitz)
    return


def _dohist(data, s, binsize, hist, revind, dorev=False):

    nbin=hist.size
    offset = nbin+1
    i=0
    binnum_old = -1
    while i < s.size:
        data_index = s[i]
        if dorev:
            revind[offset] = data_index

        val = data[data_index]

        binnum = numpy.int64( (val-dmin)/bsize )
        #print 'binnum:',binnum,' binnum old:',binnum_old, 'val:',val
        if binnum >= 0 and binnum < nbin:
        #if binnum >= 0:
            if binnum > binnum_old:
                tbin = binnum_old + 1
                while tbin <= binnum:
                    if dorev:
                        revind[tbin] = offset
                        #print '\t\trevind[%d] = %d' % (tbin,offset)
                    tbin += 1

            hist[binnum] += 1
            binnum_old = binnum
        #print 'rev:',revind[binnum]

        i += 1
        offset += 1

    if dorev:
        #pass
        # Fill in the last ones
        tbin = binnum_old + 1
        while tbin <= nbin:
            revind[tbin] = revsize
            tbin += 1
        #revind[nbin] = revsize




def histogram(data, binsize=1., min=None, max=None, rev=False, use_weave=True):
    """
    Similar to IDL histogram.

    For reverse indices, the fast version uses weave from scipy. This is the
    default.  If scipy is not available a slower version is used.

    """

    if not have_scipy:
        use_weave=False
     
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

    nbin = numpy.int64( (dmax-dmin)/bsize ) + 1
    revsize = s.size + nbin+1

    if rev:
        revind = numpy.zeros(revsize, dtype='i8')
    hist = numpy.zeros(nbin, dtype='i8')

    # populate the array from nbin+1:nbin+1+s.size
    # with the sort indices.  Simultaneosly record bin
    # edges at the beginning of reverse indices

    if use_weave:
        _weave_dohist(data, s, bsize, hist, revind, dorev=rev)
        return hist, revind
    else:
        _dohist(data, s, bsize, hist, revind, dorev=rev)

    if dorev:
        return hist, revind
    else:
        return hist

