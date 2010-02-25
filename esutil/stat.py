try:
    import numpy
    have_numpy=True
except:
    have_numpy=False

try:
    import scipy
    #from scipy.interpolate import interp1d
    import scipy.weave
    have_scipy=True
except:
    have_scipy=False

# for checking function type, method type
from types import *

import numpy_util

class RandomGenerator():
    """
    Class Name:
        RandomGenerator

    Purpose:
        A class for creating random samples from an arbitrary input probability
        distribution.  The input distribution can either be an array of points
        along with corresponding x values, or a function.

    Calling Sequence:
        import esutil
        gen = esutil.stat.RandomGenerator(pofx, x=None, xrange=None, nx=None,
                                          method='accum', cumulative=False, seed=None)

        r = gen.genrand(num)
        r = gen.genrand(num, seed=None)

    Inputs:
        pofx: Either an array of points or a function.  If p(x) is an array sample
            from the p(x) you must also enter the corresponding x values.

    Optional Inputs:
        x:  An array of x values.  If p(x) is a n array, these must correspond.
            If p(x) is a function, these values will be used to define the
            range over which randoms are generated, and if the cut method is
            used, to find the maximum of the input p(x) over the input range.

        xrange=[xmin,xmax]:  If p(x) is a function, you can enter xrange and nx
            and a set of x values will be generated.
        nx:  The number of points to generate in [xmin,xmax].  If p(x) is a
            function you can enter xrange and nx and a set of x values will
            be generated.

        method:  The method used for getting random points. 

            'accum': The cumulative, or accumulated, distribution is used to
                generate random points.

            'cut': Points are drawn randomly in the 2-d space defined by
                [min(x),max(x)] [0,max(prob)] and only those that lie underneath
                the pofx curve are kept.  This can be used to generate randoms
                from complex distributions that do not integrate easily.

        cumulative: The input distrubtion pofx is actually the accumulated distro.
            Note for method='cut' this is ignored: you must enter the differential 
            distribution.


    Examples:

        Using a sampled distribution pofx measured at values x.  In this case
        pofx and x are arrays.

            import esutil
            gen = esutil.stat.RandomGenerator(pofx, x)
            rand = gen.genrand(1000000)


        Using a function:

            def gaussfunc(x):
                return numpy.exp(-0.5*x**2)/numpy.sqrt(2.0*numpy.pi)

            gen = esutil.stat.RandomGenerator(gaussfunc, 
                                              xrange=[-4.5,4.5], nx=100)
            rand = gen.genrand(1000000)

    Revision History:
        2010-02-18: Created, Erin Sheldon, BNL.
    """


    def __init__(self, pofx, x=None, xrange=None, nx=None, 
                 method='accum', cumulative=False, seed=None):

        # make sure the method is valid
        self._check_method(method)
        self.method=method

        self.cumulative=cumulative



        # different initializations depending if p(x) array was sent or a
        # function

        if isinstance(pofx,(FunctionType,MethodType)):
            if x is None:
                # we'll generate the points from xrange and nx
                if xrange is None or nx is None:
                    raise ValueError("Enter the points x or both xrange and nx")
                x = numpy.arange(nx,dtype='f8')
                x = numpy_util.arrscl(x, xrange[0], xrange[1] )

            self.xinput = numpy.array(x, ndmin=1, copy=False)

            # input is some type of function
            self.isfunc=True
            self.pofx = pofx

            self.initialize_func()
        else:
            if x is None:
                raise ValueError("For p(x) an array, you must also enter the "
                                 "corresponding x values")

            self.xinput = numpy.array(x, ndmin=1, copy=False)

            # points were entered
            self.isfunc=False
            self.pofx = numpy.array(pofx, ndmin=1, copy=False)
            if self.xinput.size != self.pofx.size:
                raise ValueError("x and pofx must be same size")

            self.initialize_points()

        if self.method == 'cut':
            self.xmin = self.xinput.min()
            self.xmax = self.xinput.max()
            self.xwidth = self.xmax - self.xmin

            # maked rescaled with ranges to [0,1]
            self.xinput_scale = (self.xinput-self.xmin)/self.xwidth
            if not self.isfunc:
                self.pofx_max = pofx.max()
                self.pofx_scale = self.pofx/self.pofx_max
            else:
                # get max from evaluating at input x values
                p=self.pofx(self.xinput)
                self.pofx_max = p.max()



        if seed is not None:
            numpy.random.seed(seed=seed)

    def _check_method(self, method):
        if method not in ['accum','cut']:
            raise ValueError("method must be 'accum' or 'cut'")




    def genrand(self, numrand, seed=None):
        if self.method == 'accum':
            return self.genrand_accum(numrand, seed=None)
        elif self.method == 'cut':
            return self.genrand_cut(numrand, seed=None)

    def genrand_accum(self, numrand, seed=None):
        if seed is not None:
            numpy.random.seed(seed=seed)

        # this returns f8
        urand = numpy.random.random(numrand)

        # to get randoms from the distribution, we interpolate the x(pcum) at
        # the test rand values.  Clever!
        rand = interplin(self.xvals, self.pcum, urand)

        return rand


    def genrand_cut(self, numrand, seed=None):

        if seed is not None:
            numpy.random.seed(seed=seed)

        rand = numpy.zeros(numrand,dtype='f8')

        nleft=numrand
        ngood=0
        nleft=numrand
        while nleft > 0:

            # generate x,y values in a plane covering [xmin,xmax] [0,max(p(x))]
            randx, randy, pinterp = self.generate_cut_values(nleft)

            # keep ones where the random y values lie under the interpolated
            # curve
            w,=numpy.where(randy < pinterp)
            if w.size > 0:
                rand[ngood:ngood+w.size] = randx[w]
                ngood += w.size
                nleft -= w.size


        if not self.isfunc:
            # If we were not working with a function, we had used scaled versions
            # of x and p(x) for speed
            rand *= self.xwidth
            rand += self.xmin

        return rand

    def generate_cut_values(self, num):
        randx = numpy.random.random(num)
        randy = numpy.random.random(num)

        if self.isfunc:
            # get x,y on the right range and evaluate function
            randx *= self.xwidth
            randx += self.xmin
            randy *= self.pofx_max
            pinterp = self.pofx(randx)
        else:
            # for the point distribution, we have scaled versions of the
            # input x and y.  This will save some computation, but we have
            # to interpolate
            pinterp = interplin(self.pofx_scale, self.xinput_scale, randx)
        return randx, randy, pinterp



    def initialize_points(self):
        """
        Set up the case where the user sent x and p(x) points instead
        of a function for p(x)
        """

        if self.method == 'accum':
            # we need the cumulative probability distribution

            if self.cumulative:
                # we are done
                self.xvals = self.xinput
                self.norm = self.pofx[-1]
                self.pcum = self.pofx/self.norm

            else:
                # we must integrate and take a subset of x, since the
                # first value is not defined.

                import scipy.integrate

                pcum = scipy.integrate.cumtrapz(self.pofx,self.xinput)
                self.norm = pcum[-1]
                self.pcum = pcum/self.norm

                # interval is smaller, no integral in first point
                self.xvals = self.xinput[1:]



    def initialize_func(self):
        """
        Set up the case where the user sent x and p(x) as a function
        """

        if self.method == 'accum':
            # we need the cumulative probability distribution

            if self.cumulative:
                self.xvals = self.xinput
                self.norm = self.pofx(self.xinput[-1])
                self.pcum = self.pofx(self.xinput)/self.norm
            else:
                # we must integrate and take a subset of x, since the
                # first value is not defined.

                import scipy.integrate

                pofxvals = self.pofx(self.xinput)

                pcum = scipy.integrate.cumtrapz(pofxvals,self.xinput)
                self.norm = pcum[-1]
                self.pcum = pcum/self.norm

                # interval is smaller, no integral in first point
                self.xvals = self.xinput[1:]
 
def gaussfunc(x):
    """
    for testing function stuff
    """
    return numpy.exp(-0.5*x**2)/numpy.sqrt(2.0*numpy.pi)


def genrand(pofx, x=None, num=1, xrange=None, nx=None, 
            method='accum', cumulative=False, seed=None):
    """
    Name
        genrand
    Purpose

        This is a convenience function to create random samples from an input
        distribution.  The input distribution can either be an array of points
        along with corresponding x values, or a function.  This is just a 
        wrapper for the RandomGenerator class, using that class if preferable.

    Calling Sequence:
        import esutil
        rand = esutil.stat.genrand(pofx, x=None, num=1, xrange=None, nx=None,
                                   method='accum', cumulative=False, seed=None)

        See the documentation for the esutil.stat.RandomGenerator class for
        more info.

    Inputs:
        pofx: Either an array of points or a function.  If p(x) is an array
          sample from the p(x) you must also enter the corresponding x values.

    Optional Inputs:
        x:  An array of x values.  If p(x) is a n array, these must correspond.
            If p(x) is a function, these values will be used to define the
            range over which randoms are generated, and if the cut method is
            used, to find the maximum of the input p(x) over the input range.

        num=1: Number of randoms to generate. Default is 1.

        xrange=[xmin,xmax]:  If p(x) is a function, you can enter xrange and nx
            and a set of x values will be generated.
        nx:  The number of points to generate in [xmin,xmax].  If p(x) is a
            function you can enter xrange and nx and a set of x values will
            be generated.

        method:  The method used for getting random points. 

            'accum': The cumulative, or accumulated, distribution is used to
                generate random points.

            'cut': Points are drawn randomly in the 2-d space defined by
                [min(x),max(x)] [0,max(prob)] and only those that lie underneath
                the pofx curve are kept.  This can be used to generate randoms
                from complex distributions that do not integrate easily.

        cumulative: The input distrubtion pofx is actually the accumulated
          distro.  Note for method='cut' this is ignored: you must enter the
          differential distribution.  Default False.


    Examples:

        Using a sampled distribution pofx measured at values x.  In this case
        pofx and x are arrays.

            import esutil
            gen = esutil.stat.RandomGenerator(pofx, x=x)
            rand = esutil.genrand(pofx, x, 1000000)


        Using a function:

            def gaussfunc(x):
                return numpy.exp(-0.5*x**2)/numpy.sqrt(2.0*numpy.pi)

            rand = gen.genrand(gaussfunc, num=1000000,
                               xrange=[-4.5,4.5], nx=100)

    Revision History:
        2010-02-18: Created, Erin Sheldon, BNL.
    """


    x = numpy.array(x_input, ndmin=1, copy=False)

    if isinstance(pofx_input,(FunctionType,MethodType)):
        isfunc=True
        pofx = pofx_input
    else:
        isfunc=False
        pofx = numpy.array(pofx_input, ndmin=1, copy=False)
        if x.size != pofx.size:
            raise ValueError("x and pofx must be same size")

    if method == 'accum':
        import scipy.integrate

        if cumulative:
            pcum = pofx
            xvals = x
        else:
            # we have to do the integral
            if isfunc:
                pofx_vals = pofx(x)
                pcum = scipy.integrate.cumtrapz(pofx_vals,x)
                xvals = x[1:]
            else:
                pcum = scipy.integrate.cumtrapz(pofx,x)
                # interval is smaller, no integral in first point
                xvals = x[1:]
        
        # the normalization of the distribution
        norm = pcum[-1]
        pcum /= norm

        # generate some uniform random numbers
        numpy.random.seed(seed=seed)
        # this returns f8
        urand = numpy.random.random(numrand)

        # to get randoms from the distribution, we interpolate the 
        # x(pcum) at the test rand values.  Clever!

        rand = interplin(xvals, pcum/norm, urand)
    elif method=='cut':

        rand = numpy.zeros(numrand,dtype='f8')

        xmin = x.min()
        xmax = x.max()
        xrnge=xmax-xmin

        # rescale range to [0,1]
        x_scale = (x-xmin)/xrnge
        pofx_scale = pofx/pofx.max()
        pmax=pofx.max()

        nleft=numrand
        ngood=0
        nleft=numrand
        while nleft > 0:
            # because we scaled x to [0,1] and pofx to [0,1] we can generate
            # randoms in the [0,1] [0,1] plane

            randx = numpy.random.random(nleft)
            randy = numpy.random.random(nleft)

            # interpolate the scaled pofx to the random x points
            pinterp = interplin(pofx_scale, x_scale, randx)

            # keep ones where the random y values lie under the interpolated
            # curve

            w,=numpy.where(randy < pinterp)
            if w.size > 0:
                rand[ngood:ngood+w.size] = randx[w]
                ngood += w.size
                nleft -= w.size


        # rescale from [0,1] to proper range
        rand = rand*xrnge + xmin

    else:
        raise ValueError("method should be 'accum' or 'cut'")


    return rand


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

    if rev:
        revsize = s.size + nbin+1
        revind = numpy.zeros(revsize, dtype='i8')
    else:
        # this is just a dummy variable
        revind=numpy.zeros(1, dtype='i8')
    hist = numpy.zeros(nbin, dtype='i8')

    # populate the array from nbin+1:nbin+1+s.size
    # with the sort indices.  Simultaneosly record bin
    # edges at the beginning of reverse indices

    if use_weave:
        _weave_dohist(data, s, bsize, hist, revind, dorev=rev)
    else:
        _dohist(data, s, bsize, hist, revind, dorev=rev)

    if rev:
        return hist, revind
    else:
        return hist


def _weave_dohist(data, s, binsize, hist, rev, dorev=False):

    if dorev:
        dorev=1
    else:
        dorev=0
    """
    Weave version of histogram with reverse_indices

    s is an index into data, sorted and possibly a subset
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
                       type_converters = scipy.weave.converters.blitz)
    return


def _dohist(data, s, binsize, hist, revind, dorev=False):

    nbin=hist.size
    offset = nbin+1
    i=0
    binnum_old = -1

    dmin = data[s[0]]
    while i < s.size:
        data_index = s[i]
        if dorev:
            revind[offset] = data_index

        val = data[data_index]

        binnum = numpy.int64( (val-dmin)/binsize )
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
        # Fill in the last ones
        tbin = binnum_old + 1
        while tbin <= nbin:
            revind[tbin] = revind.size
            tbin += 1





def histogram2d(x, y, 
                nx=None, 
                ny=None, 
                xbin=None, 
                ybin=None, 
                xmin=None, 
                xmax=None, 
                ymin=None, 
                ymax=None, 
                rev=False):

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    # binsizes will take precedence
    dobinsizes=False
    donbin=False
    if (xbin != None) and (ybin != None):
        dobinsizes=True
    elif (nx != None) or (ny != None):
        donbin=True
    else:
        raise ValueError("Enter either nx,ny or xbin,ybin")

    if xmin is None:
        xmin=x.min()
    if xmax is None:
        xmax=x.max()
    if ymin is None:
        ymin=y.min()
    if ymax is None:
        ymax=y.max()

    w, = numpy.where( (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) )
    if w.size == 0:
        raise ValueError("No data in specified min/max range\n")

    if dobinsizes:
        # determine nx,ny from binsizes
        nx = numpy.int64(  (xmax-xmin)/xbin ) + 1
        ny = numpy.int64(  (ymax-ymin)/ybin ) + 1

    xind=numpy.floor((x[w]-xmin)*(nx/(xmax-xmin)))
    yind=numpy.floor((y[w]-ymin)*(ny/(ymax-ymin)))

    ind=xind+nx*yind


    hist=histogram(ind, min=0l, max=nx*ny-1, rev=rev)
    if rev:
        hist, revind=hist
        hist = hist.reshape(nx,ny)
        return hist, revind
    else:
        hist = hist.reshape(nx,ny)
        return hist


def wmom(arrin, weightsin, inputmean=None, calcerr=False, sdev=False):
    """
    NAME:
      wmom()
      
    PURPOSE:
      Calculate the weighted mean, error, and optionally standard deviation
      of an input array.

    CALLING SEQUENCE:
     wmean,werr = wmom(arr, weights, inputmean=None, calcerr=False, sdev=False)
    
    INPUTS:
      arr: A numpy array or a sequence that can be converted.
      weights: A set of weights for each elements in array.
    OPTIONAL INPUTS:
      inputmean: An input mean value, around which them mean is calculated.
      calcerr=False: Calculate the weighted error.  By default the error
        is calculated as 1/sqrt( weights.sum() ).  In this case it is
        calculated as sqrt( (w**2 * (arr-mean)**2).sum() )/weights.sum()
      sdev=False: If True, also return the weighted standard deviation 
        as a third element in the tuple.

    OUTPUTS:
      wmean, werr: A tuple of the weighted mean and error. If sdev=True the
         tuple will also contain sdev: wmean,werr,wsdev

    REVISION HISTORY:
      Converted from IDL: 2006-10-23. Erin Sheldon, NYU

   """
    from numpy import float64
    
    # no copy made if they are already arrays
    arr = numpy.array(arrin, ndmin=1, copy=False)
    weights = numpy.array(weightsin, ndmin=1, copy=False)
    
    # Weights is forced to be type double. All resulting calculations
    # will also be double
    if weights.dtype != float64:
        weights = numpy.array(weights, dtype=float64)
  
    wtot = weights.sum()
        
    # user has input a mean value
    if inputmean is None:
        wmean = ( weights*arr ).sum()/wtot
    else:
        wmean=float(inputmean)

    # how should error be calculated?
    if calcerr:
        werr2 = ( weights**2 * (arr-wmean)**2 ).sum()
        werr = numpy.sqrt( werr2 )/wtot
    else:
        werr = 1.0/numpy.sqrt(wtot)

    # should output include the weighted standard deviation?
    if sdev:
        wvar = ( weights*(arr-wmean)**2 ).sum()/wtot
        wsdev = numpy.sqrt(wvar)
        return wmean,werr,wsdev
    else:
        return wmean,werr




def sigma_clip(arrin, niter=4, nsig=4, extra={}, verbose=False):
    """
    NAME:
      sigma_clip()
      
    PURPOSE:
      Calculate the mean/stdev of an array with sigma clipping. Iterate
      niter times, removing elements that are outside nsig, and recalculating
      mean/stdev.

    CALLING SEQUENCE:
      mean,stdev = sigma_clip(arr, niter=4, nsig=4, extra={})
    
    INPUTS:
      arr: A numpy array or a sequence that can be converted.

    OPTIONAL INPUTS:
      niter: number of iterations, defaults to 4
      nsig: number of sigma, defaults to 4

    OUTPUTS:
      mean,stdev: A tuple containing mean and standard deviation.
    OPTIONAL OUTPUTS
      extra={}: Dictionary containing the array of used indices in
         extra['index']

    REVISION HISTORY:
      Converted from IDL: 2006-10-23. Erin Sheldon, NYU

   """
    arr = numpy.array(arrin, ndmin=1, copy=False)

    index = numpy.arange( arr.size )

    for i in numpy.arange(niter):
        m = arr[index].mean()
        s = arr[index].std()

        if verbose:
            sys.stdout.write('iter %s\tnuse: %s\tmean %s\tstdev %s\n' % \
                (i+1, index.size,m,s))

        clip = nsig*s

        w, = numpy.where( (numpy.abs(arr[index]) - m) < clip )

        if w.size == 0:
            sys.stderr.write("nsig too small. Everything clipped on iteration %d" % i+1)
            return m,s


        index = index[w]

    # Calculate final stats
    amean = arr[index].mean()
    asig = arr[index].std()

    extra['index'] = index
    return amean, asig
     

def interplin(vin, xin, uin):
    """
    NAME:
      interplin()
      
    PURPOSE:
      Perform 1-d linear interpolation.  Values outside the bounds are
      permitted unlike the scipy.interpolate.interp1d module. The are
      extrapolated from the line between the 0,1 or n-2,n-1 entries.
      This program is not as powerful as interp1d but it does provide
      this which makes it compatible with the IDL interpol() function.

    CALLING SEQUENCE:
      yint = interplin(y, x, u)

    INPUTS:
      y, x:  The y and x values of the data.
      u: The x-values to which will be interpolated.

    REVISION HISTORY:
      Created: 2006-10-24, Erin Sheldon, NYU
    """
    # Make sure inputs are arrays.  Copy only made if they are not.
    v=numpy.array(vin, ndmin=1, copy=False)
    x=numpy.array(xin, ndmin=1, copy=False)
    u=numpy.array(uin, ndmin=1, copy=False)

    # Find closest indices
    xm = x.searchsorted(u) - 1
    
    # searchsorted returns size(array) when the input is larger than xmax
    # Also, we need the index to be less than the last since we interpolate
    # *between* points.
    w, = numpy.where(xm >= (x.size-1))
    if w.size > 0:
        xm[w] = x.size-2

    w, = numpy.where(xm < 0)
    if w.size > 0:
        xm[w] = 0
        
    xmp1 = xm+1
    return (u-x[xm])*(v[xmp1] - v[xm])/(x[xmp1] - x[xm]) + v[xm]


