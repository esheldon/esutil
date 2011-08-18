"""
Package:
    stat

This is a sub-package of the esutil package. The full reference is esutil.stat

Methods:
    histogram:  
        Calculate the histogram of the input data.  The reverse indices are
        also optionally calculated.  This function behaves similarly to the
        IDL histogram function.  Also has the option to use weights, and
        to tabulate a large number of statistics for each bin.
    histogram2d:  
        Histgram two variables.
    wmom:  
        Calculate weighted mean and error for the given input data.
    sigma_clip:  
        Return the sigma-clipped mean and error for the input data.
    interplin:  
        Perform linear interpolation.  This function is less powerful than
        scipy.interpolate.interp1d but behaves like the IDL interpol()
        function, including extrapolation beyond boundaries.
"""
license="""
  Copyright (C) 2010  Erin Sheldon

    This program is free software; you can redistribute it and/or modify it
    under the terms of version 2 of the GNU General Public License as
    published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""


# the external C++ code for doing 
# histograms.
try:
    import chist
    have_chist=True
except:
    have_chist=False

# for checking function type, method type
from types import *

try:
    import numpy
    have_numpy=True
except:
    have_numpy=False

import esutil.numpy_util as numpy_util

class Binner(dict):
    """
    Bin data and calculate statistics of the bins.

    b = Binner(x, y=None, weights=None)

    Examples
    --------
    For more examples, see the dohist() and calc_stats() methods.
    1-d:
        b=Binner(x)
        # few ways to calculate hist
        b.dohist(binsize=0.1)
        b.dohist(nbin=10)
        b.dohist(nperbin=10)

        # histogram exists now
        b['hist']

        # calculate more statistics about the bins
        b.calc_stats()
        b['center'], b['low'], b['high']

        # if weights are sent, we also get 'whist' after
        # running calc_stats
        b=Binner(x, weights=w)
        b.dohist(binsize=0.1)
        b.calc_stats()

    2-d:
        b=Binner(x,y)
        b.dohist(nperbin=10)
        b.calc_stats()

        # reverse indices are always calculated when
        # y is sent, so we have the mean values for x,y
        b['center']
        b['xmean'], b['xstd'], b['xerr'], b['xerr2']
        b['ymean'], b['ystd'], b['yerr'], b['yerr2']

    """
    def __init__(self, x, y=None, weights=None):
        self.x = numpy.array(x, ndmin=1, copy=False)
        self.y = y
        self.weights = weights

        self.sort_index = None

        self.xpref=''
        if y is not None:
            self.xpref = 'x'
            self.y = numpy.array(y, ndmin=1, copy=False)
            if self.y.size != self.x.size:
                raise ValueError("y must be same len as x")

        if weights is not None:
            self.weights = numpy.array(weights, ndmin=1, copy=False)
            if self.weights.size != self.x.size:
                raise ValueError("Weights must be same len as data")



    def dohist(self, binsize=None, nbin=None, nperbin=None, min=None, max=None, rev=False):
        """
        Perform the basic histogram, optionally getting reverse indices. Note
        if weights were sent, reverse indices will always be calculated
        """

        self.clear()

        # if y is sent, we use rev to get mean y in the bins
        if self.y is not None:
            rev=True

        # get self['wsort'] and self.dmin, self.dmax
        self._get_minmax_and_indices(min=min, max=max)

        if nperbin is not None:
            self._hist_by_num(nperbin)
        elif nbin is not None or binsize is not None:
            self._hist_by_binsize_or_nbin(binsize, nbin, rev)
        else:
            raise ValueError("Send binsize or nbin or nperbin")

    def _hist_by_binsize_or_nbin(self, binsize, nbin, rev):

        if binsize is not None:
            nbin = numpy.int64( (self.dmax-self.dmin)/binsize ) + 1
        elif nbin is not None:
            binsize = float(self.dmax-self.dmin)/nbin
        else:
            raise RuntimeError("Expected binsize or nbin")
        self['binsize'] = binsize
        self['nbin'] = nbin

        h,r = self._do_hist(self.x, self.dmin, self['wsort'], binsize, nbin, rev=rev)

        self['hist'] = h
        if r is not None:
            self['rev'] = r

    def _hist_by_num(self, nperbin):
        
        # histogram indices into array
        # binsize is nperbin
        ind = numpy.arange(self['wsort'].size)
        # already sorted
        inds = ind
        bsize = float(nperbin)
        indmax = ind[-1]
        indmin = 0
        nbin = numpy.int64( (indmax-indmin)/bsize ) + 1
        hist,rev = self._do_hist(ind, 0, inds, bsize, nbin, True)

        # convert the indices in rev to the unlimited, unsorted frame
        self['low'] = numpy.zeros(nbin, dtype='f8')
        self['high'] = numpy.zeros(nbin, dtype='f8')
        for i in xrange(nbin):
            if rev[i] != rev[i+1]:
                w = rev[ rev[i]:rev[i+1] ]
                w = self['wsort'][w]

                rev[ rev[i]:rev[i+1] ] = w

                self['low'][i] = self.x[w[0]]
                self['high'][i] = self.x[w[-1]]

        self['hist'] = hist
        if rev is not None:
            self['rev'] = rev
        self['nperbin'] = nperbin

    def _do_hist(self, data, dmin, s, bsize, nbin, rev=False):
        dorev = rev
        if self.weights is not None:
            # force rev so we can add up in bins with weights
            dorev=True

        if have_chist:
            # compute using the external C++ code
            if dorev:
                hist, revind = chist.chist(data, dmin, s, bsize, nbin, dorev)
            else:
                hist         = chist.chist(data, dmin, s, bsize, nbin, dorev)
                revind=None

        else:
            # compute in a python loop

            if dorev:
                revsize = s.size + nbin+1
                revind = numpy.zeros(revsize, dtype='i8')
            else:
                # this is just a dummy variable
                revind=None
            hist = numpy.zeros(nbin, dtype='i8')

            _dohist(data, dmin, s, bsize, hist, revind=revind)

        return hist, revind


    def _get_sort_index(self):
        if self.sort_index is None:
            self.sort_index = self.x.argsort()
        # make it visible too
        self['sort_index'] = self.sort_index

    def _get_minmax_and_indices(self, min=None, max=None):
        """
        Get sort index, min/max, and w, the sorted indices in the specified
        min/max range.
        """
        self._get_sort_index()
        s=self['sort_index']

        dowhere=False
        if min is not None:
            xmin = min
            dowhere=True
        else:
            xmin = self.x[s[0]]


        if max is not None:
            xmax = max
            dowhere=True
        else:
            xmax = self.x[s[-1]]
        
        self.dmin = xmin
        self.dmax = xmax

        self[self.xpref+'min'] = xmin
        self[self.xpref+'max'] = xmax

        if dowhere:
            # where function will preserve order, so subscript with s
            w,=numpy.where( (self.x[s] >= xmin) & (self.x[s] <= xmax) )
            if w.size == 0:
                raise ValueError("No data in specified min/max range: [%s,%s]" % (xmin,xmax))
            self['wsort'] = s[w]
        else:
            self['wsort'] = s

    def calc_stats(self):
        if 'hist' not in self:
            raise ValueError("run dohist first")

        xpref=self.xpref

        nhist = len(self['hist'])
        if 'nperbin' in self:
            # we need to get the actual bin edges
            # from the reverse indices
            pass
        else:
            # if we used a binsize or nbin, we return the
            # edges and center of the bin

            low = numpy.arange(nhist, dtype='f8')
            low = self.dmin + low*self['binsize']

            high = low + self['binsize']
            center = low + 0.5*self['binsize']

            self[xpref+'low'] = low
            self[xpref+'high'] = high
            self[xpref+'center'] = center

        if 'rev' in self:
            revind = self['rev']
            # calculate the mean in the bins
            xmean   = numpy.zeros(nhist) - 9999.0
            xstd    = xmean.copy()
            xerr    = xmean.copy()
            xmedian = xmean.copy()
            if self.y is not None:
                ymean   = xmean.copy()
                ystd    = xmean.copy()
                yerr    = xmean.copy()
                ymedian = xmean.copy()

            if self.weights is not None:
                whist = xmean.copy()
                whist[:] = 0
                wxmean = xmean.copy()
                wxstd  = xmean.copy()
                wxerr  = xmean.copy()
                wxerr2 = xmean.copy()
                if self.y is not None:
                    wymean = xmean.copy()
                    wystd  = xmean.copy()
                    wyerr  = xmean.copy()
                    wyerr2 = xmean.copy()

            for i in xrange(nhist):
                if revind[i] != revind[i+1]:
                    w = revind[ revind[i]:revind[i+1] ]

                    if w.size == 1:
                        xmean[i] = self.x[w[0]]
                        xmedian[i] = xmean[i]
                        xstd[i] = 0
                        xerr[i] = xmean[i]
                        if self.y is not None:
                            ymean[i] = self.y[w[0]]
                            ymedian[i] = ymean[i]
                            ystd[i] = 0
                            yerr[i] = ymean[i]

                        if self.weights is not None:
                            whist[i] = self.x[w[0]]*self.weights[w[0]]
                            wxmean[i] = xmean[i]
                            wxstd[i] = 0
                            wxerr[i] = wxmean[i]
                            wxerr2[i] = wxmean[i]
                            if self.y is not None:
                                wymean[i] = ymean[i]
                                wystd[i] = 0
                                wyerr[i] = wymean[i]
                                wyerr2[i] = wymean[i]


                    else:
                        xmean[i] = self.x[w].mean()
                        xstd[i] = self.x[w].std()
                        xerr[i] = xstd[i]/numpy.sqrt(w.size)
                        xmedian[i] = numpy.median(self.x[w])
                        if self.y is not None:
                            ymean[i] = self.y[w].mean()
                            ystd[i] = self.y[w].std()
                            yerr[i] = ystd[i]/numpy.sqrt(w.size)
                            ymedian[i] = numpy.median(self.y[w])

                        if self.weights is not None:
                            whist[i] = self.weights[w].sum()
                            wm,we,ws=wmom(self.x[w],self.weights[w],sdev=True)
                            j1,we2=wmom(self.x[w],self.weights[w], calcerr=True)
                            wxmean[i] = wm
                            wxstd[i] = ws
                            wxerr[i] = we
                            wxerr2[i] = we2

                            if self.y is not None:
                                wm,we,ws=wmom(self.y[w],self.weights[w],sdev=True)
                                j1,we2=wmom(self.y[w],self.weights[w], calcerr=True)
                                wymean[i] = wm
                                wystd[i] = ws
                                wyerr[i] = we
                                wyerr2[i] = we2




            self[xpref+'mean']   = xmean
            self[xpref+'std']    = xstd
            self[xpref+'err']    = xerr
            self[xpref+'median'] = xmedian
            if self.y is not None:
                self['ymean']   = ymean
                self['ystd']    = ystd
                self['yerr']    = yerr
                self['ymedian'] = ymedian


            if self.weights is not None:
                self['whist']          = whist
                self['w'+xpref+'mean'] = wxmean
                self['w'+xpref+'std']  = wxstd
                self['w'+xpref+'err']  = wxerr
                self['w'+xpref+'err2'] = wxerr2
                if self.y is not None:
                    self['wymean'] = wymean
                    self['wystd']  = wystd
                    self['wyerr']  = wyerr
                    self['wyerr2'] = wyerr2



def histogram(data, weights=None, binsize=1., nbin=None, nperbin=None, 
              min=None, max=None, 
              rev=False, more=False):
    """
    Calculate the histogram of the input data.  
    
    Similar to IDL histogram.  The reverse indices are also optionally
    calculated.  This routine uses the Binner class in this module for the
    computations.  If you need something more flexible, or need to calculate
    statistics for another variable in the bin, see that class.

    Parameters
    ----------
    data: 
        A numpy array or a sequence that can be converted.

    weights: optional
        A set of weights to use for calculating some statistics.  If weights
        are sent, more=True is implied and a dictionary is returned containing
        a wide variety of statistics.  See docs for the more keyword below.

    binsize: number, optional
        Default 1.0.  The bin size for histogramming.
    nbin: integar, optional
        Choose a number of bins instead of a binsize.  Overrides
        the binsize keyword.
    nperbin: integar, optional
        Choose the number of objects that should fall in each bin.
        Overrides the binsize keyword.  Results in uneven binning.

    min, max:  number, optional
        The min and max data to use from the array.  If these are not sent, min
        and max are determined from the input.


    rev: boolean, optional
        If true, return a tuple 
            h, rev
        Where rev is the reverse indices.   Default is false.  Note if weights
        are sent, or more=True, the result is always a dictionary.  See below.

    more:

        If more is True, or weights are sent, then return more statistics, with
        all outputs in a dictionary.

            keys of the returned dictionary:
                'hist': The histogram.
                'rev':  The reverse indices if the keyword rev=True
                'low': The lower edge of the bins
                'high' Upper edge of the bins.
                'center': Center of the bins
                
            if the keyword rev=True or weights are sent:
                'mean': The mean value in the bin. -9999 if there are
                    no data in the bin.
                'std': The standard deviation in the bin.
                'err': The error on the mean.

            if weights are sent:
                'whist': The weighted histogram.
                'wmean': The weighted mean in the bin.
                'wstd': The weighted standard deviation in the bin.
                'werr': The weighted error in the bin, calculated as
                    sqrt(1/sum(weights))
                'werr2': The weighted error calculated from the
                    weighted variance: 
                      sqrt( (w**2 * (arr-mean)**2).sum() )/weights.sum()


    Using Reverse Indices:
        h,rev = histogram(data, binsize=binsize, rev=True)

        for i in range(h.size):
            if rev[i] != rev[i+1]:
                # data points were found in this bin, get their indices
                indices = rev[ rev[i]:rev[i+1] ]

                # do calculations with data[indices] ...
    """

    if nbin is not None:
        binsize=None

    b = Binner(data, weights=weights)
    b.dohist(binsize=binsize, nbin=nbin, nperbin=nperbin, min=min, max=max, rev=rev)

    if more or weights is not None:
        b.calc_stats()

    if weights is not None or more:
        return b

    if rev:
        return b['hist'], b['rev']
    else:
        return b['hist']



def _dohist(data, dmin, s, binsize, hist, revind=None):
    """
    This is the slower python-only implementation
    """
    
    dorev=False
    if revind is not None:
        dorev=True

    nbin=hist.size
    offset = nbin+1
    i=0
    binnum_old = -1

    while i < s.size:
        data_index = s[i]
        if dorev:
            revind[offset] = data_index

        val = data[data_index]

        binnum = numpy.int64( (val-dmin)/binsize )
        if binnum >= 0 and binnum < nbin:
            if binnum > binnum_old:
                tbin = binnum_old + 1
                while tbin <= binnum:
                    if dorev:
                        revind[tbin] = offset
                    tbin += 1

            hist[binnum] += 1
            binnum_old = binnum

        i += 1
        offset += 1

    if dorev:
        # Fill in the last ones
        tbin = binnum_old + 1
        while tbin <= nbin:
            revind[tbin] = revind.size
            tbin += 1






def testhist(doplot=False):
    """
    testhist(doplot=False)

    Run some tests on the histogram function.  If doplot=True is sent, you must
    have the biggles plotting program installed.

    """
    import esutil
    from sys import stdout
    d = numpy.random.random(100000)
    weights = d.copy()
    weights[:] = 0.5

    stdout.write("Testing straight hist\n")
    binsize=0.1
    h=histogram(d,binsize=binsize)
    esutil.misc.colprint(h,names='hist',min=0,max=1)
    stdout.write('\n')

    stdout.write("Testing weights=0.5\n")
    res=histogram(d,binsize=0.1,min=0,max=1,weights=weights)

    form='%12g'
    nform='%12s'
    names=['low','center','high',
           'median',
           'mean','std','err',
           'hist',
           'wmean','wstd','werr','werr2',
           'whist']
    esutil.misc.colprint(res['low'],res['center'],res['high'],
                         res['median'],
                         res['mean'],res['std'],res['err'],
                         res['hist'],
                         res['wmean'],res['wstd'],res['werr'],res['werr2'],
                         res['whist'],
                         names=names, format=form,nformat=nform)

    if doplot:
        import biggles
        plt=biggles.FramedPlot()
        d=biggles.Histogram(res['hist'], x0=min(res['low']), binsize=binsize)
        plt.add(d)
        plt.xlabel = 'data'
        plt.ylabel = 'counts'
        plt.show()



def histogram2d(x, y, 
                nx=None, 
                ny=None, 
                xbin=None, 
                ybin=None, 
                xmin=None, 
                xmax=None, 
                ymin=None, 
                ymax=None, 
                rev=False,
                more=False):
    """
    Name:
        histogram2d
    Purpose:
        Histogram two-dimensional data.

    Calling Sequence:
        histogram2d(x, y, 
                    nx=None, 
                    ny=None, 
                    xbin=None, 
                    ybin=None, 
                    xmin=None, 
                    xmax=None, 
                    ymin=None, 
                    ymax=None, 
                    rev=False,
                    more=False)

    Inputs:
        x,y:  The x and y values for the data.  Must be same length.

    Keywords:
        nx: Number of bins in the x direction.
        ny: Number of bins in the y direction.
        xbin: binsize in the x direction.
        ybin: binsize in the y direction.
        xmin: min range to use in the x direction.
        xmax: max range to use in the x direction.
        ymin: min range to use in the y direction.
        ymax: max range to use in the y direction.
        rev: if True, return a tuple hist,rev

        more: If True, return a dictionary with the histogram in the
            'hist' key, as well as xlow,xhigh,xcenter and other bin
            information.

    """

    x = numpy.array(x, ndmin=1, copy=False)
    y = numpy.array(y, ndmin=1, copy=False)

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
    else:
        xbin = (xmax-xmin)/nx
        ybin = (ymax-ymin)/ny

    xind=numpy.floor((x[w]-xmin)*(nx/(xmax-xmin)))
    yind=numpy.floor((y[w]-ymin)*(ny/(ymax-ymin)))

    ind=xind+nx*yind


    result = histogram(ind, min=0, max=nx*ny-1, rev=rev)

    if not more:
        if rev:
            hist, revind=result
            hist = hist.reshape(nx,ny)
            return hist, revind
        else:
            hist = result.reshape(nx,ny)
            return hist
    else:

        output={}
        if rev:
            hist, revind = result
            hist = hist.reshape(nx,ny)
            output['hist'] = hist
            output['rev'] = revind
        else:
            hist = result.reshape(nx,ny)
            output['hist'] = hist


        # create the bin edges and centers
        xlow = numpy.arange(nx, dtype='f8')
        xlow = xmin + xlow*xbin
        xhigh = xlow + xbin
        xcenter = xlow + 0.5*xbin

        ylow = numpy.arange(ny, dtype='f8')
        ylow = ymin + ylow*ybin
        yhigh = ylow + ybin
        ycenter = ylow + 0.5*ybin

        output['xlow'] = xlow
        output['xhigh'] = xhigh
        output['xcenter'] = xcenter

        output['ylow'] = ylow
        output['yhigh'] = yhigh
        output['ycenter'] = ycenter

        output['nx'] = nx
        output['xbin'] = xbin
        output['xmin'] = xmin
        output['xmax'] = xmax

        output['ny'] = ny
        output['ybin'] = ybin
        output['ymin'] = ymin
        output['ymax'] = ymax

        output['ranges'] = ( (xmin,ymin), (xmax,ymax) )
        # useful for biggles
        output['ranges_reverse'] = ( (ymin,xmin), (ymax,xmax) )

        return output




def wmom(arrin, weights_in, inputmean=None, calcerr=False, sdev=False):
    """
    NAME:
      wmom()
      
    PURPOSE:
      Calculate the weighted mean, error, and optionally standard deviation of
      an input array.  By default error is calculated assuming the weights are
      1/err^2, but if you send calcerr=True this assumption is dropped and the
      error is determined from the weighted scatter.

    CALLING SEQUENCE:
     wmean,werr = wmom(arr, weights, inputmean=None, calcerr=False, sdev=False)
    
    INPUTS:
      arr: A numpy array or a sequence that can be converted.
      weights: A set of weights for each elements in array.
    OPTIONAL INPUTS:
      inputmean: 
          An input mean value, around which them mean is calculated.
      calcerr=False: 
          Calculate the weighted error.  By default the error is calculated as
          1/sqrt( weights.sum() ).  If calcerr=True it is calculated as sqrt(
          (w**2 * (arr-mean)**2).sum() )/weights.sum()
      sdev=False: 
          If True, also return the weighted standard deviation as a third
          element in the tuple.

    OUTPUTS:
      wmean, werr: A tuple of the weighted mean and error. If sdev=True the
         tuple will also contain sdev: wmean,werr,wsdev

    REVISION HISTORY:
      Converted from IDL: 2006-10-23. Erin Sheldon, NYU

   """
    
    # no copy made if they are already arrays
    arr = numpy.array(arrin, ndmin=1, copy=False)
    
    # Weights is forced to be type double. All resulting calculations
    # will also be double
    weights = numpy.array(weights_in, ndmin=1, dtype='f8', copy=False)
  
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




def sigma_clip(arrin, niter=4, nsig=4, extra={}, 
               verbose=False, silent=False):
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
      Minor bug fix to error messaging: 2010-05-28. Brian Gerke, SLAC
      Added silent keyword, to shut off error messages.  BFG 2010-09-13

    """
    from sys import stdout, stderr
    arr = numpy.array(arrin, ndmin=1, copy=False)

    index = numpy.arange( arr.size )

    for i in numpy.arange(niter):
        m = arr[index].mean()
        s = arr[index].std()

        if verbose:
            stdout.write('iter %s\tnuse: %s\tmean %s\tstdev %s\n' % \
                             (i+1, index.size,m,s))

        clip = nsig*s

        w, = numpy.where( (numpy.abs(arr[index]) - m) < clip )

        if (w.size == 0):
            if (not silent):
                stderr.write("nsig too small. Everything clipped on "
                             "iteration %d\n" % (i+1))
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
      permitted unlike the scipy.interpolate.interp1d module. They are
      extrapolated from the line between the 0,1 or n-2,n-1 entries.  This
      program is not as powerful as interp1d but it does provide this feature
      which makes it compatible with the IDL interpol() function.

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


