try:
    import numpy
    from numpy import where
    have_numpy = True
except:
    have_numpy = False

import esutil
import numpy_util

# biggles plotting routines.  mostly convenience functions for interactive work

def bbox(x0, x1, y0, y1, **keys):
    """
    Name:
        bbox
    Purpose:
        Return a biggles Curve representing a box.
    Usage:
        box = bbox(minx,maxx,miny,maxy)
        plt.add( box )
        plt.show()
    """
    import biggles
    bx = numpy.array( [x0,x0,x1,x1,x0], dtype='f8' )
    by = numpy.array( [y0,y1,y1,y0,y0], dtype='f8' )

    return biggles.Curve(bx, by, **keys)

def bscatter(xin, yin, show=True, plt=None, **keywords):
    """
    Name:
        bscatter
    Purpose:
        A wrapper to perform a quick scatter plot with biggles.  For anything
        more complex, it is better to use the object oriented interface.

    Calling Sequence:
        bscatter(x, y, 
                 xerr=None, 
                 yerr=None,
                 xrange=None,
                 yrange=None,
                 type='filled_circle', 
                 color=None,
                 xlabel=None, 
                 ylabel=None, 
                 label=None,
                 title=None,
                 file=None, 
                 xsize=None, 
                 ysize=None,
                 aspect_ratio=None,
                 show=True,
                 plt=None)

    Return value is the used biggles plot object.

    For overplotting, send an existing biggles plot object in the plt= keyword

    """

    import biggles
    if plt is None:
        plt = biggles.FramedPlot()
        xlog=keywords.get('xlog',False)
        ylog=keywords.get('ylog',False)
    else:
        xlog=plt.xlog
        ylog=plt.ylog

    pdict={}

    # plot symbol or line type
    type = keywords.get('type', 'filled circle')

    xerr = keywords.get('xerr',None)
    yerr = keywords.get('yerr',None)
    x = xin
    y = yin

    xrng = keywords.get('xrange',None)
    yrng = keywords.get('yrange',None)

    # For log, Don't plot points less than zero
    w=None
    if xlog and ylog:
        xrng = get_log_plot_range(x, err=xerr, input_range=xrng)
        yrng = get_log_plot_range(y, err=yerr, input_range=yrng)
        w,=numpy.where( (x > xrng[0]) & (y > yrng[0]) )
    elif xlog:
        xrng = get_log_plot_range(x, err=xerr, input_range=xrng)
        w,=numpy.where( x > xrng[0])
    elif ylog:
        yrng = get_log_plot_range(y, err=yerr, input_range=yrng)
        w,=numpy.where( y > yrng[0])

    if w is not None:
        if w.size == 0:
            raise ValueError("no points > 0 for log plot")
        x = x[w]
        y = y[w]

    pkeywords = {}
    if 'color' in keywords:
        pkeywords['color'] = keywords['color']

    if type in ["solid","dotted","dotdashed","shortdashed",
                "longdashed","dotdotdashed","dotdotdotdashed"]:
        if 'width' in keywords:
            pkeywords['width'] = keywords['width']

        p=biggles.Curve(x, y, type=type, **pkeywords)
    else:
        size=keywords.get('size', 1)
        p=biggles.Points(x, y, type=type, size=size, **pkeywords)

    label = keywords.get('label',None)
    if label is not None:
        p.label = label

    plt.add(p)
    pdict['p'] = p

    # note for log error bars, we start with original points since
    # the bars may extend above zero even for negative points
    if yerr is not None:
        if ylog:
            pdict['p_yerr'] = add_log_error_bars(plt, 'y', xin, yin, yerr, yrng, **pkeywords)
            #p_yerr = add_log_error_bars(plt, 'y', xin, yin, yerr, yrng, **pkeywords)
        else:
            p_yerr=biggles.SymmetricErrorBarsY(x, y, yerr, **pkeywords)
            plt.add(p_yerr)
            pdict['p_yerr'] = p_yerr
    if xerr is not None:
        if xlog:
            pdict['p_xerr'] = add_log_error_bars(plt, 'y', xin, yin, xerr, xrng, **pkeywords)
            #p_xerr = add_log_error_bars(plt, 'y', xin, yin, xerr, xrng, **pkeywords)
        else:
            p_xerr=biggles.SymmetricErrorBarsX(x, y, xerr, **pkeywords)
            plt.add(p_xerr)
            pdict['p_xerr'] = p_xerr

    plt.xlog = xlog
    plt.ylog = ylog

    if xrng is not None:
        plt.xrange = xrng
    if yrng is not None:
        plt.yrange = yrng

    if 'xlabel' in keywords:
        plt.xlabel = keywords['xlabel']
    if 'ylabel' in keywords:
        plt.ylabel = keywords['ylabel']

    if 'title' in keywords:
        plt.title=keywords['title']

    if 'aspect_ratio' in keywords:
        plt.aspect_ratio =keywords['aspect_ratio']
            

    if 'file' in keywords:
        fname = keywords['file']
        if fname.find('.eps') != -1 or fname.find('.ps') != -1:
            plt.write_eps(fname)
        else:
            xsize = keywords.get('xsize',512)
            ysize = keywords.get('ysize',512)
            plt.write_image(xsize, ysize, fname)
    else:
        if show:
            plt.show()

    pdict['plt'] = plt
    if 'dict' in keywords:
        if keywords['dict']:
            return pdict
    return plt

def bhist(x, binsize=1.0, nbin=None, min=None,max=None,weights=None,plt=None,**keywords):
    """
    Name:
        bhist
    Purpose:
        A wrapper to perform a quick histogram plot with biggles.  For anything
        more complex, it is better to use the object oriented interface.

    Calling Sequence:
        bhist(x, 
              binsize=1.0,
              nbin=None,
              weights=None,
              gethist=False,
              getphist=False,
              min=None,
              max=None,
              xrange=None,
              yrange=None,
              color='black',
              xlabel=None, 
              ylabel=None, 
              label=None,
              title=None,
              file=None, 
              xsize=None, 
              ysize=None,
              aspect_ratio=None,
              show=True,
              plt=None)

    """

    import esutil
    import biggles
    hout = esutil.stat.histogram(x, 
                                 binsize=binsize, 
                                 nbin=nbin, 
                                 min=min,
                                 max=max,
                                 weights=weights,
                                 more=True)
    
    if nbin is not None:
        binsize = hout['low'][1] - hout['low'][0]

    if plt is None:
        plt = biggles.FramedPlot()

    pkeywords = {}
    if 'color' in keywords:
        color = keywords['color']
        if color is not None:
            pkeywords['color'] = color

    if weights is not None:
        ph=biggles.Histogram(hout['whist'], x0=hout['low'][0], binsize=binsize, 
                             **pkeywords)
    else:
        ph=biggles.Histogram(hout['hist'], x0=hout['low'][0], binsize=binsize, 
                             **pkeywords)

    label = keywords.get('label',None)
    if label is not None:
        ph.label = label
    plt.add(ph)

    if 'xrange' in keywords:
        plt.xrange = keywords['xrange']
    if 'yrange' in keywords:
        plt.yrange = keywords['yrange']

    if 'xlabel' in keywords:
        plt.xlabel = keywords['xlabel']
    if 'ylabel' in keywords:
        plt.ylabel = keywords['ylabel']

    if 'title' in keywords:
        plt.title=keywords['title']

    if 'aspect_ratio' in keywords:
        plt.aspect_ratio =keywords[' aspect_ratio']
 
    if 'file' in keywords:
        fname = keywords['file']
        if fname.find('.eps') != -1 or fname.find('.ps'):
            plt.write_eps(fname)
        else:
            xsize = keywords.get('xsize',512)
            ysize = keywords.get('ysize',512)
            plt.write_image(xsize, ysize, fname)
    else:
        show = keywords.get('show',True)
        if show:
            plt.show()

    gethist = keywords.get('gethist',False)
    getphist = keywords.get('getphist',False)
    if gethist:
        return plt,hout
    elif getphist:
        return plt,ph
    else:
        return plt

def bwhiskers(xin, yin, uin, vin, 
              scale=1.0, 
              file=None,
              xsize=512,
              ysize=512,
              show=None, 
              plt=None, 
              **plotting_keywords):
    """
    Name:
        bwhiskers
    Calling Sequence:
        whiskers(plt, x, y, u, v, scale=1, plt=None, **plotting_keywords)
    Plotting Context:
        biggles.  Make whiskers using matplotlib use the mwhiskers function
            Add them to the entered plot object, otherwise create a
            new plot object and display the plot (or save to a file).

    Purpose:
        Using biggles, create lines centered a the input x,y positions, with
        length 
            sqrt(u**2 + v**2) 
        and angle 
            arctan(v,u)

    Inputs:
        x,y: The x,y positions for the midpoint of each whisker.
        u,v: The vectors to draw.  You can create these vectors from
            shears, or polarizations, using the polar2whisker function.
    Optional Inputs:
        plt: A biggles plot object.  If not sent, a FramedPlot() instance 
            is created.
        show: Show the plot in a window.  
        
            If this keyword is not sent, the plot will only be shown in a
            window if these conditions hold

                1) The file keyword is not sent.
                2) A plt object is not sent.  If a plot object is
                    entered it is assumed you only want to add the
                    whiskers to the existing object but not show it.

        file: A filename to write the image, should be .eps or .png
        xsize, ysize: Keywords indicating the size of a png file in x and y.
            Defaults are each 512.

        scale: A scale to multiply the length of each whisker.  Default 1.
        **plotting keywords:  keywords to be used when creating each
            whisker.  Each whisker is represented by a biggles Curve()
            object.

    Outputs:
        The biggles plot instance.

    """

    if show is None:
        if file is None and plt is None:
            show = True

    import biggles
    if plt is None:
        plt = biggles.FramedPlot()

    if 'xrange' in plotting_keywords:
        plt.xrange = plotting_keywords['xrange']
    if 'yrange' in plotting_keywords:
        plt.yrange = plotting_keywords['yrange']

    if 'xlabel' in plotting_keywords:
        plt.xlabel = plotting_keywords['xlabel']
    if 'ylabel' in plotting_keywords:
        plt.ylabel = plotting_keywords['ylabel']

    if 'title' in plotting_keywords:
        plt.title=plotting_keywords['title']

    if 'aspect_ratio' in plotting_keywords:
        plt.aspect_ratio =plotting_keywords['aspect_ratio']
 

    x = numpy.array(xin, copy=False, ndmin=1)
    y = numpy.array(yin, copy=False, ndmin=1)
    u = numpy.array(uin, copy=False, ndmin=1)
    v = numpy.array(vin, copy=False, ndmin=1)

    if x.size != y.size or x.size != u.size or x.size != v.size:
        raise ValueError("Sizes don't match: "
                         "%s %s %s %s\n" % (x.size,y.size,u.size,v.size))

    for i in range(x.size):
        # create the line to draw.
        xvals = x[i] + numpy.array([ -u[i]/2.0, u[i]/2.0], dtype='f4')*scale
        yvals = y[i] + numpy.array([ -v[i]/2.0, v[i]/2.0], dtype='f4')*scale

        c = biggles.Curve(xvals, yvals, **plotting_keywords)
        plt.add(c)

    if file is not None:
        if file.find('.eps') != -1 or file.find('.ps') != -1:
            plt.write_eps(file)
        else:
            if xsize is None:
                xsize=512
            if ysize is None:
                ysize=512
            plt.write_image(xsize, ysize, file)
    else:
        if show:
            plt.show()

    return plt



# matplotlib related routines
def setuplot(backend=None, params=None):
    """
    Import pyplot from matplotlib and return it.  Can specify a backend
    and some params.

    Specifying backend will only work if this is the first time importing
    pyplot, which is the primary reason for this convenience function.

    """
    import matplotlib
    if backend is not None:
        try:
            matplotlib.use(backend, warn=False)
        except:
            pass

    from matplotlib import pyplot as plt

    if params is not None:
        plt.rcParams.update(params)

    return plt


def set_minor_ticks(ax, xloc=None, yloc=None):
    """
    By default minor ticks are not drawn in matplotlib.

    This function takes an axes instance (e.g. from axes or add_subplot) and
    adds minor ticks.  By default uses a simple algorithm to figure out where
    they should go based on the limits.  So best to call this program last
    right before saving the figure.

    Requires matplotlib
    """
    from math import log10, floor
    from matplotlib.ticker import MultipleLocator as ml

    ranges = ax.axis()
    if xloc is None:
        r=floor(log10(ranges[1]-ranges[0])-1)
        xloc = 10.0**r
    if yloc is None:
        r=floor(log10(ranges[3]-ranges[2])-1)
        yloc = 10.0**r

    ax.xaxis.set_minor_locator(ml(xloc))
    ax.yaxis.set_minor_locator(ml(yloc))


def mwhiskers(plt, xin, yin, uin, vin, 
              scale=1.0, linewidth=0.5, **plotting_keywords):
    """
    Name:
        mwhiskers
    Calling Sequence:
        whiskers(plt, x, y, u, v, scale=1, **plotting_keywords)
    Plotting Context:
        matplotlib.  Do make whiskers using biggles use the bwhiskers function

    Purpose:

        Using matplotlib, draw lines centered a the input x,y positions, with
        length 
            sqrt(u**2 + v**2) 
        and angle 
            arctan(v,u)

    plt could be an axes instance
        ax = pyplot.subplot(1,2,1)
    or could it self be pyplot or pylab

    """

    if not have_numpy:
        raise ImportError("numpy is not available")
    x = numpy.array(xin, copy=False, ndmin=1)
    y = numpy.array(yin, copy=False, ndmin=1)
    u = numpy.array(uin, copy=False, ndmin=1)
    v = numpy.array(vin, copy=False, ndmin=1)

    if x.size != y.size or x.size != u.size or x.size != v.size:
        raise ValueError("Sizes don't match: %s %s %s %s\n" % (x.size,y.size,u.size,v.size))

    for i in range(x.size):
        # create the line to draw.
        xvals = x[i] + numpy.array([ -u[i]/2.0, u[i]/2.0], dtype='f4')*scale
        yvals = y[i] + numpy.array([ -v[i]/2.0, v[i]/2.0], dtype='f4')*scale

        plt.plot(xvals, yvals, color=color, linewidth=linewidth, **plotting_keywords) 


def polar2whisker(e1, e2, angle=False, degrees=False):
    if not have_numpy:
        raise ImportError("numpy is not available")

    etot = numpy.sqrt( e1**2 + e2**2 )
    posangle = 0.5*numpy.arctan2(e2, e1)

    if angle:
        if degrees:
            posangle *= 180./numpy.pi
        return etot, posangle

    # x component of the "vector" version
    u = etot*numpy.cos(posangle)
    # y component of the "vector" version
    v = etot*numpy.sin(posangle)

    return u, v


def plotrand(x, y, frac=0.1, plt=None, **keys):
    import biggles
    from biggles import FramedPlot, Points
    if plt is None:
        plt = FramedPlot()

    x=numpy.array(x,ndmin=1,copy=False)
    y=numpy.array(y,ndmin=1,copy=False)
    if x.size != y.size:
        raise ValueError("x,y must be same size")
    nrand = int(x.size*frac)

    ind = numpy_util.random_subset(x.size, nrand)

    if 'type' not in keys:
        keys['type'] = 'dot'

    c = Points(x[ind], y[ind], **keys)
    plt.add(c)

    if 'xlabel' in keys:
        plt.xlabel = keys['xlabel']
    if 'ylabel' in keys:
        plt.ylabel = keys['ylabel']


    show = keys.get('show',True)
    if show:
        plt.show()

    return plt

def transform_box(lonmin, lonmax, latmin, latmax, fromsys, tosys, **keys):
    """
    Name:
        transform_box
    Purpose:
        Transform the box specified in system1 to system2.  npts points will be
        used to represent each line segment, and these will be transformed
        to the new system.

    Calling Sequence:
        bx, by = transform_box(lonmin, lonmax, latmin, latmax, fromsys, tosys,
                               npts=40)

        plt = biggles.FramedPlot()
        plt.add( biggles.Curve(bx, by, color='red') )
        plt.show()
    """

    npts = keys.get('npts',40)
    blon = numpy.zeros( 4*npts, dtype='f8' )
    blat = numpy.zeros( 4*npts, dtype='f8' )

    blon[0:npts] = lonmin
    blat[0:npts] = esutil.numpy_util.arrscl(numpy.arange(npts), latmin, latmax)

    blon[npts:2*npts] = esutil.numpy_util.arrscl(numpy.arange(npts), lonmin, lonmax)
    blat[npts:2*npts] = latmax

    blon[2*npts:3*npts] = lonmax
    blat[2*npts:3*npts] = esutil.numpy_util.arrscl(numpy.arange(npts), latmax, latmin)

    blon[3*npts:4*npts] = esutil.numpy_util.arrscl(numpy.arange(npts), lonmax, lonmin)
    blat[3*npts:4*npts] = latmin

    if fromsys == 'eq' and tosys in ['survey','sdss']:
        return esutil.coords.eq2sdss(blon,blat)
    if fromsys in ['survey','sdss'] and tosys == 'eq':
        return esutil.coords.sdss2eq(blon,blat)
    else:
        raise ValueError("dont' yet support '%s' to '%s'" % (fromsys,tosys))


def asinh_scale(image, alpha=0.02, nonlinearity=8.0):
    image_out=numpy.array(image, dtype='f8', copy=True)

    image_out[:] = \
        numpy.arcsinh( alpha*nonlinearity*image )/nonlinearity

    return image_out

def image_norm(image, reverse=False):
    image_out=numpy.array(image, dtype='f8', copy=True)
    image_out /= image_out.max()

    if reverse:
        image_out = 1.0 - image_out

    return image_out

def get_log_plot_range(x, err=None, input_range=None):
    if input_range is not None:
        if input_range[0] <= 0. or input_range[1] <= 0.:
            raise ValueError("cannot use plot range < 0 for log plots, got [%s,%s]" % tuple(input_range))
        return input_range

    w,=where(x > 0.)
    if w.size == 0:
        raise ValueError("No values are greater than zero in log plot")

    minval = min(x[w])
    if err is not None:
        w2, = where( (x[w] - err[w]) > 0 )
        minval2 =  min(x[w[w2]] - err[w[w2]])
        minval = min(minval,minval2)

        maxval = max(x+err)
    else:
        maxval = max(x)

    minval *= 0.5
    maxval *= 2

    return [minval,maxval]

def add_log_error_bars(plt, type, x, y, err, prange, **pkeywords):
    import biggles
    if type == 'x':
        low = x-err
        high = x+err
    else:
        low = y-err
        high = y+err

    w,=where(high > 0)
    if w.size > 0:
        high = high[w]

        # outside range to avoid seeing hat
        low = low[w].clip(0.5*prange[0], 2.0*max(max(high),prange[1]) )

        if type == 'x':
            p=biggles.ErrorBarsX(y[w], low, high, **pkeywords)
        else:
            p=biggles.ErrorBarsY(x[w], low, high, **pkeywords)
        plt.add(p)

        return p


def fake_filled_circles(labels, colors=None, x=-9.99e12, y=-9.99e12):
    """

    When using a dot as plot symbol, the PlotKey is not useful because the dot
    is too small to see.  This creates a filled circle point in specified
    location (should be off the plot region) and returns the Point objects in a
    list with the specified labels and possibly colors.

    Then add these to your PlotKey

    """
    from biggles import Point

    if colors is not None:
        if len(colors) != len(labels):
            raise ValueError("colors must be same len as labels")

    points=[]
    keys={}
    for i in xrange(len(labels)):

        if colors is not None:
            keys['color'] = colors[i]

        p = Point(x,y,type='filled circle', **keys)
        p.label = labels[i]

        points.append(p)
    return points
