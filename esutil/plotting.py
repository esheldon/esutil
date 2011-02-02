try:
    import numpy
    have_numpy = True
except:
    have_numpy = False

import esutil

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

def bscatter(x, y, show=True, plt=None, **keywords):
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

    type = keywords.get('type', 'filled circle')

    pkeywords = {}
    if 'color' in keywords:
        pkeywords['color'] = keywords['color']

    if type in ["solid","dotted","dotdashed","shortdashed",
                "longdashed","dotdotdashed","dotdotdotdashed"]:
        p=biggles.Curve(x, y, type=type, **pkeywords)
    else:
        p=biggles.Points(x, y, type=type, **pkeywords)

    label = keywords.get('label',None)
    if label is not None:
        p.label = label

    plt.add(p)

    if 'yerr' in keywords:
        p_yerr=biggles.SymmetricErrorBarsY(x, y, keywords['yerr'], **pkeywords)
        plt.add(p_yerr)
    if 'xerr' in keywords:
        p_xerr=biggles.SymmetricErrorBarsX(x, y, keywords['yerr'], **pkeywords)
        plt.add(p_xerr)

    if 'xlog' in keywords:
        plt.xlog = keywords['xlog']
    if 'ylog' in keywords:
        plt.ylog = keywords['ylog']
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


def plotrand(x, y, marker, frac=0.1, plt=None, backend=None, params=None, **keys):
    if plt is None:
        plt=setuplot(backend=backend,params=params)

    x=numpy.array(x,ndmin=1,copy=False)
    y=numpy.array(y,ndmin=1,copy=False)
    if x.size != y.size:
        raise ValueError("x,y must be same size")

    nuse = long( x.size*frac)
    ind=numpy.zeros(nuse,dtype='i4')

    rnd=numpy.random.random(nuse)
    ind[:]=esutil.numpy_util.arrscl( rnd, 0, x.size-1, arrmin=0.0, arrmax=1.0)
    
    plt.plot(x[ind],y[ind],marker,**keys)

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
