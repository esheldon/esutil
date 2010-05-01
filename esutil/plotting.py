try:
    import numpy
    have_numpy = True
except:
    have_numpy = False

import esutil

# biggles plotting routines.  mostly convenience functions for interactive work

def bscatter(x, y, plt=None, **keywords):
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
                 type='diamond', 
                 color='black',
                 xlabel=None, 
                 ylabel=None, 
                 title=None,
                 file=None, 
                 xsize=None, 
                 ysize=None,
                 aspect_ratio=None,
                 plt=None)

    Return value is the used biggles plot object.

    For overplotting, send an existing biggles plot object in the plt= keyword

    """

    import biggles

    if plt is None:
        plt = biggles.FramedPlot()

    type = keywords.get('type', 'diamond')
    color = keywords.get('color', 'black')
    

    p=biggles.Points(x, y, type=type, color=color)
    plt.add(p)

    if 'yerr' in keywords:
        p_yerr = biggles.SymmetricErrorBarsY(x, y, keywords['yerr'], color=color)
        plt.add(p_yerr)
    if 'xerr' in keywords:
        p_xerr = biggles.SymmetricErrorBarsX(x, y, keywords['yerr'], color=color)
        plt.add(p_xerr)

    if 'xrange' in keywords:
        plt.xrange = xrange
    if 'yrange' in keywords:
        plt.yrange = yrange

    if 'xlabel' in keywords:
        plt.xlabel = xlabel
    if 'ylabel' in keywords:
        plt.ylabel = ylabel

    if 'title' in keywords:
        plt.title=title

    if 'aspect_ratio' in keywords:
        plt.aspect_ratio = aspect_ratio
            

    if 'file' in keywords:
        fname = keywords['file']
        if fname.find('.eps') != -1 or fname.find('.ps'):
            plt.write_eps(fname)
        else:
            xsize = keywords.get('xsize',512)
            ysize = keywords.get('ysize',512)
            plt.write_image(xsize, ysize, fname)
    else:
        plt.show()

    return plt

def bhist(x, binsize=1.0, min=None, max=None, weights=None, plt=None, **keywords):

    import esutil
    import biggles
    hout = esutil.stat.histogram(x, 
                                 binsize=binsize, 
                                 min=min,
                                 max=max,
                                 weights=weights,
                                 more=True)
    

    if plt is None:
        plt = biggles.FramedPlot()

    ph=biggles.Histogram(hout['hist'], x0=hout['low'][0], binsize=binsize)

    plt.add(ph)

    plt.show()


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


def whiskers(plt, xin, yin, uin, vin, 
             scale=1.0, color='black', linewidth=0.5, **plotting_keywords):
    """

    Draw lines centered a the input x,y positions, with length 
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


def polar2whisker(e1, e2, angle=False):
    if not have_numpy:
        raise ImportError("numpy is not available")

    etot = numpy.sqrt( e1**2 + e2**2 )
    posangle = 0.5*numpy.arctan2(e2, e1)*180./numpy.pi

    if angle:
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
