try:
    import numpy
    have_numpy = True
except:
    have_numpy = False

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




