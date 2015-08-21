try:
    import numpy
    from numpy import where
    have_numpy = True
except:
    have_numpy = False

import esutil
import numpy_util


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
                 type='filled circle', 
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

    if 'width' in keywords:
        pkeywords['width'] = keywords['width']
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

def compare_hist(data1, data2, names=None, nsig=10.0, **kw):
    """
    Compare the normalized histograms for the two data sets.  Make a grid of
    plots if the data are multi-dimensional

    parameters
    ----------
    data1: array
        a [N] or [N,dim] array
    data2: array
        a [M] or [M,dim] array
    names: list, optional
        Optional list of names for each dimension
    nsig: float, optional
        Optional number of standard deviations to clip histograms,
        default 10.0
    """
    import biggles
    from numpy import newaxis
    from .stat import sigma_clip

    if len(data1.shape)==1:
        data1=data1[:,newaxis]
    if len(data2.shape)==1:
        data1=data2[:,newaxis]

    n1,d1 = data1.shape
    n2,d2 = data2.shape

    if d1 != d2:
        raise ValueError("data must have same number of dims. "
                         "got %d and %d" % (d1,d2))
    
    if names is not None:
        if len(names) != d1:
            raise ValueError("names must have len equal to number of dims. "
                             " in data, got %d and %d" % (d1,len(names)))

    else:
        names = ['par%d' % i for i in xrange(d1)]

    if nsig is None:
        nsig=100.0

    grid = Grid(d1)
    tab = biggles.Table(grid.nrow, grid.ncol)

    pkw = {}
    pkw.update(kw)
    for dkeys in ['width','height']:
        del pkw[dkeys]

    pkw['visible']=False
    for i in xrange(d1):
        mn1,st1,ind1=sigma_clip(data1[:,i], nsig=nsig, get_indices=True)
        mn2,st2,ind2=sigma_clip(data2[:,i], nsig=nsig, get_indices=True)

        min_std = min(st1,st2)
        binsize = 0.2*min_std

        plt=biggles.FramedPlot()
        plt.xlabel=names[i]

        pkw['binsize']=binsize
        pkw['color']='blue'
        h1=biggles.make_histc(data1[ind1,i], **pkw)
        pkw['color']='red'
        h2=biggles.make_histc(data2[ind2,i], **pkw)

        plt.add(h1,h2)

        if i==0:
            h1.label='dataset 1'
            h2.label='dataset 2'
            key=biggles.PlotKey(0.1,0.9,[h1,h2],halign='left')
            plt.add(key)

        row,col=grid(i)
        tab[row,col] = plt

    if 'show' in kw:
        show=kw['show']
    elif 'visible' in kw:
        show=kw['visible']
    else:
        show=True

    if show:
        width=kw.get('width',1000)
        height=kw.get('height',1000)
        tab.show(width=width, height=height)

    return tab

def bhist(x, binsize=1.0, nbin=None, min=None,max=None,weights=None,plt=None,**keywords):
    """
    This is now superceded by biggles.plot_hist

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

    norm=keywords.get('norm',None)

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
        pltsent=False
    else:
        pltsent=True

    pkeywords = {}
    pkeywords.update(**keywords)
    if 'color' in keywords:
        color = keywords['color']
        if color is not None:
            pkeywords['color'] = color

    if weights is not None:
        hist = hout['whist'].copy()
    else:
        hist = hout['hist'].copy()
        
    if norm is not None:
        if norm == True:
            norm=1.0
        hist = norm*hist.astype('f8')*(1./hist.sum())

    xlog = keywords.get('xlog',False)
    ylog = keywords.get('ylog',False)
    xrng = keywords.get('xrange',None)
    yrng = keywords.get('yrange',None)


    if ylog:
        yrng,wy = get_log_plot_range(hist, input_range=yrng,get_good=True)
        plt.ylog=True
        miny = yrng[0]
    else:
        wy=numpy.arange(hist.size)
        miny = 0

        if yrng is None and not pltsent:
            yrng=[0, 1.1*hist.max()]

    if len(wy) != len(x):
        hplot = numpy.zeros(hist.size,dtype='f8') + miny
        hplot[wy] = hist[wy]
        h = hplot

    xvals=numpy.zeros(2*hist.size + 2)
    yvals=numpy.zeros(2*hist.size + 2)
    for i in xrange(xvals.size):
        if i == 0:
            xvals[i]=hout['low'][0]
            yvals[i]=miny
        elif i == (xvals.size-1):
            xvals[i]=hout['high'][-1]
            yvals[i]=miny
        elif i == (xvals.size-2):
            xvals[i] = hout['high'][-1]
            yvals[i] = hist[-1]
        else:
            iix = i/2
            iiy = (i-1)/2
            xvals[i] = hout['low'][iix]
            yvals[i] = hist[iiy]


    if xlog:
        xrng,wx = get_log_plot_range(xvals, input_range=xrng, get_good=True)
        ph = biggles.Curve(xvals[wx], yvals[wx], **pkeywords)
        plt.xlog=True
    else:
        ph = biggles.Curve(xvals, yvals, **pkeywords)

    label = keywords.get('label',None)
    if label is not None:
        ph.label = label
    plt.add(ph)

    if xrng is not None:
        plt.xrange = xrng

    if yrng is not None:
        plt.yrange = yrng
    elif pltsent:
        # if two data sets are present, we should auto-adjust
        plt.yrange=None

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

def bhist_vs(data, *fields, **keys):
    """
    Plot data from an array with fields or dictionary.
    
    If only xfield is sent, a histogram of that field is the only plot.   If
    other arguments are sent, these name other fields in the input data to
    plot vs x in the same bins.

    parameters
    ----------
    data: numpy array with fields or dict
        Must have field names.  This can be a recarray or ordinary
        array with field names, or even a dict as long as the arrays
        all have the same length.
    field1, field2, ...:  string
        A set of fields names to plot.  The first is the "x" variable The data
        are binned according to this variable.  If only a single field is sent,
        a simple histogram is shown.  If multiple are given, the average as a
        function of x is shown in separate plots.  
        
        Note if nperbin= is given, no histogram is shown unless binsize is
        *also* given (to be implemented).  In that case a histogram of "x" is
        also shown in light grey on the background.

    stype: string
        The type of statistic to plot
            if 'mean', plot the mean with errors as a function of the 
                binning field.
            if 'sdev', plot the standard deviation as a function of the 
                binning field.
    names: dict
        Dictionary with names for plotting, e.g. if a field name is 'x'
        this could be {'x':'new name for x'}.
    clip: bool
        If clip=true and weights are not sent for the histogram, the 
        data are sigma clipped at 4 sigma with 4 iterations.
    extra keywords:
        Extra keywords for the histogram program and for plotting.
    """
    import biggles
    from itertools import izip
    import copy
    from esutil.stat import wmom

    if len(fields) == 0:
        raise ValueError("Send at least one field name")

    fields=list(fields)

    stype=keys.get('stype','mean')

    # names for the fields in the plots
    knames=keys.get('names',{})
    plabels = {}
    for k in fields:
        if k in knames:
            plabels[k] = knames[k]
        else:
            plabels[k] = k

    xfield = fields.pop(0)
    x=data[xfield]

    keys['more'] = True
    hout = esutil.stat.histogram(x, **keys)
 
    plots=[]
    if 'nperbin' not in keys:
        if 'weights' in keys:
            hcurve = make_hist_curve(hout['low'],hout['high'],hout['whist'])
        else:
            hcurve = make_hist_curve(hout['low'],hout['high'],hout['hist'])

        hplt = biggles.FramedPlot()
        hplt.add(hcurve)
        hplt.xlabel = plabels[xfield]
        hplt.show()
        plots.append(hplt)

    nfields=len(fields)
    if nfields == 0:
        return
    
    nx=len(x)
    bindata=[]

    nbin = hout['hist'].size
    nonempty,=numpy.where(hout['hist'] > 0)

    # now make a data set for each argument
    for f in fields:
        if len(data[f]) != nx:
            raise ValueError("field %s is not same size as field %s" % (f,xfield))
        d = {'name':f,
             'plabel':plabels[f]}
        if stype == 'mean':
            d['mean'] = numpy.zeros(nbin)
            d['err'] = numpy.zeros(nbin)
        else:
            d['sdev'] = numpy.zeros(nbin)
        bindata.append(d)

    # get averages for each argument in each bin
    rev=hout['rev']
    weights=keys.get('weights',None)
    # this only applies if weights are None
    clip=keys.get('clip',False)
    for i in xrange(nbin):
        if rev[i] != rev[i+1]:
            w=rev[ rev[i]:rev[i+1] ]

            for bd in bindata:
                ydata = data[bd['name']][w]
                if weights is not None:
                    mn,err,sdev = wmom(ydata, weights[w], sdev=True)
                else:
                    if clip:
                        mn,sdev=esutil.stat.sigma_clip(ydata)
                    else:
                        mn = ydata.mean()
                        sdev = ydata.std()
                    err = sdev/numpy.sqrt(w.size)
                if stype == 'mean':
                    bd['mean'][i] = mn
                    bd['err'][i] = err
                else:
                    bd['sdev'][i] = sdev

    # now run through and make all the plots
    keys['xlabel'] = plabels[xfield]
    for bd in bindata:
        keys['ylabel'] = bd['plabel']
        if 'mean' in hout:
            xh = hout['mean'][nonempty]
        else:
            xh = hout['center'][nonempty]

        if stype == 'mean':
            plt=bscatter(xh, bd['mean'][nonempty], yerr=bd['err'][nonempty], **keys)
        else:
            plt=bscatter(xh, bd['sdev'][nonempty], **keys)
        plots.append(plt)

    return plots

def make_hist_curve(xlow, xhigh, y, ymin=None, ymax=None, **keys):
    """
    Make a curve corresponding to the input edge locations and y values, that
    will draw the usual "box-like" histogram shape

    extra plotting keywords can be sent in the keys
    """
    import biggles


    xvals=numpy.zeros(2*y.size + 2)
    yvals=numpy.zeros(2*y.size + 2)
    for i in xrange(xvals.size):
        if i == 0:
            xvals[i]=xlow[0]
            yvals[i]=0
        elif i == (xvals.size-1):
            xvals[i]=xhigh[-1]
            yvals[i]=0
        elif i == (xvals.size-2):
            xvals[i] = xhigh[-1]
            yvals[i] = y[-1]
        else:
            iix = i/2
            iiy = (i-1)/2
            xvals[i] = xlow[iix]
            yvals[i] = y[iiy]

    if ymin is not None or ymax is not None:
        if ymin is None:
            ymin = 0.0
        if ymax is None:
            ymax = yvals.max()
        yvals = esutil.numpy_util.arrscl(yvals, ymin, ymax)


    ph = biggles.Curve(xvals, yvals, **keys)
    return ph

def bwhiskers(xin, yin, uin, vin, 
              scale=1.0, 
              file=None,
              xsize=512,
              ysize=512,
              show=None, 
              plt=None, 
              **keys):
    """
    Create a "whisker" plot from the input polarizations 
    
    Polarizations are headless vectors, rotating as 2*theta, as found in weak
    lensing (e1,e2).

    The plot is made using biggles.

    parameters
    ----------
    x,y: 
        The x,y positions for the midpoint of each whisker.
    u,v: 
        The vectors to draw.  You can create these vectors from shears, or
        polarizations, using the polar2whisker function in this module.

    scale: 
        A scale to multiply the length of each whisker.  Default 1.
    wkeyval:
        Make a key for the plot showing a whisker of this length.
        This value will get multiplied by scale.

    plt: optional 
        A biggles plot object on which to draw.  If not sent, a new
        FramedPlot() instance is created.

    show: bool, optional 
        Show the plot in a window.  
        
        If this keyword is not sent, the plot will only be shown in a
        window if these conditions hold
            1) The file keyword is not sent.
            2) A plt object is not sent.  If a plot object is entered it is
            assumed you only want to add the whiskers to the existing object
            but not show it.

    file: string, optional
        A filename to write the image, should be .eps or .png
    xsize, ysize:
        Keywords indicating the size of a png file in x and y.  Defaults are
        each 512.


    **keys:  
        keywords to be used when creating each whisker.  Each whisker is
        represented by a biggles Curve() object.

    return value
    ------------
    The biggles plot instance.

    """

    if show is None:
        if file is None and plt is None:
            show = True

    import biggles
    if plt is None:
        plt = biggles.FramedPlot()

    if 'xrange' in keys:
        plt.xrange = keys['xrange']
    if 'yrange' in keys:
        plt.yrange = keys['yrange']

    if 'xlabel' in keys:
        plt.xlabel = keys['xlabel']
    if 'ylabel' in keys:
        plt.ylabel = keys['ylabel']

    if 'title' in keys:
        plt.title=keys['title']

    if 'aspect_ratio' in keys:
        plt.aspect_ratio =keys['aspect_ratio']
 

    x = numpy.array(xin, copy=False, ndmin=1)
    y = numpy.array(yin, copy=False, ndmin=1)
    u = numpy.array(uin, copy=False, ndmin=1)
    v = numpy.array(vin, copy=False, ndmin=1)

    if x.size != y.size or x.size != u.size or x.size != v.size:
        raise ValueError("Sizes don't match: "
                         "%s %s %s %s\n" % (x.size,y.size,u.size,v.size))

    if 'wkeyval' in keys:
        minx=x.min()
        maxx=x.max()
        miny=y.min()
        maxy=y.max()

        px=minx + 0.05*(maxx-minx)
        py=miny + 0.95*(maxy-miny)
        laby=miny + 0.925*(maxy-miny)

        kc=biggles.Curve([px,px+keys['wkeyval']*scale],
                         [py,py],
                         color='red')
        kclab=biggles.PlotLabel(0.05,0.925,'%.2g' % keys['wkeyval'],
                                halign='left')
        plt.add(kc,kclab)

    for i in range(x.size):
        # create the line to draw.
        xvals = x[i] + numpy.array([ -u[i]/2.0, u[i]/2.0], dtype='f4')*scale
        yvals = y[i] + numpy.array([ -v[i]/2.0, v[i]/2.0], dtype='f4')*scale

        c = biggles.Curve(xvals, yvals, **keys)
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

def get_binned_whiskers(x, y, u, v, **keys):
    import esutil as eu

    keys['more']=True
    keys['rev']=True
    hdict=eu.stat.histogram2d(x, y, **keys)

    nbin=hdict['hist'].size
    rev=hdict['rev']


    xcen=hdict['xcenter']
    ycen=hdict['ycenter']

    xmeans=numpy.zeros(nbin)
    ymeans=numpy.zeros(nbin)
    umeans=numpy.zeros(nbin)
    vmeans=numpy.zeros(nbin)

    i=0
    for ix in xrange(len(xcen)):
        for iy in xrange(len(ycen)):

            xmeans[i] = xcen[ix]
            ymeans[i] = ycen[iy]

            if rev[i] != rev[i+1]:
                w=rev[ rev[i]:rev[i+1] ]

                umeans[i] = u[w].mean()
                vmeans[i] = v[w].mean()

            i += 1

    return xmeans, ymeans, umeans, vmeans
            
def get_grid(ntot):
    """
    Get a 2-d grid layout given the total number of plots

    returns nrow,ncol

    e.g.
       p1 p2
       
       p1 p2
       p3

       p1 p2
       p3 p4

       p1 p2 p3
       p4 p5

       etc.
    """
    from math import sqrt
    sq=int(sqrt(ntot))
    if ntot==sq*sq:
        return (sq,sq)
    elif ntot <= sq*(sq+1):
        return (sq,sq+1)
    else:
        return (sq+1,sq+1)




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


def plotrand(x, y, frac=0.1, get_indices=False, **keys):
    """
    plot a random subset of the points
    """
    import biggles
    from biggles import FramedPlot, Points

    x=numpy.array(x,ndmin=1,copy=False)
    y=numpy.array(y,ndmin=1,copy=False)
    if x.size != y.size:
        raise ValueError("x,y must be same size")
    nrand = int(x.size*frac)
    if nrand < 1:
        nrand=1
    elif nrand > x.size:
        nrand=x.size

    #ind = esutil.stat.util.random_indices(x.size, nrand, **keys)
    ind = esutil.random.random_indices(x.size, nrand, **keys)

    plt=bscatter(x[ind], y[ind], **keys)

    if get_indices:
        return plt, ind
    else:
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

def get_log_plot_range_xy(x, y, xerr=None, yerr=None, 
                          xlog=False, ylog=False, xrng=None, yrng=None,
                          get_good=False):
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
    else:
        w=numpy.arange(x.size)

    if get_good:
        return xrng, yrng, w
    else:
        return xrng, yrng


def get_log_plot_range(x, err=None, input_range=None, get_good=False):
    if input_range is not None:
        if len(input_range) < 2:
            raise ValueError("expected [xmin,xmax] for input range")
        if input_range[0] <= 0. or input_range[1] <= 0.:
            raise ValueError("cannot use plot range < 0 for log plots, got [%s,%s]" % tuple(input_range))
        if get_good:
            w,=where((x >= input_range[0]) & (x <= input_range[1]))
            return input_range, w
        else:
            return input_range

    w,=where(x > 0.)
    if w.size == 0:
        raise ValueError("No values are greater than zero in log plot")

    minval = min(x[w])
    if err is not None:
        w2, = where( (x[w] - err[w]) > 0 )
        if w2.size > 0:
            minval2 =  min(x[w[w2]] - err[w[w2]])
            minval = min(minval,minval2)

        maxval = max(x+err)
    else:
        maxval = max(x)

    minval *= 0.5
    maxval *= 2

    if get_good:
        return [minval,maxval], w
    else:
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


def fake_points(symbols, labels, colors=None, sizes=None, x=-9.99e12, y=-9.99e12):
    """
    fake points for use with plot legends when the points object is not
    available

    Add these to a biggles.PlotKey object
    """
    from biggles import Point

    if len(symbols) != len(labels):
        raise ValueError("symbols must be same len as labels")
    if colors is not None:
        if len(colors) != len(labels):
            raise ValueError("colors must be same len as labels")
    if sizes is not None:
        if len(sizes) != len(labels):
            raise ValueError("sizes must be same len as labels")

    points=[]
    for i in xrange(len(labels)):

        keys = {'type':symbols[i]}

        if colors is not None:
            keys['color'] = colors[i]

        if sizes is not None:
            keys['size'] = sizes[i]

        p = Point(x,y, **keys)
        p.label = labels[i]

        points.append(p)
    return points

def fake_filled_circles(labels, **keys):
    """

    When using a dot as plot symbol, the PlotKey is not useful because the dot
    is too small to see.  This creates a filled circle point in specified
    location (should be off the plot region) and returns the Point objects in a
    list with the specified labels and possibly colors.

    Then add these to your PlotKey

    """

    return fake_points(['filled circle']*len(labels), labels,
                       **keys)

class Grid(object):
    """
    represent plots in a grid.  The grid is chosen
    based on the number of plots
    
    example
    -------
    grid=Grid(n)

    for i in xrange(n):
        row,col = grid(i)

        # equivalently grid.get_rowcol(i)

        plot_table[row,col] = plot(...)
    """
    def __init__(self, nplot):
        self.set_grid(nplot)
    
    def set_grid(self, nplot):
        """
        set the grid given the number of plots
        """
        from math import sqrt

        self.nplot=nplot

        # first check some special cases
        if nplot==8:
            self.nrow, self.ncol = 2,4
        else:

            sq=int(sqrt(nplot))
            if nplot==sq*sq:
                self.nrow, self.ncol = sq,sq
            elif nplot <= sq*(sq+1):
                self.nrow, self.ncol = sq,sq+1
            else:
                self.nrow, self.ncol = sq+1,sq+1

        self.nplot_tot=self.nrow*self.ncol

    def get_rowcol(self, index):
        """
        get the grid position given the number of plots

        move along columns first

        parameters
        ----------
        index: int
            Index in the grid

        example
        -------
        nplot=7
        grid=Grid(nplot)
        arr=biggles.FramedArray(grid.nrow, grid.ncol)

        for i in xrange(nplot):
            row,col=grid.get_rowcol(nplot, i)
            arr[row,col].add( ... )
        """

        imax=self.nplot_tot-1
        if index > imax:
            raise ValueError("index too large %d > %d" % (index,imax))

        row = index/self.ncol
        col = index % self.ncol

        return row,col

    def __call__(self, index):
        return self.get_rowcol(index)
