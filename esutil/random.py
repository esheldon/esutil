"""
Module:
    random

functions:
    srandu(num=1)
        Generate random numbers in the symmetric distribution [-1,1]


Classes:

    Class Name:
        Generator

    Purpose:

        A class for creating random samples from an arbitrary input probability
        distribution.  The input distribution can either be an array of points
        along with corresponding x values, or a function.

    ** See docs for the Generator class for more details.  To do so in IPython:
        import esutil
        esutil.random.Generator?

    Examples:

        Generate random points from a "sampled" pofx, measured at values x.  In
        this case pofx is an array of data instead of a function.

            import esutil
            gen = esutil.random.Generator(pofx, x)
            rand = gen.genrand(1000000)


        Generate random points from an arbitrary function. In this case we can
        send a range over which x will be generated and the number of points in
        x to use for the integration.

            def gaussfunc(x):
                return numpy.exp(-0.5*x**2)/numpy.sqrt(2.0*numpy.pi)

            gen = esutil.random.Generator(gaussfunc, 
                                          xrange=[-4.5,4.5], nx=100)
            rand = gen.genrand(1000000)

Modification History:
    Created:  2010-02, Erin Sheldon, BNL.
    2010-03-03: Moved to new module random.py, Erin Sheldon, BNL

"""
try:
    import numpy
    from numpy import log,exp,sqrt,any,pi
    have_numpy=True
except:
    have_numpy=False

try:
    import scipy
    have_scipy=True
except:
    have_scipy=False

# for checking function type, method type
from types import *

import numpy_util
import stat
import esutil as eu

class Generator(object):
    """
    Class Name:
        Generator

    Purpose:
        A class for creating random samples from an arbitrary input probability
        distribution.  The input distribution can either be an array of points
        along with corresponding x values, or a function.

    Calling Sequence:
        import esutil
        gen = esutil.random.Generator(pofx, x=None, xrange=None, nx=None,
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
            function you must enter xrange and nx and a set of x values will
            be generated to find max(p(x)).  This could also be done with
            a maximizer from scipy

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


        Generate random points from a "sampled" pofx, measured at values x.  In
        this case pofx is an array of data instead of a function.

            import esutil
            gen = esutil.random.Generator(pofx, x)
            rand = gen.genrand(1000000)


        Generate random points from an arbitrary function. In this case we can
        send a range over which x will be generated and the number of points in
        x to use for the integration.

            def gaussfunc(x):
                return numpy.exp(-0.5*x**2)/numpy.sqrt(2.0*numpy.pi)

            gen = esutil.random.Generator(gaussfunc, 
                                          xrange=[-4.5,4.5], nx=100)
            rand = gen.genrand(1000000)


    Revision History:
        2010-02-18: Created, Erin Sheldon, BNL.
    """


    def __init__(self, 
                 pofx, 
                 x=None, 
                 xrange=None, 
                 nx=None, 
                 method='accum', 
                 cumulative=False, 
                 seed=None):

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
                x = numpy.linspace(xrange[0], xrange[1], nx)

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
            if self.xinput.shape != self.pofx.shape:
                raise ValueError("x and pofx must be same shape")

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
        """
        Class:
            random.Genrand

        Purpose:
            Generate random points from the current probability distribution.

        Calling Sequence:
            import esutil

            # see docs on Genrand for info about constructor.
            generator = esutil.random.Generator(pofx, ...)

            rand = generator.genrand(numrand)

        """
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
        rand = stat.interplin(self.xvals, self.pcum, urand)

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

    def get_yvals(self, x):
        if self.cumulative:
            raise ValueError("dont' support getting y vals for cumulative")
        if self.isfunc:
            return self.pofx(x)
        else:
            rand = stat.interplin(self.xvals, self.pofx, x)

    def test(self, nrand=500000):
        """
        Generate some randoms and compare to input distribution
        """

        import biggles

        x = self.xinput
        if self.isfunc:
            y = self.pofx(x)
        else:
            y = self.pofx

        # generate some randoms
        rand = self.genrand(nrand)

        # make the histogram at the binsize of the
        # xinput
        binsize = x[1]-x[0]
        h = eu.stat.histogram(rand, min=x[0], max=x[-1], binsize=binsize)

        # since on same grid can normalize simply
        h = h/float(h.sum())
        y = y/float(y.sum())

        plt=biggles.FramedPlot()
        py = biggles.Histogram(y, x0=x[0], binsize=binsize)
        ph = biggles.Histogram(h, x0=x[0], binsize=binsize, color='red')
            
        plt.add(py,ph)
        plt.show()


class CutGenerator(object):
    """
    Generate random points from an arbitrary input probability distribution.

    The method is the "cut" method: Points are generated in the plane
    [xmin,xmax] [0,max(pofx)] and points below the curve are kept.

    This class is specialized to the case where you have a function for pofx
    and you know the maximum of the function.

    parameters
    ----------
    pofx: Either an array of points or a function.  If p(x) is an array sample
        from the p(x) you must also enter the corresponding x values.
    xrange: 2-element sequence
        The range over which random points will be generated.
        [xmin,xmax]
    pofx_max:
        The maximum of the function over the input range.
    seed: optional
        the seed for the random number generator

    Examples:
        import esutil
        def gaussfunc(x):
            return numpy.exp(-0.5*x**2)
        pofx_max=1.0
        xrange=[-5,5]
        gen = esutil.random.CutGenerator(gaussfunc, xrange, pofx_max)
        rand = gen.genrand(1000000)

    Revision History:
        2012-11-06: Created, Erin Sheldon, BNL.
    """


    def __init__(self, 
                 pofx, 
                 xrange,
                 pofx_max,
                 seed=None):

        self.pofx=pofx
        self.xmin=xrange[0]
        self.xmax=xrange[1]
        self.xwidth=xrange[1]-xrange[0]
        self.pofx_max=pofx_max
        self.seed=seed
    
        if self.seed is not None:
            numpy.random.seed(seed=seed)

    def genrand(self, numrand, seed=None):
        """
        Generate random points from the input probability distribution.

        parameters
        ----------
        numrand: integer
            The number of randoms to generate
        seed: integer, optional
            A new seed for the random number generator
        """

        if seed is not None:
            numpy.random.seed(seed=seed)

        rand = numpy.zeros(numrand,dtype='f8')

        nleft=numrand
        ngood=0
        nleft=numrand
        while nleft > 0:

            # generate x,y values in a plane covering [xmin,xmax] [0,max(p(x))]
            randx, randy, pvals = self.generate_cut_values(nleft)

            # keep ones where the random y values lie under the interpolated
            # curve
            w,=numpy.where(randy < pvals)
            if w.size > 0:
                rand[ngood:ngood+w.size] = randx[w]
                ngood += w.size
                nleft -= w.size

        return rand

    def generate_cut_values(self, num):
        randx = numpy.random.random(num)
        randy = numpy.random.random(num)

        # get x,y on the right range and evaluate function
        randx *= self.xwidth
        randx += self.xmin
        randy *= self.pofx_max
        pvals = self.pofx(randx)

        return randx, randy, pvals

    def test(self, nrand=500000):
        """
        Generate some randoms and compare to input distribution
        """

        import biggles

        # generate some randoms
        rand = self.genrand(nrand)

        std=rand.std()
        binsize=std*0.05

        xvals = numpy.arange(self.xmin, self.xmax, binsize)
        yvals = self.pofx(xvals)
        h = eu.stat.histogram(rand, 
                              min=xvals[0]-binsize/2, 
                              max=xvals[-1]+binsize/2, 
                              binsize=binsize)

        # since on same grid can normalize simply
        h = h/float(h.sum())
        yvals = yvals/float(yvals.sum())

        plt=biggles.FramedPlot()
        py = biggles.Histogram(yvals, x0=xvals[0], binsize=binsize)
        ph = biggles.Histogram(h, x0=xvals[0], binsize=binsize, color='red')
            
        plt.add(py,ph)
        plt.show()

def get_dist(typ, pars):
    typ=typ.lower()
    if typ=="normal":
        return Normal(pars[0], pars[1])
    elif typ=="lognormal":
        return LogNormal(pars[0], pars[1])
    else:
        raise ValueError("unsupported dist: %s" % typ)

class Normal(object):
    """
    Lognormal distribution

    parameters
    ----------
    mean, sigma

    methods
    -------
    sample(nrand):
        Get nrand random deviates from the distribution
    lnprob(x):
        Get the natural logarithm of the probability of x.  x can
        be an array
    prob(x):
        Get the probability of x.  x can be an array
    """
    def __init__(self, mean, sigma):
        self.mean=float(mean)
        self.sigma=float(sigma)
        self.ivar = 1.0/sigma**2
        self.maxval=1.0
        self.maxval_lnprob=0.0

        self.dist="Normal"

    def __call__(self, x):
        return self.prob(x)

    def get_dist_name(self):
        """
        Get the name of this distribution
        """
        return self.dist

    def get_mean(self):
        """
        Get the mean of the distribution
        """
        return self.mean

    def get_sigma(self):
        """
        Get the width sigma of the distribution
        """
        return self.sigma

    def get_mode(self):
        """
        Get the location of the peak
        """
        return self.mean

    def get_max(self):
        """
        Get maximum value of this distribution
        """
        return self.maxval

    def get_max_lnprob(self):
        """
        Get maximum value ln(prob) of this distribution
        """
        return self.maxval_lnprob


    def lnprob(self, x):
        """
        Get the natural logarithm of the probability of x.  x can
        be an array
        """

        lnp = -0.5*self.ivar*(x-self.mean)**2
        return lnp

    def prob(self, x):
        """
        Get the probability of x.  x can be an array
        """
        
        return exp(self.lnprob(x))

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution

        If z is drawn from a normal random distribution, then exp(logmean+logsigma*z)
        is drawn from lognormal
        """
        if nrand is None:
            z=numpy.random.randn()
        else:
            z=numpy.random.randn(nrand)
        z *= self.sigma
        z += self.mean
        return z


class LogNormal(object):
    """
    Lognormal distribution

    parameters
    ----------
    mean:
        such that <x> in linear space is mean.  This implies the mean in log(x)
        is 
            <log(x)> = log(mean) - 0.5*log( 1 + sigma**2/mean**2 )
    sigma:
        such than the variace in linear space is sigma**2.  This implies
        the variance in log space is
            var(log(x)) = log( 1 + sigma**2/mean**2 )
    norm: optional
        When calling eval() the return value will be norm*prob(x)


    methods
    -------
    sample(nrand):
        Get nrand random deviates from the distribution
    lnprob(x):
        Get the natural logarithm of the probability of x.  x can
        be an array
    prob(x):
        Get the probability of x.  x can be an array
    """
    def __init__(self, mean, sigma):
        from math import log,exp,sqrt
        mean=float(mean)
        sigma=float(sigma)

        self.dist="LogNormal"

        if mean <= 0:
            raise ValueError("mean %s is < 0" % mean)

        self.mean=mean
        self.sigma=sigma

        self.logmean = log(mean) - 0.5*log( 1 + sigma**2/mean**2 )
        self.logvar = log(1 + sigma**2/mean**2 )
        self.logsigma = sqrt(self.logvar)
        self.logivar = 1./self.logvar

        self.nconst = 1/sqrt(2*pi*self.logvar)
        self.logofnconst = log(self.nconst)

        self.mode=exp(self.logmean - self.logvar)
        self.maxval = self.prob(self.mode)
        self.maxval_lnprob=log(self.maxval)

    def __call__(self, x):
        return self.prob(x)

    def get_dist_name(self):
        """
        Get the name of this distribution
        """
        return self.dist
 
    def get_mean(self):
        """
        Get the mean of the distribution
        """
        return self.mean

    def get_sigma(self):
        """
        Get the width sigma of the distribution
        """
        return self.sigma

    def get_mode(self):
        """
        Get the location of the peak
        """
        return self.mode

    def get_max(self):
        """
        Get maximum value of this distribution
        """
        return self.maxval

    def get_max_lnprob(self):
        """
        Get maximum value ln(prob) of this distribution
        """
        return self.maxval_lnprob

    def lnprob(self, x):
        """
        Get the natural logarithm of the probability of x.  x can
        be an array
        """
        if isinstance(x,numpy.ndarray):
            if numpy.any(x <= 0):
                raise ValueError("values of x must be > 0")
        else:
            if x <= 0:
                raise ValueError("values of x must be > 0")
        return self._lnprob(x)

    def _lnprob(self, x):
        """
        This one no error checking
        """
        logx = log(x)

        chi2 = self.logivar*(logx-self.logmean)**2

        lnprob = self.logofnconst - 0.5*chi2 - logx
        return lnprob

    def prob(self, x):
        """
        Get the probability of x.  x can be an array
        and can go < 0 since no logs are taken
        """
        if isinstance(x,numpy.ndarray):
            prob=numpy.zeros(x.size)
            w,=numpy.where(x > 0)
            if w.size > 0:
                lnprob = self._lnprob(x[w])
                prob[w] = exp(lnprob)
        else:
            if x <= 0:
                prob=0.0
            else:
                prob=exp(self._lnprob(x))

        return prob

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution

        If z is drawn from a normal random distribution, then exp(logmean+logsigma*z)
        is drawn from lognormal
        """
        if nrand is None:
            z=numpy.random.randn()
        else:
            z=numpy.random.randn(nrand)
        return exp(self.logmean + self.logsigma*z)

def srandu(num=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]
    """
    return 2*(numpy.random.random(num)-0.5)


def test_generator(doplot=False):
    """
    Do some tests of the random generator
    """

    # first a function
    funcgen = Generator(gaussfunc, xrange=[-4.5,4.5], nx=100)

    nrand = 100000
    func_rand = funcgen.genrand(nrand)
 


    # now points
    x = numpy_util.arrscl( numpy.arange(100,dtype='f4'), -4.5,4.5 )
    pofx = gaussfunc(x)

    pointgen = Generator(pofx, x=x)
    point_rand = pointgen.genrand(nrand)


    if doplot:
        import plotting
        plt=plotting.setuplot()
        plt.subplot(2,1,1)
        plt.hist(func_rand,bins=100,normed=True)
        plt.plot(x,pofx)

        plt.subplot(2,1,2)
        plt.hist(point_rand,bins=100,normed=True,color='red',edgecolor='black')
        plt.plot(x,pofx,color='blue')

        leg = plt.legend()
        leg.draw_frame(False)

        plt.show()


def gaussfunc(x):
    """
    for testing function stuff
    """
    return numpy.exp(-0.5*x**2)/numpy.sqrt(2.0*numpy.pi)


