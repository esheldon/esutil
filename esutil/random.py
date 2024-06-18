"""
Module:
    random

functions:
    srandu(num=1)
        Generate random numbers in the symmetric distribution [-1,1]
    cholesky_sample
        sample a multivariate covariant distribution using cholesky
        decomposition.  Uses the CholeskySampler
    random_indices:
        Get a unique random selection of indices in [0,imax)

Classes:

    Generator
        A class for creating random samples from an arbitrary input probability
        distribution.  The input distribution can either be an array of points
        along with corresponding x values, or a function.

    CutGenerator
        Generate random points from an arbitrary input probability
        distribution.

    CholeskySampler
        sample a multivariate covariant distribution using cholesky
        decomposition
"""
from __future__ import print_function

import numpy
from numpy import log, exp, pi


# for checking function type, method type
from types import FunctionType, MethodType

from . import stat
from .numpy_util import arrscl

LOWVAL = -9999.0e47


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
        gen = esutil.random.Generator(
            pofx, x=None, xrange=None, nx=None,
            method='accum', cumulative=False, rng=None,
            seed=None
        )

        r = gen.sample() # single scalar value
        r = gen.sample(num)

    Inputs:
        pofx:
            Either an array of points or a function.  If p(x) is an array
            sample from the p(x) you must also enter the corresponding x
            values.

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
                [min(x),max(x)] [0,max(prob)] and only those that lie
                underneath the pofx curve are kept.  This can be used to
                generate randoms from complex distributions that do not
                integrate easily.

        cumulative:
            The input distrubtion pofx is actually the accumulated distro.
            Note for method='cut' this is ignored: you must enter the
            differential distribution.

        rng: random number generator
            e.g. numpy.random.RandomState

        seed: integer
            Used to start a new random number generator if rng is not sent

    Examples:


        Generate random points from a "sampled" pofx, measured at values x.  In
        this case pofx is an array of data instead of a function.

            import esutil
            gen = esutil.random.Generator(pofx, x)
            rand = gen.sample(1000000)


        Generate random points from an arbitrary function. In this case we can
        send a range over which x will be generated and the number of points in
        x to use for the integration.

            def gaussfunc(x):
                return numpy.exp(-0.5*x**2)/numpy.sqrt(2.0*numpy.pi)

            gen = esutil.random.Generator(gaussfunc,
                                          xrange=[-4.5,4.5], nx=100)
            rand = gen.sample(1000000)


    Revision History:
        2010-02-18: Created, Erin Sheldon, BNL.
    """

    def __init__(
        self,
        pofx,
        x=None,
        xrange=None,
        nx=None,
        method="accum",
        cumulative=False,
        seed=None,
        rng=None,
    ):

        # make sure the method is valid
        self._check_method(method)
        self.method = method

        self.cumulative = cumulative

        if rng is None:
            self.rng = numpy.random.RandomState(seed=seed)
        else:
            self.rng = rng

        # different initializations depending if p(x) array was sent or a
        # function

        if isinstance(pofx, (FunctionType, MethodType)):
            if x is None:
                # we'll generate the points from xrange and nx
                if xrange is None or nx is None:
                    raise ValueError(
                        "Enter the points x or both xrange and nx"
                    )
                x = numpy.linspace(xrange[0], xrange[1], nx)

            self.xinput = numpy.atleast_1d(x)

            # input is some type of function
            self.isfunc = True
            self.pofx = pofx

            self.initialize_func()
        else:
            if x is None:
                raise ValueError(
                    "For p(x) an array, you must also enter the "
                    "corresponding x values"
                )

            self.xinput = numpy.atleast_1d(x)

            # points were entered
            self.isfunc = False
            self.pofx = numpy.atleast_1d(pofx)
            if self.xinput.shape != self.pofx.shape:
                raise ValueError("x and pofx must be same shape")

            self.initialize_points()

        if self.method == "cut":
            self.xmin = self.xinput.min()
            self.xmax = self.xinput.max()
            self.xwidth = self.xmax - self.xmin

            # maked rescaled with ranges to [0,1]
            self.xinput_scale = (self.xinput - self.xmin) / self.xwidth
            if not self.isfunc:
                self.pofx_max = pofx.max()
                self.pofx_scale = self.pofx / self.pofx_max
            else:
                # get max from evaluating at input x values
                p = self.pofx(self.xinput)
                self.pofx_max = p.max()

    def _check_method(self, method):
        if method not in ["accum", "cut"]:
            raise ValueError("method must be 'accum' or 'cut'")

    def sample(self, numrand=None, **kw):
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

        if numrand is None:
            numrand = 1
            is_scalar = True
        else:
            is_scalar = False
        if self.method == "accum":
            vals = self._genrand_accum(numrand)
        elif self.method == "cut":
            vals = self._genrand_cut(numrand)

        if is_scalar:
            vals = vals[0]

        return vals

    genrand = sample

    def _genrand_accum(self, numrand):

        # this returns f8
        urand = self.rng.uniform(size=numrand)

        # to get randoms from the distribution, we interpolate the x(pcum) at
        # the test rand values.  Clever!
        rand = stat.interplin(self.xvals, self.pcum, urand)

        return rand

    def _genrand_cut(self, numrand):

        rand = numpy.zeros(numrand, dtype="f8")

        nleft = numrand
        ngood = 0
        nleft = numrand
        while nleft > 0:

            # generate x,y values in a plane covering [xmin,xmax] [0,max(p(x))]
            randx, randy, pinterp = self.generate_cut_values(nleft)

            # keep ones where the random y values lie under the interpolated
            # curve
            (w,) = numpy.where(randy < pinterp)
            if w.size > 0:
                rand[ngood: ngood + w.size] = randx[w]
                ngood += w.size
                nleft -= w.size

        if not self.isfunc:
            # If we were not working with a function, we had used scaled
            # versions of x and p(x) for speed
            rand *= self.xwidth
            rand += self.xmin

        return rand

    def generate_cut_values(self, num):
        randx = self.rng.uniform(size=num)
        randy = self.rng.uniform(size=num)

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
            pinterp = stat.interplin(self.pofx_scale, self.xinput_scale, randx)
        return randx, randy, pinterp

    def initialize_points(self):
        """
        Set up the case where the user sent x and p(x) points instead
        of a function for p(x)
        """

        if self.method == "accum":
            # we need the cumulative probability distribution

            if self.cumulative:
                # we are done
                self.xvals = self.xinput
                self.norm = self.pofx[-1]
                self.pcum = self.pofx / self.norm

            else:
                # we must integrate and take a subset of x, since the
                # first value is not defined.

                import scipy.integrate

                pcum = scipy.integrate.cumulative_trapezoid(self.pofx, self.xinput)
                self.norm = pcum[-1]
                self.pcum = pcum / self.norm

                # interval is smaller, no integral in first point
                self.xvals = self.xinput[1:]

    def initialize_func(self):
        """
        Set up the case where the user sent x and p(x) as a function
        """

        if self.method == "accum":
            # we need the cumulative probability distribution

            if self.cumulative:
                self.xvals = self.xinput
                self.norm = self.pofx(self.xinput[-1])
                self.pcum = self.pofx(self.xinput) / self.norm
            else:
                # we must integrate and take a subset of x, since the
                # first value is not defined.

                import scipy.integrate

                pofxvals = self.pofx(self.xinput)

                pcum = scipy.integrate.cumulative_trapezoid(pofxvals, self.xinput)
                self.norm = pcum[-1]
                self.pcum = pcum / self.norm

                # interval is smaller, no integral in first point
                self.xvals = self.xinput[1:]

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
        binsize = x[1] - x[0]
        h = stat.histogram(rand, min=x[0], max=x[-1], binsize=binsize)

        # since on same grid can normalize simply
        h = h / float(h.sum())
        y = y / float(y.sum())

        plt = biggles.FramedPlot()
        py = biggles.Histogram(y, x0=x[0], binsize=binsize)
        ph = biggles.Histogram(h, x0=x[0], binsize=binsize, color="red")

        plt.add(py, ph)
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

    def __init__(self, pofx, xrange, pofx_max, seed=None):

        self.pofx = pofx
        self.xmin = xrange[0]
        self.xmax = xrange[1]
        self.xwidth = xrange[1] - xrange[0]
        self.pofx_max = pofx_max
        self.seed = seed

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

        rand = numpy.zeros(numrand, dtype="f8")

        nleft = numrand
        ngood = 0
        nleft = numrand
        while nleft > 0:

            # generate x,y values in a plane covering [xmin,xmax] [0,max(p(x))]
            randx, randy, pvals = self.generate_cut_values(nleft)

            # keep ones where the random y values lie under the interpolated
            # curve
            (w,) = numpy.where(randy < pvals)
            if w.size > 0:
                rand[ngood: ngood + w.size] = randx[w]
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

        std = rand.std()
        binsize = std * 0.05

        xvals = numpy.arange(self.xmin, self.xmax, binsize)
        yvals = self.pofx(xvals)
        h = stat.histogram(
            rand,
            min=xvals[0] - binsize / 2.0,
            max=xvals[-1] + binsize / 2.0,
            binsize=binsize,
        )

        # since on same grid can normalize simply
        h = h / float(h.sum())
        yvals = yvals / float(yvals.sum())

        plt = biggles.FramedPlot()
        py = biggles.Histogram(yvals, x0=xvals[0], binsize=binsize)
        ph = biggles.Histogram(h, x0=xvals[0], binsize=binsize, color="red")

        plt.add(py, ph)
        plt.show()


def get_dist(typ, pars):
    typ = typ.lower()
    if typ == "normal":
        return Normal(pars[0], pars[1])
    elif typ == "lognormal":
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
        self.mean = float(mean)
        self.sigma = float(sigma)
        self.ivar = 1.0 / sigma ** 2
        self.maxval = 1.0
        self.maxval_lnprob = 0.0

        self.dist = "Normal"

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

        lnp = -0.5 * self.ivar * (x - self.mean) ** 2
        return lnp

    def prob(self, x):
        """
        Get the probability of x.  x can be an array
        """

        return exp(self.lnprob(x))

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution

        If z is drawn from a normal random distribution,
        then exp(logmean+logsigma*z) is drawn from lognormal
        """
        if nrand is None:
            z = numpy.random.randn()
        else:
            z = numpy.random.randn(nrand)
        z *= self.sigma
        z += self.mean
        return z


class NormalND:
    """
    Currently no covariance
    """

    def __init__(self, mean, sigma):
        self.mean = numpy.array(mean)
        self.sigma = numpy.array(sigma)
        self.sigma2 = numpy.array([s ** 2 for s in sigma])
        self.ivar = 1.0 / self.sigma2

        self.ndim = self.mean.size

    def get_max(self):
        return 1.0

    def lnprob(self, pos):
        if len(pos.shape) > 1:
            lnprob = numpy.zeros(pos.shape[0])
            for i in range(self.ndim):
                lnprob += -0.5 * (self.mean[i] - pos[:, i]) ** 2 * self.ivar[i]

        else:
            lnprob = 0.0
            for i in range(self.ndim):
                lnprob += -0.5 * (self.mean[i] - pos[i]) ** 2 * self.ivar[i]

        return lnprob

    def sample(self, n=None):
        """
        Get a single sample
        """
        if n is None:
            rand = self.mean + self.sigma * numpy.random.randn(self.ndim)
        else:
            rand = numpy.random.randn(n, self.ndim).reshape(n, self.ndim)
            for i in range(self.ndim):
                rand[:, i] *= self.sigma[i]
                rand[:, i] += self.mean[i]

        return rand


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
        from math import log, exp, sqrt

        mean = float(mean)
        sigma = float(sigma)

        self.dist = "LogNormal"

        if mean <= 1.0e-10:
            raise ValueError("mean %s is < 0" % mean)

        self.mean = mean
        self.sigma = sigma

        self.logmean = log(mean) - 0.5 * log(1 + sigma ** 2 / mean ** 2)
        self.logvar = log(1 + sigma ** 2 / mean ** 2)
        self.logsigma = sqrt(self.logvar)
        self.logivar = 1.0 / self.logvar

        self.nconst = 1.0 / sqrt(2 * pi * self.logvar)
        if self.nconst <= 1.0e-10:
            raise ValueError("logvar %s is too large" % self.logvar)

        self.logofnconst = log(self.nconst)

        self.mode = exp(self.logmean - self.logvar)
        self.maxval = self.prob(self.mode)
        self.maxval_lnprob = log(self.maxval)

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
        if isinstance(x, numpy.ndarray):
            if numpy.any(x <= 1.0e-10):
                raise ValueError("values of x must be > 0")
            return self._lnprob_array(x)
        else:
            if x <= 1.0e-10:
                raise ValueError("values of x must be > 0")
            return self._lnprob_scalar(x)

    def _lnprob_array(self, x):
        """
        This one no error checking
        """
        logx = log(x)

        chi2 = self.logivar * (logx - self.logmean) ** 2

        lnprob = self.logofnconst - 0.5 * chi2 - logx
        return lnprob

    def _lnprob_scalar(self, x):
        """
        This one no error checking
        """
        from math import log

        logx = log(x)

        chi2 = self.logivar * (logx - self.logmean) ** 2

        lnprob = self.logofnconst - 0.5 * chi2 - logx
        return lnprob

    def prob(self, x):
        """
        Get the probability of x.  x can be an array
        and can go < 0 since no logs are taken
        """
        if isinstance(x, numpy.ndarray):
            prob = numpy.zeros(x.size)
            (w,) = numpy.where(x > 0)
            if w.size > 0:
                lnprob = self._lnprob_array(x[w])
                prob[w] = exp(lnprob)
        else:
            if x <= 0:
                prob = 0.0
            else:
                prob = exp(self._lnprob_scalar(x))

        return prob

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution

        If z is drawn from a normal random distribution,
        then exp(logmean+logsigma*z)
        is drawn from lognormal
        """
        if nrand is None:
            z = numpy.random.randn()
        else:
            z = numpy.random.randn(nrand)
        return exp(self.logmean + self.logsigma * z)


def srandu(num=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]
    """
    return 2 * (numpy.random.random(num) - 0.5)


class CholeskySampler(object):
    """
    sample a multivariate covariant distribution using cholesky decomposition

    example
    -------
    means=[20.0, 40.0]
    cov=[[1.0,0.5],[0.5,2.0]]

    cs=CholeskySampler(means,cov)
    n=100000
    rand=cs.sample(n)

    s.mean(axis=0)
    array([ 20.00139558,  50.00419912])

    s.var(axis=0)
    array([ 1.00076388,  2.00251013])

    mm=s.mean(axis=0)
    ( (s[:,0]-mm[0])*(s[:,1]-mm[1]) ).sum()/(n-1)
    0.50052647916418957
    """

    def __init__(self, mean, cov, dist=None):
        self.mean = numpy.array(mean, ndmin=1)
        self.cov = numpy.array(cov, ndmin=2)

        if dist is None:
            dist = numpy.random.randn
        self.dist = dist

        npar = mean.size
        n1, n2 = cov.shape[0: 0 + 2]
        if npar != cov.shape[0] or npar != cov.shape[1]:
            raise ValueError(
                "mean shape [%d] inconsistent "
                "with cov shape [%d,%d]" % (npar, n1, n2)
            )

        self.M = numpy.linalg.cholesky(self.cov)
        self.npar = npar

    def sample(self, n=None):
        """
        sample the distribution

        parameters
        ----------
        n: integer, optional
            the number of samples.  If not sent, a single
            sample is returned, otherwise an array [n,npars]
            is returned.
        """

        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        npar = self.npar
        r = self.dist(npar * n).reshape(npar, n)

        V = numpy.dot(self.M, r)

        mean = self.mean
        for i in range(npar):
            V[i, :] += mean[i]

        samples = V.T
        if is_scalar:
            return samples[0, :]
        else:
            return samples


def cholesky_sample(cov, n, means=None, dist=None):
    """
    Sample the input covariance using a cholesky decomposition.  The idea is
    that in each dimension we draw from the standard distribution, and then
    transform them to have the specified covariance matrix.

    This can be used to produce the mean and errors on combined parameters,
    taking into account the covariance.

    parameters
    ----------
    cov: array
        A 2-d array representing the covariance of the parameters
    n: integer
        The number of random points to generate
    means: array, optional
        The mean values to add to the random points; by default
        the randoms are centered on 0
    dist: function, optional
        The distribution function.  Default is numpy.random.randn.

    example:
        cov = array([[1.5,0.3],
                     [0.3,2.7]])
        means=array([5.6, 12.3])
        r = cholesky_sample(cov, 100000, means=means)

        x = (r[0,:]
        erand = (r[1,:]-r[0,:])/(r[1,:]+r[0,:])
        e_mean = erand.mean()
        e_err = erand.std()

    History
        - output is now (npoints,npar) instead of (npar,npoints) to match
        expectation from rec arrays
    """
    if dist is None:
        dist = numpy.random.randn

    npar = cov.shape[0]
    if means is not None:
        nm = len(means)
        if nm != cov.shape[0]:
            raise ValueError("expected %d mean values, got %d" % (npar, nm))

    M = numpy.linalg.cholesky(cov)

    r = dist(npar * n).reshape(npar, n)

    V = numpy.dot(M, r)

    if means is not None:
        for i in range(npar):
            V[i, :] += means[i]

    return V.T


def random_indices(imax, nrand, unique=True, rng=None, seed=None):
    """
    Get a unique random selection of indices in [0,imax)

    Now just calls numpy random choice

    parameters
    ----------
    imax:
        range to draw from is [0,imax)
    nrand:
        Number of randoms to create.
    unique:
        If False, the sample will have replacement, and nrand
        can be greater than imax
    rng: np.default_rng, optional
        Optional random number generator
    seed: int, optional
        A seed to create a new rng
    """

    if rng is None:
        rng = numpy.random.default_rng(seed)

    if not unique:
        replace = True
    else:
        replace = False

    return rng.choice(imax, size=nrand, replace=replace)


def randind(nmax, nrand, dtype=None):
    """
    OBSOLETE, use numpy.random.randint

    Name:
        randind
    Calling Sequence:
        ind = randind(nmax, nrand, dtype=)
    Purpose:
        Generate random indices, with replacement, in the open interval
        [0,nmax)
    Inputs:
        nmax: Indices will be generated to nmax-1
        nrand: Number of randoms to create.
    Optional Inputs:
        dtype:  If not sent, will be unsigned 8-byte integer if
            nmax > 2**32-1 else will be unsigned 4-byte.

    """

    if dtype is None:
        if nmax > (2 ** 32 - 1):
            dtype = "u8"
        else:
            dtype = "u4"

    rnd = numpy.random.random(nrand)
    if nrand == 1:
        ind = int(rnd * nmax)
    else:
        ind = numpy.zeros(nrand, dtype=dtype)
        ind[:] = arrscl(rnd, 0, nmax - 1, arrmin=0.0, arrmax=1.0)

    return ind
