"""
Module:
    random

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

        Using a sampled distribution pofx measured at values x.  In this case pofx
        is an array of data instead of a function.

            import esutil
            gen = esutil.random.Generator(pofx, x)
            rand = gen.genrand(1000000)


        Using a function:

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

class Generator():
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
            gen = esutil.random.Generator(pofx, x)
            rand = gen.genrand(1000000)


        Using a function:

            def gaussfunc(x):
                return numpy.exp(-0.5*x**2)/numpy.sqrt(2.0*numpy.pi)

            gen = esutil.random.Generator(gaussfunc, 
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


