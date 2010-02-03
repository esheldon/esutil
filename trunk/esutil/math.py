from sys import stdout
import numpy
import esutil.stat
import esutil.numpy_util



class qgauss():
    def __init__(self, npts=None):

        self.npts = None
        self.xxi = None
        self.wii = None
        self.f1 = None
        self.f2 = None

        self.setup(npts=npts)

    def setup(self, npts=None):
        if npts is not None:
            if self.npts != npts:
                self.npts=npts
                self.xxi, self.wii = gauleg(-1.0, 1.0, self.npts)


    def integrate_func(self, xvals, func, npts=None):
        """
        Integrate a function
        """
        self.setup(npts=npts)
        if self.npts is None:
            raise ValueError("Set npts on construction or in this call")

        x1 = xvals.min()
        x2 = xvals.max()

        f1 = (x2-x1)/2.
        f2 = (x2+x1)/2.

        xi = self.xxi*f1 + f2

        yvals = func(xi)

        integrand = yvals*self.wii
        isum = integrand.sum()
        return f1*isum

    def integrate_data(self, xvals, yvals, npts=None):
        """
        Integrate over data, using linear interpolation
        """
        self.setup(npts=npts)
        if self.npts is None:
            raise ValueError("Set npts on construction or in this call")

        x1 = xvals.min()
        x2 = xvals.max()

        f1 = (x2-x1)/2.
        f2 = (x2+x1)/2.

        xi = self.xxi*f1 + f2

        # interpolate the yvalues to the right x values for gauss legendre
        # integration
        yi = esutil.stat.interplin(yvals, xvals, xi)

        integrand = yi*self.wii
        isum = integrand.sum()
        return f1*isum

    def test_gauss(self, npts=None):
        mean = 0.0
        sigma = 1.0

        num = 100
        xvals = numpy.arange(100)
        xvals = esutil.numpy_util.arrscl(xvals,mean-4.0*sigma,mean+4.0*sigma)

        norm = 1.0/numpy.sqrt(2.0*numpy.pi*sigma**2)
        gauss = norm*numpy.exp(-0.5*(xvals - mean)**2/sigma**2 )

        expected = 1.0

        ival = self.integrate_data(xvals, gauss, npts=npts)

        stdout.write("Expected value: %s\n" % expected)
        stdout.write("Got value: %s\n" % ival)

        pdiff = (ival - expected)/expected
        stdout.write("%% diff: %s\n" % pdiff)



def gauleg(x1, x2, npts):
    """
    NAME:
      gauleg()
      
    PURPOSE:
      Calculate the weights and abscissa for Gauss-Legendre integration.
    
    CALLING SEQUENCE:
      x,w = gauleg(x1,x2,npts)

    INPUTS:
      x1,x2: The range for the integration.
      npts: Number of points to use in the integration.

    REVISION HISTORY:
      Created: 2006-10-24. Adapted from Numerial recipes in C. Uses
        scipy.weave.inline for the C loops.  2006-10-24 Erin Sheldon NYU
    """

    try:
        from scipy import weave
    except:
        raise ImportError("scipy.weave could not be imported")
    # outputs
    x = numpy.zeros(npts, dtype='f8')
    w = numpy.zeros(npts, dtype='f8')

    # Code string for weave
    code = \
         """
         int i, j, m;
         double xm, xl, z1, z, p1, p2, p3, pp, pi, EPS, abszdiff;
         
         EPS = 3.e-11;
         pi=3.1415927;

         m = (npts + 1)/2;

         xm = (x1 + x2)/2.0;
         xl = (x2 - x1)/2.0;
         z1 = 0.0;

         for (i=1; i<= m; ++i) 
         {
      
           z=cos( pi*(i-0.25)/(npts+.5) );

           abszdiff = fabs(z-z1);

           while (abszdiff > EPS) 
           {
             p1 = 1.0;
             p2 = 0.0;
             for (j=1; j <= npts;++j)
             {
                p3 = p2;
                p2 = p1;
                p1 = ( (2.0*j - 1.0)*z*p2 - (j-1.0)*p3 )/j;
             }
             pp = npts*(z*p1 - p2)/(z*z -1.);
             z1=z;
             z=z1 - p1/pp;

             abszdiff = fabs(z-z1);

           }
      
           x(i-1) = xm - xl*z;
           x(npts+1-i-1) = xm + xl*z;
           w(i-1) = 2.0*xl/( (1.-z*z)*pp*pp );
           w(npts+1-i-1) = w(i-1);


         }

         return_val = 0;
         

         """
    
    weave.inline(code, ['x1', 'x2', 'npts', 'x', 'w'],
                       type_converters = weave.converters.blitz)

    return x,w


