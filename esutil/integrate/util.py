"""
Module:
    integrate


See docs for individual classes and functions for more detail.

Classes:
    QGauss:
        A class to perform gauss-legendre integration.

Functions:
    gauleg:
        Calculate the weights and abscissa for Gauss-Legendre integration.
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




from sys import stdout
import numpy
import esutil.stat
import esutil.numpy_util

# for checking function type, method type
from types import *

try:
    import cgauleg
    have_cgauleg=True
except:
    have_cgauleg=False

class QGauss():
    """
    Module:
        integrate

    Class Name:
        QGauss
    
    Purpose:
        Perform gauss-legendre integration of points or functions.

    Methods:
        integrate: Perform the integration.
    Examples:
        from esutil.integrate import QGauss
        npoints = 30
        qg = QGauss(npoints)

        # integrate x-y point data
        result = qg.integrate(x, y)

        # integrate a function or method over the range xmin,xmax
        result = qg.integrate([xmin,xmax], some_function)

    """
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


    def integrate(self, xvals, yvals_or_func, npts=None):
        """
        Integrate a function or points
        """

        if isinstance(yvals_or_func,(FunctionType,MethodType)):
            return self.integrate_func(xvals,yvals_or_func,npts)
        else:
            return self.integrate_data(xvals,yvals_or_func,npts)

    def integrate_func(self, xvals, func, npts=None):
        """
        Integrate a function
        """
        self.setup(npts=npts)
        if self.npts is None:
            raise ValueError("Set npts on construction or in this call")

        if len(xvals) != 2:
            raise ValueError("When integrating a function, send the "
                             "x range [xmin,xmax] ")
        x1 = xvals[0]
        x2 = xvals[1]

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

    def test_gauss_data(self, npts=None):
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

    def test_gauss_func(self, npts=None):
        xrange = [-4.0,4.0]


        expected = 1.0

        ival = self.integrate_func(xrange, self.gaussfunc)

        stdout.write("Expected value: %s\n" % expected)
        stdout.write("Got value: %s\n" % ival)

        pdiff = (ival - expected)/expected
        stdout.write("%% diff: %s\n" % pdiff)


    def gaussfunc(self,xvals):
        mean=0.0
        sigma=1.0

        norm = 1.0/numpy.sqrt(2.0*numpy.pi*sigma**2)
        gauss = norm*numpy.exp(-0.5*(xvals - mean)**2/sigma**2 )

        return gauss


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
      Created: 2010-04-18. Use the new C++ extension and only 
      drop back to python only version if necessary.
    """

    if have_cgauleg:
        x,w = cgauleg.cgauleg(x1,x2,npts)
    else:
        x,w = _gauleg_weave(x1,x2,npts)

    return x,w

def _gauleg_weave(x1, x2, npts):
    """
    NAME:
      _gauleg_weave()
      
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


