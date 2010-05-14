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

class QGauss:
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

        ival = self.integrate_func(xrange, self.gaussfunc, npts=npts)

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
        raise ValueError("gauleg C++ extension not found")

    return x,w


