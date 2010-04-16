"""
Package:
    stat

This is a sub-package of the esutil package. The full reference is esutil.stat

Methods:
    (for full docs, see the individual methods.  E.g. esutil.stat.histogram

    histogram:  
        Calculate the histogram of the input data.  The reverse indices are
        also optionally calculated.  This function behaves similarly to the
        IDL histogram funcion.
    histogram2d:  
        Histgram two variables.
    wmom:  
        Calculate weighted mean and error for the given input data.
    sigma_clip:  
        Return the sigma-clipped mean and error for the input data.
    interplin:  
        Perform linear interpolation.  This function is less powerful than
        scipy.interpolate.interp1d but behaves like the IDL interpol()
        function, including extrapolation beyond boundaries.
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


import util
from util import *
import unit_tests
