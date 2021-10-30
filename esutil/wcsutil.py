"""
Module:
    wcsutil

Contains the class WCS to perform world coordinate system transformations.
See documentation for the WCS class for more information.

Examples:
    # Use a fits header as initialization to a WCS class and convert
    # image (x,y) to equatorial longitude,latitude (ra,dec)
    from esutil import wcsutil
    import pyfits
    hdr=pyfits.getheader(fname)
    wcs = wcsutil.WCS(hdr)

    # convert x,y to ra,dec. x,y can be scalars or numpy arrays.
    # The returned ra,dec are always numpy arrays.

    ra,dec = wcs.image2sky(x,y)

    # the inverse.  When there is a distortion model prosent, by default it
    # finds the root of the forward transform, which is most accurate way to do
    # the inversion.  Send find=False to attempt to use an inverse polynomial.

    x,y = wcs.sky2image(ra,dec)


Modification History:
    Early 2009, created. Erin Sheldon, NYU

    Now included in esutil.  Cleaned up imports so the module can be imported
    without numpy/scipy even though nothing will work.  2009-11-01. E.S.S. BNL

    2011-06-18: If input are scalars, return scalars

"""

import numpy as np
from numpy import isscalar

import math
import sys

r2d = 180.0 / math.pi
d2r = math.pi / 180.0
DEFTOL = 1e-8

# map the odd scamp naming scheme onto a matrix
# I didn't figure out the formula

_scamp_max_order = 3
_scamp_max_ncoeff = 11
_scamp_skip = [3]

_scamp_map = {}
_scamp_map["pv1_0"] = (0, 0)
_scamp_map["pv1_1"] = (1, 0)
_scamp_map["pv1_2"] = (0, 1)
_scamp_map["pv1_4"] = (2, 0)
_scamp_map["pv1_5"] = (1, 1)
_scamp_map["pv1_6"] = (0, 2)
_scamp_map["pv1_7"] = (3, 0)
_scamp_map["pv1_8"] = (2, 1)
_scamp_map["pv1_9"] = (1, 2)
_scamp_map["pv1_10"] = (0, 3)

_scamp_map["pv2_0"] = (0, 0)
_scamp_map["pv2_1"] = (0, 1)
_scamp_map["pv2_2"] = (1, 0)
_scamp_map["pv2_4"] = (0, 2)
_scamp_map["pv2_5"] = (1, 1)
_scamp_map["pv2_6"] = (2, 0)
_scamp_map["pv2_7"] = (0, 3)
_scamp_map["pv2_8"] = (1, 2)
_scamp_map["pv2_9"] = (2, 1)
_scamp_map["pv2_10"] = (3, 0)

_allowed_projections = ["-TAN", "-TPV", "-TAN-SIP"]

_ap = {}
dname = "-TAN"
_ap[dname] = {}
_ap[dname]["name"] = "scamp"
_ap[dname]["aprefix"] = "pv1"
_ap[dname]["bprefix"] = "pv2"
_ap[dname]["apprefix"] = "pvi1"
_ap[dname]["bpprefix"] = "pvi2"

_ap["-TPV"] = _ap["-TAN"]

dname = "-TAN-SIP"
_ap[dname] = {}
_ap[dname]["name"] = "sip"
_ap[dname]["aprefix"] = "a"
_ap[dname]["bprefix"] = "b"
_ap[dname]["apprefix"] = "ap"
_ap[dname]["bpprefix"] = "bp"

_allowed_units = ["deg"]

# same mapping for the inverse
smkeys = list(_scamp_map.keys())
for key in smkeys:
    newkey = key.replace("pv", "pvi")
    _scamp_map[newkey] = _scamp_map[key]


class WCS(object):
    """
    A class to do WCS transformations.  Currently supports TAN projections
    for

        RA---TPV, DEC--TPV
        RA---TAN and DEC--TAN
        RA---TAN--SIP,DEC--TAN--SIP

    ctypes in degrees.  The first two are both actually TPV, but old versions
    of scamp wrote them simply as TAN.

    Usage:

    from esutil import wcsutil
    wcs = wcsutil.WCS(wcs_structure, longpole=180.0, latpole=90.0, theta0=90.0)

    The input is a wcs structure.  This could be a dictionary or numpy array
    that can be addressed like wcs['cunit1'] for example, or something that
    supports iteration like a fitsio header, or have an items() method such as
    for a pyfits header.  It is converted to a dictionary internally.  The
    structure is exactly that as would be written to a FITS header, so for
    example distortion fields are not converted to a matrix form, rather each
    field in the header gets a field in this structure.

    When there is a distortion model the inverse transformation is gotten
    by solving the for the roots of the transformation by default.  This
    is slow, so if you care about speed and not precision you can set
    find=False in sky2image() and it will use a polynomial fit to the inverse,
    which is calculated if not already in the header.

    The solve is done using scipy.optimize.fsolve

    Examples:
        # Use a fits header as initialization to a WCS class and convert
        # image (x,y) to equatorial longitude,latitude (ra,dec)
        from esutil import wcsutil
        import pyfits
        hdr=pyfits.getheader(fname)
        wcs = wcsutil.WCS(hdr)

        # convert x,y to ra,dec. x,y can be scalars or numpy arrays.
        # The returned ra,dec are always numpy arrays.
        ra,dec = wcs.image2sky(x,y)

        # the inverse.  When there is a distortion model prosent, by default
        # it finds the root of the forward transform, which is most accurate
        # way to do the inversion.  Send find=False to attempt to use an
        # inverse polynomial.

        x,y = wcs.sky2image(ra,dec)

    """

    def __init__(self, wcs, longpole=180.0, latpole=90.0, theta0=90.0):

        # Convert to internal dictionary and set some attributes of this
        # instance
        self.wcs = self.ConvertWCS(wcs)
        self._set_naxis()
        self._inverse_computed = False

        # Set these as attributes, either from above keywords or from the
        # wcs header
        self.SetAngles(longpole, latpole, theta0)

        # Now set a bunch more instance attributes from the wcs in a form
        # that is easier to work with
        self.ExtractFromWCS()

        # for finding the inverse trans
        self.lonlat_answer = np.zeros(2, dtype="f8")
        self.xyguess = np.zeros(2, dtype="f8")

    def __repr__(self):
        import pprint

        return pprint.pformat(self.wcs)

    def __getitem__(self, key):
        return self.wcs[key]

    def __setitem__(self, key, val):
        self.wcs[key] = val

    def keys(self):
        return self.wcs.keys()

    def get_naxis(self):
        """
        get [nx,ny], properly accounting for compressed data that
        use znaxis*

        returns
        -------
        [nx,ny] as an array
        """
        return self.naxis.copy()

    def get_jacobian(self, x, y, distort=True, step=1.0):
        """
        Get the elementes of the jacobian matrix at the specified locations
        This method currently assumes the system is ra,dec

        parameters
        ----------
        x,y: scalars or arrays
            x and y coords in the image
        distort:  bool, optional
            Use the distortion model if present.  Default is True
        step: float
            Step used for central difference formula, in pixels.  Default is
            1.0 pixels.

        returns
        -------
        jacobian elements: tuple of arrays
            dra_dx, dra_dy, ddec_dx, ddec_dy

        method
        ------
        Finite difference
        """

        fac = 1.0 / (2 * step)

        ra, dec = self.image2sky(x, y, distort=distort)

        xp = x + step
        xm = x - step
        yp = y + step
        ym = y - step

        ra_p0, dec_p0 = self.image2sky(xp, y, distort=distort)
        ra_m0, dec_m0 = self.image2sky(xm, y, distort=distort)

        ra_0p, dec_0p = self.image2sky(x, yp, distort=distort)
        ra_0m, dec_0m = self.image2sky(x, ym, distort=distort)

        # in arcsec/pixel
        dra_dx = fac * 3600.0 * wrap_ra_diff(ra_p0 - ra_m0)
        dra_dy = fac * 3600.0 * wrap_ra_diff(ra_0p - ra_0m)
        ddec_dx = fac * 3600.0 * (dec_p0 - dec_m0)
        ddec_dy = fac * 3600.0 * (dec_0p - dec_0m)

        # need to scale dra b -cos(dec), minus sign since ra increases
        # to the left
        cosdec = -np.cos(dec * d2r)
        dra_dx *= cosdec
        dra_dy *= cosdec

        return dra_dx, dra_dy, ddec_dx, ddec_dy

    def image2sky(self, x, y, distort=True):
        """
        Convert between image x,y and sky coordinates lon,lat e.g. ra,dec.

        parameters
        ----------
        x,y: scalars or arrays
            x and y coords in the image
        distort:  bool, optional
            Use the distortion model if present.  Default is True

        returned values
        ---------------
        longitude,latitude:  tupple of arrays
            Probably ra,dec.  Will have the same shape as x,y

        examples
        --------
        from esutil import wcsutil
        import fitsio
        hdr=fitsio.read_header(fname)
        wcs = wcsutil.WCS(hdr)
        ra,dec = wcs.image2sky(x,y)
        """

        xdiff = x - self.crpix[0]
        ydiff = y - self.crpix[1]

        p = self.projection.upper()
        if p in ["-TAN", "-TPV"]:
            u, v = self.ApplyCDMatrix(xdiff, ydiff)
            if distort and self.distort["name"] != "none":
                # Assuming PV distortions
                u, v = self.Distort(u, v)

        elif p == "-TAN-SIP":
            if distort and self.distort["name"] != "none":
                u, v = self.Distort(xdiff, ydiff)
            u, v = self.ApplyCDMatrix(u, v)
        else:
            raise ValueError("projection '%s' not supported" % p)

        longitude, latitude = self.image2sph(u, v)

        return longitude, latitude

    def sky2image(
        self, longitude, latitude, distort=True, find=True, xtol=DEFTOL,
    ):
        """
        Usage:
            x,y=sky2image(longitude, latitude, distort=True, find=True)

        Purpose:
            Convert between sky (lon,lat) and image coordinates (x,y)

        Inputs:
            longitude,latitude:  Probably ra,dec. Can be arrays.
        Optional Inputs:
            distort:  Use the distortion model if present.  Default is True
            find: When the distortion model is present, simply find the
                roots of the polynomial rather than using an inverse
                polynomial.  This is more accurate but slower. Default True.
            xtol: tolerance to use when root finding with find=True Default is
                1e-8.
        Outputs:
            x,y: x and y coords in the image.  Will have the same shape as
                lon,lat
        Example:
            from esutil import wcsutil
            import pyfits
            hdr=pyfits.getheader(fname)
            wcs = wcsutil.WCS(hdr)
            x,y = wcs.image2sky(ra,dec)
        """

        # Only do this if there is distortion
        if find and self.distort["name"] != "none":
            x, y = self._findxy(longitude, latitude, xtol=xtol)
        else:
            u, v = self.sph2image(longitude, latitude)

            p = self.projection.upper()
            if p in ["-TAN", "-TPV"]:
                if distort and self.distort["name"] != "none":
                    u, v = self.Distort(u, v, inverse=True)
                xdiff, ydiff = self.ApplyCDMatrix(u, v, inverse=True)

            elif p == "-TAN-SIP":
                u, v = self.ApplyCDMatrix(u, v, inverse=True)
                if distort and self.distort["name"] != "none":
                    xdiff, ydiff = self.Distort(u, v, inverse=True)
                else:
                    xdiff, ydiff = u, v

            else:
                raise ValueError("projection '%s' not supported" % p)

            x = xdiff + self.crpix[0]
            y = ydiff + self.crpix[1]

        return x, y

    def ExtractProjection(self, wcs):
        projection = wcs["ctype1"][4:].strip().upper()
        if projection not in _allowed_projections:
            err = (
                "Projection type %s unsupported.  Only [%s] projections "
                "currently supported"
            )
            err = err % (projection, ", ".join(_allowed_projections))
            raise ValueError(err)

        return projection

    def ApplyCDMatrix(self, x, y, inverse=False):
        if not inverse:
            cd = self.cd
            xp = cd[0, 0] * x + cd[0, 1] * y
            yp = cd[1, 0] * x + cd[1, 1] * y
        else:
            cdinv = self.cdinv
            xp = cdinv[0, 0] * x + cdinv[0, 1] * y
            yp = cdinv[1, 0] * x + cdinv[1, 1] * y

        return xp, yp

    def image2sph(self, x, y):
        """
        Convert x,y projected coordinates to spherical coordinates
        Currently only supports tangent plane projections.
        The conventions assumed are that of the WCS
        Works in the native system currently
        """

        latitude = np.zeros_like(x) + math.pi / 2

        # radius in radians
        r = np.sqrt(x ** 2 + y ** 2) * math.pi / 180.0

        if isscalar(r):
            scalar = True
        else:
            scalar = False

        if scalar:
            if r > 0:
                latitude = np.arctan(1.0 / r)
        else:
            (w,) = np.where(r > 0)
            if w.size > 0:
                latitude[w] = np.arctan(1.0 / r[w])

        longitude = np.arctan2(x, -y)

        longitude *= r2d
        latitude *= r2d

        longitude, latitude = self.Rotate(longitude, latitude, reverse=True)

        # Make sure the result runs from 0 to 360
        if scalar:
            if longitude < 0.0:
                longitude += 360.0

            if longitude >= 360.0:
                longitude -= 360.0

        else:
            (w,) = np.where(longitude < 0.0)
            if w.size > 0:
                longitude[w] += 360.0
            (w,) = np.where(longitude >= 360.0)
            if w.size > 0:
                longitude[w] -= 360.0

        return longitude, latitude

    def sph2image(self, longitude, latitude):
        """
        Must be a tangent plane projection
        """

        longitude, latitude = self.Rotate(longitude, latitude)
        longitude *= d2r
        latitude *= d2r

        x = np.zeros_like(longitude)
        y = np.zeros_like(longitude)

        if isscalar(longitude):
            if latitude > 0.0:
                rdiv = r2d / np.tan(latitude)
                x = rdiv * np.sin(longitude)
                y = -rdiv * np.cos(longitude)
        else:
            (w,) = np.where(latitude > 0.0)
            if w.size > 0:
                rdiv = r2d / np.tan(latitude[w])
                x[w] = rdiv * np.sin(longitude[w])
                y[w] = -rdiv * np.cos(longitude[w])

        return x, y

    def Rotate(self, lon, lat, reverse=False, origin=False):

        longitude = lon * d2r
        latitude = lat * d2r

        r = self.rotation_matrix
        if reverse:
            r = r.transpose()

        return self._rotate(longitude, latitude, r)

    def CreateRotationMatrix(self):
        # If Theta0 = 90 then CRVAL gives the coordinates of the origin in the
        # native system.   This must be converted (using Eq. 7 in Greisen &
        # Calabretta with theta0 = 0) to give the coordinates of the North
        # pole (longitude_p, latitude_p)

        # Longpole is the longitude in the native system of the North Pole in
        # the standard system (default = 180 degrees).
        sp = math.sin(self.longpole * d2r)
        cp = math.cos(self.longpole * d2r)

        sa = math.sin(self.native_longpole)
        ca = math.cos(self.native_longpole)
        sd = math.sin(self.native_latpole)
        cd = math.cos(self.native_latpole)

        # calculate rotation matrix

        # IDL array construction is transposed compared to python apparently
        # So this is reversed from the idl routines
        r = np.array(
            [
                [-sa * sp - ca * cp * sd, sa * cp - ca * sp * sd, ca * cd],
                [ca * sp - sa * cp * sd, -ca * cp - sa * sp * sd, sa * cd],
                [cp * cd, sp * cd, sd],
            ],
            dtype="f8",
        )

        return r

    def _rotate(self, longitude, latitude, r):
        """
        Apply a rotation matrix to the input longitude and latitude
        inputs must be numpy arrays
        """
        l = np.cos(latitude) * np.cos(longitude)  # noqa
        m = np.cos(latitude) * np.sin(longitude)
        n = np.sin(latitude)

        # find solution to the system of equations and put it in b
        # Can't use matrix notation in case l,m,n are rrays

        b0 = r[0, 0] * l + r[1, 0] * m + r[2, 0] * n
        b1 = r[0, 1] * l + r[1, 1] * m + r[2, 1] * n
        b2 = r[0, 2] * l + r[1, 2] * m + r[2, 2] * n

        # Account for possible roundoff
        b2 = np.clip(b2, -1.0, 1.0)
        """
        w, = np.where( b2 > 1.0 )
        if w.size > 0:
            b2[w] = 1.0
        w, = np.where( b2 < -1.0 )
        if w.size > 0:
            b2[w] = -1.0
        """

        # looks like arctan2 has less roundoff error
        # old code used arcsin
        # lat_new = np.arcsin(b2)*r2d
        lat_new = np.arctan2(b2, np.sqrt(b0 * b0 + b1 * b1)) * r2d
        lon_new = np.arctan2(b1, b0) * r2d

        return lon_new, lat_new

    def _lonlatdiff(self, xy):
        x = xy[0]
        y = xy[1]
        lon, lat = self.image2sky(x, y)
        lonlat = np.zeros(2)
        lonlat[0] = lon
        lonlat[1] = lat
        diff = lonlat - self.lonlat_answer
        return diff

    def _fsolve_xy(self, xyguess, xtol=DEFTOL):
        import scipy.optimize

        xy = scipy.optimize.fsolve(self._lonlatdiff, xyguess, xtol=xtol)
        return xy

    def _lmfind_xy(self, xyguess):
        from scipy.optimize import leastsq

        lm_tup = leastsq(self._lonlatdiff, xyguess, full_output=1)
        xy, pcov0, infodict, errmsg, ier = lm_tup
        if ier > 4:
            raise RuntimeError(
                "failed to find inverse transform: '%s'" % errmsg
            )
        return xy

    def _findxy(self, lon, lat, xtol=DEFTOL):
        """
        This is the simplest way to do the inverse of the (x,y)->(lon,lat)
        transformation when there are distortions.  Simply find the x,y
        that give the input lon,lat from the actual distortion function.

        Uses scipy.optimize.fsolve to find the roots of the transformation
        """

        if isscalar(lon):
            x, y = self._findxy_one(lon, lat, xtol=xtol)
        else:
            x = np.zeros_like(lon)
            y = np.zeros_like(lon)

            for i in range(lon.size):
                x[i], y[i] = self._findxy_one(lon[i], lat[i], xtol=xtol)

        return x, y

    def _findxy_one(self, lon, lat, xtol=DEFTOL):
        """
        This is the simplest way to do the inverse of the (x,y)->(lon,lat)
        transformation when there are distortions.  Simply find the x,y
        that give the input lon,lat from the actual distortion function.

        Uses scipy.optimize.fsolve to find the roots of the transformation
        """

        self.lonlat_answer[0] = lon
        self.lonlat_answer[1] = lat

        xyguess = self.xyguess

        # Use inversion without distortion as our guess
        xyguess[0], xyguess[1] = self.sky2image(
            lon, lat, find=False, distort=False,
        )
        xy = self._fsolve_xy(xyguess, xtol=xtol)
        # print 'using lm'
        # xy = self._lmfind_xy(xyguess)
        x, y = xy[0], xy[1]

        return x, y

    def Distort(self, x, y, inverse=False):
        """
        Apply a distortion map to the data.  This follows the SIP convention,
        but if the scamp PV coefficients were found by the ConvertWCS code
        they are converted to the SIP convention.  The only difference is
        the order of operations:  for image to sky PV distortions come after
        the application of the CD matrix as opposed to SIP.

        """
        if not self._inverse_computed and inverse:
            self._inverse_computed = True
            self.InvertDistortion()
            self.distort["ap_order"] = self.distort["a_order"] + 1
            self.distort["bp_order"] = self.distort["b_order"] + 1

        # Sometimes there is no distortion model present
        if self.distort is None or self.distort["name"] == "none":
            # return copies
            return x * 1.0, y * 1.0

        if inverse:
            a = self.distort["ap"]
            b = self.distort["bp"]
        else:
            a = self.distort["a"]
            b = self.distort["b"]

        sx, sy = a.shape

        if self.distort["name"] == "scamp":
            xp = 0 * x
            yp = 0 * y
        elif self.distort["name"] == "sip":
            xp = x * 1.0
            yp = y * 1.0
        else:
            raise ValueError(
                "Unsupported distortion model '%s'" % self.distort["name"]
            )

        xp += Apply2DPolynomial(a, x, y)
        yp += Apply2DPolynomial(b, x, y)

        return xp, yp

    def _compare_inversion(
        self, x, y, xback, yback, verbose=False, doplot=False, units=""
    ):
        # Get rms differences
        t = (xback - x) ** 2 + (yback - y) ** 2
        rms = np.sqrt(t.sum() / t.size)
        if verbose:
            mess = "rms error"
            if units != "":
                mess += "(" + units + ")"
            mess += ":"
            sys.stdout.write("%s %s\n" % (mess, rms))
        if doplot:
            import pylab

            pylab.clf()
            pylab.hist(x - xback, 50, edgecolor="black", fill=False)
            pylab.hist(y - yback, 50, edgecolor="red", fill=False)
            pylab.show()

        return rms

    def InvertDistortion(self, fac=5, order_increase=1,
                         verbose=False, doplot=False):
        if self.distort["name"] == "scamp":
            return self.InvertPVDistortion(
                fac=fac, order_increase=order_increase,
                verbose=verbose, doplot=doplot
            )
        elif self.distort["name"] == "sip":
            return self.InvertSipDistortion(
                fac=fac, order_increase=order_increase,
                verbose=verbose, doplot=doplot
            )
        else:
            raise ValueError("Can only invert scamp and sip distortions")

    def InvertPVDistortion(self, fac=5, order_increase=1, verbose=False,
                           doplot=False):
        """
        Invert the distortion model.  Must contain a,b matrices
        """

        # Order of polynomial
        sx, sy = self.distort["a"].shape
        porder = sx - 1

        ng = 2 * (porder + 2)
        ng *= fac

        # Assuming 1 offset
        xrang = np.array([1.0, self.naxis[0]], dtype="f8") - self.crpix[0]
        yrang = np.array([1.0, self.naxis[1]], dtype="f8") - self.crpix[1]

        xdiff, ydiff = make_xy_grid(ng, xrang, yrang)

        # same to here
        u, v = self.ApplyCDMatrix(xdiff, ydiff)

        # This is what we will invert
        # up,vp = self.Distort(u,v)
        up = Apply2DPolynomial(self.distort["a"], u, v)
        vp = Apply2DPolynomial(self.distort["b"], u, v)

        # Find polynomial from up,vp to u,v
        ainv, binv = Invert2DPolynomial(up, vp, u, v, porder + order_increase)
        self.distort["ap"] = ainv
        self.distort["bp"] = binv

        # newu, newv = self.Distort(up, vp, inverse=True)
        newu = Apply2DPolynomial(ainv, up, vp)
        newv = Apply2DPolynomial(binv, up, vp)
        ufrac = (u - newu) / u
        vfrac = (v - newv) / v
        if verbose:
            sys.stdout.write("\ntesting inverse now:\n")
            sys.stdout.write("\n  ufrac=%s\n" % ufrac)
            sys.stdout.write("  vfrac=%s\n" % vfrac)
            sys.stdout.write("\n  median ufrac=%s\n" % np.median(ufrac))
            sys.stdout.write("  median vfrac= %s\n\n" % np.median(vfrac))

        self._compare_inversion(
            u, v, newu, newv, verbose=verbose, doplot=doplot
        )

        x = xdiff + self.crpix[0]
        y = ydiff + self.crpix[1]
        lon, lat = self.image2sky(x, y)
        xback, yback = self.sky2image(lon, lat, find=False)

        rms = self._compare_inversion(
            x, y, xback, yback, verbose=verbose, doplot=doplot, units="pixels"
        )
        return rms

    def InvertSipDistortion(self, fac=5, verbose=False,
                            doplot=False, order_increase=1):
        """
        Invert the distortion model.  Must contain a,b matrices
        """

        # Order of polynomial
        sx, sy = self.distort["a"].shape
        porder = sx - 1

        ng = 2 * (porder + 2)
        ng *= fac

        xrang = np.array([1.0, self.naxis[0]])
        yrang = np.array([1.0, self.naxis[1]])

        x, y = make_xy_grid(ng, xrang, yrang)

        # Use distortion for getting sky coords
        lon, lat = self.image2sky(x, y)
        # Don't use distortion to get back image coords.  We will use
        # the difference to fit for new coefficients.
        xback, yback = self.sky2image(lon, lat, distort=False, find=False)
        self._compare_inversion(
            x, y, xback, yback, verbose=verbose, doplot=doplot,
        )

        xdiff = xback - self.crpix[0]
        ydiff = yback - self.crpix[1]

        constant = False
        ainv, binv = Invert2DPolynomial(
            xdiff,
            ydiff,
            x - xback,
            y - yback,
            porder + order_increase,
            constant=constant,
        )

        if "ap" in self.distort:
            xback2, yback2 = self.sky2image(lon, lat, find=False)
            rms = self._compare_inversion(
                x, y, xback2, yback2, verbose=verbose, doplot=doplot,
                units="pixels"
            )

        self.distort["ap"] = ainv
        self.distort["bp"] = binv

        xback2, yback2 = self.sky2image(lon, lat, find=False)

        rms = self._compare_inversion(
            x, y, xback2, yback2, verbose=verbose, doplot=doplot
        )
        return rms

    def GetPole(self):

        longitude_0 = self.wcs["crval1"] * d2r
        latitude_0 = self.wcs["crval2"] * d2r

        if self.theta0 == 90.0:
            return longitude_0, latitude_0

        # Longpole is the longitude in the native system of the North Pole
        # in the standard system (default = 180 degrees).
        phi_p = np.deg2rad(self.longpole)
        # phi_p = self.longpole / radeg
        sp = math.sin(phi_p)
        cp = math.cos(phi_p)
        sd = math.sin(latitude_0)
        cd = math.cos(latitude_0)
        tand = math.tan(latitude_0)

        if self.theta0 == 0.0:
            if latitude_0 == 0 and self.longpole == 90.0:
                latitude_p = self.latpole
            else:
                latitude_p = math.acos(sd / cp)

            if self.latpole != 90.0:
                if math.fabs(self.latpole + latitude_p) < math.fabs(
                    self.latpole - latitude_p
                ):
                    latitude_p = -latitude_p

            if (self.longpole == 180.0) or (cd == 0.0):
                longitude_p = longitude_0
            else:
                longitude_p = longitude_0 - math.atan2(
                    sp / cd, -math.tan(latitude_p) * tand
                )
        else:
            ctheta = math.cos(self.theta0 * d2r)
            stheta = math.sin(self.theta0 * d2r)

            term1 = math.atan2(stheta, ctheta * cp)
            term2 = math.acos(
                sd / (math.sqrt(1.0 - ctheta * ctheta * sp * sp))
            )

            if term2 == 0.0:
                latitude_p = term1
            else:
                latitude_p1 = math.fabs((term1 + term2) * r2d)
                latitude_p2 = math.fabs((term1 - term2) * r2d)

                if (latitude_p1 > 90.0) and (latitude_p2 > 90.0):
                    raise ValueError("No valid solution")
                elif (latitude_p1 < 90.0) and (latitude_p2 > 90.0):
                    latitude_p = term1 + term2
                elif (latitude_p1 > 90.0) and (latitude_p2 < 90.0):
                    latitude_p = term1 - term2
                else:
                    # Two valid solutions
                    latitude_p1 = (term1 + term2) * r2d
                    latitude_p2 = (term1 - term2) * r2d
                    if math.fabs(self.latpole - latitude_p1) < math.fabs(
                        self.latpole - latitude_p2
                    ):
                        latitude_p = term1 + term2
                    else:
                        latitude_p = term1 - term2

                if cd == 0.0:
                    longitude_p = longitude_0
                else:
                    sdelt = math.sin(latitude_p)
                    if sdelt == 1.0:
                        longitude_p = longitude_0 - phi_p - math.pi
                    else:
                        if sdelt == -1.0:
                            longitude_p = longitude_0 - phi_p
                        else:
                            sdp = math.sin(latitude_p)
                            cdp = math.cos(latitude_p)
                            longitude_p = longitude_0 - math.atan2(
                                (stheta - sdp * sd) / (cdp * cd), sp * ctheta / cd  # noqa
                            )
        return longitude_p, latitude_p

    def ConvertWCS(self, wcs_in):
        """
        Convert to a dictionary
        """

        self.wcs = None
        self.distort = {"name": "none"}
        self.cd = None
        self.crpix = None
        self.crval = None
        self.projection = None

        # Convert the wcs to a local dictionary

        wcs = {}
        if type(wcs_in) == np.ndarray or hasattr(wcs_in, "dtype"):
            if wcs_in.dtype.fields is None:
                raise ValueError("wcs array must have fields")

            for f in wcs_in.dtype.fields:
                fl = f.lower()
                val = wcs_in[f]
                if val.ndim == 0:
                    wcs[fl] = val
                else:
                    # only scalars
                    wcs[fl] = val[0]

        elif isinstance(wcs_in, dict):
            wcs = wcs_in.copy()
        elif hasattr(wcs_in, "__iter__"):
            wcs = {}
            for k in wcs_in:
                if k is None:
                    continue
                wcs[k.lower()] = wcs_in[k]
        else:
            # Try to use the items() method to get what we want
            wcs = {}
            try:
                for k, v in wcs_in.items():
                    if k is None:
                        continue
                    wcs[k.lower()] = v
            except Exception:
                raise ValueError(
                    "Input wcs must be a numpy array "
                    + "with fields or a dictionary or support "
                    + "iteration or an items() method"
                )

        return wcs

    def SetAngles(self, longpole, latpole, theta0):
        # These can get set if they were not in the WCS header
        if "longpole" not in self.wcs:
            self.longpole = longpole
        else:
            self.longpole = self.wcs["longpole"]

        if "latpole" not in self.wcs:
            self.latpole = latpole
        else:
            self.latpole = self.wcs["latpole"]

        if "theta0" not in self.wcs:
            self.theta0 = theta0
        else:
            self.theta0 = self.wcs["theta0"]

    def ExtractUnits(self, wcs):

        if "cunit1" in wcs:
            units = wcs["cunit1"].strip().lower()
            if units not in _allowed_units:
                err = "Unsupported units %s.  Only [%s] supported"
                raise ValueError(err % (units, ", ".join(_allowed_units)))
        else:
            units = None
        return units

    def ExtractDistortCoeffs(self, dname, wcs, prefix):
        if dname == "scamp":
            return self.ExtractPVCoeffs(wcs, prefix)
        elif dname == "sip":
            return self.ExtractSIPCoeffs(wcs, prefix)

    def ExtractPVCoeffs(self, wcs, prefix):
        order = _scamp_max_order
        dim = order + 1
        matrix = np.zeros((dim, dim), dtype="f8")
        count = 0
        for i in range(_scamp_max_ncoeff):
            if i not in _scamp_skip:
                key = prefix + "_" + str(i)
                if key in wcs:
                    indices = _scamp_map[key]
                    matrix[indices[0], indices[1]] = wcs[key]
                    count += 1
        return matrix, count, order

    def ExtractSIPCoeffs(self, wcs, prefix):
        order = _dict_get(wcs, prefix + "_order")
        matrix = np.zeros((order + 1, order + 1), dtype="f8")
        count = 0
        for ix in range(order + 1):
            for iy in range(order + 1):
                key = prefix + "_" + str(ix) + "_" + str(iy)
                if key in wcs:
                    matrix[ix, iy] = wcs[key]
                    count += 1
        return matrix, count, order

    def ExtractDistortionModel(self):
        if self.projection not in _allowed_projections:
            raise ValueError(
                "Projection must be on of %s " % ", ".join(_allowed_projections)  # noqa
            )
        else:
            # look for forward coeffs first
            dinfo = _ap[self.projection]
            dname = dinfo["name"]
            a, ca, aorder = self.ExtractDistortCoeffs(
                dname, self.wcs, dinfo["aprefix"]
            )

            if ca != 0:
                self.distort["name"] = dname

                b, cb, border = self.ExtractDistortCoeffs(
                    dname, self.wcs, dinfo["bprefix"]
                )
                ap, cap, aporder = self.ExtractDistortCoeffs(
                    dname, self.wcs, dinfo["apprefix"]
                )
                bp, cbp, bporder = self.ExtractDistortCoeffs(
                    dname, self.wcs, dinfo["bpprefix"]
                )

                self.distort["a"] = a
                self.distort["a_order"] = aorder
                self.distort["b"] = b
                self.distort["b_order"] = border

                # these coeffs will be zeros if not found above
                self.distort["ap"] = ap
                self.distort["ap_order"] = aporder
                self.distort["bp"] = bp
                self.distort["bp_order"] = bporder

                # If inverse can't be found, make sure we computed it later
                if cap == 0 or cbp == 0:
                    self._inverse_computed = False

    def ExtractFromWCS(self):

        # for easier notation
        wcs = self.wcs

        # set these to little arrays
        self.crpix = np.array([wcs["crpix1"], wcs["crpix2"]], dtype="f8")
        self.crval = np.array([wcs["crval1"], wcs["crval2"]], dtype="f8")
        self.ctype = np.array(
            [wcs["ctype1"].strip().upper(), wcs["ctype2"].strip().upper()]
        )

        # Get the projection from ctype
        self.projection = self.ExtractProjection(wcs)

        # Get units
        self.units = self.ExtractUnits(wcs)

        # CTYPE[0] - first four characters specify standard system
        #       ('RA--','GLON' or 'ELON' for right ascension, galactic
        #       longitude or ecliptic longitude respectively), second four
        #       letters specify the type of map projection (eg '-AIT' for
        #       Aitoff projection)
        # CTYPE[1] - first four characters specify standard system
        #       ('DEC-','GLAT' or 'ELAT' for declination, galactic latitude
        #       or ecliptic latitude respectively; these must match
        #       the appropriate system of ctype1), second four letters of
        #       ctype2 must match second four letters of ctype1.

        system1 = self.wcs["ctype1"][0:4]
        system2 = self.wcs["ctype2"][0:4]
        self.system = np.array([system1, system2], dtype="S4")

        # Add a 2x2 array for the cd matrix
        if "cd1_1" in wcs:
            cd = np.zeros((2, 2), dtype="f8")
            cd[0, 0] = wcs["cd1_1"]
            cd[0, 1] = wcs["cd1_2"]
            cd[1, 0] = wcs["cd2_1"]
            cd[1, 1] = wcs["cd2_2"]
            self.cd = cd

            try:
                self.cdinv = np.linalg.inv(cd)
            except np.linalg.LinAlgError:
                raise ValueError("Could not find inverse of CD matrix")

        # Get the poles for the inputs.  Assumes we already ran
        # SetAngles() before calling this method
        self.native_longpole, self.native_latpole = self.GetPole()

        # Create the rotation matrix for later.  Requires that the
        # native system be set up using GetPole()
        self.rotation_matrix = self.CreateRotationMatrix()

        # Extract the distortion model
        self.ExtractDistortionModel()

    def _set_naxis(self):
        wcs = self.wcs
        if "znaxis1" in wcs:
            self.naxis = np.array([wcs["znaxis1"], wcs["znaxis2"]])
        else:
            self.naxis = np.array([wcs["naxis1"], wcs["naxis2"]])


def _dict_get(d, key, default=None):
    if key not in d:
        if default is not None:
            return default
        else:
            raise ValueError("key '%s' must be present" % key)
    return d[key]


def arrscl(arr, minval, maxval, arrmin=None, arrmax=None):
    # makes a copy either way (asarray would not if it was an array already)
    output = np.array(arr)

    if arrmin is None:
        arrmin = output.min()
    if arrmax is None:
        arrmax = output.max()

    if output.size == 1:
        return output

    if arrmin == arrmax:
        sys.stdout.write("arrmin must not equal arrmax\n")
        return None

    try:
        a = (maxval - minval) / (arrmax - arrmin)
        b = (arrmax * minval - arrmin * maxval) / (arrmax - arrmin)
    except Exception as err:
        sys.stdout.write(
            "Error calculating a,b: %s" % str(err)
        )
        return None

    # in place
    np.multiply(output, a, output)
    np.add(output, b, output)

    return output


def Apply2DPolynomial(a, x, y):
    v = np.zeros_like(x)

    sx, sy = a.shape
    for ix in range(sx):
        for iy in range(sy):
            xpow = x ** ix
            ypow = y ** iy
            if a[ix, iy] != 0.0:
                addval = a[ix, iy] * xpow * ypow
                v += addval

    return v


def make_xy_grid(n, xrang, yrang):
    # Create a grid on input ranges
    rng = np.arange(n, dtype="f8")
    ones = np.ones(n, dtype="f8")

    x = arrscl(rng, xrang[0], xrang[1])
    y = arrscl(rng, yrang[0], yrang[1])

    x = np.outer(x, ones)
    y = np.outer(ones, y)
    x = x.flatten("F")
    y = y.flatten("F")

    return x, y


def make_amatrix(u, v, order, constant=True):
    # matrix for inversion.
    # coeffs_u = A^{-1} x = (a^Ta)^{-1} A^T x
    # coeffs_v = A^{-1} v

    # n = (order+1)*2
    # n = n*n
    n = u.size

    tshape = [(order + 1) * (order + 2) // 2 - 1, n]
    if constant:
        # Extra column with ones in it for the constant term
        tshape[0] += 1
        kstart = 1
    else:
        kstart = 0
    # amatrix = np.zeros( tshape )
    amatrix = np.ones(tshape)

    kk = kstart
    for order in range(1, order + 1):
        for jj in range(order + 1):
            amatrix[kk, :] = u ** (order - jj) * v ** jj
            kk += 1

    return amatrix


def invert_for_coeffs(amatrix, x, y, lsolve=True):
    # a^T a
    ata = np.inner(amatrix, amatrix)
    # a^T x
    atx = np.inner(amatrix, x)
    # a^T y
    aty = np.inner(amatrix, y)

    if lsolve:
        # More stable solver
        xcoeffs = np.linalg.solve(ata, atx)
        ycoeffs = np.linalg.solve(ata, aty)

    else:
        atainv = np.linalg.inv(ata)
        # atainv = np.linalg.pinv(ata)
        xcoeffs = np.inner(atainv, atx)
        ycoeffs = np.inner(atainv, aty)

    return xcoeffs, ycoeffs


def pack_coeffs(xcoeffs, ycoeffs, porder, constant=True):
    """
    pack coeffs into a matrix form
    """

    if constant:
        ostart = 0
    else:
        ostart = 1

    kk = 0
    shape = (porder + 1, porder + 1)
    ainv = np.zeros(shape)
    binv = np.zeros(shape)
    for order in range(ostart, porder + 1):
        for jj in range(order + 1):
            ainv[order - jj, jj] = xcoeffs[kk]
            binv[order - jj, jj] = ycoeffs[kk]
            kk += 1
    return ainv, binv


# Find the polynomial coeffs that take us from u,v to x,y
def Invert2DPolynomial(u, v, x, y, porder, pack=True, constant=True):
    # matrix for inversion.
    # coeffs_u = A^{-1} x = (A^TA)^{-1} A^T x
    # coeffs_v = A^{-1} v
    amatrix = make_amatrix(u, v, porder, constant=constant)

    # Now we know the inverse must equal x,y so we use that as the
    # constraint vector
    xcoeffs, ycoeffs = invert_for_coeffs(amatrix, x, y)

    if pack:
        # now pack the coefficients into a matrix
        ainv, binv = pack_coeffs(xcoeffs, ycoeffs, porder, constant=constant)
        return ainv, binv
    else:
        return xcoeffs, ycoeffs


def Ncoeff(order, constant=True):
    ncoeff = (order + 1) * (order + 2) // 2
    if not constant:
        ncoeff -= 1
    return ncoeff


def wrap_ra_diff(dra):
    """Given an input ra difference, wrap it to range -180, 180.

    Parameters
    ----------
    dra : float or np.ndarray
        The input difference in degrees.

    Returns
    -------
    wrapped_dra : float or np.ndarray
        Thje wrapped difference in degrees in the range [-180, 180].
    """
    if np.ndim(dra) == 0:
        if not np.isfinite(dra):
            return dra

        while dra < -180.0:
            dra += 360.0
        while dra > 180.0:
            dra -= 360.0
    else:
        msk_finite = np.isfinite(dra)
        msk = (dra < -180.0) & msk_finite
        while np.any(msk):
            dra[msk] = dra[msk] + 360.0
            msk = (dra < -180.0) & msk_finite

        msk = (dra > 180.0) & msk_finite
        while np.any(msk):
            dra[msk] = dra[msk] - 360.0
            msk = (dra > 180.0) & msk_finite

    return dra
