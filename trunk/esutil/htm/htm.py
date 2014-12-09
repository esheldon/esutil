"""

Please consult the docs for the main htm package.  For example, in IPython:

>>> import esutil
>>> esutil.htm?


"""
import htmc
from esutil import stat
import numpy
from sys import stdout

class HTM(htmc.HTMC):

    def get_depth(self):
        """
        get the depth of the HTM tree
        """
        return super(HTM,self).depth()

    def area(self):
        """

        Get the mean area of triangles at the current depth. The units are
        square degrees.

        >>> import esutil
        >>> h=esutil.htm.HTM(10)
        >>> h.area()
        0.0049177362024091812

        """
        pi=numpy.pi
        area0=4.0*pi/8.0

        areadiv = 4.0**self.get_depth()
        area = area0/areadiv*(180.0/pi)**2
        return area

    def intersect(self, ra, dec, radius, inclusive=True):
        """
        look up all triangles that are contained within or intersect a circle
        centered on the input point.

        parameters
        ----------
        ra: float
            RA of central point in degrees
        dec: float
            DEC of central point in degrees
        radius: float
            radius of circle in degrees
        inclusive: bool, optional
            If False, only include triangles fully enclosed within the circle.
            If True, include those that intersect as well.  Default True.
        """
        if inclusive:
            inc=1
        else:
            inc=0

        return super(HTM,self).intersect(ra, dec, radius, inc)

    def match(self, ra1, dec1, ra2, dec2, radius,
              maxmatch=1, 
              htmid2=None, 
              htmrev2=None,
              minid=None,
              maxid=None,
              file=None,
              verbose=False):
        """
        Match two sets of ra/dec points using the Hierarchical Triangular
        Mesh code.
          
        This is very efficient for large search angles and large lists.

        This method is a simple wrapper around the Matcher class

        If you need to match against the same points many times, use
        a htm.Matcher object.  don't use the old htmrev2 method.
        
        parameters
        ----------
        ra1: array or scalar
        dec1: array or scalar
        ra2: array or scalar
        dec2:  array or scalar
            ra,dec in degrees.  Can be scalars or arrays but require
            size(ra) == size(dec) in each set.

        radius: 
            The search radius in degrees.  May be a scalar or an array same
            length as ra1,dec1.

        maxmatch: integer, optional 
            The maximum number of allowed matches per point. Defaults to return
            the closest match, maxmatch=1.  Use maxmatch<=0 to return all
            matches

        file: string, optional
            A file into which will be written the indices and distances.
            When this keyword is sent, None,None,None is returned. This is
            useful when the match data will not fit into memory.  
            
            The file is in text format of the form
                i1 i2 d12
            Where i1,i2 are the match indices and d12 is the distance between
            them in degrees

            The file can be read using the read() method.

        returns
        -------
            m1,m2,d12: 

                A tuple of m1,m2,d12.  m1 and m2 are the match indices for
                list1 and list2.  d12 is the distance between them in degrees.

                You can subscript the arrays ra1,dec1 with the m1 array, and
                ra2,dec2 with the m2 array.   If you do so the data "line-up"
                so that points in list one and list two at the same index are
                matches.
                
                If you write the results to a file, the returned value is
                simply the match count.

        examples
        --------
        
        # try the matching two lists of ra/dec points
        # Matching by ra/dec, expect 10 matches ordered by distance....

        # match within two arcseconds
        two = 2.0/3600.

        # offset second list by fraction of 2 arcsec in dec
        # but last one won't match anything
        ra1 = [200.0, 200.0, 200.0, 175.23, 21.36]
        dec1 = [24.3,  24.3,  24.3,  -28.25, -15.32]
        ra2 = [200.0, 200.0, 200.0, 175.23, 55.25]
        dec2 = [24.3+0.75*two, 24.3 + 0.25*two, 24.3 - 0.33*two, -28.25 + 0.58*two, 75.22]

        m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,two,maxmatch=0)

        for i in xrange(m1.size):
            print m1[i],m2[i],d12[i]

        # this produces
        0 1 0.00013888984367
        0 2 0.00018333285694
        0 0 0.000416666032158
        1 1 0.00013888984367
        1 2 0.00018333285694
        1 0 0.000416666032158
        2 1 0.00013888984367
        2 2 0.00018333285694
        2 0 0.000416666032158
        3 3 0.000322221232243

        """

        ra1=numpy.array(ra1, dtype='f8', ndmin=1, copy=False)
        dec1=numpy.array(dec1, dtype='f8', ndmin=1, copy=False)
        ra2=numpy.array(ra2, dtype='f8', ndmin=1, copy=False)
        dec2=numpy.array(dec2, dtype='f8', ndmin=1, copy=False)
        radius=numpy.array(radius, dtype='f8', ndmin=1, copy=False)

        if (ra1.size != dec1.size
                or ra2.size != ra2.size):
            stup=(ra1.size,dec1.size,ra2.size,dec2.size)
            raise ValueError("ra1 must equal dec1 in size "
                             "and ra2 must equal dec2 in size, "
                             "got %d,%d and %d,%d" % stup)

        if radius.size != 1 and radius.size != ra2.size:
            raise ValueError("radius size (%d) != 1 and"
                             " != ra2,dec2 size (%d)" % (radius.size,ra2.size))

        file=check_filename(file)

        if htmrev2 is None:
            # new way using a Matcher
            depth=self.get_depth()
            matcher=Matcher(depth, ra2, dec2)
            return matcher.match(ra1,
                                 dec1,
                                 radius,
                                 maxmatch=maxmatch,
                                 file=file)

        else:
            # deprecated way
            stdout.write("you are using the old reverse indices style, "
                         "this will work but is deprecated\n")

            if minid is None:
                minid = htmid2.min()
            if maxid is None:
                maxid = htmid2.max()
            if verbose:
                stdout.write("calling cmatch\n");stdout.flush()
            return self.cmatch(radius,
                               ra1,
                               dec1,
                               ra2,
                               dec2,
                               htmrev2,
                               minid,
                               maxid,
                               maxmatch,
                               file)



    def match_prepare(self, ra, dec, verbose=False):
        """
        deprecated.  Use an htm.Matcher instead
        """

        print 'deprecated: use a htm.Matcher instead'

        if verbose:
            stdout.write("looking up ids\n")

        htmid = self.lookup_id(ra, dec)
        minid = htmid.min()
        maxid = htmid.max()

        if verbose:
            stdout.write("Getting reverse indices\n");stdout.flush()
        hist, htmrev = stat.histogram(htmid-minid,rev=True)

        return htmrev, minid, maxid



    def cylmatch(self, ra1, dec1, z1, ra2, dec2, z2, 
                 radius, dz,  
                 maxmatch = 50, 
                 unique=False, nkeep=1, 
                 **kw):

        '''
        Class:
           HTM

        Method Name:
           cylmatch

        Purpose:

            Perform cylindrical RA-Dec matching of two catalogs (called cat1
            and cat2 in this document) by finding the nearest N neighbors
            within a fixed search aperture and within a fixed window in some
            arbitrary third parameter (called z for the purposes of this
            document).  The M closest neighbors in the z direction are returned
            (with M <= N)

        Syntax:

            matchind, adist, zdist = cylmatch(ra1, dec1, z1, ra2, dec2, z2,
                                              radius., dz, 
                                              maxmatch = 50, 
                                              unique=False, nkeep=1, 
                                              **kw)

        Inputs:

            ra1, dec1, z1: RA, Dec and z values for the first catalog
            ra2, dec2, z2: As above for the second catalog.

          radius: angular radius of search aperture in degrees.  
                  Can either be a scalar or an array of values the 
                  same length as cat1.

          dz: half-length of the search cylinder.  Can either be a scalar or
              an array of values the same length as cat1.


        Keywords:

            maxmatch: 
                Maximum number of neighbors to find within the
                search radius.  Note that this maximum is applied to
                the *total* number of matches within the search
                aperture, before applying the cut in the z parameter.
                Therefore, one wants this to be something reasonably
                large (much larger than nkeep) to ensure that matches
                within the dz cut are included.  However, larger
                values of magmatch may create memory issues for very
                large catalogs.  Default value: 10.

            nkeep: 
                Number of matches to keep (and return) for each object in cat1
                (M in the summary description above). If the number of matches
                is less than nkeep then the rest of the  output arrays will be
                filled with the bad value -999.  nkeep is automatically set to
                1 and ignored if unique = True. Default value: 1.

            radius: 
                angular radius of search aperture in degrees.  Can either be a 
                scalar or an array of values the same length as cat1.
            dz: 
                half-length of the search cylinder.  Can either be a scalar or
                an array of values the same length as cat1.

            unique: 
                if this is True, the matching is done uniquely--i.e., members
                of catalog 2 are excluded from future matching once they are
                matched to something in catalog 1 (the matching proceeds by
                stepping through catalog 1 in the order in which it is passed
                to cylmatch).



        **kw: keyword arguments passed through to htm.match. 


        Returns:

        matchind: 
            An LIST of arrays containing indices of the matches in cat2 for
            each element of cat1, with a maximum of nkeep matches returned per
            element.

        adist: 
            angular distance to each of these matches

        zdist: 
            distance to each of these matches in the z dimension 
            (catalog 1 minus catalog 2).  


        Revision History:

        Written by Brian F. Gerke at SLAC in May-June 2010.
        Added to HTM class in July 2010.

        '''

        npts = numpy.size(ra1)
        npts2 = numpy.size(ra2)
        #check input
        if ((numpy.size(dec1) != npts) | (numpy.size(dec2) != npts2) | \
                (numpy.size(z1) != npts) | (numpy.size(z2) != npts2)):
            print npts, numpy.size(ra1), numpy.size(dec1), numpy.size(z1)
            print npts2, numpy.size(ra2), numpy.size(dec2), numpy.size(z2)
            raise ValueError('RA Dec and z input arrays to cylmatch must' 
                             ' have the same length for each catalog.')
        if (numpy.size(dz) > 1) & (numpy.size(dz) < npts):
            raise ValueError('dz must either be a scalar or have the '
                             'same length as the input arrays in cylmatch.')

        if unique: 
            nkeep = 1


        
        #Match up catalogs on the sky.
        m1, m2, d12 = self.match(ra1, dec1, 
                                 ra2, dec2, radius,
                                 maxmatch = maxmatch, **kw)
        

        #Now limit to matches that are within +/- dz of each object

        if numpy.size(dz) == 1:

            w, = numpy.where((z2[m2] > (z1[m1] - dz)) &
                          (z2[m2] < (z1[m1] + dz)))

        else:

            w, = numpy.where((z2[m2] > (z1[m1] - dz[m1])) &
                          (z2[m2] < (z1[m1] + dz[m1])))


        m1=m1[w]
        m2=m2[w]
        d12=d12[w]

        #Now in the case of multiple matches, take the one with the minimum
        #difference in z.

        matchindex = []
        angdist = []
        zdist = []

        flag_matched = numpy.zeros(numpy.size(ra2), dtype='i4') #for ensuring unique matching

        j1=numpy.searchsorted(m1,numpy.arange(npts),'left') #j1 and j2 are the start and end indices for each unique value of m1
        j2=numpy.searchsorted(m1,numpy.arange(npts),'right')

        for i in range(npts):

            if j1[i] == j2[i]: 

                #First, check to see if the ith object got a match at all...
                #If so, check and see if the best match has already been used
                if (j1[i] != i) or ((flag_matched[m2[j1[i]]] == 1) and unique):
                    matchis = numpy.array([],dtype='i4')
                    angdists = numpy.array([])
                    zdiff = numpy.array([])
                else:
                    #if there's a good match, save it.
                    matchis = numpy.array(m2[[j1[i]]])
                    angdists = numpy.array(d12[[j1[i]]])
                    zdiff = numpy.array([(z1[m1[j1[i]]] - 
                                       z2[m2[j1[i]]])])
                    flag_matched[m2[j1[i]]] = 1
            else:
                #compute difference in z-direction for the different matches.
                zdiff = z1[m1[j1[i]:j2[i]]] - \
                        z2[m2[j1[i]:j2[i]]]

                angdists = d12[j1[i]:j2[i]]
                matchis = m2[j1[i]:j2[i]]

                #Remove objects that have already been used
                if unique: 
                    zdiff = zdiff[where(flag_matched[m2[j1[i]:j2[i]]] == 0)]

                #If none remains, there's no match
                if len(zdiff) > 0:
                    # sort matches by absolute distance in z direction
                    isort = (numpy.abs(zdiff)).argsort()
                    angdists = angdists[isort]
                    zdiff = zdiff[isort]
                    matchis = matchis[isort]
                    flag_matched[m2[j1[i]+isort[0]]] = 1




            matchindex.append(matchis[0:nkeep])
            angdist.append(angdists[0:nkeep])
            zdist.append(zdiff[0:nkeep])




        return(matchindex, angdist, zdist)
    

    def read(self, filename, verbose=False):
        """
        read pair info from a file written by match()

        parameters
        ----------
        filename: string
            the file name
        verbose: bool, optional
            print some info

        returns
        -------

        A structured array with fields
            'i1': The index of matches into list 1
            'i2': The index of matches into list 2
            'd12': The distance between the matched points
                in degrees.

        These are equivalent to m1,m2,d12 returned by the
            match() program when no file is sent.
        """

        return read_pairs(filename, verbose=verbose)

    def bincount(self,
                 rmin, rmax, nbin, ra1, dec1, ra2, dec2, scale=None,
                 htmid2=None, 
                 htmrev2=None,
                 minid=None,
                 maxid=None,
                 getbins=True):
        """
        Class:
            HTM

        Method Name:
            bincount 
        
        Purpose:

            Count number of pairs between two ra/dec lists as a function of
            their separation.  The binning is equal spaced in the log10 of the
            separation.  By default the bin sizes are in degrees, unless the
            scale= keyword is sent, in which case the units are angle*scale
            with angle in radians.

            This code can be used to calculate correlation functions by
            calling it on the data as well as random points.
        
        
        Calling Sequence:
            import esutil
            depth = 10
            h=esutil.htm.HTM(depth)
            rlower, rupper, counts = h.bincount(
                 rmin, rmax, nbin, ra1, dec1, ra2, dec2, 
                 scale=None,
                 htmid2=None, 
                 htmrev2=None,
                 minid=None,
                 maxid=None,
                 getbins=True)

        Inputs:
            rmin,rmax: Smallest and largest separations to consider.  This
                is in degrees unless the scale= keyword is sent, in which
                case the units are angle*scale with angle in radians.
            nbin:  The number of bins to use.  Bins will be equally spaced
                in the log10 of the separation.
            ra1,dec1,ra2,dec2: 
                ra,dec lists in degrees.  Can be scalars or arrays but require
                len(ra) == len(dec) in each set.
        
        Keyword Parameters:

            scale:  
                A scale to apply to the angular separations.  Must be the same
                length as ra1/dec1 or a scalar.  This is useful for converting
                angle to physical distance.  For example, scale could be the
                angular diameter distance to cosmological objects in list 1.

                If scale is sent, rmin,rmax must be in units of angle*scale
                where angle is in *radians*, as opposed to degrees when scale
                is not sent.

            htmid2=None: 
                the htm indexes for the second list.  If not sent they are
                generated internally.  You can generate these with 

                    htmid = h.lookup_id(ra, dec)

            htmrev2=None: 
                The result of
                    import esutil
                    htmid2 = h.lookup_id(ra, dec)
                    minid=htmid2.min()
                    hist2,htmrev2=\\
                        esutil.stat.histogram(htmid2-minid,rev=True) 

                If not sent it is calculated internally for fast lookups.  You
                can save time on successive calls by generating these your
                self.

            getbins: 
                If True, return a tuple 
                    rlower,rupper,counts 

                instead of just counts.  rlower,rupper are the lower and upper
                limits of each bin.  getbins=True is the default.
        
        Outputs:

            if getbins=True:
                rlower,rupper,counts:  rlower,rupper are the lower
                and upper limits of each bin.  getbins=True is the default.
            if getbins=False:
                counts:  The pair counts in equally spaced logarithmic bins
                    in separation.


        
        Restrictions:
            The C++ wrapper must be compiled.  This will happend automatically
            during installation of esutil.
        

         EXAMPLE:
            import esutil

            # simple angular counts, no scaling
            # cross correlate with second catalog
            h=esutil.htm.HTM()
            rmin=10/3600. # degrees
            rmax=1000/3600. # degrees
            nbin=25
            rlower,rupper,counts = h.bincount(rmin,rmax,nbin,
                                              cat1['ra'],cat1['dec'],
                                              cat2['ra'],cat2['dec'])



            # counts using scaling of the angular separations with
            # the angular diameter distance to get projected
            # physical separations.
            c=esutil.cosmology.Cosmo()

            # get angular diameter distance to catalog 1 objects
            DA=c.Da(0.0, cat1['z'])

            # cross correlate with second catalog
            h=esutil.htm.HTM()
            rmin=0.025 # Mpc
            rmax=30.0 # Mpc
            nbin=25
            rlower,rupper,counts = h.bincount(rmin,rmax,nbin,
                                              cat1['ra'],cat1['dec'],
                                              cat2['ra'],cat2['dec'],
                                              scale=DA)
        
         MODIFICATION HISTORY:
             Created:  2010-03-31, Erin Sheldon, BNL


        """


        if htmid2 is None:
            stdout.write("Generating HTM ids\n")
            htmid2 = self.lookup_id(ra2, dec2)
            minid = htmid2.min()
            maxid = htmid2.max()
        else:
            if minid is None:
                minid = htmid2.min()
            if maxid is None:
                maxid = htmid2.max()

        if htmrev2 is None:
            stdout.write("Generating reverse indices\n")
            hist2, htmrev2 = stat.histogram(htmid2-minid,rev=True)

        counts = self.cbincount(rmin,rmax,nbin,ra1,dec1,ra2,dec2,
                                htmrev2,minid,maxid,scale)
        if getbins:
            lower,upper = log_bins(rmin, rmax, nbin)
            return lower,upper,counts
        else:
            return counts

class Matcher(htmc.Matcher):
    """
    Object to match arrays of ra,dec

    The object is initialized with a set of ra,dec and can
    then be matched to other sets

    parameters
    ----------
    depth: int
        Depth for HTM tree.
    ra: scalar or array
        right ascension in degrees
    dec: scalar or array
        declination in degrees
    """
    def __init__(self, depth, ra, dec):

        ra=numpy.array(ra, dtype='f8', ndmin=1, copy=False)
        dec=numpy.array(dec, dtype='f8', ndmin=1, copy=False)

        if ra.size != dec.size:
            raise ValueError("ra size (%d) != "
                             "dec size (%d)" % (ra.size, dec.size))

        super(Matcher, self).__init__(depth, ra, dec)

    def get_depth(self):
        """
        get the depth of the HTM tree
        """
        return super(Matcher,self).get_depth()
    depth=get_depth

    def match(self, ra, dec, radius, maxmatch=1, file=None):
        """
        match to the input set of ra,dec points

        ra: scalar or array
            right ascension in degrees to match against
        dec: scalar or array in degrees to match against
            declination
        radius: scalar or array
            search radius in degrees.  Can be a scalar or an array the
            same size as ra,dec
        maxmatch: int, optional
            Maximum number of matches to return per point, default 1.  Set
            maxmatch <= 0 to return all matches
        file: string, optional
            If sent, write pairs to the file instead of returning the pair
            data.  This can use much less memory for large match sets.
            The file is in text format of the form
                i1 i2 d12
            Where i1,i2 are the match indices and d12 is the distance between
            them in degrees

            The file can be read using the read() method.

        returns
        -------
        If file= is not sent, a tuple (m1, m2, d):
            m1:
                The match indices for the input ra,dec
            m2:
                The match indices for the internal ra,dec of
                the Matcher object
            d:
                Distance between the pairs in degrees

        if file= is sent then then number of matches is returned.
        """

        ra=numpy.array(ra, dtype='f8', ndmin=1, copy=False)
        dec=numpy.array(dec, dtype='f8', ndmin=1, copy=False)
        radius=numpy.array(radius, dtype='f8', ndmin=1, copy=False)

        if ra.size != dec.size:
            raise ValueError("ra size (%d) != "
                             "dec size (%d)" % (ra.size, dec.size))

        if radius.size != 1 and radius.size != ra.size:
            raise ValueError("radius size (%d) != 1 and"
                             " != ra,dec size (%d)" % (radius.size,ra.size))

        file=check_filename(file)
        return super(Matcher, self).match(ra, dec, radius, maxmatch, file)

def read_pairs(filename, verbose=False):
    """
    Read the pair info written by the match code

    parameters
    -----------
    filename: string
        filename holding the pair data
    verbose: bool, optional
        print what is happening
    returns
    -------
    Outputs:
        A structured array with fields
            'i1': The index of matches into list 1
            'i2': The index of matches into list 2
            'd12': The distance between the matched points
                in degrees.

        These are equivalent to m1,m2,d12 returned by the
        match() program when no file is sent.

    Example:
        import esutil
        h=esutil.htm.HTM(depth)

        h.match(ra1,dec1,ra2,dec2,radius,filename='some-path')

        data = esutil.htm.read_pairs('some-path')
    """

    import esutil as eu
    from esutil.recfile import Recfile

    dtype=[('i1','i8'),('i2','i8'),('d12','f8')]

    if verbose:
        stdout.write("Reading pairs from file: %s\n" % filename)

    filename=check_filename(filename)
    with Recfile(filename, "r", dtype=dtype, delim=' ') as robj:
        data=robj.read()

    if verbose:
        stdout.write("    read %d pairs\n" % data.size)

    return data


def gmean(r1, r2, dim):
    e1 = dim + 1
    e2 = dim

    frac = (1.0*e2)/e1
    gm = frac*( r1**e1 - r2**e1)/( r1**e2 - r2**e2)
    return gm

def log_bins(rmin, rmax, nbin):
    log_rmin = numpy.log10(rmin)
    log_rmax = numpy.log10(rmax)
    log_binsize = (log_rmax-log_rmin)/nbin

    log_lower_edges = log_rmin + log_binsize*numpy.arange(nbin)
    log_upper_edges = log_lower_edges + log_binsize
    #print 'log_lower_edges:',log_lower_edges
    #print 'log_upper_edges:',log_upper_edges

    lower_edges = 10**log_lower_edges
    upper_edges = 10**log_upper_edges
    #print 'lower_edges:',lower_edges
    #print 'upper_edges:',upper_edges

    #gm = gmean(upper_edges, lower_edges, 2)
    #print 'gmean: ',gm
    return lower_edges, upper_edges

def check_filename(filename):
    if filename is not None:
        if isinstance(filename,unicode):
            print 'htm: warning: filename is unicode, converting to string"
            filename=str(filename)

    return filename
