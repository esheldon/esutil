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

        areadiv = 4.0**self.depth()
        area = area0/areadiv*(180.0/pi)**2
        return area



    def match(self, ra1, dec1, ra2, dec2, radius,
              maxmatch=1, 
              htmid2=None, 
              htmrev2=None,
              minid=None,
              maxid=None,
              file=None,
              verbose=False):
        """
        Class:
            HTM

        Method Name:
            match
        
        Purpose:

          Match two sets of ra/dec points using the Hierarchical Triangular
          Mesh code.  This is very efficient for large search angles and large
          lists.  May seem slow otherwise due to overhead creating htm indices.
        
        
        Calling Sequence:
            import esutil
            depth = 10
            h=esutil.htm.HTM(depth)
            m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,radius,
                                maxmatch=1,
                                htmid2=None,
                                htmrev2=None,
                                minid=None,
                                maxid=None,
                                file=None)
        
        Inputs:
            ra1,dec1,ra2,dec2: 
                ra,dec lists in degrees.  Can be scalars or arrays but require
                len(ra) == len(dec) in each set.

          radius: 
              The search radius in degrees.  May be a scalar or an array same
              length as ra1,dec1.
        
        
        Keyword Parameters:

            maxmatch=1: 
                The maximum number of allowed matches per point. Defaults to
                return the closest match, maxmatch=1

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

            minid=None, maxid=None: 
                If htmrev2 is sent along with these, there is no need to
                calculate htmid2.

            file=None: 
                A file into which will be written the indices and distances.
                When this keyword is sent, None,None,None is returned. This is
                useful when the match data will not fit into memory.  
                
                The file is an unformatted binary file. It can be read with the
                read() method. 

                The format is a 64-bit signed integer representing the number
                of rows, followed by rows of 
                  
                  i1 i2 d12

                Where i1,i2 are the match indices as 64-bit signed integers and
                d12 is the distance between them in degrees as a 64-bit float.
        
        Outputs:

            m1,m2,d12: 

                A tuple of m1,m2,d12.  m1 and m2 are the match indices for
                list1 and list2.  d12 is the distance between them in degrees.

                You can subscript the arrays ra1,dec1 with the m1 array, and
                ra2,dec2 with the m2 array.   If you do so the data "line-up"
                so that points in list one and list two at the same index are
                matches.
                
                If you write the results to a file, the returned value is
                simply the match count.

        
        Restrictions:
            The C++ wrapper must be compiled.  This will happend automatically
            during installation of esutil.
        

         EXAMPLE:
        
            # try the matching two lists of ra/dec points
            # Matching by ra/dec, expect 10 matches ordered by distance....

            # match within two arcseconds
            two = 2.0/3600.

            # offset second list by fraction of 2 arcsec in dec
            # but last one won't match anything
            ra1 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 21.36])
            dec1 = numpy.array( [24.3,          24.3,            24.3,  -28.25, -15.32])
            ra2 = numpy.array(  [200.0, 200.0, 200.0, 175.23, 55.25])
            dec2 = numpy.array( [24.3+0.75*two, 24.3 + 0.25*two, 24.3 - 0.33*two, -28.25 + 0.58*two, 75.22])

            m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,two)

            for i in range(m1.size):
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


        
         MODIFICATION HISTORY:
            SWIG Wrapper and matching code working 2010-03-03, 
                Erin Sheldon, BNL.
            2010-03-19: Default to maxmatch=1, return the closest match.


        """

        if (len(ra1) != len(dec1)) or (len(ra2) != len(dec2)):
            raise ValueError("require len(ra)==len(dec) for "
                             "both sets of inputs")

        if htmid2 is None:
            if verbose:
                stdout.write("looking up ids\n");stdout.flush()
            htmid2 = self.lookup_id(ra2, dec2)
            minid = htmid2.min()
            maxid = htmid2.max()
        else:
            if minid is None:
                minid = htmid2.min()
            if maxid is None:
                maxid = htmid2.max()

        if htmrev2 is None:
            if verbose:
                stdout.write("Getting reverse indices\n");stdout.flush()
            hist2, htmrev2 = stat.histogram(htmid2-minid,rev=True)

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

    def read(self, filename, verbose=False):
        """
        Class:
            HTM

        Method Name:
            read

        Purpose:
            Read the binary file format written by the match() code.

        Calling Sequence:
            data = h.read(filename)

        Inputs:
            filename: A filename as a python string.

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

            h.match(ra1,dec1,ra2,dec2,radius,filename='some-path'

            data = h.read('some-path')


         MODIFICATION HISTORY:
            SWIG Wrapper and matching code working 2010-03-03, 
                Erin Sheldon, BNL.

        """

        fobj=open(filename,'r')

        nrows=numpy.fromfile(fobj,count=1,dtype='i8')

        if verbose:
            stdout.write("Reading %s rows from file: %s\n" % (nrows[0],filename))

        dtype=[('i1','i8'),('i2','i8'),('d12','f8')]

        data = numpy.fromfile(fobj, count=nrows[0], dtype=dtype)

        fobj.close()

        return data



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

            Count pairs number of pairs between two ra/dec lists as a function
            of their separation.  The binning is done bins equal spaced in the
            log10 of the separation.  By default the bin sizes are in degrees,
            unless the scale= keyword is sent, in which case the units are
            angle*scale with angle in radians.

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

