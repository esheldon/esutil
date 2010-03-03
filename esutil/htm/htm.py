import htmc
from esutil import stat

class HTM(htmc.HTMC):

    def match(self, ra1, dec1, ra2, dec2, radius, distance=False,
              maxmatch=0, 
              htmid2=None, 
              htmrev2=None,
              minid=None,
              maxid=None,
              file=None):

        if (len(ra1) != len(dec1)) or (len(ra2) != len(dec2)):
            raise ValueError("require len(ra)==len(dec) for "
                             "both sets of inputs")

        if htmid2 is None:
            htmid2 = self.lookup_id(ra2, dec2)
            minid = htmid2.min()
            maxid = htmid2.max()
        else:
            if minid is None:
                minid = htmid2.min()
            if maxid is None:
                maxid = htmid2.max()

        if htmrev2 is None:
            hist2, htmrev2 = stat.histogram(htmid2-minid,rev=True)

        return self.cmatch(radius,
                           ra1,
                           dec1,
                           ra2,
                           dec2,
                           htmrev2,
                           minid,
                           maxid,
                           maxmatch)
