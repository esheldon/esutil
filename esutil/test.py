import esutil

def test_histogram(nrand=None, fixed=False, use_weave=False, display=True):

    min=None
    max=None
    if fixed:
        data= \
            numpy.array([11, 62,  4, 69, 56, 
                         32, 80, 31, 19,  4, 
                         19, 81, 37, 89, 14, 
                         55,  2, 36, 81, 76])
        binsize=1
        min = -10
        max=85
    elif nrand is not None:
        data = numpy.random.normal(size=nrand)
        binsize=0.1
    else:
        data = 0.7*numpy.array(
            [0.01233105,0.1967937,0.56444893,0.57024105,0.95079757,
             0.80643415,0.36878545,0.5514516,0.87020452,0.27447491,
             0.30478068,0.10748157,0.12471763,0.45594969,0.80562055,
             0.62194449,0.38645207,0.54947444,0.56945579,0.44712302])
        binsize=0.05


    hist,rev = esutil.histogram(data, binsize=binsize, min=min, max=max,
                                rev=True,use_weave=use_weave)

    if display:
        print hist
        print rev[0:hist.size+1]

        for i in range(hist.size):
            if rev[i] != rev[i+1]:
                w = rev[rev[i]:rev[i+1]]
                w.sort()
                #print 'rev[%d]: %d rev[%d]: %d' % (i,rev[i],i+1,rev[i+1])
                print '\tbin:',i,' w:',w, 'vals:',data[w]
        print 'nbin =',hist.size


