def ptime(seconds, fobj=None, format='%s\n'):
    """
    ptime(seconds, fobj=None, format='%s\n')

    Print a pretty version of the input seconds.  
    
        fobj: A file object in which to write the result.
        format: The format for printing.  The default is '%s\n'

    Examples:
        import time
        tm1=time.time()
        ...do somethign
        tm2=time.time()
        ptime(tm2-tm1)

    """
    from sys import stdout

    min, sec = divmod(seconds, 60.0)
    hr, min = divmod(min, 60.0)
    days, hr = divmod(hr, 24.0)
    yrs,days = divmod(days, 365.0)

    if yrs > 0:
        tstr="%d years %d days %d hours %d min %f sec" % (yrs,days,hr,min,sec)
    elif days > 0:
        tstr="%d days %d hours %d min %f sec" % (days,hr,min,sec)
    elif hr > 0:
        tstr="%d hours %d min %f sec" % (hr,min,sec)
    elif min > 0:
        tstr="%d min %f sec" % (min,sec)
    else:
        tstr="%f sec" % sec

    if fobj is None:
        stdout.write(format % tstr)
    else:
        fobj.write(format % tstr)



