import json

have_cjson=False
try:
    import cjson
    have_cjson=True
except:
    pass

def read(fname):
    """
    Read data from a json file.  If the faster cjson is available it is used,
    else json is imported.
    """
    fobj=open(fname)
    if have_cjson:
        data = cjson.decode(fobj.read())
    else:
        data = json.load(fobj)

    fobj.close()
    return data

def write(obj, fname, pretty=True):
    """
    Write the object to a json file.  
    
    By default the ordinary json is used since it supports pretty printing.
    If pretty=False and the faster cjson is available, it is used for printing.

    """

    fobj=open(fname,'w')
    
    if not pretty and have_cjson:
        jstring = cjson.encode(obj)
        fobj.write(jstring)
    else:
        json.dump(obj, fobj, indent=1, separators=(',', ':'))
    fobj.close()

