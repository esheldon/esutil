"""

Some convenience functions for working with json files.  

    read(file):  
        
        Read from the file name or opened file object. If the faster cjson is
        available, an attempt to use it is made.  If this fails (cjson is known
        to fail in certain corner cases) ordinary json is tried.  

    write(obj, file, pretty=True):

        Write the object to a fson file.  The "file" input can be either a file
        name or opened file object.  Ordinary json is the default since it
        supports human readable writing.  Sending pretty=False to the write
        program will force use of cjson if it is available.


"""

try:
    import json
    have_json=True
except:
    have_json=False

try:
    import cjson
    have_cjson=True
except:
    have_cjson=False

def read(fname):
    """
    obj = json_util.read(file):  

    Read from the file name or opened file object. If the faster cjson is
    available, an attempt to use it is made.  If this fails (cjson is known
    to fail in certain corner cases) ordinary json is tried.  
    """

    if not have_json and not have_cjson:
        raise ImportError("Neither cjson or json could be imported")

    input_fileobj=False
    if isinstance(fname, file):
        input_fileobj=True
        fobj=fname
    else:
        fobj=open(fname)

    if have_cjson:
        try:
            data = cjson.decode(fobj.read())
        except: 
            # fall back to using json
            fobj.seek(0)
            data = json.load(fobj)
    else:
        data = json.load(fobj)

    if not input_fileobj:
        fobj.close()
    return data

def write(obj, fname, pretty=True):
    """
    json_util.write(obj, fname, pretty=True)

    Write the object to a fson file.  The "file" input can be either a file
    name or opened file object.  Ordinary json is the default since it
    supports human readable writing.  Sending pretty=False to the write
    program will force use of cjson if it is available.
    """

    if not have_json and not have_cjson:
        raise ImportError("Neither cjson or json could be imported")

    input_fileobj=False
    if isinstance(fname,file):
        input_fileobj=True
        fobj=fname
    else:
        fobj=open(fname,'w')
    
    if not pretty and have_cjson:
        jstring = cjson.encode(obj)
        fobj.write(jstring)
    else:
        json.dump(obj, fobj, indent=1, separators=(',', ':'))

    if not input_fileobj:
        fobj.close()

