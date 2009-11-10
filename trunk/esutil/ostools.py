import os

def path_join(*paths):
    """

    path=path_join(any number of paths)

    Join path elements using the system path separator.  Any number of inputs
    can be given.  These must be strings or sequences.  This is similar to the
    os.path.join function but can join any number of path elements and supports
    sequences.

    Examples:
        # Join three path elements
        p=path_join('/tmp', 'test', 'file.txt') # gives /tmp/test/file.txt


        # join a list of path elements
        p=path_join(['/tmp','file.txt']) # gives /tmp/file.txt
        
        # Join a path element with a list of path elements
        p=path_join('/tmp', ['test','file.txt']) # gives /tmp/test/file.txt
        p=path_join(['/tmp','test'], 'file.txt') # gives /tmp/test/file.txt

        # nested sequences.  Gives /tmp/test1/test2/file.txt
        p=path_join(['/tmp',['test1','test2']], 'file.txt') 
    """

    plist=[]
    for path in paths:
        # for py3k unicode will disappear
        if isinstance(path, str) or isinstance(path, unicode):
            plist.append(path)
        elif isinstance(path, list) or isinstance(path, tuple):
            for p in path:
                tpath = path_join(p)
                plist.append( tpath )
        else:
            raise ValueError('paths must be strings or sequences of strings')

    # We now have a list of strings.
    path = os.sep.join( plist )

    return path

def getenv_check(name):
    """
    getenv_check(name)

    Check for the envrionment variable and raise a RuntimeError if not
    found
    """
    val=os.getenv(name)
    if val is None:
        raise RuntimeError("Environment variable '%s' is not set" % name)
    return val


def expand_path(filename):
    """
    expand all user info such as ~userid and environment
    variables such as $SOMEVAR.
    """
    fname = os.path.expanduser(filename)
    fname = os.path.expandvars(fname)
    return fname

# synonym
expand_filename=expand_path

