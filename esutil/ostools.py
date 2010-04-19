"""
Module:
    ostools
Purpose:
    A set of tools for working with the operating system.

Classes:
    Class Name:
        DirStack
    Purpose:
        This is a directory simple stack that works like the
        directory stack in Unix shells.  See the documentation
        for the DirStack class for more details.

    Example:
        >>> ds=esutil.ostools.DirStack(verbose=True)
        >>> ds.push('~/data')
        ~/data ~
        >>> ds.push('/usr/bin')
        /usr/bin ~/data ~
        >>> ds.pop()
        ~/data ~
        >>> ds.pop()
        ~

Functions:
    See docs for the individual functions for more info.

    path_join(*paths):  
        Join path elements using the system path separator.  Any number of
        inputs can be given.  These must be strings or sequences.  This is
        similar to the os.path.join function but can join any number of path
        elements and supports sequences.

    getenv_check(environment variable name):
        Check for the envrionment variable and raise a RuntimeError if not
        found.  This differs from os.getenv() in that it raises a RuntimeError
        if the variable is not found instead of returning None.

    expand_path:
        Expand all user info such as ~userid and environment variables such as
        $SOMEVAR.  this simple uses a call to both os.path.expanduser and
        os.path.expandvars

"""
import os

import os
import sys
from sys import stdout

class DirStack(object):
    """
    Class:
        DirStack
    Purpose:
        A simple directory stack.

    Construction:
        ds=DirStack(verbose=False):  If verbose=True a message is
            printed for push and pop similar to that printed on 
            unix systems.
    Methods:
        push(directory): Change to the input directory.  Push the
            current working directory onto the stack.
        pop: Pop the last directory from the stack and change to
            that directory.

    Example:
        >>> ds=esutil.ostools.DirStack(verbose=True)
        >>> ds.push('~/data')
        ~/data ~
        >>> ds.push('/usr/bin')
        /usr/bin ~/data ~
        >>> ds.pop()
        ~/data ~
        >>> ds.pop()
        ~

    """
    def __init__(self, verbose=False):
        self.verbose=verbose
        self._home = os.path.expanduser('~')
        self._dirs = []

    def push(self, dir):
        """
        push(dir):  Change to the indicated dir and push the current
            working directory onto the stack
        """
        dir=os.path.expandvars(dir)
        dir=os.path.expanduser(dir)

        old_dir = os.getcwd()

        os.chdir(dir)
        
        # only do this *after* we successfully chdir
        self._dirs.append(old_dir)
        if self.verbose:
            self.print_stack()


    def pop(self):
        """
        pop(): Pop the last directory from the stack and change to
            that directory.
        """
        if len(self._dirs) == 0:
            stdout.write("Directory stack is empty\n")
            return

        dir = self._dirs.pop()
        os.chdir(dir)

        if self.verbose:
            self.print_stack()

    def getstack(self):
        """
        getstack(): Return the current stack.
        """
        return self._dirs

    def print_stack(self):
        self.print_dir(os.getcwd())
        for i in xrange(len(self._dirs)-1,-1,-1):
            d=self._dirs[i]
            self.print_dir(d)
        stdout.write('\n')
    def print_dir(self, dir):
        dir = dir.replace(self._home, '~')
        stdout.write('%s ' % dir)




def path_join(*paths):
    """
    Name:
        path_join
    Calling Sequence:
        path=path_join(any number of paths)

    Purpose:

        Join path elements using the system path separator.  Any number of
        inputs can be given.  These must be strings or sequences.  This is
        similar to the os.path.join function but can join any number of path
        elements and supports sequences.

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
    Name:
        getenv_check
    Calling Sequence:
        val = getenv_check(name)

    Purpose:
        Check for the envrionment variable and raise a RuntimeError if not
        found.  This differs from os.getenv() in that it raises a RuntimeError
        if the variable is not found.

    """
    val=os.getenv(name)
    if val is None:
        raise RuntimeError("Environment variable '%s' is not set" % name)
    return val


def expand_path(filename):
    """
    Name:
        expand_path
    Purpose:
        Expand all user info such as ~userid and environment variables such as
        $SOMEVAR.  this simple uses a call to both os.path.expanduser and
        os.path.expandvars
    Calling Sequence:
        fullpath = expand_path(path)

    """
    fname = os.path.expanduser(filename)
    fname = os.path.expandvars(fname)
    return fname

# synonym
expand_filename=expand_path

