"""
Miscellaneous functions that may be convenient:

center_text
  Print text centered in a field of a given width.

colprint
  Print sequences out in columnar format (or write to a text file).

dict_select
  Select a subset of keys from the input dict.

isstring
  Returns True if the input object is a string.

ptime
  Prints an input time in seconds in a more human-friendly format

"""

license="""
  Copyright (C) 2010  Erin Sheldon

    This program is free software; you can redistribute it and/or modify it
    under the terms of version 2 of the GNU General Public License as
    published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""

import os
from sys import stdout, stderr

try:
    import numpy
    have_numpy=True
except:
    have_numpy=False

import pydoc


def center_text(text, width):
    text = text.strip()
    space = width - len(text)
    return ' '*(space/2) + text + ' '*(space/2 + space%2)

def colprint(*args, **keys):
    """
    Name:
        colprint
    Purpose:
        print the input sequences or arrays in columns.  All must be the
        same length.
    Calling Sequence:
        colprint(var1, var2, ..., nlines=all, sep=' ', format=None, 
                 names=None, nformat=None, file=None, page=False)

    Inputs:
        A set of python objects.  Each must be a sequence or array and all must
        be the same length.

    Optional Inputs:
        nlines:  
            Number of lines to print.  Default is all.
        sep: 
            Separator, default is ' '
        file:  
            A file path or file object.  Default is to print to standard
            output. Ignored if paging.

        format: 
            A format string to apply to every argument.  E.g. format='%15s'
            Since every arg gets the same format, only %s type formats should
            be used unless the types are homogeneous.

        names: 
            A list of names for each argument.  There must be an entry for
            each argument. The names are printed above each column.
        nformat:
            A Format to apply to the names.  By default, the same format used
            for the arguments is tried.  If formatting fails, a simple '%s' is
            used for the names.

        page: If True, run the output through a pager.

    Revision History:
        Create: 2010-04-05, Erin Sheldon, BNL
    """
    nargs = len(args)
    if nargs == 0:
        return

    try:
        n1 = len(args[0])
    except:
        raise ValueError("Could not get len() of argument 1")

    # Should we print only a subset?
    nlines = keys.get('nlines',n1)
    if nlines is None:
        nlines = n1
    elif nlines > n1:
        nlines = n1

    # what separator should be used?
    sep = keys.get('sep',' ')

    # should we page the results?
    page = keys.get('page', False)

    if not page:
        # should we print to a file?
        f = keys.get('file', stdout)
        if isinstance(f, file):
            fobj = f
        else:
            f = os.path.expandvars(f)
            f = os.path.expanduser(f)
            fobj = open(f,'w')

    # make sure all the arguments are the same length.
    for i in range(nargs):
        try:
            l=len(args[i])
        except:
            raise ValueError("Could not get len() of argument %s" % (i+1))
        if l != n1:
            e="argument %s has non-matching length.  %s instead of %s" \
                    % (i+1, l, n1)
            raise ValueError(e)

    # if we are paging, we will store the lines, otherwise this won't be used
    lines = []

    # print a header
    names = keys.get('names',None)
    if names is not None:
        if isinstance(names, basestring):
            names = [names]
        nnames = len(names)
        if len(names) != nargs:
            raise ValueError("Expected %s names, got %s" % (nargs,nnames))
        
        # see if explicit format has been requested.
        nformat =keys.get('nformat',None)

        if nformat is not None:
            nformat = [nformat]*nnames
        else:
            # try to use the other format
            fmt=keys.get('format','%s')
            if fmt is None:
                fmt='%s'
            nformat=[fmt]*nnames

        nformat = sep.join(nformat)
        try:
            line = nformat % tuple(names)
        except:
            nformat = ['%s']*nnames
            nformat = sep.join(nformat)
            line = nformat % tuple(names)

        if page:
            lines.append(line)
        else:
            fobj.write(line)
            fobj.write('\n')


    # format for columns.  Same is used for all.
    format = keys.get('format','%s')
    if format is not None:
        format = [format]*nargs
    else:
        format = ['%s']*nargs

    format = sep.join(format)

    # loop over and print columns
    for i in range(nlines):
        data = []
        for iarg in range(nargs):
            data.append(args[iarg][i])
        
        data = tuple(data)

        line = format % data
        line = line.replace('\n','')

        if page:
            lines.append(line)
        else:
            fobj.write(line)
            fobj.write('\n')

    if page:
        lines = '\n'.join(lines)
        pydoc.pager(lines)
    else:
        # close if this is not stdout
        if fobj != stdout:
            fobj.close()


def ptime(seconds, fobj=None, format='%s\n'):
    """
    Name:
        ptime(seconds, fobj=None, format='%s\n')
    Purpose:
        Print a pretty version of the input seconds.  
    Calling Sequence:
        ptime(seconds, fobj=None, format='%s\n')
    
    Inputs:
        Time in seconds.

    Optional Inputs:
        fobj: A file object in which to write the result.
        format: The format for printing.  The default is '%s\n'

    Examples:
        import time
        tm1=time.time()
        ...do somethign
        tm2=time.time()
        ptime(tm2-tm1)

        5 min 23.210000 sec
    """

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



def dict_select(input_dict, keep=None, remove=None):
    """
    Name:
        dict_select
    Purpose:
        Select a subset of keys from the input dict.

    Calling Sequence:
        newdict = dict_select(input_dict, keep=all, remove=[])

    Inputs:
        dict: the input dictionary.

    Optional Inputs:
        keep=None: 
            A list of keys to keep. If the input is None or [] all keys are
            returned that are not in the remove list.  Default [].

        remove=None: 
            A list of keys to ignore.  Defaults to [].

    """

    outdict={}

    if keep is None:
        keep = []
    if remove is None:
        remove = []

    if len(keep) == 0:
        # wrap in list() for py3k in which keys() does not return a list.
        keep = list( input_dict.keys() )

    for key in keep:
        if key in input_dict and key not in remove:
            outdict[key] = input_dict[key]

    return outdict



def isstring(obj):
    if have_numpy:
        string_types=(basestring, numpy.string_)
    else:
        string_types=basestring

    if isinstance(obj, string_types):
        return True
    else:
        return False


