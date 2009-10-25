# version info
# You need to run 'svn propset svn:keywords HeadURL' on the file and commit
# before this works.
#
# Don't edit these svn properties by hand

_property_headurl='$HeadURL$'

import sys

def version():
    from sys import stderr

    thisname='/esutil/__init__.py'
    badvers="NOTAG: unparseable"

    psplit=_property_headurl.split()
    if len(psplit) != 3:
        mess="headurl did not split into 3: '%s'\n" % _property_headurl
        stderr.write(mess)
        return badvers

    url=psplit[1]

    if url.find(thisname) == -1:
        mess="url '%s' does not contain string '%s'\n" % \
                (_property_headurl, thisname)
        stderr.write(mess)
        return badvers

    urlfront = url.replace(thisname, '')

    tag=urlfront.split('/')[-1]
    return tag

def get_python_version(numerical=False):
    if numerical:
        v=sys.version_info[0:3]
        pyvers=v[0] + 0.1*v[1] + 0.01*v[2]
    else:
        pyvers='v%s.%s.%s' % sys.version_info[0:3]
    return pyvers


# imports are the ony things that will have to be converted for py3k
# other than extension modules
import xmltools
import ostools
import misc
import json_util
import stat
import numpy_util
import oracle_util
import sfile
import io

import test
