# we don't import oracle_util by default as it probably will fail for 
# most people


# version info
# You need to run 'svn propset svn:keywords HeadURL' on the file and commit
# before this works.
#
# Don't edit these svn properties by hand

_property_headurl='$HeadURL$'

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


import xmltools
import ostools
import misc
import numpy_util


