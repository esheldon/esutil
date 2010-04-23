import sys

try:
    # works in python 2.5
    from xml.etree import cElementTree as ElementTree
except:
    try:
        # works in python 2.4
        import cElementTree as ElementTree
    except:
        stderr.write('Failed to import ElementTree')

# calling example
def testxml():
    configdict = xml2dict('config.xml')

    # you can access the data as a dictionary
    configdict['settings']['color'] = 'red'

    # or you can access it like object attributes
    configdict.settings.color = 'red'

    # just return the root
    # can also do dict2xml(configdict, filename_or_obj)
    root = dict2xml(configdict)

    # could also have written to a file with converter above
    tree = ElementTree.ElementTree(root)
    tree.write('config.new.xml')



class XmlDictObject(dict):
    """
    Adds object like functionality to the standard dictionary.
    """

    def __init__(self, initdict=None):
        if initdict is None:
            initdict = {}
        dict.__init__(self, initdict)
    
    def __getattr__(self, item):
        return self.__getitem__(item)
    
    def __setattr__(self, item, value):
        self.__setitem__(item, value)
    
    def __str__(self):
        if '_text' in self:
            return self.__getitem__('_text')
        else:
            return ''

    @staticmethod
    def Wrap(x):
        """
        Static method to wrap a dictionary recursively as an XmlDictObject
        """

        if isinstance(x, dict):
            return XmlDictObject((k, XmlDictObject.Wrap(v)) for (k, v) in list(x.items()))
        elif isinstance(x, list):
            return [XmlDictObject.Wrap(v) for v in x]
        else:
            return x

    @staticmethod
    def _UnWrap(x):
        if isinstance(x, dict):
            return dict((k, XmlDictObject._UnWrap(v)) for (k, v) in list(x.items()) )
        elif isinstance(x, list):
            return [XmlDictObject._UnWrap(v) for v in x]
        else:
            return x
        
    def UnWrap(self):
        """
        Recursively converts an XmlDictObject to a standard dictionary and returns the result.
        """

        return XmlDictObject._UnWrap(self)


def xml2dict(root, dictclass=XmlDictObject, seproot=False, noroot=False):
    """

    d=xml2dict(element or filename, dictclass=XmlDictObject, 
               noroot=False, seproot=False)

    Converts an XML file or ElementTree Element to a dictionary

    If noroot=True then the root tag is not included in the dictionary and
        xmldict[roottag] 
    is returned.

    If seproot=True then the root tag is not included in the dictionary, and
    instead the tuple 
        (xmldict[roottag], roottag) 
    is returned.  The name of the roottag is lost in this case.
    """

    if not have_element_tree:
        raise ImportError("Neither cElementTree or ElementTree could "
                          "be imported")

    # If a string is passed in, try to open it as a file
    if type(root) == type(''):
        root = ElementTree.parse(root).getroot()
    elif not isinstance(root, ElementTree.Element):
        raise TypeError('Expected ElementTree.Element or file path string')

    xmldict = dictclass({root.tag: _xml2dict_recurse(root, dictclass)})

    keys = list( xmldict.keys() )
    roottag = keys[0]
    if seproot:
        return xmldict[roottag], roottag
    elif noroot:
        return xmldict[roottag]
    else:
        return xmldict

def _xml2dict_recurse(node, dictclass):
    nodedict = dictclass()
    
    if len(list(node.items())) > 0:
        # if we have attributes, set them
        nodedict.update(dict( list(node.items()) ))
    
    for child in node:
        # recursively add the element's children
        newitem = _xml2dict_recurse(child, dictclass)
        if child.tag in nodedict:
            # found duplicate tag, force a list
            if type(nodedict[child.tag]) is type([]):
                # append to existing list
                nodedict[child.tag].append(newitem)
            else:
                # convert to list
                nodedict[child.tag] = [nodedict[child.tag], newitem]
        else:
            # only one, directly set the dictionary
            nodedict[child.tag] = newitem

    if node.text is None: 
        text = ''
    else: 
        text = node.text.strip()
    
    if len(nodedict) > 0:            
        # if we have a dictionary add the text as a dictionary value (if there is any)
        if len(text) > 0:
            nodedict['_text'] = text
    else:
        # if we don't have child nodes or attributes, just set the text
        nodedict = text
        
    return nodedict




def dict2xml(xmldict, filename_or_obj=None, roottag=None):
    """
    dict2xml(xmldict, [optional filename or obj], roottag=None)

    Converts a dictionary to an XML ElementTree Element and returns the
    result.  Optionally prints to file if input.

    If roottag is not sent, it is assumed that the dictionary is keyed by
    the root, and this roottag is gotten with roottag = xmldict.keys()[0]

    If roottag is sent, the root will be created with that name and the
    input dictionary will be placed under that tag.
    """

    if not have_element_tree:
        raise ImportError("Neither cElementTree or ElementTree could "
                          "be imported")

    if roottag is None:
        keys = list( xmldict.keys() )
        roottag = keys[0]
        root = ElementTree.Element(roottag)
        _dict2xml_recurse(root, xmldict[roottag])
    else:
        root = ElementTree.Element(roottag)
        _dict2xml_recurse(root, xmldict)

    xml_indent(root)
    # just return the xml if no file is given
    if filename_or_obj is not None:
        tree = ElementTree.ElementTree(root)
        tree.write(filename_or_obj)
    return root


def xml_indent(elem, level=0):
    """
    xml_indent(element, level=0)
    From http://infix.se/2007/02/06/gentlemen-indent-your-xml
    Input should be an element (perhaps root) of an element tree.
    e.g.
        tree = ElementTree.parse(somefile) 
        xml_indent(tree.getroot())
        tree.write(filename)
    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            xml_indent(e, level+1)
            if not e.tail or not e.tail.strip():
                e.tail = i + "  "
        if not e.tail or not e.tail.strip():
            e.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _dict2xml_recurse(parent, dictitem):
    assert type(dictitem) is not type([])

    if isinstance(dictitem, dict):
        for (tag, child) in list(dictitem.items() ):
            if str(tag) == '_text':
                parent.text = str(child)
            elif type(child) is type([]):
                # iterate through the array and convert
                for listchild in child:
                    elem = ElementTree.Element(tag)
                    parent.append(elem)
                    _dict2xml_recurse(elem, listchild)
            else:                
                elem = ElementTree.Element(tag)
                parent.append(elem)
                _dict2xml_recurse(elem, child)
    else:
        parent.text = str(dictitem)
    

       

