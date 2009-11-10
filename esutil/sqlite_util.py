try:
    import sqlite3 as sqlite
    have_sqlite=True
except:
    have_sqlite=False




# dict sqlite tools
def py2sqlite(data):
    """
    Return an sqlite type name based on the type of the input data

    Can try to support numpy later
    """

    # will have to be converted to py3k
    int_types = (int, long)
    text_types = (str, unicode)

    if isinstance(data, int_types):
        return 'integer'
    elif isinstance(data, float):
        return 'real'
    elif isinstance(data, (str,unicode)):
        return 'text'
    else:
        message="Error: python data must be one of the following types:\n"
                "    integer\n"
                "    float\n"
                "    string\n"
                "    None\n"
        raise ValueError(message)

def dict2coldefs(data, types=None):
    """
    Create a table definition based on the entries in the input dict

    Get the types from the dict unless types= is sent
    """

    coldefs=[]
    i=0
    for key in data:
        if types is not None:
            typedef = types[i]
            i += 1
        else:
            typedef = py2sqlite(data[key])

        coldef = key+' '+typedef
        coldefs.append(coldef)

    return coldefs

def dict2tabledef(data, tablename, types=None):
    coldefs = dict2coldefs(data, types=types)

    tabledef = "create table %s (" + ",".join(coldefs)+")" % tablename

    return tabledef

def dict2table(data, dbfile, tablename, indices=None, types=None):
    """

    Convert a dict or list of dicts to an sqlite table, creating the table if
    needed.  If the table exists, the data are appended.

    Note appending is not yet supported.

    """

    
    dbpath=os.path.expanduser(dbfile)
    dbpath=os.path.expandvars(dbpath)
    
    if os.path.exists(dbpath):
        os.remove(dbpath)

    conn = sqlite.connect(dbpath)
    conn.row_factory = sqlite.Row

    tabledef = dict2tabledef(data, types=types)
    conn.execute(tabledef)
