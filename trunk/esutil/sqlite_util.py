import os
import tempfile
import shutil
from sys import stdout

import esutil

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
        message="""
        Error: python data must be one of the following types:
            integer
            float
            string
            None"""
        raise ValueError(message)

def dict2coldefs(data, types=None, keys=None):
    """
    Create a table definition based on the entries in the input dict

    Get the types from the dict unless types= is sent
    """

    if keys is None:
        keys = list( data.keys() )

    coldefs=[]
    i=0
    for key in keys:
        if types is not None:
            typedef = types[i]
            i += 1
        else:
            typedef = py2sqlite(data[key])

        coldef = key+' '+typedef
        coldefs.append(coldef)

    return coldefs

def dict2tabledef(data, tablename, types=None, keys=None):
    coldefs = dict2coldefs(data, types=types, keys=keys)

    coldefs = ",\n            ".join(coldefs)
    tabledef = """
        create table %s (
            %s
        )
    \n""" % (tablename, coldefs)

    return tabledef


def dict2csv(data, filename, keys=None):
    """
    Write a sequence of dictionaries to a csv file
    """

    data = dict_ensurelist(data)

    if keys is None:
        keys = list(data[0].keys())

    import csv
    fobj=open(filename,'w')

    writer = csv.DictWriter(fobj, keys)
    writer.writerows(data)

    fobj.close()

def dict_ensurelist(data):
    errormess="Input data must be dict or sequence of dicts"
    if isinstance(data, list) or isinstance(data, tuple):
        if not isinstance(data[0], dict):
            raise ValueError(errormess)
    elif isinstance(data, dict):
        data = [data]
    else:
        raise ValueError(errormess)
    return data

def add_index(dbfile, tablename, columns, verbose=False):

    if not isinstance(columns,(list,tuple)):
        columns = [columns]

    dbpath=esutil.ostools.expand_path(dbfile)
    conn = sqlite.connect(dbpath, isolation_level=None)
    curs=conn.cursor()

    index_name = '_'.join(columns) + '_index'
    column_list = ','.join(columns)

    query="""
    create index if not exists %s on %s (%s)
    \n""" % (index_name, tablename, column_list)

    if verbose:
        stdout.write("Adding index: \n")
        stdout.write(query)

    curs.execute(query)

    conn.close()


def dict2table(data, dbfile, tablename, 
               keys=None, types=None,
               indices=None, 
               tmpdir='.', clobber=True, 
               cleanup=True,
               verbose=False):
    """

    Convert a dict or list of dicts to an sqlite table, creating the table if
    needed.  If the table exists, the data are appended.

    """
    
    # ensure we have a list of dicts
    data = dict_ensurelist(data)

    # check paths
    dbpath=os.path.expanduser(dbfile)
    dbpath=os.path.expandvars(dbpath)

    if verbose:
        stdout.write("database file: %s\n" % dbfile)

    existing_table=False
    if os.path.exists(dbpath):
        if clobber:
            if verbose:
                stdout.write("Removing existing database file: %s\n" % dbpath)
            os.remove(dbpath)
        else:
            # it exists and we'll keep it
            if verbose:
                stdout.write("Using existing database file: %s\n" % dbpath)
            existing_table = True

    # open. This will create if new or open for updating if exists.
    conn = sqlite.connect(dbpath, isolation_level=None)
    #conn.row_factory = sqlite.Row

    curs=conn.cursor()

    # see if the table exists
    if existing_table:
        curs.execute("select name from sqlite_master where type='table' and name = '%s'" % tablename)
        res = curs.fetchall()
        if len(res) == 0:
            # this will be a new table in an existing database
            existing_table=False

    if not existing_table:
        # Get the table definition
        tabledef = dict2tabledef(data[0], tablename, types=types, keys=keys)
        if verbose:
            stdout.write("Creating table '%s'\n" % tablename)
            stdout.write(tabledef)

        curs.execute(tabledef)
    else:
        if verbose:
            stdout.write("Appending to existing table '%s'\n" % tablename)

    conn.close()

    # write data to a temporary csv file and then import
    csvtmp = tempfile.mktemp(dir=tmpdir, prefix=tablename+'-', suffix='.csv')

    if verbose:
        stdout.write("Writing to temporary file: %s\n" % csvtmp)

    dict2csv(data, csvtmp, keys=keys)

    #  Now execute the import statement
    if verbose:
        stdout.write("Importing data\n")
    comm="""
        sqlite3 -separator ',' %s ".import %s %s" 
    """ % (dbpath, csvtmp, tablename)

    esutil.misc.exec_process(comm, verbose=verbose)

    if cleanup:
        if verbose:
            stdout.write("Cleaning up temporary file %s\n" % csvtmp)
        os.remove(csvtmp)

    if indices is not None:
        for index in indices:
            add_index(dbpath, tablename, index, verbose=verbose)
