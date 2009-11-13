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

try:
    import numpy
    have_numpy=True
except:
    have_numpy=False


def table2array(dbfile, tablename):
    pass


def tabledef2dtype(table_info, size=None):
    """
    Take output of table_info() and convert to a numpy descriptor.

    Todo: 
        variable length columns
        Implement sizes keyword
    """
    dtype = []
    for info in table_info:
        name = str( info['name'].lower() )
        typ = coltype2numpy(info['type'])

        if typ == 'S?':
            raise ValueError("Dont' support variable length columns yet")

        dtype.append( (name, typ) )

    return dtype

_np2col={}
_np2col['i1'] = 'int8'

def numpy2coltype(type_description):
    """
    Convert a numpy data type name to an appropriate column type in sqlite.
    Note sqlite doesn't actualy use this information; it will store any
    value into any column.  This is to facilitate reading the data back
    into numpy arrays.

    The mapping:
        numpy                       column declaration
        ----------------------------------------------------------
        i1                          int8
        i2                          int16
        i4                          int32
        i8                          int64

        f4                          float32
        f8                          float64

        SN                          char(N)
    """

    tname = type_description.strip.lower()

    if tname[0] == 'S':
        try:
            lenstr = int( tname[1:] )
        except:
            raise ValueError("Could not extract length from %s" % tname)

    if tname not in ['i1','i2','i4','i8','f4','f8']

def coltype2numpy(typename, size=None):
    """

    Convert a column type declaration from sqlite to a numpy data type. 
    Determine sizes as best as possible from the declaration, or use the
    sizes= keyword to set sizes explicitly.

    We have to deal with som limitations.  Primarily:
        1) sqlite columsn have no data type.
        2) numpy fields are fixed length.

    In sqlite, columns don't actually have data types.  When data are inserted
    the type is determined at run time and storage is determined based on
    the value.

    So integers are stored in a variable length encoding.  Even if a column is
    declared i4 it can store 8 bytes.  The actual storage depends on the value
    of the data.  
    
    But if a column is declared to have specific size, we will honor this and
    assume the user knows what they are doing.  Since "integer" is how most
    people will declare all integer columns, we cannot be tempted to assign
    this 4 bytes.  Rather we will require explicit declaration of size.  We
    will honor tinyint,smallint,bigint as well but these are less clear.

    This goes for floating point too.  Internally, all floating point values
    are stored as 8 byte.  So we reqire an identifier such as f4 or f8, float
    or double.

    Note for text fields, which have no practical limit, a similar principle
    will hold.  If a field is simply declaed as "text" we must determine the
    size from the first row.  But if a size is specifically declared, we will
    honor it.  Note, character varying is treated like a text field.  Since
    n is meant to be a maximum size, this could potentially waste a lot of
    space, so we will guess from the first row.

        declared type           numpy conversion
        ----------------------------------------------------------
        text:                       S  size determined from the first result
        real:                       f8
        integer:                    i8

        f4, float32, float:         f4
        f8, float64, double:        f8

        i1, int8,  tinyint:         i1
        i2, int16, smallint:        i2
        i4, int32                   i4
        i8, int64, bigint:          i8

        char(n), character(n):      Sn
        varchar(n), character varying(n), character, char, string:
                                    S  size determined from the first result

    """
    typename = typename.strip().lower()
    if typename   in ['i1','int8','tinyint']:
        return 'i1'
    elif typename in ['i2','int16','smallint']:
        return 'i2'
    elif typename in ['i4','int32']:
        return 'i4'
    elif typename in ['i8','int64','bigint','int','integer']:
        return 'i8'
    elif typename in ['f4','float32','float']:
        return 'f4'
    elif typename in ['f8','float64','double','real']:
        return 'f8'
    else:
        # if size is sent, allow it to override any size indicators
        if size is not None:
            try:
                szint = int(size)
            except:
                raise ValueError("could not convert input size "
                                 "'%s' to int" % size)
            return 'S%d' % szint


        # no size given in name.  Treat as text.
        if typename in ['char','character','text','string']:
            return 'S?'


        # varchar and character varying only have max limits, so we will
        # treat them like a "text" field even if they have sizes declared.
        if typename.find('varying') != -1 or typename.find('varchar') != -1:
            return 'S?'


        # Try to infer the size
        left=typename.find('(')
        if left != -1:
            right = typename.find(')')
            if right != -1:
                substr = typename[left+1:right]

                try:
                    szint = int(substr)
                except:
                    raise ValueError("could not convert size "
                                     "indicator '%s' to int in" % typename)
                return 'S%d' % szint

        # if we get here something went wrong
        raise ValueError("Unable to convert type name: '%s'" % typename)





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

def add_index(dbfile_or_conn, tablename, columns, verbose=False):

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

    existing_db=False
    if os.path.exists(dbpath):
        if clobber:
            if verbose:
                stdout.write("Removing existing database file: %s\n" % dbpath)
            os.remove(dbpath)
        else:
            # it exists and we'll keep it
            if verbose:
                stdout.write("Using existing database file: %s\n" % dbpath)
            existing_db = True

    # open. This will create if new or open for updating if exists.
    conn = sqlite.connect(dbpath, isolation_level=None)

    curs=conn.cursor()

    # see if the table exists
    existing_table=False
    if existing_db:
        curs.execute("select name from sqlite_master where type='table' and name = '%s'" % tablename)
        res = curs.fetchall()
        if len(res) != 0:
            # this will be a new table in an existing database
            existing_table=True

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

class SqliteConnection(sqlite.Connection):
    """
    Inherits from the Connection class of sqlite and adds some functionality
    """

    def __init__(self, dbfile, **keys):
        sqlite.Connection.__init__(self, dbfile, **keys)
        self.row_factory = sqlite.Row

    def describe(self, tablename=None):
        """
        Describe the database or the table if tablename is given.

        currently does nothing if tablename is not sent.

        indices on tables not yet implemented.
        """

        if tablename is not None:
            info = self.table_info(tablename)

            stdout.write("%-15s %15s\n" % ("field","type"))
            stdout.write("-"*32+"\n")
            for row in info:
                stdout.write("%-15s %15s\n" % (row['name'],row['type']))


    def table_info(self, tablename):
        """
        Retrieve the table info.  Each row returned has fields:
            'cid': column id number
            'name': name of column
            'type': declared type. Ignored by sqlite but useful for converting
                to numpy.
            'notnull': If 1 can not be Null.
            'dflt_value': default value for column
            'pk': Do not know.
        """

        query="pragma table_info(%s)" % tablename

        curs = self.cursor()
        curs.execute(query)

        return curs.fetchall()

    def asarray(self, tablename, query=None):
        """
        Instead maybe override execute and have a asarray= keyword?
        """
        if not have_numpy:
            raise RuntimeError("numpy is not available")

        if query is None:
            query="select * from %s" % tablename


        tinfo = self.table_info(tablename)
        dtype = tabledef2dtype(tinfo)


        # when row factor is Row, the proper iteration is not supported
        # for use with fromiter.  Temporarily turn it off.
        row_factory_old = self.row_factory
        self.row_factory=None

        curs = self.cursor()
        curs.execute(query)
        
        res = numpy.fromiter(curs, dtype=dtype)
        self.row_factory = row_factory_old

        return res




def table_info(dbfile, tablename):
    """
    Convenience function
    """
    conn = SqliteConnection(dbfile)
    return conn.table_info(tablename)

def describe(dbfile, tablename=None):
    """
    Convenience function
    """
    conn = SqliteConnection(dbfile)
    conn.describe(tablename=tablename)

def asarray(dbfile, tablename, query=None):
    """
    Convenience function
    """
    conn = SqliteConnection(dbfile)
    return conn.asarray(tablename, query=query)

