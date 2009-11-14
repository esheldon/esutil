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


def tabledef2dtype(table_info, columns=None, size=None):
    """
    Take output of table_info() and convert to a numpy descriptor.

    Todo: 
        variable length columns
        Implement sizes keyword
    """
    dtype = []
    for info in table_info:
        name = str( info['name'].lower() )
        
        keepcol=True
        if columns is not None:
            if name not in columns:
                keepcol=False
        
        if keepcol:
            typ = coltype2numpy(info['type'])

            if typ == 'S?':
                raise ValueError("Dont' support variable length columns yet")

            dtype.append( (name, typ) )

    return dtype

_np2col={}

_np2col['i1'] = 'int8'
_np2col['int8'] = 'int8'
_np2col['i2'] = 'int16'
_np2col['int16'] = 'int16'
_np2col['i4'] = 'int32'
_np2col['int32'] = 'int32'
_np2col['i8'] = 'int64'
_np2col['int64'] = 'int64'
_np2col['f4'] = 'float32'
_np2col['float32'] = 'float32'
_np2col['f8'] = 'float64'
_np2col['float64'] = 'float64'

_np2col['i1'] = 'integer'
_np2col['int8'] = 'integer'
_np2col['i2'] = 'integer'
_np2col['int16'] = 'integer'
_np2col['i4'] = 'integer'
_np2col['int32'] = 'integer'
_np2col['i8'] = 'integer'
_np2col['int64'] = 'integer'
_np2col['f4'] = 'real'
_np2col['float32'] = 'real'
_np2col['f8'] = 'real'
_np2col['float64'] = 'real'


def descr2tabledef(descr, tablename):
    """
    Convert a numpy type descriptor to a create table statement. Numpy
    type descriptors have the following form
        [(name1, type1), (name2,type2),...]

    Where names are strings and types are strings such as '<f4' or '|S20'.
    See numpy2coltype() for how these type strings are converted to column
    type definitions.

    These can be retrieved from a recarray, or numpy array with fields, via
        arr.dtype.descr
    """

    coldefs = descr2coldefs(descr)

    coldefs = ",\n        ".join(coldefs)

    tabledef = """
    create table %s (
        %s
    )
    \n""" % (tablename, coldefs)

    return tabledef

def descr2coldefs(descr):
    """
    Convert a numpy type descriptor to a set of column definitions.  Numpy
    type descriptors have the following form
        [(name1, type1), (name2,type2),...]

    Where names are strings and types are strings such as '<f4' or '|S20'.
    See numpy2coltype() for how these type strings are converted to column
    type definitions.

    The descr can be retrieved from a recarray, or numpy array with fields, via
        arr.dtype.descr
    """

    coldefs=[]
    for d in descr:
        name = d[0]
        tname = d[1]

        coltype = numpy2coltype(tname)

        coldef = '%s %s' % (name, coltype)
        coldefs.append(coldef)

    return coldefs

def numpy2coltype(typename):
    """

    Convert a numpy data type name to an appropriate column type in sqlite.
    Note sqlite doesn't actualy use this information; it will store any value
    into any column.  This is to facilitate reading the data back into numpy
    arrays.  Byte order characters are ignored at the beginning, e.g '>i4'.

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

    tname = typename.strip().lower()
    tname = _remove_byteorder(tname)

    if tname[0] == 's':
        # !!!!! FIX  THIS!!!!
        return 'TEXT'
        try:
            lenstr = int( tname[1:] )
        except:
            raise ValueError("Could not extract length from %s" % tname)
        return 'char(%s)' % lenstr
    
    if tname not in _np2col:
        raise ValueError("unrecognized typename: %s" % tname)

    return _np2col[tname]

def _remove_byteorder(tname):
    if tname[0] == '>' or tname[0] == '<' or tname[0] == '|':
        return tname[1:]
    else:
        return tname

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

def add_index(dbfile, tablename, columns, verbose=False):
    """
    Convenience function
    """
    conn = SqliteConnection(dbfile, isolation_level=None)
    conn.add_index(tablename, columns, verbose=verbose)
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
    csvtmp = tempfile.mktemp(dir=tmpdir, 
                             prefix=tablename+'-temp-', 
                             suffix='.csv')

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

        dbpath=esutil.ostools.expand_path(dbfile)
        sqlite.Connection.__init__(self, dbpath, **keys)
        self.row_factory = sqlite.Row
        self.dbfile = dbpath

    def describe(self, typ=None, tablename=None):
        """
        Describe the database or the table if tablename is given.

        currently does nothing if tablename is not sent.

        indices on tables not yet implemented.
        """

        if tablename is not None:
            # describe this table
            info = self.table_info(tablename)

            stdout.write("%-15s %15s\n" % ("field","type"))
            stdout.write("-"*32+"\n")
            for row in info:
                stdout.write("%-15s %15s\n" % (row['name'],row['type']))

        else:
            # describe all tables
            infolist = self.info(typ=typ)
            head = "%-15s %15s %15s %s" %  ('type','name','tbl_name','sql')
            stdout.write(head) 
            stdout.write("\n" + "-"*len(head)+"\n")
            for info in infolist:
                stdout.write("%-15s %15s %15s %s\n" % \
                    (info['type'],info['name'],info['tbl_name'],info['sql'])) 


    def info(self, typ=None):
        """

        Query the sqlite_master table.  This holds info about objects in the
        database, such as tables and indexes.  By default return all entries.
        If typ is not None, return only entries of that type, e.g. 'table' or
        'index'

        Each rows contains:
            type: e.g. 'table' or 'index'
            name: name of object
            tbl_name: name of associated table
            rootpage: ?
            sql:  sql query used to create object.

        """

        query = "select * from sqlite_master"
        if typ is not None:
            query += " where type = '%s'" % typ
        curs = self.cursor()
        return curs.execute(query).fetchall()

    def table_info(self, tablename=None, columns=None):
        """
        info = conn.table_info(tablename=None, columns=None)

        if tablename is sent:

            Retrieve the table info.  Each row returned has fields:
                'cid': column id number
                'name': name of column
                'type': declared type. Ignored by sqlite but useful for 
                    converting to numpy.
                'notnull': If 1 can not be Null.
                'dflt_value': default value for column
                'pk': Do not know.
            
            if columns is sent, a subset of columns will be returned.

        if tablename is not sent:
            Retreive info about all tables from the sqlite_master table.
            Each rows contains:
                type: This will always be 'table'
                name: name of object
                tbl_name: name of associated table
                rootpage: ?
                sql:  sql query used to create object.
        """

        curs = self.cursor()
        if tablename is not None:
            query="pragma table_info(%s)" % tablename
        else:
            query = "select * from sqlite_master where type='table'"
        res = curs.execute(query).fetchall()

        if tablename is not None and columns is not None:
            # extract a subset of columns
            newres = []
            for info in res:
                if info['name'] in columns:
                    newres.append(info)
            return newres
        else:
            return res

    def table_exists(self, tablename):
        query = """
            select 
                name 
            from 
                sqlite_master 
            where 
                type='table' and name = '%s'
        """ % tablename

        curs = self.cursor()
        res = curs.execute(query).fetchall()
        if len(res) == 0:
            return False
        else:
            return True


    def asarray(self, tablename, query=None):
        """
        Instead maybe override execute and have a asarray= keyword?
        """
        if not have_numpy:
            raise RuntimeError("numpy is not available")

        if query is None:
            query="select * from %s" % tablename



        # when row factory is Row, the proper iteration is not supported
        # for use with fromiter.  Temporarily turn it off.
        row_factory_old = self.row_factory
        self.row_factory=None
        curs = self.cursor()
        curs.execute(query)

        # now figure out the data types for the returned columns
        self.row_factory = row_factory_old
        colnames = [d[0].lower() for d in curs.description]
        tinfo = self.table_info(tablename, columns=colnames)
        dtype = tabledef2dtype(tinfo)


        
        res = numpy.fromiter(curs, dtype=dtype)

        return res
    
    def drop(self, tablename, verbose=False):
        query = "drop table %s" % tablename
        if verbose:
            stdout.write("%s\n" % query)
        curs=self.cursor()
        curs.execute(query)


    def fromarray(self, arr, tablename, indices=None,
                  create=False, tmpdir='.', 
                  cleanup=True,
                  verbose=False):

        from esutil import recfile
        exists = self.table_exists(tablename)
        if exists:
            if create:
                if verbose:
                    stdout.write("Dropping existing table: %s\n" % tablename)
                self.drop(tablename)
            else:
                if verbose:
                    stdout.write("Appending to existing "
                                 "table: %s\n" % tablename)

        
        tabledef = descr2tabledef(arr.dtype.descr, tablename)
        
        curs=self.cursor()

        if verbose:
            stdout.write(tabledef)

        curs.execute(tabledef)


        # write data to a temporary csv file and then import note we ill
        # ignore nulls in writing because sqlite can't handle them
        csvtmp = tempfile.mktemp(dir=tmpdir, 
                                 prefix=tablename+'-temp-', 
                                 suffix='.csv')

        if verbose:
            stdout.write("Writing to temporary file: %s\n" % csvtmp)
        r = recfile.Open(csvtmp, mode='w', delim=',', dtype=arr.dtype)
        r.Write(arr.view(numpy.ndarray), ignorenull=True)
        r.Close()

        if verbose:
            stdout.write("Importing data\n")
        comm="""
            sqlite3 -separator ',' %s ".import %s %s" 
        """ % (self.dbfile, csvtmp, tablename)

        status, stdo, stde = esutil.misc.exec_process(comm, verbose=verbose)
        if status != 0 or stde != "":
            mess="""
            Error occurred:
                exit_status: %s
                stdout: %s
                stderr: %s
            """ % (status,stdo,stde)
            raise RuntimeError(mess)

        if cleanup:
            if verbose:
                stdout.write("Cleaning up temporary file %s\n" % csvtmp)
            os.remove(csvtmp)

        if indices is not None:
            for index in indices:
                self.add_index(tablename, index, verbose=verbose)

        
    def add_index(self, tablename, columns, verbose=False):

        if not isinstance(columns,(list,tuple)):
            columns = [columns]

        curs=self.cursor()

        index_name = '_'.join(columns) + '_index'
        column_list = ','.join(columns)

        query="""
        create index if not exists %s on %s (%s)
        \n""" % (index_name, tablename, column_list)

        if verbose:
            stdout.write("Adding index: \n")
            stdout.write(query)

        curs.execute(query)



def table_info(dbfile, tablename=None, columns=None):
    """
    Convenience function
    """
    conn = SqliteConnection(dbfile)
    return conn.table_info(tablename=tablename, columns=columns)

def describe(dbfile, typ=None, tablename=None):
    """
    Convenience function
    """
    conn = SqliteConnection(dbfile)
    conn.describe(typ=typ, tablename=tablename)

def asarray(dbfile, tablename, query=None):
    """
    Convenience function
    """
    conn = SqliteConnection(dbfile)
    return conn.asarray(tablename, query=query)

