
import sys, os

try:
    import cx_Oracle as cxo
    have_oracle=True
except:
    have_oracle=False

try:
    import numpy
    have_numpy=True
except:
    have_numpy=False

lower_default=True

__doc__="""
    NAME
        oracle_util
    PURPOSE

        Defines a wrapper class for cx_Oracle and some utility functions,
        including the ability to convert the results of an oracle query to
        numerical python (NumPy) arrays.  

        This is pure python code.  This is rather inefficient because 
        cx_Oracle must first convert the results to python objects and then
        they must be converted to NumPy arrays.  A more efficient version 
        would wrap the C API and convert directly to NumPy arrays.

        Also, I had to hack cx_Oracle to replace None objects with something
        that Numpy can convert.  I chose generically -9999 which seems to
        work for the use cases I've seen.  Ask me for the hacked code.

    CLASSES
        Connection(conninfo=defconn):  Wrapper for a cx_Oracle connection.
        CLASS METHODS
            Execute(query)
                Execute the query and return the results in a NumPy array
                if appropriate
            Quick(query, limit=None)
                Execute the query and display the results in a nice format.
                Use the limit= keyword to limit the number of results 
                displayed.
            Describe(table, columns=None)
                Print a description of the requested table
            Connect(conninfo=defconn):
                Make a new connection, closing the existing one if necessary
            Close():
                Close the connection

    FUNCTIONS
        Cursor2Array(oracle_cursor, dtype=None)
            Convert an cx_ Oracle cursor object into a NumPy array.  If the 
            dtype is not given, the description field is converted to a NumPy 
            type list using the NumpyDescriptor() function.
        Res2Array(fetched_results, dtype=None)
            Convert fetched results from a cx_ Oracle cursor object into a NumPy
            array.  The dtype can be gotten by running the NumpyDescriptor()
            function on the cursor.description field.  This is less efficient
            that using Cursor2Array since extra memory is used in the fetching
            process.
        NumpyDescriptor(oracle descriptioin list)
            Convert a list of cx_ Oracle descriptions to a list of NumPy type 
            descriptions.  This cx_Oracle description list is gotten from the 
            cursor description field
                cursor.description
        NumpyType(oracle description)
            Convert a cx_Oracle description to a NumPy type description.  This
            is an element of the cursor.description list.
        PrintCursor(cursor, limit=None, maxwidth=30)
            Print a pretty formatted version of the data contained in the input
            cursor.  Limit the number of printed rows with limit, and alter
            the maximum width of a row with maxwidth.  Note if limit is sent
            the cursor might still have rows to be fetched.

    REQUIREMENTS
        cx_Oracle http://cx-oracle.sourceforge.net/
        NumPy if you want that functionality http://www.scipy.org/NumPy
    REVISION HISTORY
        Created 2008-07-09, Erin Sheldon, NYU

"""

if 'ORACLE_CONNINFO' in os.environ:
    defconn=os.environ['ORACLE_CONNINFO']
else:
    defconn=""

_defs={}
_defs['f4_digits'] = 6
_defs['f8_digits'] = 15


_binary_err='size of %s not allowed for BINARY floating point types'

_flt_digits_err=\
"""WARNING: Digits for field "%s" of %s exceeds that of an
8-byte floating point
Setting to type "f16" which may or may not exceed 8-bytes in size, 
depending on your platform\n"""

_int_digits_err=\
"""WARNING: Digits for field "%s" of %s exceeds largest available 
(18 digits for 8-byte binary integer).  Setting to 8-byte integer\n"""

_string_err='The size of field "%s" is %s but must be greater than zero'

class Connection(object):
    """
    CLASS
        oracle_util.Connection(conninfo=default, f4_digits=6, f8_digits=15)
    PURPOSE
        A wrapper class for a cx_Oracle connection that defines some methods
        for working quickly with queries and tables.
    INPUTS
        conninfo:  A string containing the connection info.  The format
            should be 
                user/password@host.address/dsn
            Where dsn is the data source name.  The default connection info
            is taken from the ORACLE_CONNINFO environment variable if it
            exists, otherwise it is set to ""
        f4_digits, f8_digits:  The number of digits to demand when converting
            to these types from number(digits,n).  The default is 6 or less
            for floats and 7-15 for double, e.g. f4_digits=6, f8_digits=15  
            For example if you want everything to be double use f4_digits=0
    CLASS METHODS
        Execute(query)
            Execute the query and return the results in a NumPy array
            if appropriate
        Quick(query, limit=None)
            Execute the query and display the results in a nice format.
            Use the limit= keyword to limit the number of results displayed.
        Describe(table, columns=None)
            Print a description of the requested table
        Connect(conninfo=defconn):
            Make a new connection, closing the existing one if necessary

    Examples
        # read from a table into a NumPy array and work with the data
        import oracle_util
        o=oracle_util.Connection()
        o.Describe('objects')
        arr=o.Execute('select field1, field3 from objects')
        print(arr['field1'])
        newarray = arr['field1']*arr['field2']
        prit newarray
    """
    def __init__(self, f4_digits=_defs['f4_digits'], f8_digits=_defs['f8_digits'], conninfo=defconn):
        self.conn = cxo.connect(conninfo)
        self.f4_digits = f4_digits
        self.f8_digits = f8_digits

    def test(self):

        tilename='BCS2316-5455'
        run='BCS20080710_BCS2316-5455'
        query="""
            SELECT 
                cobj.coadd_objects_id, cobj.alpha_j2000, cobj.delta_j2000,
                cobj.x_image, cobj.y_image 
            FROM 
                coadd_objects cobj, catalog cat
            WHERE
                cat.id = cobj.catalogid_i 
                    AND cat.tilename = '%s' 
                    AND cat.run = '%s' 
                    AND rownum <= 10
            """ % (tilename, run)

        sys.stdout.write('\nExecuting query: '+query+'\n\n\n')
        self.Quick(query)


        query="""
            SELECT 
                id,tilename,band,run 
            FROM 
                coadd 
            WHERE
                tilename='%s' 
                    AND band='g' 
                    AND run='%s'
            """ % (tilename, run)

        sys.stdout.write('\nExecuting query: '+query+'\n\n\n')
        self.Quick(query)

        query="""
            SELECT 
                image.parentid 
            FROM 
                image,coadd_src 
            WHERE
                coadd_src.coadd_imageid=5427611 
                    AND coadd_src.src_imageid=image.id
            """

        sys.stdout.write('\nExecuting query: '+query+'\n\n\n')
        self.Quick(query)

    def Connect(self, conninfo=defconn):
        """
        Create a new connection, closing the old one if necessary
        """
        # might not be open
        try:
            self.conn.close()
        except:
            pass
        self.conn = cxo.connect(conninfo)

    def Close(self):
        """
        Close the connection.
        """
        try:
            self.conn.close()
        except:
            pass


    def Describe(self, table, columns=None, verbose=False):
        """
        NAME
            Describe(table, columns=None, verbose=False)
        PURPOSE
            Print a simple description of the input table.  If the columns
            keyword is sent, then just print the description of that column
            or list of columns
        """
        q="""
            SELECT
                column_name, 
                CAST(data_type as VARCHAR2(15)) as type, 
                CAST(data_length as VARCHAR(6)) as length, 
                CAST(data_precision as VARCHAR(9)) as precision, 
                CAST(data_scale as VARCHAR(5)) as scale, 
                CAST(nullable as VARCHAR(8)) as nullable
            FROM
                all_tab_columns
            WHERE
                table_name = '%s'
                AND column_name <> 'TNAME'
                AND column_name <> 'CREATOR'
                AND column_name <> 'TABLETYPE'
                AND column_name <> 'REMARKS'"""

        if columns != None:
            q = q + """
                AND column_name IN ("""
            if type(columns) == type(''):
                tc = [columns]
            else:
                tc = columns
            tc=["'"+c.upper()+"'" for c in tc]
            tc = ",".join(tc)
            q += tc + ")"

        q = q + """
            ORDER BY 
                column_id
        """

        q = q % (table.upper(),)

        if verbose:
            sys.stdout.write('%s\n' % q)

        curs = self.conn.cursor()
        curs.execute(q) 
        PrintCursor(curs)

        # now indexes

        q = """
            select
                index_name, column_name, column_position, descend
            from
                all_ind_columns
            where
                table_name = '%s' order by index_name, column_position
        """ % table.upper()



        curs.execute(q)
        PrintCursor(curs)

        curs.close()


    def Execute(self, query, cursor=False,
                f4_digits=None,
                f8_digits=None,
                dictlist=False,
                lower=lower_default):
        """

        Does job of executing the query, converting to a numpy array or list of
        dictionaries if appropriate, and returns the result.  If cursor=True
        then the cursor object is returned instead of a numpy array.

        """

        if f4_digits is None:
            f4_digits=self.f4_digits
        if f8_digits is None:
            f8_digits=self.f8_digits

        curs=self.conn.cursor()
        curs.execute(query)
        if cursor:
            return curs

        if dictlist:
            return Cursor2Dictlist(curs, lower=lower)

        result = Cursor2Array(curs, 
                              f4_digits=f4_digits, 
                              f8_digits=f8_digits,
                              lower=lower)
        if result.size == 0:
            result=None

        curs.close()
        return result



    def Quick(self, query, fname=None, limit=None, maxwidth=30):
        """
        NAME
            Quick(query, fname=None, limit=None)
        PURPOSE 
            Execute the query and display the results in a nice format.
            Use the limit= keyword to limit the number of results 
            displayed.

        """

        curs = self.conn.cursor()
        curs.execute(query)

        PrintCursor(curs, fname=fname, limit=None, maxwidth=maxwidth)
        curs.close()

        return None


    def GetConnection(self):
        return self.conn
    def GetCursor(self):
        return self.conn.cursor()


def NumpyType(odesc, f4_digits=_defs['f4_digits'], f8_digits=_defs['f8_digits']):
    """
    NAME
        NumpyType
    PURPOSE
        Convert a cx_Oracle field description list into a NumPy type.
    USAGE
        nt = NumpyType(oracle_field_description, f4_digits=6, f8_digits=15)

    INPUTS
        oracle_field_description:  This is an element of the cx_Oracle
            description list.  This list is gotten from the cursor object:
                cursor.description
            An element of this list contains the following:
                (name, cx_Oracle_type, display_size, internal_size, 
                precision, scale, null_ok)
        f4_digits, f8_digits:  The number of digits to demand when converting
            to these types from number(digits,n).  The default is 6 or less
            for floats and 7-15 for double, e.g. f4_digits=6, f8_digits=15  
            For example if you want everything to be double use f4_digits=0
    Currently recognizes the following cx_Oracle types
        NATIVE_FLOAT.  This corresponds to the Oracle types 
            BINARY_FLOAT and BINARY_DOUBLE
        NUMBER with various precision, both floating point and fixed point
            NUMBER(p,s) is floating point, NUMBER(p) is integer
        STRING and character arrays of variable and fixed length. Some
            maximum length must be specified, but this is always the case
            for cx_Oracle description lists

        Be warned that Oracle supports precisions of both integers and
        floats that is far beyond the standard data types.  In these cases
        the integer size is set to 64-bit and the floating type is set to
        128 bit, but note that the 128 bit float type in numerical python is
        in practice usually limited to less precision.
    """

    err='size of %s not allowed for type %s'
    name = odesc[0]
    otype = odesc[1]
    size = odesc[3]
    digits = odesc[4]
    scale = odesc[5]
    if otype == cxo.NATIVE_FLOAT:
        # This one is easy: sizes indicate everything!
        if size == 4:
            Ntype='f4'
        elif size==8:
            Ntype='f8'
        else:
            raise ValueError(_binary_err % (size,))
    elif otype == cxo.NUMBER:
        if scale != 0:
            if digits <= f4_digits:
                Ntype='f4'
            elif digits <= f8_digits:
                Ntype='f8'
            else:
                sys.stdout.write(_flt_digits_err % (name,digits))
                Ntype='f16'
        else:
            if digits == 0:
                Ntype = 'i8'
            elif digits <= 4:
                Ntype = 'i2'
            elif digits <= 9:
                Ntype = 'i4'
            elif digits <= 18:
                Ntype= 'i8'
            else:
                sys.stdout.write(_int_digits_err % (name,digits))
                Ntype='i8'

    elif otype == cxo.STRING:
        if size <= 0:
            raise ValueError(_string_err % (name, size))
        Ntype= 'S'+str(size)
    else:
        if size <= 0:
            raise ValueError(_string_err % (name, size))
        Ntype= 'S'+str(size)
        #raise ValueError,'Unsupported data type: '+repr(otype)

    return Ntype


def NumpyDescriptor(odesc, f4_digits=_defs['f4_digits'], f8_digits=_defs['f8_digits'], lower=lower_default):
    """
    NAME
        NumpyDescriptor(cx_Oracle_description, f4_digits=6, f8_digits=15, 
                        lower=True)
    PURPOSE
        Convert a list of cx_ Oracle descriptions to a list of NumPy type 
        descriptions.  This cx_Oracle description list is gotten from the 
        cursor description field
            cursor.description
        See NumpyType for the the conversion process.

        f4_digits, f8_digits:  The number of digits to demand when converting
            to these types from number(digits,n).  The default is 6 or less
            for floats and 7-15 for double, e.g. f4_digits=6, f8_digits=15  
            For example if you want everything to be double use f4_digits=0
        lower: If True then all names are converted to lower case
    """
    dtype=[]

    for d in odesc:
        name = d[0]
        if lower:
            name=name.lower()
        Ntype = NumpyType(d, f4_digits=f4_digits, f8_digits=f8_digits)
        dtype.append( (name, Ntype) )

    return dtype

def Res2Array(res, dtype):
    """
    NAME
        Res2Array(res, dtype)
    PURPOSE
        Convert fetched results from a cx_ Oracle cursor object into a NumPy
        array.  The dtype can be gotten by running the NumpyDescriptor()
        function on the cursor.description field.  This is less efficient that
        using Cursor2Array since extra memory is used in the fetching process.
    EXAMPLES
        curs=conn.cursor()
        curs.execute(query)
        dtype=NumpyDescriptor(curs.description)
        res=curs.fetchall()
        arr = Res2Array(curs, dtype)
    """

    arr = numpy.array(res, dtype=dtype)
    return arr

def Cursor2Array(curs, dtype=None, f4_digits=_defs['f4_digits'], f8_digits=_defs['f8_digits'], lower=lower_default):
    """
    NAME
        Cursor2Array(curs, dtype=None, f4_digits=6, f8_digits=15, lower=True)
    PURPOSE
        Convert an cx_ Oracle cursor object into a NumPy array.  If the 
        dtype is not given, the description field is converted to a NumPy 
        type list using the NumpyDescriptor() function.  This is more 
        efficient than using Res2Array since no extra memory is used.

        f4_digits, f8_digits:  The number of digits to demand when converting
            to these types from number(digits,n).  The default is 6 or less
            for floats and 7-15 for double, e.g. f4_digits=6, f8_digits=15  
            For example if you want everything to be double use f4_digits=0
    EXAMPLES
        curs=conn.cursor()
        curs.execute(query)
        arr = Cursor2Array(curs)
    """
    if dtype is None:
        dtype=NumpyDescriptor(curs.description, 
                              f4_digits=f4_digits, f8_digits=f8_digits,
                              lower=lower)
    arr = numpy.fromiter(curs, dtype=dtype)
    return arr

def Cursor2Dictlist(curs, lower=lower_default):
    if curs is None:
        return None

    keys=[]
    for d in curs.description:
        key=d[0]
        if lower:
            key=key.lower()
        keys.append(key)
        
    output=[]
    for row in curs:
        tmp={}
        i=0
        for val in row:
            tmp[keys[i]] = val    
            i+=1
        output.append(tmp)

    return output

def PrintCursor(curs, delim=' ', fname=None, limit=None, maxwidth=30):
    """ 
    NAME
        PrintCursor(curs, limit=None, maxwidth=30)
    PURPOSE
        Print a pretty formatted version of the data in the input
        cursor.  Limit the number of printed rows with limit, and alter
        the maximum width of a row with maxwidth.  Note if limit is sent
        the cursor might still have rows to be fetched.
    """

    if fname is None:
        isfile=False
        fout=sys.stdout
    else:
        isfile=True
        fout=open(fname,'w')

    # build up a format string
    formats=[]
    separators=[]
    names=[]
    for d in curs.description:
        dsize = d[2]
        if dsize > maxwidth:
            dsize=maxwidth

        formats.append('%'+repr(dsize)+'s')
        names.append(d[0])
        separators.append('-'*dsize)

    format=delim.join(formats)

    count = 0
    for row in curs:
        if ((count % 50) == 0) and (not isfile):
            fout.write('\n')
            fout.write(format % tuple(names))
            fout.write('\n')
            fout.write(format % tuple(separators))
            fout.write('\n')

        fout.write(format % row)
        fout.write('\n')

        count += 1
        if (limit is not None) and (count == limit):
            break


