import os

def exists(hdfs_url):
    """
    Test if the url exists.
    """
    return test(hdfs_url, test='e')
 
def test(hdfs_url, test='e'):
    """
    Test the url.
        
    parameters
    ----------
    hdfs_url: string
        The hdfs url
    test: string, optional
        'e': existence
        'd': is a directory
        'z': zero length

        Default is an existence test, 'e'
    """
    command="""hadoop fs -test -%s %s""" % (test, hdfs_url)

    exit_code, stdo, stde = exec_command(command)

    if exit_code != 0:
        return False
    else:
        return True


def stat(hdfs_url):
    """
    stat the hdfs URL, return None if does not exist.

    Returns a dictionary with keys
        filename: base name of file
        blocks: number of blocks
        block_size: size of each block
        mod_date: last modification
        replication: number of copies in hdfs
    """

    command="""hadoop fs -stat "{'blocks': %b, 'mod_date': '%y', 'replication': %r, 'filename':'%n'}" """
    command += hdfs_url

    exit_code, stdo, stde = exec_command(command)

    if exit_code != 0:
        return None
    else:
        return eval(stdo.strip())


def ls(hdfs_url='', recurse=False):
    """
    List the hdfs URL.  If the URL is a directory, the contents are returned.
    """
    if recurse:
        cmd='lsr'
    else:
        cmd='ls'

    command = "hadoop fs -%s %s | awk 'NF==8 {print $8}'" % (cmd,hdfs_url)

    exit_code, stdo, stde = exec_command(command)

    if exit_code != 0:
        raise RuntimeError("hdfs failed to list url '%s'" % hdfs_url)

    fl = stdo.strip().split('\n')
    return fl

def lsr(hdfs_url=''):
    """
    Recursively List the hdfs URL.  This is equivalent to hdfs.ls(url, recurse=True)
    """
    ls(hdfs_url, recurse=True)

def read(hdfs_url, reader, verbose=False, **keys):
    with HDFSFile(hdfs_url, verbose=verbose) as fobj:
        return fobj.read(reader, **keys)

def put(local_file, hdfs_url, verbose=False):
    """
    Copy the local file to the hdfs_url.
    """

    if verbose:
        print 'hdfs',local_file,'->',hdfs_url

    command = 'hadoop fs -put %s %s' % (local_file, hdfs_url)
    exit_code, stdo, stde = exec_command(command)
    if exit_code != 0:
        raise RuntimeError("Failed to copy to hdfs %s -> %s: %s" % (local_file,hdfs_url,stde))

def opent(hdfs_url, tmpdir=None, verbose=False):
    """
    pipe the file from hadoop fs -cat into a temporary file, and then return
    the file object for the temporary file.

    The temporary file is automatically cleaned up when it is closed.
    """
    import subprocess
    from subprocess import PIPE
    import tempfile
    bname = os.path.basename(hdfs_url)
    temp_file = tempfile.NamedTemporaryFile(prefix='hdfs-', suffix='-'+bname, dir=tmpdir)

    if verbose:
        print 'hdfs opening: ',hdfs_url,'for reading, staging in temp file:',temp_file.name

    command = 'hadoop fs -cat %s' % hdfs_url
    pobj = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

    buffsize = 2*1024*1024
    while True:
        data = pobj.stdout.read(buffsize)
        if len(data) == 0:
            break
        temp_file.write(data)

    # we're done, just need to wait for exit
    ret=pobj.wait()
    if ret != 0:
        raise RuntimeError("Failed to copy to hdfs %s -> %s: %s" % (temp_file.name,hdfs_url,pobj.stderr.read()))

    temp_file.seek(0)
    return temp_file

def rm(hdfs_url, recurse=False, verbose=False):
    """
    Remove the specified hdfs url
    """
    mess='hdfs removing '+hdfs_url
        

    if recurse:
        cmd='rmr'
        mess+=' recursively'
    else:
        cmd='rm'

    if verbose:
        print mess

    command = 'hadoop fs -%s %s' % (cmd, hdfs_url)
    exit_code, stdo, stde = exec_command(command)
    if exit_code != 0:
        raise RuntimeError("hdfs %s" % stde)

def rmr(hdfs_url, verbose=False):
    """

    Remove the specified hdfs url recursively.  Equivalent to rm(url,
    recurse=True)
    """
    rm(hdfs_url, recurse=True, verbose=verbose)

def mkdir(hdfs_url, verbose=False):
    """
    Equivalent of mkdir -p in unix
    """
    if verbose:
        print 'hdfs mkdir',hdfs_url

    command = 'hadoop fs -mkdir '+hdfs_url
    exit_code, stdo, stde = exec_command(command)
    if exit_code != 0:
        raise RuntimeError("hdfs %s" % stde)


class HDFSFile:
    """
    A class to help get files in and out of the Hadoop Distributed File System.


    parameters
    ----------
    hdfs_url: string
        The URL of an hdfs file. Note if it begins with hdfs:// it must be an
        absolute path name, otherwise it references your home directory.

    verbose: bool
        Tell what is going on.  Default False.


    Examples
    --------

        General usage
        -------------

        The best way to use an HDFSFile:

            with hdfs.HDFSFile(hdfs_url) as fobj:
                do things with fobj

        This guarantees the cleanup is run after the with block, and guarantees no
        problems with reference counting.

        In the following examples we will use this approach.  But if you are
        using an older python you'll have to run the cleanup() method to make
        sure things get cleaned up.


        Staging a file for local read
        -----------------------------

        This copies the hdfs file to a local file.  The name of the local file
        is stored in the "localfile" attribute.

            # stage the file to a local temporary file
            with hdfs.HDFSFile(hdfs_url) as hdfile:
                hdfile.stage()

                # this will read from the local file
                with open(hdfile.localfile) as fobj:
                    data = fobj.read()
        
        The file local is automatically cleaned up after exiting the "with"
        context. You can clean up the local file manually by running the
        .cleanup() method.


        Reading a file with a reader object
        -----------------------------------

        Suppose you a function or method that has the following signature:

            (filename, **keys)

        then you can do the following:

            with hdfs.HDFSFile(hdfs_url) as hdfile:
                data = hdfile.read(reader, **keys)

        Under the hood the file is staged locally and the reader is used to
        grab the data.  The local file is cleaned up.


    """

    def __init__(self, hdfs_url, verbose=False):
        self.hdfs_url = hdfs_url
        self.verbose=verbose

        self.localfile=None

    def stage(self, tmpdir=None):
        """
        Stage a file out of hdfs to a temporary file.
        """
        import subprocess

        local_file = self.temp_filename(self.hdfs_url, tmpdir=tmpdir)
        command = 'hadoop fs -get %s %s' % (self.hdfs_url,local_file)

        if self.verbose:
            print "hdfs staging",self.hdfs_url,"->",local_file

        exit_code, stdo, stde = exec_command(command)

        if exit_code != 0:
            raise RuntimeError("Failed to copy from hdfs %s -> %s: %s" % (self.hdfs_url,local_file,stde))

        if not os.path.exists(local_file):
            raise RuntimeError("In copy from hdfs %s -> %s, local copy not found" % (self.hdfs_url,local_file))

        self.localfile = local_file
        return local_file


    def read(self, reader, **keys):
        """
        Use the input reader to read the hdfs file.

        The file is first staged locally; the temporary file will be cleaned up
        unless you send cleanup=False.  Not cleaning is useful for debugging
        your reader.

        parameters
        ----------
        reader: method
            The reader must have a (fname, **keys) signature.
        cleanup: bool
            If True, the temporary file is removed before exiting.
            Default is True.

        other keywords:
            These are passed on to the reader.
        """

        self.stage()

        try:
            data = reader(self.localfile, **keys)
        finally:
            cleanup=keys.get('cleanup',True)
            if cleanup:
                self.cleanup()

        return data

    def write(self, writer, data, **keys):
        """
        Write the input data to the output file using the specified
        writer.

        The file is first written locally; the temporary file will be cleaned
        up unless you send cleanup=False.  Not cleaning is useful for debugging
        your writer.

        parameters
        ----------
        writer: method
            The writer must have a (fname, data, **keys) signature.
        data: object
            An object to be written.
        cleanup: bool, optional
            If True, the temporary file is removed before exiting.
            Default is True.
        other keywords:
            These are passed on to the writer.
        """

        if exists(self.hdfs_url):
            clobber=keys.get('clobber',False)
            if clobber:
                if self.verbose:
                    print 'removing existing hdfs file:',self.hdfs_url
                rm(self.hdfs_url)
            else:
                raise ValueError("hdfs file already exists: %s, "
                                 "send clobber=True to remove" % self.hdfs_url)
            
        tmpdir=keys.get('tmpdir',None)
        self.localfile = self.temp_filename(self.hdfs_url, tmpdir=tmpdir)

        try:
            writer(self.localfile, data, **keys)
            put(self.localfile, self.hdfs_url, verbose=self.verbose)
        finally:
            cleanup=keys.get('cleanup',True)
            if cleanup:
                self.cleanup()


    def temp_filename(self, fname, tmpdir=None):
        import tempfile
        bname = os.path.basename(fname)
        tfile = tempfile.mktemp(prefix='hdfs-', suffix='-'+bname, dir=tmpdir)
        return tfile

    def cleanup(self):
        if self.localfile is not None:
            if os.path.exists(self.localfile):
                if self.verbose:
                    print "hdfs removing staged file", self.localfile
                os.remove(self.localfile)

            self.localfile=None


    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.cleanup()
    def __del__(self):
        self.cleanup()



def exec_command(command):
    """
    Execute the command and return the exit status.
    """
    import subprocess
    from subprocess import PIPE
    pobj = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

    stdo, stde = pobj.communicate()
    exit_code = pobj.returncode

    return exit_code, stdo, stde


