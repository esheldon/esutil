"""
Module:
    ostools
Purpose:
    A set of tools for working with the operating system.

Classes:
    Class Name:
        DirStack
    Purpose:
        This is a directory simple stack that works like the
        directory stack in Unix shells.  See the documentation
        for the DirStack class for more details.

    Example:
        >>> ds=esutil.ostools.DirStack(verbose=True)
        >>> ds.push('~/data')
        ~/data ~
        >>> ds.push('/usr/bin')
        /usr/bin ~/data ~
        >>> ds.pop()
        ~/data ~
        >>> ds.pop()
        ~

    Class Name:
        StagedOutFile
    Purpose:
        A context manager for staging files from temporary directories to
        a final destination.

    Example:
        >>> fname = "/home/jill/output.dat"
        >>> tmpdir = "/tmp"
        >>> with StagedOutFile(fname, tmpdir=tmpdir) as sf:
        ...     with open(sf.path, 'w') as fobj:
        ...         fobj.write("some data")

    Class Name:
        StagedInFile
    Purpose:
        A class to stage a file in to local disk for reading.

    Example:
        >>> fname="/home/jill/output.dat"
        >>> tmpdir="/tmp"
        >>> with StagedInFile(fname,tmpdir=tmpdir) as sf:
        ...     with open(sf.path) as fobj:
        ...         # read some data from fobj

Functions:
    See docs for the individual functions for more info.

    path_join(*paths):
        Join path elements using the system path separator.  Any number of
        inputs can be given.  These must be strings or sequences.  This is
        similar to the os.path.join function but can join any number of path
        elements and supports sequences.

    getenv_check(environment variable name):
        Check for the envrionment variable and raise a RuntimeError if not
        found.  This differs from os.getenv() in that it raises a RuntimeError
        if the variable is not found instead of returning None.

    expand_path:
        Expand all user info such as ~userid and environment variables such as
        $SOMEVAR.  this simple uses a call to both os.path.expanduser and
        os.path.expandvars

    expand_filename:
        Synonym for `expand_path`.

    exec_process:
        Execute a command on the operating system with a possible timeout in
        seconds

    makedirs_fromfile:
        Extract the directory from a file name and create it if it doesn't
        exist.
"""
from __future__ import print_function

import os
import shutil
from sys import stdout, stderr
import subprocess


class DirStack(object):
    """
    Class:
        DirStack
    Purpose:
        A simple directory stack.

    Construction:
        ds=DirStack(verbose=False):  If verbose=True a message is
            printed for push and pop similar to that printed on
            unix systems.
    Methods:
        push(directory): Change to the input directory.  Push the
            current working directory onto the stack.
        pop: Pop the last directory from the stack and change to
            that directory.

    Example:
        >>> ds=esutil.ostools.DirStack(verbose=True)
        >>> ds.push('~/data')
        ~/data ~
        >>> ds.push('/usr/bin')
        /usr/bin ~/data ~
        >>> ds.pop()
        ~/data ~
        >>> ds.pop()
        ~

    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self._home = os.path.expanduser('~')
        self._dirs = []

    def push(self, dir):
        """
        push(dir):  Change to the indicated dir and push the current
            working directory onto the stack
        """
        dir = os.path.expandvars(dir)
        dir = os.path.expanduser(dir)

        old_dir = os.getcwd()

        os.chdir(dir)

        # only do this *after* we successfully chdir
        self._dirs.append(old_dir)
        if self.verbose:
            self.print_stack()

    def pop(self):
        """
        pop(): Pop the last directory from the stack and change to
            that directory.
        """
        if len(self._dirs) == 0:
            stderr.write("Directory stack is empty\n")
            return

        dir = self._dirs.pop()
        os.chdir(dir)

        if self.verbose:
            self.print_stack()

    def getstack(self):
        """
        getstack(): Return the current stack.
        """
        return self._dirs

    def print_stack(self):
        self.print_dir(os.getcwd())
        for i in range(len(self._dirs)-1, -1, -1):
            d = self._dirs[i]
            self.print_dir(d)
        stdout.write('\n')

    def print_dir(self, dir):
        dir = dir.replace(self._home, '~')
        stdout.write('%s ' % dir)


def path_join(*paths):
    """
    Name:
        path_join
    Calling Sequence:
        path=path_join(any number of paths)

    Purpose:

        Join path elements using the system path separator.  Any number of
        inputs can be given.  These must be strings or sequences.  This is
        similar to the os.path.join function but can join any number of path
        elements and supports sequences.

    Examples:
        # Join three path elements
        p=path_join('/tmp', 'test', 'file.txt') # gives /tmp/test/file.txt


        # join a list of path elements
        p=path_join(['/tmp','file.txt']) # gives /tmp/file.txt

        # Join a path element with a list of path elements
        p=path_join('/tmp', ['test','file.txt']) # gives /tmp/test/file.txt
        p=path_join(['/tmp','test'], 'file.txt') # gives /tmp/test/file.txt

        # nested sequences.  Gives /tmp/test1/test2/file.txt
        p=path_join(['/tmp',['test1','test2']], 'file.txt')
    """

    plist = []
    for path in paths:
        # for py3k unicode will disappear
        if isinstance(path, str):
            plist.append(path)
        elif isinstance(path, (list, tuple)):
            for p in path:
                tpath = path_join(p)
                plist.append(tpath)
        else:
            raise ValueError('paths must be strings or sequences of strings')

    # We now have a list of strings.
    path = os.sep.join(plist)

    return path


def getenv_check(name):
    """
    Name:
        getenv_check
    Calling Sequence:
        val = getenv_check(name)

    Purpose:
        Check for the envrionment variable and raise a RuntimeError if not
        found.  This differs from os.getenv() in that it raises a RuntimeError
        if the variable is not found.

    """
    val = os.getenv(name)
    if val is None:
        raise RuntimeError("Environment variable '%s' is not set" % name)
    return val


def expand_path(filename):
    """
    Name:
        expand_path
    Purpose:
        Expand all user info such as ~userid and environment variables such as
        $SOMEVAR.  this simple uses a call to both os.path.expanduser and
        os.path.expandvars
    Calling Sequence:
        fullpath = expand_path(path)

    """
    fname = os.path.expanduser(filename)
    fname = os.path.expandvars(fname)
    return fname


# synonym
expand_filename = expand_path


def exec_process(command,
                 timeout=None,
                 poll=1,
                 stdout_file=subprocess.PIPE,
                 stderr_file=subprocess.PIPE,
                 shell=True,
                 verbose=False):
    """
    Name:
        exec_process
    Purpose:
        Execute a command on the operating system with a possible timeout in
        seconds

    Calling Sequence:

        exit_status, stdout_returned, stderr_returned = \
           execute_command(command,
                           timeout=None,
                           poll=1,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           shell=True,
                           verbose=False)
    Inputs:
        command: A command to run.

    Keywords:
        timeout:
            If this argument is sent, the process will be killed if it runs for
            longer than timeout seconds.
        poll:
            How often to poll the process while waiting for a timeout.  Default
            is 1.
        verbose:
            print the command.

    The rest of the keywords are subprocess.Popen keywords, see docs for
    that module.

    """

    # the user can send file names, PIPE, or a file object
    if isinstance(stdout_file, str):
        stdout_was_entered = False
        stdout_fileobj = open(stdout_file, 'w')
    else:
        stdout_was_entered = True
        stdout_fileobj = stdout_file

    if isinstance(stderr_file, str):
        stderr_was_entered = False
        stderr_fileobj = open(stderr_file, 'w')
    else:
        stderr_was_entered = True
        stderr_fileobj = stderr_file

    # if a list was entered, convert to a string.  Also print the command
    # if requested
    if verbose:
        print('Executing command', file=stderr)
    if isinstance(command, list):
        cmd = ' '.join(command)
        if verbose:
            print(command[0], '   \\', file=stderr)
            for c in command[1:]:
                print('   '+c+'    \\', file=stderr)
    else:
        cmd = command
        if verbose:
            print(cmd, file=stderr)

    stdout.flush()
    pobj = subprocess.Popen(
        cmd,
        stdout=stdout_fileobj,
        stderr=stderr_fileobj,
        shell=shell,
    )

    if timeout is not None:
        exit_status, stdout_ret, stderr_ret = _poll_subprocess(
            pobj, timeout, poll
        )
    else:
        # this just waits for the process to end
        stdout_ret, stderr_ret = pobj.communicate()
        # this is not set until we call pobj.communicate()
        exit_status = pobj.returncode

    # close them if we opened them
    if hasattr(stdout_fileobj, 'close') and not stdout_was_entered:
        stdout_fileobj.close()
    if hasattr(stderr_fileobj, 'close') and not stderr_was_entered:
        stderr_fileobj.close()

    return exit_status, stdout_ret, stderr_ret


def _poll_subprocess(pobj, timeout, poll):
    import time
    import signal

    if poll < 0.1:
        poll = 0.1
    if timeout < 0:
        timeout = poll

    try:
        tm0 = time.time()
        while 1:
            time.sleep(poll)

            exit_status = pobj.poll()
            if exit_status is not None:
                break
            tm = time.time()-tm0
            if tm > timeout:
                break
    except KeyboardInterrupt:
        mess = 'Keyboard Interrupt encountered, halted process %s' % pobj.pid
        os.kill(pobj.pid, signal.SIGTERM)
        raise KeyboardInterrupt(mess)

    # exit status will not be None upon completion.  If we passed
    # the timeout we want to kill the process.
    if exit_status is None:
        stderr.write("Process is taking longer than %s seconds.  "
                     "Ending process\n" % timeout)
        os.kill(pobj.pid, signal.SIGTERM)
        exit_status = 1024
        stdout_ret, stderr_ret = None, None
    else:
        stdout_ret, stderr_ret = pobj.communicate()

    return exit_status, stdout_ret, stderr_ret


def makedirs_fromfile(f, verbose=False, allow_fail=False):
    """
    Extract the directory from a file name and create it if it doesn't exist.

    parameters
    ----------
    filename: string
        The file name
    verbose: boolean, optional
        Optionally print that the dir is being created.
    allow_fail: boolean, optional
        If True, raise an error if the directory cannot be made and it does
        not already exist. Default is False.
    """
    import errno
    from esutil import hdfs

    d = os.path.dirname(f)
    if d == '':
        return

    if hdfs.is_in_hdfs(f):
        if not hdfs.exists(d):
            if verbose:
                print('creating dir:', d)
            hdfs.mkdir(d)
    else:
        if not os.path.exists(d):
            if verbose:
                print('creating dir:', d)
            try:
                os.makedirs(d)
            except OSError as ex:
                if ex.errno == errno.EEXIST and os.path.isdir(d):
                    pass
                else:
                    if not allow_fail:
                        raise


class StagedOutFile(object):
    """A context manager for staging files from temporary directories to
    a final destination.

    Parameters
    ----------
    fname : str
        Final destination path for file.
    tmpdir : str, optional
        If not sent, or `None`, the final path is used and no staging
        is performed.
    must_exist : bool, optional
        If `True`, the file to be staged must exist at the time of staging
        or an `IOError` is thrown. If `False`, this is silently ignored.
        Default `False`.

    Examples
    --------
    >>> fname = "/home/jill/output.dat"
    >>> tmpdir = "/tmp"
    >>> with StagedOutFile(fname, tmpdir=tmpdir) as sf:
    ...     with open(sf.path, 'w') as fobj:
    ...         fobj.write("some data")
    """
    def __init__(self, fname, tmpdir=None, must_exist=False):
        self.must_exist = must_exist
        self.was_staged_out = False
        self._set_paths(fname, tmpdir=tmpdir)

    def _set_paths(self, fname, tmpdir=None):
        fname = os.path.realpath(expand_path(fname))

        self.final_path = fname

        if tmpdir is not None:
            self.tmpdir = os.path.realpath(expand_path(tmpdir))
        else:
            self.tmpdir = tmpdir

        fdir = os.path.dirname(self.final_path)

        if self.tmpdir is None:
            self.is_temp = False
            self.path = self.final_path
        else:
            if not os.path.exists(self.tmpdir):
                os.makedirs(self.tmpdir)

            bname = os.path.basename(fname)
            self.path = os.path.join(self.tmpdir, bname)

            if self.tmpdir == fdir:
                # the user sent tmpdir as the final output dir, no
                # staging is performed
                self.is_temp = False
            else:
                self.is_temp = True

    def stage_out(self):
        """If a tempdir was used, move the file to its final destination.

        Note that you normally would not call this yourself, but rather use a
        context manager, in which case this method is called for you.
        """
        if self.is_temp and not self.was_staged_out:
            if not os.path.exists(self.path):
                if self.must_exist:
                    mess = "temporary file not found: %s" % self.path
                    raise IOError(mess)
                else:
                    return

            if os.path.exists(self.final_path):
                print("removing existing file:", self.final_path)
                os.remove(self.final_path)

            makedirs_fromfile(self.final_path)

            print(
                "staging out '%s' -> '%s'" % (self.path, self.final_path))
            shutil.move(self.path, self.final_path)

        self.was_staged_out = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stage_out()


class StagedInFile(object):
    """
    A class to stage a file in to local disk for reading.

    parameters
    ----------
    fname: string
        original file location
    tmpdir: string, optional
        If not sent or None, no staging is done and the original file
        path is used.

    examples
    --------
    # using a context for the staged file
    fname="/home/jill/output.dat"
    tmpdir="/tmp"
    with StagedInFile(fname,tmpdir=tmpdir) as sf:
        with open(sf.path) as fobj:
            # read some data

    """
    def __init__(self, fname, tmpdir=None):

        self._set_paths(fname, tmpdir=tmpdir)
        self.stage_in()

    def _set_paths(self, fname, tmpdir=None):
        fname = os.path.realpath(expand_path(fname))

        self.original_path = fname

        if tmpdir is not None:
            self.tmpdir = os.path.realpath(expand_path(tmpdir))
        else:
            self.tmpdir = tmpdir

        self.was_staged_in = False
        self._stage_in = False

        if self.tmpdir is not None:
            bdir, bname = os.path.split(self.original_path)
            self.path = os.path.join(self.tmpdir, bname)

            if self.tmpdir == bdir:
                # the user sent tmpdir as the source dir, no
                # staging is performed
                self._stage_in = False
            else:
                self._stage_in = True
        else:
            self.path = self.original_path

    def stage_in(self):
        """
        make a local copy of the file
        """
        import shutil

        if self._stage_in:
            if not os.path.exists(self.original_path):
                raise IOError("file not found:", self.original_path)

            if os.path.exists(self.path):
                print("removing existing file:", self.path)
                os.remove(self.path)
            else:
                makedirs_fromfile(self.path)

            print("staging in", self.original_path, "->", self.path)
            shutil.copy(self.original_path, self.path)

            self.was_staged_in = True

    def cleanup(self):
        if self.was_staged_in and os.path.exists(self.path):
            print("removing temporary file:", self.path)
            os.remove(self.path)
            self.was_staged_in = False

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.cleanup()
