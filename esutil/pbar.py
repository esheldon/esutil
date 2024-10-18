"""
heavily simplified version of the original simple tqdm from

https://github.com/noamraph/tqdm
"""
__all__ = ['pbar', 'prange', 'PBar']

import sys
import time


def pbar(iterable, desc='', total=None, leave=True, file=sys.stderr,
         mininterval=0.5, miniters=1, n_bars=20, simple=False):
    """
    Get an iterable object, and return an iterator which acts exactly like the
    iterable, but prints a progress meter and updates it every time a value is
    requested.

    parameters
    ----------
    iterable: iterable
        An iterable that is iterated over; the objects are yielded
    desc: string, optional
        An optional short string, describing the progress, that is added
        in the beginning of the line.
    total: int, optional
        Optional number of expected iterations. If not given,
        len(iterable) is used if it is defined.
    file: file-like object, optional
        A file-like object to output the progress message to. Default
        stderr
    leave: bool, optional
        If True, leave the remaining text from the progress.  If False,
        delete it.
    mininterval: float, optional
        default 0.5
    miniters: int, optional
        default 1

        If less than mininterval seconds or miniters iterations have passed
        since the last progress meter update, it is not updated again.
    n_bars: int
        Number of bars to show
    simple: bool
        If set to True, a simple countup 0 to 9 is show, independent
        of the other inputs.  Useful when you don't want to spam
        a log file with lots of data
            |0123456789|
    """
    if simple:
        return sbar(iterable, desc=desc, total=total, file=file)
    else:
        return _pbar_full(
            iterable, desc=desc, total=total, file=file,
            leave=leave, mininterval=mininterval, miniters=miniters,
            n_bars=n_bars, simple=simple,
        )


PBar = pbar


def _pbar_full(
    iterable, desc='', total=None, leave=True, file=sys.stderr,
    mininterval=0.5, miniters=1, n_bars=20, simple=False,
):
    """
    See docs for pbar
    """
    prefix = desc+': ' if desc else ''

    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    if simple:
        # we need enbed this because pbar is a generator
        assert total is not None, (
            'iterable must have len or send total for simple pbar'
        )
        _pnn(prefix + '|', file)
        plast = -1
        for i, obj in enumerate(iterable):
            yield obj
            i += 1

            p = int(i / total * 10)
            if p > plast:
                _pnn(p, file)
                plast = p

        print('|', file=file, flush=True)
        return

    sp = StatusPrinter(file)
    sp.print_status(prefix + format_meter(0, total, 0, n_bars=n_bars))

    start_t = last_print_t = time.time()
    last_print_n = 0
    n = 0
    for obj in iterable:
        yield obj
        # Now the object was created and processed, so we can print the meter.
        n += 1
        if n - last_print_n >= miniters:
            # We check the counter first, to reduce the overhead of time.time()
            cur_t = time.time()
            if cur_t - last_print_t >= mininterval:
                pstat = format_meter(
                    n,
                    total,
                    cur_t-start_t,
                    n_bars=n_bars,
                )
                sp.print_status(prefix + pstat)

                last_print_n = n
                last_print_t = cur_t

    if not leave:
        sp.print_status('')
        file.write('\r')
    else:
        if last_print_n < n:
            cur_t = time.time()

            pstat = format_meter(
                n,
                total,
                cur_t-start_t,
                n_bars=n_bars,
            )
            sp.print_status(prefix + pstat)
        file.write('\n')


def sbar(iterable, desc='', total=None, file=sys.stderr):
    """
    Get an iterable object, and return an iterator which acts exactly like the
    iterable, but prints progress meter.  This simple version does a countup 0
    to 9, independent of the other inputs.  Useful when you don't want to spam
    a log file with lots of data

            |0123456789|

    parameters
    ----------
    iterable: iterable
        An iterable that is iterated over; the objects are yielded
    desc: string, optional
        An optional short string, describing the progress, that is added
        in the beginning of the line.
    total: int, optional
        Optional number of expected iterations. If not given,
        len(iterable) is used if it is defined.
    file: file-like object, optional
        A file-like object to output the progress message to. Default
        stderr
    """

    prefix = desc+': ' if desc else ''

    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            raise RuntimeError(
                'for sbar you must send total= '
                'if the iterable does not provide length'
            )

    def pnn(d):
        print(d, end='', file=file, flush=True)

    pnn(prefix + '|')
    plast = -1
    for i, obj in enumerate(iterable):
        yield obj
        i += 1

        p = int(i / total * 10)

        if p > plast:
            pnn(p)
            plast = p

    print('|', file=file, flush=True)


def prange(*args, **kwargs):
    """
    A shortcut for writing pbar(range(...))

    Parameters
    ----------
    Same args as for range.   Extra keywords are sent to
    Pbar

    e.g.

    for i in prange(20):
        print(i)
        time.sleep(0.1)
    """
    return pbar(range(*args), **kwargs)


def pmap(fn, iterable, chunksize=1, nproc=1, **kw):
    """
    Execute the function on the inputs using multiple processes, while showing
    a progress bar.  The result is equivalent to doing

        list(map(fn, iterable))

    Parameters
    ----------
    fn: function
        The function to execute
    iterable: iterable data
        The data over which to iterate
    chunksize: int, optional
        Default 1. It is often must faster to send large
        chunks of data rather than 1.
    nproc: int, optional
        Number of processes to use, default 1
    **kw:
        Additional keyword arguments for the progress bar.
        See pbar for details

    Returns
    -------
    An list of data, the equivalent of
        list(map(fn, iterable))
    """
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=nproc) as ex:
        res = list(pbar(ex.map(fn, iterable, chunksize=chunksize), **kw))

    return res


def format_interval(t):
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return '%d:%02d:%02d' % (h, m, s)
    else:
        return '%02d:%02d' % (m, s)


def format_meter(n, total, elapsed, n_bars=20):
    # n - number of finished iterations
    # total - total number of iterations, or None
    # elapsed - number of seconds passed since start
    if n > total:
        total = None

    elapsed_str = format_interval(elapsed)

    if total:
        frac = float(n) / total

        bar_length = int(frac*n_bars)
        bar = '#'*bar_length + '-'*(n_bars-bar_length)

        percentage = '%3d%%' % (frac * 100)

        if elapsed > 0:
            it_per_second = n / elapsed  # iterations per second
            if it_per_second > 1:
                rate_str = f'{it_per_second:.3g} it/s'
            else:
                second_per_it = elapsed / n
                rate_str = f'{second_per_it:.3g} s/it'
        else:
            rate_str = '---'

        left_str = format_interval(elapsed / n * (total-n)) if n else '?'

        totstr = str(total)
        nfmt = '%' + str(len(totstr)) + 'd'
        meter_fmt = '|%s| ' + nfmt + '/' + nfmt + ' %s [%s<%s %s]'

        return meter_fmt % (
            bar, n, total, percentage, elapsed_str, left_str, rate_str
        )

    else:
        return '%d [elapsed: %s]' % (n, elapsed_str)


class StatusPrinter(object):
    def __init__(self, file):
        self.file = file
        self.last_printed_len = 0

    def print_status(self, s):
        self.file.write('\r'+s+' '*max(self.last_printed_len-len(s), 0))
        self.file.flush()
        self.last_printed_len = len(s)


def _pnn(d, file):
    print(d, end='', file=file, flush=True)
