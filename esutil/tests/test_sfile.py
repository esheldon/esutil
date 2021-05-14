import sys
import os
from esutil import sfile
import numpy as np
import tempfile
import pytest

if sys.version_info > (3, 0, 0):
    stype = (str, bytes)
else:
    stype = str


def get_data():
    nvec = 2
    ashape = (21, 21)
    Sdtype = "S6"
    # all currently available types, scalar, 1-d and 2-d array columns
    dtype = [
        ("u1scalar", "u1"),
        ("i1scalar", "i1"),
        ("u2scalar", "u2"),
        ("i2scalar", "i2"),
        ("u4scalar", "u4"),
        ("i4scalar", "i4"),
        ("i8scalar", "i8"),
        ("f4scalar", "f4"),
        ("f8scalar", "f8"),
        ("u1vec", "u1", nvec),
        ("i1vec", "i1", nvec),
        ("u2vec", "u2", nvec),
        ("i2vec", "i2", nvec),
        ("u4vec", "u4", nvec),
        ("i4vec", "i4", nvec),
        ("i8vec", "i8", nvec),
        ("f4vec", "f4", nvec),
        ("f8vec", "f8", nvec),
        ("u1arr", "u1", ashape),
        ("i1arr", "i1", ashape),
        ("u2arr", "u2", ashape),
        ("i2arr", "i2", ashape),
        ("u4arr", "u4", ashape),
        ("i4arr", "i4", ashape),
        ("i8arr", "i8", ashape),
        ("f4arr", "f4", ashape),
        ("f8arr", "f8", ashape),
        ("Sscalar", Sdtype),
        ("Svec", Sdtype, nvec),
        ("Sarr", Sdtype, ashape),
    ]

    nrows = 4
    data = np.zeros(nrows, dtype=dtype)
    names = data.dtype.names

    dtypes = [
        "u1", "i1", "u2", "i2", "u4", "i4",
        "i8", "f4", "f8", "c8", "c16",
    ]
    for t in dtypes:

        sname = t + "scalar"
        vname = t + "vec"
        aname = t + "arr"

        if t in ["c8", "c16"]:
            if sname in names:
                data[sname] = [
                    complex(i + 1, (i + 1) * 2) for i in range(nrows)
                ]

            if vname in names:
                for row in range(nrows):
                    for i in range(nvec):
                        index = (row + 1) * (i + 1)
                        data[vname][row, i] = complex(index, index * 2)

            if aname in names:
                for row in range(nrows):
                    for i in range(ashape[0]):
                        for j in range(ashape[1]):
                            index = (row + 1) * (i + 1) * (j + 1)
                            data[aname][row, i, j] = complex(index, index * 2)

        else:
            if sname in names:
                data[sname] = 1 + np.arange(nrows, dtype=t)

            if vname in names:
                data[vname] = (
                    1 + np.arange(nrows * nvec, dtype=t).reshape(nrows, nvec)
                )

            if aname in names:
                arr = 1 + np.arange(nrows * ashape[0] * ashape[1], dtype=t)
                data[aname] = arr.reshape(nrows, ashape[0], ashape[1])

    for t in ["b1"]:
        sname = t + "scalar"
        vname = t + "vec"
        aname = t + "arr"

        if sname in names:
            data[sname] = (np.arange(nrows) % 2 == 0).astype("?")

        if vname in names:
            data[vname] = (
                (np.arange(nrows * nvec) % 2 == 0).astype("?").reshape(nrows, nvec)  # noqa
            )

        if aname in names:
            arr = (np.arange(nrows * ashape[0] * ashape[1]) % 2 == 0).astype("?")  # noqa
            data[aname] = arr.reshape(nrows, ashape[0], ashape[1])

    if "Sscalar" in data.dtype.names:
        data["Sscalar"] = ["hello", "world", "good", "bye"]
    if "Svec" in data.dtype.names:
        data["Svec"][:, 0] = "hello"
        data["Svec"][:, 1] = "world"

    if "Sarr" in data.dtype.names:
        s = 1 + np.arange(nrows * ashape[0] * ashape[1])
        s = ["%s" % el for el in s]
        data["Sarr"] = np.array(s).reshape(nrows, ashape[0], ashape[1])

    swap_dtype = []
    for d in dtype:
        name = d[0]
        if "S" not in name:
            tp = ">" + d[1]
        else:
            tp = d[1]

        if len(d) == 3:
            nd = (d[0], tp, d[2])
        else:
            nd = (d[0], tp)

        swap_dtype.append(nd)

    swap_data = np.zeros(nrows, dtype=swap_dtype)

    for n in swap_data.dtype.names:
        swap_data[n] = data[n]
    swap_data = data

    header = {
        "name": "Janet Smith",
        "age": 32,
        "longitude": 124.325,
        "latitude": -18.584,
    }

    return data, swap_data, header


def is_object(arr):
    if arr.dtype.descr[0][1][1] == 'O':
        return True
    else:
        return False


def get_prefix(name, delim, doswap):
    if delim is not None:
        delimstr = delim.replace(" ", "space")
        delimstr = delim.replace(" ", "space")
        delimstr = delimstr.replace("\t", "tab")
    else:
        delimstr = "binary"

    prefix = "sfile-%s-%s-%s-" % (name, delimstr, doswap)
    return prefix


def get_testfile(name, delim, doswap):
    prefix = get_prefix(name, delim, doswap)
    fname = tempfile.mktemp(prefix=prefix, suffix=".rec")
    return fname


def compare_names(read_names, true_names, lower=False, upper=False):
    for nread, ntrue in zip(read_names, true_names):
        if lower:
            tname = ntrue.lower()
            mess = "lower: '%s' vs '%s'" % (nread, tname)
        else:
            tname = ntrue.upper()
            mess = "upper: '%s' vs '%s'" % (nread, tname)
        assert nread == tname, mess


def compare_header(h1, h2):
    """
    The headers as dictionaries
    """

    for key in h1:
        assert h1[key] == h2[key], "testing header key: '%s'" % key

        v1 = h1[key]
        v2 = h2[key]
        assert v1 == v2, "testing header key '%s'" % key


def compare_array_tol(arr1, arr2, tol, name):
    mess = "testing arrays '%s' shapes are equal: " "input %s, read: %s" % (
        name,
        arr1.shape,
        arr2.shape,
    )

    assert arr1.shape == arr2.shape, mess

    adiff = np.abs((arr1 - arr2) / arr1)
    maxdiff = adiff.max()
    res = np.where(adiff > tol)
    for i, w in enumerate(res):
        mess = (
            "testing array '%s' dim %d are "
            "equal within tolerance %e, found "
            "max diff %e" % (name, i, tol, maxdiff)
        )

        assert w.size == 0, mess


def compare_array(arr1, arr2, name):
    # print("\narr1:",arr1)
    # print("arr2:",arr2)
    mess = "testing arrays '%s' shapes are equal: " "input %s, read: %s" % (
        name,
        arr1.shape,
        arr2.shape,
    )

    assert arr1.shape == arr2.shape, mess

    res = np.where(arr1 != arr2)
    for i, w in enumerate(res):
        assert w.size == 0, "testing array '%s' dim %d are equal" % (name, i)


def compare_rec(rec1, rec2, name):
    for f in rec1.dtype.names:
        mess = (
            "testing '%s' field '%s' shapes are equal: "
            "input %s, read: %s"
        )
        mess = mess % (
            name,
            f,
            rec1[f].shape,
            rec2[f].shape,
        )

        assert rec1[f].shape == rec2[f].shape, mess

        res = np.where(rec1[f] != rec2[f])
        for w in res:
            assert w.size == 0, "testing column %s" % f


def compare_rec_subrows(rec1, rec2, rows, name):
    for f in rec1.dtype.names:
        mess = (
            "testing '%s' field '%s' shapes are equal: "
            "input %s, read: %s"
        )
        mess = mess % (
            name,
            f,
            rec1[f].shape,
            rec2[f].shape,
        )
        assert rec1[f][rows].shape == rec2[f].shape, mess

        res = np.where(rec1[f][rows] != rec2[f])
        for w in res:
            assert w.size == 0, "testing column %s" % f


def compare_rec_with_var(rec1, rec2, name, rows=None):
    """

    First one *must* be the one with object arrays

    Second can have fixed length

    both should be same number of rows

    """

    if rows is None:
        rows = np.arange(rec2.size)
        assert rec1.size == rec2.size, (
            "testing '%s' same number of rows" % name
        )

    # rec2 may have fewer fields
    for f in rec2.dtype.names:

        # f1 will have the objects
        if is_object(rec1[f]):
            compare_object_array(
                rec1[f],
                rec2[f],
                "testing '%s' field '%s'" % (name, f),
                rows=rows,
            )
        else:
            compare_array(
                rec1[f][rows],
                rec2[f],
                "testing '%s' num field '%s' equal" % (name, f),
            )


def compare_object_array(arr1, arr2, name, rows=None):
    """
    The first must be object
    """
    if rows is None:
        rows = np.arange(arr1.size)

    for i, row in enumerate(rows):
        if isinstance(arr2[i], stype):
            assert arr1[row] == arr2[i], "%s str el %d equal" % (name, i)
        else:
            delement = arr2[i]
            orig = arr1[row]
            s = len(orig)
            compare_array(
                orig,
                delement[0:s],
                "%s num el %d equal" % (name, i),
            )


def compare_rec_with_var_subrows(rec1, rec2, name, rows):
    """

    Second one must be the one with object arrays

    """
    for f in rec1.dtype.names:
        if is_object(rec2[f]):

            for i in range(rec2.size):
                if isinstance(rec2[f][i], stype):
                    assert (
                        rec1[f][i] == rec2[f][i]
                    ), "testing '%s' str field '%s' el %d equal" % (name, f, i)
                else:
                    delement = rec1[f][i]
                    orig = rec2[f][i]
                    s = orig.size
                    tname = (
                        "testing '%s' num field '%s' "
                        "el %d equal" % (name, f, i)
                    )
                    compare_array(
                        orig,
                        delement[0:s],
                        tname,
                    )
        else:
            compare_array(
                rec1[f],
                rec2[f],
                "testing '%s' num field '%s' equal" % (name, f),
            )


@pytest.mark.parametrize('delim', [None, ",", ":", "\t", " "])
def test_sfile_writeread(delim):
    """
    Test a basic table write, data and a header, then reading back in to
    check the values
    """

    data, swap_data, header = get_data()

    for doswap in [False, True]:

        if doswap:
            data = swap_data
        else:
            data = data

        try:
            fname = get_testfile("testWriteRead", delim, doswap)
            with sfile.SFile(fname, mode="w", delim=delim) as sf:

                try:
                    sf.write(data, header=header)
                    write_success = True
                except:  # noqa
                    write_success = False

                assert write_success, (
                    "testing write does not raise an "
                    "error doswap: %s" % doswap
                )

            with sfile.SFile(fname) as sf:
                d, h = sf.read(header=True)

            compare_rec(data, d, "table read/write")
            compare_header(header, h)

            # see if our convenience functions are working
            sfile.write(fname, data, delim=delim, header=header)
            d, h = sfile.read(fname, header=True)
            tname = (
                "test read/write with convenience "
                "functions doswap: %s" % doswap
            )
            compare_rec(data, d, tname)
            compare_header(header, h)

        finally:
            if os.path.exists(fname):
                os.remove(fname)


def test_sfile_subsets():
    """
    Test reading subsets and slices
    """

    data, swap_data, header = get_data()
    for delim in [None, ",", ":", "\t", " "]:
        for doswap in [False, True]:
            sstr = " doswap: %s" % doswap

            if doswap:
                data = swap_data
            else:
                data = data

            try:
                fname = get_testfile("testSubsets", delim, doswap)
                with sfile.SFile(fname, mode="w", delim=delim) as sf:
                    # initial write
                    sf.write(data)

                with sfile.SFile(fname) as sf:

                    # row slices
                    d = sf[:]
                    compare_rec(data, d, "row range all" + sstr)

                    d = sf[1:3]
                    compare_rec(data[1:3], d, "row range" + sstr)

                    d = sf[0:4:2]
                    compare_rec(data[0:4:2], d, "row range step 2" + sstr)

                    # test reading single columns
                    for f in data.dtype.names:
                        d = sf[f][:]

                        d = sf.read(columns=f)
                        tname = (
                            "test read all rows %s column subset "
                            "scalar name %s" % (f, sstr)
                        )
                        compare_array(
                            data[f],
                            d,
                            tname,
                        )

                        d = sf.read(columns=[f])
                        tname = (
                            "test read all rows %s column "
                            "subset %s" % (f, sstr)
                        )
                        compare_array(
                            data[f],
                            d[f],
                            tname,
                        )

                        d = sf[f][:]
                        compare_array(
                            data[f],
                            d,
                            ("test read all rows %s column "
                             "subset slice %s" % (f, sstr)),
                        )

                        rows = [1, 3]
                        d = sf.read(columns=f, rows=rows)
                        compare_array(
                            data[f][rows],
                            d,
                            "test read rows %s column subset scalar name %s"
                            % (f, sstr),
                        )

                        d = sf.read(columns=[f], rows=rows)
                        compare_array(
                            data[f][rows],
                            d[f],
                            "test read rows %s column subset %s" % (f, sstr),
                        )

                    cols = ["u2scalar", "f4vec", "Sarr"]

                    # column subset
                    d = sf.read(columns=cols)
                    for f in d.dtype.names:
                        compare_array(
                            data[f], d[f], "test column list %s %s" % (f, sstr)
                        )

                    # column subset and rows subset
                    rows = [1, 3]
                    d = sf[cols][rows]
                    for f in d.dtype.names:
                        compare_array(
                            data[f][rows],
                            d[f],
                            ("test column list %s row "
                             "subset slice %s" % (f, sstr)),
                        )

                    d = sf.read(rows=rows, columns=cols)
                    for f in d.dtype.names:
                        compare_array(
                            data[f][rows],
                            d[f],
                            "test column list %s row subset %s" % (f, sstr),
                        )

                    # combined with row slices

                    d = sf[cols][:]
                    for f in d.dtype.names:
                        compare_array(
                            data[f], d[f],
                            "test column list %s slice %s" % (f, sstr)
                        )

                    d = sf[cols][1:3]
                    for f in d.dtype.names:
                        compare_array(
                            data[f][1:3],
                            d[f],
                            "test column list %s row slice %s" % (f, sstr),
                        )

                    d = sf[cols][0:4:2]
                    for f in d.dtype.names:
                        compare_array(
                            data[f][0:4:2],
                            d[f],
                            ("test column list %s row slice "
                             "step 2 %s" % (f, sstr)),
                        )

            finally:
                if os.path.exists(fname):
                    os.remove(fname)


def test_sfile_append():
    """
    Test creating a table and appending new rows.
    """

    data, swap_data, header = get_data()
    for delim in [None, ",", ":", "\t", " "]:
        for doswap in [False, True]:

            if doswap:
                data = swap_data
            else:
                data = data

            try:
                fname = get_testfile("testAppend", delim, doswap)
                with sfile.SFile(fname, mode="w", delim=delim) as sf:
                    # initial write
                    sf.write(data)

                    # appending
                    data2 = data.copy()
                    data2["f4scalar"] = 3
                    sf.write(data2)

                with sfile.SFile(fname) as sf:

                    d = sf.read()
                    assert d.size == data.size * 2

                    compare_rec(
                        data, d[0: data.size], "Comparing initial write"
                    )
                    compare_rec(
                        data2, d[data.size:], "Comparing appended data"
                    )

            finally:
                if os.path.exists(fname):
                    os.remove(fname)
