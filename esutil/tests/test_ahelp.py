import numpy as np
from esutil.numpy_util import ahelp


_SCHEMA = [
    ("c_b1", "?"),
    ("c_u1", "u1"),
    ("c_u2", "u2"),
    ("c_u4", "u4"),
    ("c_u8", "u8"),
    ("c_i2", "i2"),
    ("c_i4", "i4"),
    ("c_i8", "i8"),
    ("c_f4", "f4"),
    ("c_f8", "f8"),
    ("c_c8", "c8"),
    ("c_c16", "c16"),
    ("c_str", "S16"),
    ("c_ustr", "U16"),
    ("a_f4_3", "f4", (3,)),
    ("a_f4_6", "f4", (6,)),
    ("a_f8_4", "f8", (4,)),
    ("a_f4_2x3", "f4", (2, 3)),
    ("a_f4_6x6", "f4", (6, 6)),
    ("a_f8_3x3", "f8", (3, 3)),
    ("a_f8_5x2", "f8", (5, 2)),
    ("x_f4_0", "f4"),
    ("x_f8_0", "f8"),
    ("x_i4_0", "i4"),
    ("x_i8_0", "i8"),
    ("x_f4_1", "f4"),
    ("x_f8_1", "f8"),
    ("x_i2_0", "i2"),
    ("x_u4_0", "u4"),
    ("x_f8_2", "f8"),
    ("x_f4_2", "f4"),
    ("v_str", "O"),
    ("v_ustr", "O"),
    ("v_f4", "O"),
]
VAR_DTYPES = {"v_str": "S", "v_ustr": "U", "v_f4": "f4"}


def catalog_arrays(nrows, seed=0, exclude=()):
    """
    Build a ``nrows``-row structured array for the catalog schema plus
    the ``var_dtypes`` sidecar for the VLA columns.  Integer/float
    content is random (irrelevant to read speed); VLA cells are
    variable-length (strings 5-20 bytes, f4 arrays 1-10 elements).
    ``exclude`` drops named columns (see :func:`catalog_dtype`).
    """
    rng = np.random.default_rng(seed)
    dt = catalog_dtype(exclude)
    data = np.empty(nrows, dtype=dt)
    vd = {k: v for k, v in VAR_DTYPES.items() if k not in exclude}
    for name in dt.names:
        if name in vd:
            continue
        base = dt[name].base
        shape = (nrows,) + dt[name].shape
        k = base.kind
        if k == "b":
            data[name] = rng.integers(0, 2, size=shape).astype(bool)
        elif k in ("i", "u"):
            data[name] = rng.integers(0, 1000, size=shape).astype(base)
        elif k == "f":
            data[name] = rng.standard_normal(shape).astype(base)
        elif k == "c":
            re = rng.standard_normal(shape)
            im = rng.standard_normal(shape)
            data[name] = (re + 1j * im).astype(base)
        elif k == "S":
            w = base.itemsize
            sb = rng.integers(65, 91, size=(nrows, w), dtype=np.uint8)
            data[name] = np.ascontiguousarray(sb).view(f"S{w}").reshape(nrows)
        elif k == "U":
            w = base.itemsize // 4
            sb = rng.integers(65, 91, size=(nrows, w), dtype=np.uint8)
            s = np.ascontiguousarray(sb).view(f"S{w}").reshape(nrows)
            data[name] = np.char.decode(s)
    if "v_str" in vd:
        _fill_vla(data, "v_str", rng, 5, 21, "s")
    if "v_ustr" in vd:
        _fill_vla(data, "v_ustr", rng, 5, 21, "u")
    if "v_f4" in vd:
        _fill_vla(data, "v_f4", rng, 1, 11, "f")
    return data, vd


def catalog_dtype(exclude=()):
    """
    The structured dtype for the type-exhaustive catalog schema.
    ``exclude`` drops named columns (e.g. ``("c_c16",)`` for the
    compressed-table benchmark, since rustfits's ZTABLE codecs don't
    support 16-byte elements).
    """
    fields = []
    for col in _SCHEMA:
        if col[0] in exclude:
            continue
        if len(col) == 3:
            fields.append((col[0], col[1], col[2]))
        else:
            fields.append((col[0], col[1]))
    return np.dtype(fields)


def _fill_vla(data, name, rng, lo, hi, kind):
    n = len(data)
    lens = rng.integers(lo, hi, size=n)
    off = np.concatenate(([0], np.cumsum(lens)))
    total = int(off[-1])
    out = np.empty(n, dtype=object)
    if kind in ("s", "u"):
        pool = rng.integers(97, 123, size=total, dtype=np.uint8).tobytes()
        for i in range(n):
            cell = pool[off[i]: off[i + 1]]
            out[i] = cell.decode() if kind == "u" else cell
    else:
        pool = rng.standard_normal(total).astype("f4")
        for i in range(n):
            out[i] = pool[off[i]: off[i + 1]]
    data[name] = out


def test_structured():

    cat, _ = catalog_arrays(nrows=3, seed=3232)

    print('-' * 70)
    ahelp(cat)


def test_dict():
    cat, _ = catalog_arrays(nrows=3, seed=3232)

    d = {}
    for name in cat.dtype.names:
        d[name] = cat[name]

    print('-' * 70)
    ahelp(d)


if __name__ == '__main__':
    test_structured()
    test_dict()
