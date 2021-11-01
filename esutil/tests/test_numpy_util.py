import pytest
import numpy as np
import esutil as eu


def test_combine_fields():

    num = 3
    a1 = np.zeros(num, dtype=[('x', 'f4'), ('y', 'f4')])
    a2 = np.zeros(num, dtype=[('id', 'i4'), ('name', 'U4')])

    a = eu.numpy_util.combine_fields((a1, a2))

    assert a.size == a1.size

    for name in a1.dtype.names:
        assert np.all(a[name] == a1[name])
    for name in a2.dtype.names:
        assert np.all(a[name] == a2[name])


def test_combine_fields_errors():
    a1 = np.zeros(3, dtype=[('x', 'f4'), ('y', 'f4')])
    a2 = np.zeros(2, dtype=[('id', 'i4'), ('name', 'U4')])

    with pytest.raises(ValueError):
        eu.numpy_util.combine_fields((a1, a2))

    a1 = np.zeros(3, dtype=[('x', 'f4'), ('y', 'f4')])
    a2 = np.zeros(3, dtype=[('x', 'f8'), ('id', 'i4'), ('name', 'U4')])

    with pytest.raises(ValueError):
        eu.numpy_util.combine_fields((a1, a2))
