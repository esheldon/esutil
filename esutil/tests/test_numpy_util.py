import pytest
import numpy as np
import esutil as eu


def test_combine_fields():

    num = 3
    a1 = np.zeros(num, dtype=[('x', 'f4'), ('y', 'f4')])
    a2 = np.zeros(num, dtype=[('id', 'i4'), ('name', 'U4')])
    a3 = np.zeros(num, dtype=[('q', 'i8')])

    a1['x'] = np.linspace(1, 2, num) * 3
    a1['y'] = np.linspace(1, 2, num)
    a2['id'] = np.arange(num)
    a2['name'] = ['stuff', 'and', 'things']
    a3['q'] = -np.arange(num)

    a = eu.numpy_util.combine_fields((a1, a2, a3))

    assert a.size == a1.size

    for name in a1.dtype.names:
        assert np.all(a[name] == a1[name])
    for name in a2.dtype.names:
        assert np.all(a[name] == a2[name])
    for name in a3.dtype.names:
        assert np.all(a[name] == a3[name])


def test_combine_fields_errors():
    a1 = np.zeros(3, dtype=[('x', 'f4'), ('y', 'f4')])
    a2 = np.zeros(2, dtype=[('id', 'i4'), ('name', 'U4')])

    with pytest.raises(ValueError):
        eu.numpy_util.combine_fields((a1, a2))

    a1 = np.zeros(3, dtype=[('x', 'f4'), ('y', 'f4')])
    a2 = np.zeros(3, dtype=[('x', 'f8'), ('id', 'i4'), ('name', 'U4')])

    with pytest.raises(ValueError):
        eu.numpy_util.combine_fields((a1, a2))


def test_split_array():
    arr = np.arange(25)
    nper = 3

    chunks = eu.numpy_util.splitarray(nper, arr)
    assert len(chunks) == 9

    assert np.all(chunks[0] == [0, 1, 2])
    assert np.all(chunks[1] == [3, 4, 5])
    assert np.all(chunks[2] == [6, 7, 8])
    assert np.all(chunks[3] == [9, 10, 11])
    assert np.all(chunks[4] == [12, 13, 14])
    assert np.all(chunks[5] == [15, 16, 17])
    assert np.all(chunks[6] == [18, 19, 20])
    assert np.all(chunks[7] == [21, 22, 23])
    assert np.all(chunks[8] == [24])


@pytest.mark.parametrize('presorted', [True, False])
def test_match_int(presorted):
    a1 = np.array([3, 10, 8, 4, 7])
    a2 = np.array([8, 3])

    if not presorted:
        ind = np.array([4, 1, 0, 2, 3])
        m1, m2 = eu.numpy_util.match(a1[ind], a2)
        assert np.all(m1 == [3, 2])
    else:
        m1, m2 = eu.numpy_util.match(a1, a2)
        assert np.all(m1 == [2, 0])


@pytest.mark.parametrize('presorted', [True, False])
def test_match_float(presorted):
    a1 = np.array([1.25, 6.61, 8.51, 9.91, 11.25])
    a2 = np.array([6.61, 9.91])

    if not presorted:
        ind = np.array([4, 1, 0, 2, 3])
        m1, m2 = eu.numpy_util.match(a1[ind], a2)
        assert np.all(m1 == [1, 4])
    else:
        m1, m2 = eu.numpy_util.match(a1, a2)
        assert np.all(m1 == [1, 3])


@pytest.mark.parametrize('presorted', [True, False])
def test_match_str(presorted):
    a1 = np.array(['blah', 'goodbye', 'hello', 'stuff', 'things'])
    a2 = np.array(['goodbye', 'things', 'zz'])

    if not presorted:
        ind = np.array([3, 4, 0, 2, 1])
        m1, m2 = eu.numpy_util.match(a1[ind], a2)
        assert np.all(m1 == [4, 1])
    else:
        m1, m2 = eu.numpy_util.match(a1, a2)
        assert np.all(m1 == [1, 4])


@pytest.mark.parametrize('presorted', [True, False])
def test_match_nomatch(presorted):
    a1 = np.array(['blah', 'goodbye', 'hello', 'stuff', 'things'])
    a2 = np.array(['zz', 'bb'])

    if not presorted:
        ind = np.array([3, 4, 0, 2, 1])
        m1, m2 = eu.numpy_util.match(a1[ind], a2)
    else:
        m1, m2 = eu.numpy_util.match(a1, a2)

    assert m1.size == 0 and m2.size == 0


def test_match_scalar():
    a1 = np.array([3, 4, 5])
    a2 = 4

    m1, m2 = eu.numpy_util.match(a1, a2)
    assert m1.size == 1 and m2.size == 1

    m1, m2 = eu.numpy_util.match(a2, a1)
    assert m1.size == 1 and m2.size == 1

    a1 = 5
    a2 = 5
    m1, m2 = eu.numpy_util.match(a2, a1)
    assert m1.size == 1 and m2.size == 1

    a1 = 5
    a2 = 4
    m1, m2 = eu.numpy_util.match(a2, a1)
    assert m1.size == 0 and m2.size == 0
