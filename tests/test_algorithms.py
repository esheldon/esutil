import numpy as np
import esutil as eu


def test_quicksort():
    rng = np.random.RandomState(458)
    num = 1000
    data = rng.uniform(size=num)
    data_orig = data.copy()

    eu.algorithm.quicksort(data)

    s = data_orig.argsort()

    assert np.all(data == data_orig[s])


def test_isplit():
    num = 135
    nchunks = 11

    subs = eu.algorithm.isplit(num=num, nchunks=nchunks)
    assert subs.size == 11

    assert np.all(
        subs['start'] == [0, 13, 26, 39, 51, 63, 75, 87, 99, 111, 123]
    )

    assert np.all(
        subs['end'] == [13, 26, 39, 51, 63, 75, 87, 99, 111, 123, 135]
    )
