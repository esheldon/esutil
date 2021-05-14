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
