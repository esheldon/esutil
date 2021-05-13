import numpy as np
import esutil as eu


def test_hist():
    rng = np.random.RandomState(35)
    n = 100
    data = rng.uniform(low=-1.0, high=1.0, size=n)

    h, rev = eu.stat.histogram(data, binsize=0.1, rev=True)

    hexpect = np.array(
        [5, 3, 8, 3, 7, 5, 7, 5, 1, 6, 8, 3, 7, 5, 4, 4, 4, 6, 4, 5],
        dtype='i8',
    )
    revexpect = np.array(
        [21, 26, 29, 37, 40, 47, 52, 59, 64, 65, 71, 79, 82, 89, 94, 98, 102, 106, 112, 116, 121, 75, 25, 94, 80, 11, 23, 95, 29, 5, 73, 66, 44, 57, 64, 69, 20, 58, 33, 16, 13, 81, 19, 2, 49, 42, 39, 71, 7, 3, 99, 41, 14, 1, 56, 98, 24, 45, 82, 48, 87, 86, 22, 34, 68, 77, 78, 0, 61, 84, 52, 31, 60, 21, 89, 28, 76, 43, 88, 92, 72, 96, 51, 37, 85, 6, 46, 40, 93, 54, 70, 55, 9, 8, 79, 91, 18, 62, 59, 26, 10, 50, 4, 30, 47, 35, 90, 74, 38, 27, 32, 97, 12, 53, 65, 63, 83, 36, 17, 15, 67],  # noqa
        dtype='i8',
    )

    assert np.all(hexpect == h)
    assert np.all(revexpect == rev)
