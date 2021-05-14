import pytest
import numpy as np
import esutil as eu


def gaussfunc(x):
    """
    for testing function stuff
    """
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def test_random_generator_smoke():
    """
    Do some tests of the random generator
    """

    # first a function
    funcgen = eu.random.Generator(gaussfunc, xrange=[-4.5, 4.5], nx=100)

    nrand = 100000

    _ = funcgen.genrand(nrand)

    # now points
    x = eu.numpy_util.arrscl(np.arange(100, dtype='f4'), -4.5, 4.5)
    pofx = gaussfunc(x)

    pointgen = eu.random.Generator(pofx, x=x)
    _ = pointgen.genrand(nrand)


@pytest.mark.parametrize('seed', [12, 988, 131, 7711])
def test_cholesky(seed):

    rng = np.random.RandomState(seed)
    cov = np.array(
        [[1.0, 0.1, 0.1],
         [0.1, 2.0, 0.1],
         [0.1, 0.1, 3.0]]
    )
    means = [5.0, 4.0, 8.0]

    n = 1000000

    r = eu.random.cholesky_sample(cov, n, means=means, dist=rng.randn)

    dcov = np.cov(r.T)
    print(cov)
    print(dcov)

    tol = 0.02
    for i in range(3):
        for j in range(3):
            assert abs(dcov[i, j] - cov[i, j]) < tol, '%d, %d' % (i, j)
