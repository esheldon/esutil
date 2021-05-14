import esutil as eu


def test_integrate():
    npts = 10
    x, w = eu.integrate.gauleg(1.0, 2.0, npts)

    assert eu.integrate.have_cgauleg
    assert x.size == npts
    assert w.size == npts
