import os
import tempfile
import esutil as eu
import pickle
import copy


def test_cosmology_smoke():
    cosmo = eu.cosmology.Cosmo(
        H0=70, flat=True, omega_m=0.27, omega_l=0.73,
        omega_k=0.0,
    )
    zmin, zmax = 0.1, 0.2
    cosmo.Dc(zmin, zmax)
    cosmo.Dm(zmin, zmax)
    cosmo.Da(zmin, zmax)
    cosmo.Dl(zmin, zmax)
    cosmo.dV(zmax)
    cosmo.V(zmin, zmax)
    cosmo.distmod(zmax)
    cosmo.sigmacritinv(zl=zmin, zs=zmax)
    cosmo.Ez_inverse(zmax)
    cosmo.Ezinv_integral(zmin, zmax)


def test_cosmology_copy():
    cosmo = eu.cosmology.Cosmo(
        H0=70, flat=True, omega_m=0.27, omega_l=0.73,
        omega_k=0.0,
    )

    cosmo_copy = cosmo.copy()
    _compare(cosmo, cosmo_copy)

    cosmo_copy = copy.copy(cosmo)
    _compare(cosmo, cosmo_copy)

    cosmo_copy = copy.deepcopy(cosmo)
    _compare(cosmo, cosmo_copy)


def _compare(cosmo1, cosmo2):
    assert cosmo1.H0() == cosmo2.H0()
    assert cosmo1.DH() == cosmo2.DH()
    assert cosmo1.flat() == cosmo2.flat()
    assert cosmo1.omega_m() == cosmo2.omega_m()
    assert cosmo1.omega_l() == cosmo2.omega_l()
    assert cosmo1.omega_k() == cosmo2.omega_k()


def _pickle_compare(cosmo):
    with tempfile.TemporaryDirectory() as temp_dir:
        fn = os.path.join(temp_dir, 'cosmo.pkl')
        with open(fn, 'wb') as f:
            pickle.dump(cosmo, f)
        with open(fn, 'rb') as f:
            _cosmo = pickle.load(f)

        _compare(cosmo, _cosmo)


def test_pickle_compare1():
    """
    Test case 1
    """
    cosmo = eu.cosmology.Cosmo(
        H0=70, flat=True, omega_m=0.27, omega_l=0.73,
        omega_k=0.0,
    )
    _pickle_compare(cosmo=cosmo)


def test_pickle_compare2():
    """
    Test case 2
    """
    cosmo = eu.cosmology.Cosmo(
        H0=67, flat=False, omega_m=0.32, omega_l=0.67,
        omega_k=0.01,
    )
    _pickle_compare(cosmo=cosmo)


def test_pickle_compare3():
    """
    Test case 3
    """
    cosmo = eu.cosmology.Cosmo(
        h=1, omega_m=0.3, omega_l=0.7,
        omega_k=None,
    )
    _pickle_compare(cosmo=cosmo)
