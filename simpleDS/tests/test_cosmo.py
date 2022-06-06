# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Tests for cosmo module.

Relevent Cosmology units and transforms for Power Spectrum estimation.

All cosmological calculations and converions follow from Liu et al 2014a
Phys. Rev. D 90, 023018 or 	arXiv:1404.2596
"""
from __future__ import print_function

import numpy as np
import pytest
from astropy import constants as const
from astropy import units
from astropy.cosmology import Planck15
from simpleDS import cosmo

# the emission frequency of 21m photons
f21 = 1420405751.7667 * units.Hz


def test_calc_z_freq_unitless():
    """Test redshift calculation fails for non-quantity freqs."""
    test_freq = 0.15e9
    pytest.raises(TypeError, cosmo.calc_z, test_freq)


def test_calc_z_freq_wrong_unit():
    """Test redshift calculation fails for non-quantity freqs."""
    test_freq = 0.15e9 * units.m
    pytest.raises(units.UnitsError, cosmo.calc_z, test_freq)


def test_calc_z_value():
    """Test redshift calculation value."""
    test_freq = 0.15 * units.GHz
    test_z = f21 / test_freq - 1
    assert np.isclose(test_z, cosmo.calc_z(test_freq))


def test_calc_freq_value():
    """Test frequency calculation value."""
    test_freq = 0.15 * units.GHz
    test_z = f21 / test_freq.to("Hz") - 1
    assert np.isclose(test_freq.si.value, cosmo.calc_freq(test_z).value)


def test_calc_freq_unit():
    """Test calc_freq returns a Quantity with units Hz."""
    test_z = 10
    test_freq = cosmo.calc_freq(test_z)
    assert units.Hz == test_freq.si.unit


def test_u2kperp_unit():
    """Test units of u2kperp."""
    test_z = 7.6363125
    test_u = 10
    test_kperp = cosmo.u2kperp(test_u, test_z)
    assert (1.0 / units.Mpc).unit == test_kperp.unit


def test_u2kperp_val():
    """Test value of kperp calculation."""
    test_z = 7.6363125
    test_u = 10
    test_kperp = cosmo.u2kperp(test_u, test_z)
    test_val = 2 * np.pi * test_u / Planck15.comoving_transverse_distance(test_z)
    assert np.isclose(test_val.value, test_kperp.value)


def test_kperp2u_error():
    """Test kperp must be a wavenumber Quantity."""
    test_kperp = 0.01 * units.s
    test_z = 7.6363125
    pytest.raises(units.UnitsError, cosmo.kperp2u, test_kperp, test_z)


def test_kperp2u_no_unit():
    """Test kperp must be a Quantity."""
    test_kperp = 0.01
    test_z = 7.6363125
    pytest.raises(TypeError, cosmo.kperp2u, test_kperp, test_z)


def test_kperp2u_unit():
    """Test kperp2u units."""
    test_kperp = 0.01 * 1.0 / units.Mpc
    test_z = 7.6363125
    test_u = cosmo.kperp2u(test_kperp, test_z)
    assert test_u.unit.bases == []


def test_kperp2u_value():
    """Test kperp2u value."""
    test_kperp = 0.01 * 1.0 / units.Mpc
    test_z = 7.6363125
    test_u = cosmo.kperp2u(test_kperp, test_z)
    test_val = (
        test_kperp.value * Planck15.comoving_transverse_distance(test_z) / (2 * np.pi)
    )
    assert np.isclose(test_val.value, test_u.value)


def test_kperp2u2kperp_equal():
    """Test that kperp2u and u2kperp undo each other."""
    test_kperp = 0.01 / units.Mpc
    test_z = 7.6363125
    test_u = cosmo.kperp2u(test_kperp, test_z)
    test_u2kperp = cosmo.u2kperp(test_u, test_z)
    assert test_kperp == test_u2kperp


def test_eta2kparr_error():
    """Test eta must be Quantity."""
    test_eta = 200 * 1e-9
    test_z = 9.19508
    pytest.raises(TypeError, cosmo.eta2kparr, test_eta, test_z)


def test_eta2kparr_wrong_unit():
    """Test eta must be frequncy Quantity."""
    test_eta = 200 * 1e-9 * units.m
    test_z = 9.19508
    pytest.raises(units.UnitsError, cosmo.eta2kparr, test_eta, test_z)


def test_eta2kparr_val():
    """Test eta2kaprr val."""
    test_eta = 200 * 1e-9 * units.s
    test_z = 9.19508
    test_kparr = cosmo.eta2kparr(test_eta, test_z)
    test_val = (
        test_eta
        * (2 * np.pi * Planck15.H0 * f21 * Planck15.efunc(test_z))
        / (const.c * (1 + test_z) ** 2)
    ).to("1/Mpc")
    assert np.isclose(test_val.value, test_kparr.value)


def test_eta2kparr_unit():
    """Test eta must be Quantity."""
    test_eta = 200 * 1e-9 * units.s
    test_z = 9.19508
    test_kparr = cosmo.eta2kparr(test_eta, test_z)
    assert (1.0 / units.Mpc).unit == test_kparr.unit


def test_kparr2eta_error():
    """Test kparr2eta errors for non quantity objects."""
    test_kparr = 0.1
    test_z = 9.19508
    pytest.raises(TypeError, cosmo.kparr2eta, test_kparr, test_z)


def test_kparr2eta_wrong_unit():
    """Test kparr2eta errors for non quantity objects."""
    test_kparr = 0.1 * units.s
    test_z = 9.19508
    pytest.raises(units.UnitsError, cosmo.kparr2eta, test_kparr, test_z)


def test_kparr2eta_val():
    """Test kparr2eta value matches."""
    test_kparr = 0.1 / units.Mpc
    test_z = 9.19508
    test_eta = cosmo.kparr2eta(test_kparr, test_z)
    test_val = (
        test_kparr
        * const.c
        * (1 + test_z) ** 2
        / (2 * np.pi * Planck15.H0 * f21 * Planck15.efunc(test_z))
    ).to("s")
    assert np.isclose(test_val.value, test_eta.value)


def test_kparr2eta_unit():
    """Test kparr2eta unit matches."""
    test_kparr = 0.1 / units.Mpc
    test_z = 9.19508
    test_eta = cosmo.kparr2eta(test_kparr, test_z)
    assert units.s == test_eta.unit


def test_kparr2eta_eta2kparr():
    """Test that kparr2eta and eta2kparr undo each other."""
    test_kparr = 0.1 / units.Mpc
    test_z = 9.19508
    test_eta = cosmo.kparr2eta(test_kparr, test_z)
    test_eta2kparr = cosmo.eta2kparr(test_eta, test_z)
    assert np.isclose(test_kparr.value, test_eta2kparr.value)


def test_X2Y_unit():
    """Test unit on X2Y are 1/(Hz/Mpc)^3 or Mpc^3 * s."""
    test_z = 7
    test_x2y = cosmo.X2Y(test_z)
    assert units.Mpc**3 * units.s / units.sr == test_x2y.unit


def tests_X2Y_val():
    """Test value of X2Y."""
    test_z = 7
    test_x2y = cosmo.X2Y(test_z)
    compare_x2y = (2 * np.pi) ** 3 / (
        cosmo.u2kperp(1, test_z) ** 2 * units.sr * cosmo.eta2kparr(1 * units.s, test_z)
    )
    assert np.isclose(compare_x2y.value, test_x2y.value)
