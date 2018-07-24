"""Test Delay Spectrum calculations."""
from __future__ import print_function

import os
import sys
import numpy as np
import nose.tools as nt
import pyuvdata
from pyuvdata import UVData, utils as uvutils
from simpleDS import delay_spectrum as dspec
from simpleDS.data import DATA_PATH
from builtins import range, zip
from astropy import constants as const
from astropy import units
from scipy.signal import windows


def test_jy_to_mk_value():
    """Test the Jy to mK conversion factor."""
    test_fq = np.array([.1])*units.GHz
    jy_to_mk = dspec.jy_to_mk(test_fq)
    test_conversion = const.c**2 / (2 * test_fq.to('1/s')**2 * const.k_B)
    test_conversion = test_conversion.to('mK/Jy')
    nt.assert_true(np.allclose(test_conversion.value, jy_to_mk.value))


def test_jy_to_mk_units():
    """Test the Jy to mK conversion factor."""
    test_fq = np.array([.1])*units.GHz
    jy_to_mk = dspec.jy_to_mk(test_fq)
    test_conversion = const.c**2 / (2 * test_fq.to('1/s')**2 * const.k_B)
    test_conversion = test_conversion.to('mK/Jy')
    nt.assert_equal(test_conversion.unit.to_string(),
                    jy_to_mk.unit.to_string())


def test_delay_transform():
    """Test the delay transform and cross-multiplication function."""
    fake_data = np.zeros((1, 21, 13))
    fake_data[0, 11, 7] += 1
    fake_corr = dspec.delay_transform(fake_data, window=windows.boxcar)
    test_corr = np.fft.fft(fake_data, axis=1)
    test_corr = np.fft.fftshift(test_corr, axes=1)
    test_corr = test_corr[None, ...].conj() * test_corr[:, None, ...]
    fake_corr = fake_corr.value
    nt.assert_true(np.allclose(test_corr, fake_corr))


def test_units_delay_transform():
    """Test units are returned squared from delay_transform."""
    fake_data = np.zeros((1, 21, 13)) * units.m
    fake_data[0, 11, 7] += 1 * units.m
    fake_corr = dspec.delay_transform(fake_data, window=windows.boxcar)
    test_units = (units.m*units.Hz)**2
    nt.assert_equal(test_units, fake_corr.unit)


def test_delta_f_unitless():
    """Test delta_f is unitless raises exception."""
    fake_data = np.zeros((1, 21, 13)) * units.m
    fake_data[0, 11, 7] += 1 * units.m
    nt.assert_raises(ValueError, dspec.delay_transform, fake_data, delta_f=2.)
