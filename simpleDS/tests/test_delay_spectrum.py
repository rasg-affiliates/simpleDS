"""Test Delay Spectrum calculations."""
from __future__ import print_function

import os
import sys
import numpy as np
import nose.tools as nt
from simpleDS import delay_spectrum as dspec
from simpleDS.data import DATA_PATH
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


def test_data_2_wrong_shape():
    """Test Exception is raised if shapes do not match."""
    fake_data_1 = np.zeros((1, 13, 21))
    fake_data_2 = np.zeros((2, 13, 21))
    nt.assert_raises(ValueError, dspec.delay_transform,
                     fake_data_1, fake_data_2)


def test_delay_transform():
    """Test the delay transform and cross-multiplication function."""
    fake_data = np.zeros((1, 13, 21))
    fake_data[0, 7, 11] += 1
    fake_corr = dspec.delay_transform(fake_data, window=windows.boxcar)
    test_corr = np.fft.fft(fake_data, axis=-1)
    test_corr = np.fft.fftshift(test_corr, axes=-1)
    test_corr = test_corr[None, ...].conj() * test_corr[:, None, ...]
    fake_corr = fake_corr.value
    nt.assert_true(np.allclose(test_corr, fake_corr))


def test_delay_with_pols():
    """Test delay transform is correct shape when polarizations are present."""
    fake_data = np.zeros((3, 2, 13, 31))
    fake_data[:, 0, 7, 11] += 1.
    fake_corr = dspec.delay_transform(fake_data, window=windows.boxcar)
    nt.assert_equal((3, 2, 2, 13, 31), fake_corr.shape)


def test_delay_vals_with_pols():
    """Test values in delay_transform when pols present."""
    fake_data = np.zeros((3, 2, 13, 31))
    fake_data[:, 0, 7, 11] += 1.
    fake_corr = dspec.delay_transform(fake_data, window=windows.boxcar)
    test_corr = np.fft.fft(fake_data, axis=-1)
    test_corr = np.fft.fftshift(test_corr, axes=-1)
    test_corr = test_corr[:, None, ...].conj() * test_corr[:, :, None, ...]
    fake_corr = fake_corr.value
    nt.assert_true(np.allclose(test_corr, fake_corr))


def test_units_delay_transform():
    """Test units are returned squared from delay_transform."""
    fake_data = np.zeros((1, 13, 21)) * units.m
    fake_data[0, 7, 11] += 1 * units.m
    fake_corr = dspec.delay_transform(fake_data, window=windows.boxcar)
    test_units = (units.m*units.Hz)**2
    nt.assert_equal(test_units, fake_corr.unit)


def test_delta_f_unitless():
    """Test delta_f is unitless raises exception."""
    fake_data = np.zeros((1, 13, 21)) * units.m
    fake_data[0, 7, 11] += 1 * units.m
    nt.assert_raises(ValueError, dspec.delay_transform, fake_data, delta_f=2.)


def test_combine_nsamples_different_shapes():
    """Test an error is raised if nsample_arrays have different shapes."""
    test_sample_1 = np.ones((2, 13, 21))
    test_sample_2 = np.ones((3, 13, 21))
    nt.assert_raises(ValueError, dspec.combine_nsamples,
                     test_sample_1, test_sample_2)


def test_combine_nsamples_one_array():
    """Test that if only one array is given the samples are the same."""
    test_samples = np.ones((2, 13, 21)) * 3
    samples_out = dspec.combine_nsamples(test_samples)
    test_full_samples = np.ones((2, 2, 13, 21)) * 3
    nt.assert_true(np.all(test_full_samples == samples_out))


def test_combine_nsamples_with_pols():
    """Test that if only one array is given the samples are the same."""
    test_samples_1 = np.ones((3, 2, 13, 21)) * 3
    test_samples_2 = np.ones((3, 2, 13, 21)) * 2
    samples_out = dspec.combine_nsamples(test_samples_1, test_samples_2)
    test_full_samples = np.ones((3, 2, 2, 13, 21)) * np.sqrt(6)
    nt.assert_true(np.all(test_full_samples == samples_out))


def test_remove_autos():
    """Test that the remove auto_correlations function returns right shape."""
    test_array = np.ones((3, 3, 11, 21))
    out_array = dspec.remove_auto_correlations(test_array)
    nt.assert_equal((6, 11, 21), out_array.shape)


def test_remove_autos_with_pols():
    """Test remove auto_correlations function returns right shape with pols."""
    test_array = np.ones((4, 3, 3, 11, 21))
    out_array = dspec.remove_auto_correlations(test_array)
    nt.assert_equal((4, 6, 11, 21), out_array.shape)


def test_remove_autos_small_shape():
    """Test Exception is raised on an array which is too small."""
    test_array = np.ones((3))
    nt.assert_raises(ValueError, dspec.remove_auto_correlations, test_array)


def test_remove_autos_small_shape():
    """Test Exception is raised on an array which is too big."""
    test_array = np.ones((3, 12, 12, 21, 6, 7))
    nt.assert_raises(ValueError, dspec.remove_auto_correlations, test_array)


def test_noise_power_inttime_unit():
    """Test Exception is raised if inttime is not a Quantity object."""
    test_sample = np.ones((2, 13, 21))
    test_freqs = np.linspace(.1, .2, 3) * units.GHz
    test_temp = 400 * units.K
    test_inttime = 100
    nt.assert_raises(ValueError, dspec.calculate_noise_power,
                     nsamples=test_sample, freqs=test_freqs,
                     inttime=test_inttime, trcvr=test_temp)


def test_noise_power_freq_unit():
    """Test Exception is raised if freq is not a Quantity object."""
    test_sample = np.ones((2, 13, 21))
    test_freqs = np.linspace(.1, .2, 3)
    test_temp = 400 * units.K
    test_inttime = 100 * units.s
    nt.assert_raises(ValueError, dspec.calculate_noise_power,
                     nsamples=test_sample, freqs=test_freqs,
                     inttime=test_inttime, trcvr=test_temp)


def test_noise_power_trcvr_unit():
    """Test Exception is raised if trcvr is not a Quantity object."""
    test_sample = np.ones((2, 13, 21))
    test_freqs = np.linspace(.1, .2, 3) * units.GHz
    test_temp = 400
    test_inttime = 100 * units.s
    nt.assert_raises(ValueError, dspec.calculate_noise_power,
                     nsamples=test_sample, freqs=test_freqs,
                     inttime=test_inttime, trcvr=test_temp)


def test_noise_power_shape():
    """Test shape of noise power is as expected."""
    test_sample = np.ones((2, 13, 21))
    test_freqs = np.linspace(.1, .2, 21) * units.GHz
    test_temp = 400 * units.K
    test_inttime = 100 * units.s
    test_noise_power = dspec.calculate_noise_power(nsamples=test_sample,
                                                   freqs=test_freqs,
                                                   trcvr=test_temp,
                                                   inttime=test_inttime)
    nt.assert_equal(test_sample.shape, test_noise_power.shape)


def test_noise_shape():
    """Test shape of generate_noise matches nsample array."""
    test_sample = np.ones((2, 13, 21)) * 3
    test_noise = dspec.generate_noise(test_sample)
    nt.assert_equal(test_sample.shape, test_noise.shape)


def test_noise_amplitude():
    """Ensure noise amplitude is reasonable within 1 percent."""
    rtol = 1e-2
    test_sample = np.ones((5, 13, 200)) * 3
    test_noise = dspec.generate_noise(test_sample)
    nt.assert_true(np.isclose(test_noise.std(), 3, rtol=rtol))
