"""Test Delay Spectrum calculations."""
from __future__ import print_function

import os
import sys
import numpy as np
import copy
import nose.tools as nt
from simpleDS import delay_spectrum as dspec
from simpleDS import utils
from simpleDS.data import DATA_PATH
from pyuvdata import UVBeam
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



def test_normalized_fourier_transform():
    """Test the delay transform and cross-multiplication function."""
    fake_data = np.zeros((1, 13, 21))
    fake_data[0, 7, 11] += 1
    fake_corr = dspec.normalized_fourier_transform(fake_data,
                                                   window=windows.boxcar,
                                                   axis=2)
    test_corr = np.fft.fft(fake_data, axis=-1)
    test_corr = np.fft.fftshift(test_corr, axes=-1)
    fake_corr = fake_corr.value
    nt.assert_true(np.allclose(test_corr, fake_corr))


def test_ft_with_pols():
    """Test fourier transform is correct shape when pols are present."""
    fake_data = np.zeros((3, 2, 13, 31))
    fake_data[:, 0, 7, 11] += 1.
    fake_corr = dspec.normalized_fourier_transform(fake_data,
                                                   window=windows.boxcar,
                                                   axis=3)
    nt.assert_equal((3, 2, 13, 31), fake_corr.shape)


def test_delay_vals_with_pols():
    """Test values in normalized_fourier_transform when pols present."""
    fake_data = np.zeros((3, 2, 13, 31))
    fake_data[:, 0, 7, 11] += 1.
    fake_corr = dspec.normalized_fourier_transform(fake_data,
                                                   window=windows.boxcar,
                                                   axis=3)
    test_corr = np.fft.fft(fake_data, axis=-1)
    test_corr = np.fft.fftshift(test_corr, axes=-1)
    fake_corr = fake_corr.value
    nt.assert_true(np.allclose(test_corr, fake_corr))


def test_units_normalized_fourier_transform():
    """Test units are returned squared from normalized_fourier_transform."""
    fake_data = np.zeros((1, 13, 21)) * units.m
    fake_data[0, 7, 11] += 1 * units.m
    fake_corr = dspec.normalized_fourier_transform(fake_data,
                                                   window=windows.boxcar,
                                                   axis=2)
    test_units = units.m * units.Hz
    nt.assert_equal(test_units, fake_corr.unit)


def test_delta_x_unitless():
    """Test delta_x is unitless raises exception."""
    fake_data = np.zeros((1, 13, 21)) * units.m
    fake_data[0, 7, 11] += 1 * units.m
    nt.assert_raises(ValueError, dspec.normalized_fourier_transform, fake_data,
                     delta_x=2., axis=2)


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
    test_inttime = np.ones_like(test_sample) * 100
    nt.assert_raises(ValueError, dspec.calculate_noise_power,
                     nsamples=test_sample, freqs=test_freqs,
                     inttime=test_inttime, trcvr=test_temp, npols=1)


def test_noise_power_freq_unit():
    """Test Exception is raised if freq is not a Quantity object."""
    test_sample = np.ones((2, 13, 21))
    test_freqs = np.linspace(.1, .2, 3)
    test_temp = 400 * units.K
    test_inttime = np.ones_like(test_sample) * 100 * units.s
    nt.assert_raises(ValueError, dspec.calculate_noise_power,
                     nsamples=test_sample, freqs=test_freqs,
                     inttime=test_inttime, trcvr=test_temp, npols=1)


def test_noise_power_trcvr_unit():
    """Test Exception is raised if trcvr is not a Quantity object."""
    test_sample = np.ones((2, 13, 21))
    test_freqs = np.linspace(.1, .2, 3) * units.GHz
    test_temp = 400
    test_inttime = np.ones_like(test_sample) * 100 * units.s
    nt.assert_raises(ValueError, dspec.calculate_noise_power,
                     nsamples=test_sample, freqs=test_freqs,
                     inttime=test_inttime, trcvr=test_temp, npols=1)


def test_noise_power_shape():
    """Test shape of noise power is as expected."""
    test_sample = np.ones((2, 13, 21))
    test_freqs = np.linspace(.1, .2, 21) * units.GHz
    test_temp = 400 * units.K
    test_inttime = np.ones_like(test_sample) * 100 * units.s
    test_noise_power = dspec.calculate_noise_power(nsamples=test_sample,
                                                   freqs=test_freqs,
                                                   trcvr=test_temp,
                                                   inttime=test_inttime,
                                                   npols=1)
    nt.assert_equal(test_sample.shape, test_noise_power.shape)


def test_noise_power_unit():
    """Test unit of noise power is as expected."""
    test_sample = np.ones((2, 13, 21))
    test_freqs = np.linspace(.1, .2, 21) * units.GHz
    test_temp = 400 * units.K
    test_inttime = np.ones_like(test_sample) * 100 * units.s
    test_noise_power = dspec.calculate_noise_power(nsamples=test_sample,
                                                   freqs=test_freqs,
                                                   trcvr=test_temp,
                                                   inttime=test_inttime,
                                                   npols=1)
    nt.assert_equal(units.mK, test_noise_power.unit)


def test_noise_shape():
    """Test shape of generate_noise matches nsample array."""
    test_sample = np.ones((2, 13, 21)) * 3
    test_noise = dspec.generate_noise(test_sample)
    nt.assert_equal(test_sample.shape, test_noise.shape)


def test_noise_amplitude():
    """Ensure noise amplitude is reasonable within 1 percent."""
    rtol = 1e-2
    test_sample = np.ones((100, 1000)) * 3
    test_noise = dspec.generate_noise(test_sample)
    noise_power = test_noise.std(1)
    noise_power_uncertainty = noise_power.std()
    nt.assert_true(np.isclose(test_noise.std(), 3,
                              atol=noise_power_uncertainty))


def test_calculate_delay_spectrum_mismatched_freqs():
    """Test Exception is raised when freq arrays are not equal."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    # add 10MHz to make frequencies different
    test_uv_2.freq_array += np.ones_like(test_uv_2.freq_array) * 10e6
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')
    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    nt.assert_raises(ValueError, dspec.calculate_delay_spectrum,
                     uv_even=test_uv_1, uv_odd=test_uv_2, uvb=uvb,
                     trcvr=144 * units.K, reds=reds)


def test_calculate_delay_spectrum_mismatched_inttimes():
    """Test Exception is raised when integration times are not equal."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    # change the integration_time so they do not match
    test_uv_2.integration_time += test_uv_2.integration_time * 2
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')
    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    nt.assert_raises(ValueError, dspec.calculate_delay_spectrum,
                     uv_even=test_uv_1, uv_odd=test_uv_2, uvb=uvb,
                     trcvr=144 * units.K, reds=reds)


def test_delay_spectrum_units_delays_units():
    """Test the units on the output of calculate_delay_spectrum are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(units.s, delays.unit)


def test_delay_spectrum_power_units():
    """Test the units on the output power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(units.mK**2 * units.Mpc**3, delay_power.unit)


def test_delay_spectrum_noise_power_units():
    """Test the units on the output noise power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(units.mK**2 * units.Mpc**3, noise_power.unit)


def test_delay_spectrum_thermal_power_units():
    """Test the units on the output thermal power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(units.mK**2 * units.Mpc**3, thermal_power.unit)


def test_delay_spectrum_power_shape():
    """Test the shape on the output delay power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))
    Nbls = len(reds)
    Ntimes = test_uv_2.Ntimes
    Nfreqs = test_uv_2.Nfreqs
    out_shape = (Nbls, Nbls, Ntimes, Nfreqs)

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(out_shape, thermal_power.shape)


def test_delay_spectrum_noise_shape():
    """Test the shape on the output noise power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))
    Nbls = len(reds)
    Ntimes = test_uv_2.Ntimes
    Nfreqs = test_uv_2.Nfreqs
    out_shape = (Nbls, Nbls, Ntimes, Nfreqs)

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(out_shape, delay_power.shape)


def test_delay_spectrum_noise_shape_one_pol():
    """Test the shape on the output noise power are correct with one pol."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    test_uv_1.polarization_array = np.array([-4])
    test_uv_2.polarization_array = np.array([-4])

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))
    Nbls = len(reds)
    Ntimes = test_uv_2.Ntimes
    Nfreqs = test_uv_2.Nfreqs
    out_shape = (Nbls, Nbls, Ntimes, Nfreqs)

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(out_shape, delay_power.shape)

def test_delay_spectrum_thermal_power_shape():
    """Test the shape on the output thermal power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))
    Nbls = len(reds)
    Ntimes = test_uv_2.Ntimes
    Nfreqs = test_uv_2.Nfreqs
    out_shape = (Nbls, Nbls, Ntimes, Nfreqs)

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(out_shape, noise_power.shape)


def test_delay_spectrum_power_shape_pols():
    """Test the shape on the output delay power are correct with pols."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))
    Nbls = len(reds)
    Ntimes = test_uv_2.Ntimes
    Nfreqs = test_uv_2.Nfreqs
    out_shape = (1, Nbls, Nbls, Ntimes, Nfreqs)

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds, squeeze=False)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(out_shape, thermal_power.shape)


def test_delay_spectrum_noise_shape_pols():
    """Test the shape on the output noise power are correct with pols."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))
    Nbls = len(reds)
    Ntimes = test_uv_2.Ntimes
    Nfreqs = test_uv_2.Nfreqs
    out_shape = (1, Nbls, Nbls, Ntimes, Nfreqs)

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds, squeeze=False)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(out_shape, delay_power.shape)


def test_delay_spectrum_thermal_power_shape_pols():
    """Test the shape on the output thermal power are correct with pols."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv_1 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)
    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))
    Nbls = len(reds)
    Ntimes = test_uv_2.Ntimes
    Nfreqs = test_uv_2.Nfreqs
    out_shape = (1, Nbls, Nbls, Ntimes, Nfreqs)

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds,
                                                  squeeze=False)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(out_shape, noise_power.shape)
