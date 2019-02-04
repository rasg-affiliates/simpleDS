# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Matthew Kolopanis
# Licensed under the 3-clause BSD License
"""Test Delay Spectrum calculations."""
from __future__ import print_function

import os
import sys
import numpy as np
import copy
import nose.tools as nt
import unittest
from simpleDS import DelaySpectrum, delay_spectrum as dspec
from simpleDS import utils
from simpleDS.data import DATA_PATH
from pyuvdata.data import DATA_PATH as UVDATA_PATH
from pyuvdata import UVBeam, UVData
import pyuvdata.tests as uvtest
from astropy import constants as const
from astropy import units
from scipy.signal import windows


class TestClass(object):
    """A Dummy object for comparison."""

    def __init__(self):
        """Do Nothing."""
        pass


class TestDealySpectrumInit(object):
    """A test class to check DelaySpectrum objects."""

    def setUp(self):
        """Setup for basic parameter, property and iterator tests."""
        self.required_parameters = ['_Ntimes', '_Nbls', '_Nfreqs',
                                    '_Npols', '_vis_units', '_Ndelays',
                                    '_freq_array', '_delay_array',
                                    '_data_array', '_nsample_array',
                                    '_flag_array', '_lst_array', '_ant_1_array',
                                    '_ant_2_array', '_baseline_array',
                                    '_polarization_array', '_uvw', '_trcvr',
                                    '_redshift', '_k_perpendicular',
                                    '_k_parallel', '_beam_area',
                                    '_beam_sq_area', '_taper']

        self.required_properties = ['Ntimes', 'Nbls', 'Nfreqs', 'Npols',
                                    'vis_units', 'Ndelays',
                                    'freq_array', 'delay_array',
                                    'data_array', 'nsample_array',
                                    'flag_array', 'lst_array',
                                    'ant_1_array', 'ant_2_array',
                                    'baseline_array', 'polarization_array', 'uvw',
                                    'trcvr', 'redshift', 'k_perpendicular',
                                    'k_parallel', 'beam_area',
                                    'beam_sq_area', 'taper']
        self.extra_parameters = ['_power_array']
        self.extra_properties = ['power_array']
        self.dspec_object = DelaySpectrum()

    def teardown(self):
        """Test teardown: delete object."""
        del(self.dspec_object)

    def test_required_parameter_iter(self):
        """Test expected required parameters."""
        required = []
        for prop in self.dspec_object.required():
            required.append(prop)
        for a in self.required_parameters:
            nt.assert_true(a in required, msg='expected attribute ' + a
                           + ' not returned in required iterator')

    def test_properties(self):
        """Test that properties can be get and set properly."""
        prop_dict = dict(list(zip(self.required_properties,
                                  self.required_parameters)))
        for k, v in prop_dict.items():
            rand_num = np.random.rand()
            setattr(self.dspec_object, k, rand_num)
            this_param = getattr(self.dspec_object, v)
            try:
                nt.assert_equal(rand_num, this_param.value)
            except(AssertionError):
                print('setting {prop_name} to a random number failed'.format(prop_name=k))
                raise(AssertionError)


def test_errors_when_taper_not_function():
    """Test that init errors if taper not a function."""
    nt.assert_raises(ValueError, DelaySpectrum, taper='test')


def test_error_for_multiple_baselines():
    """Test an error is raised if there are more than one unique baseline in input UVData."""
    # testfile = os.path.join(UVDATA_PATH, 'hera19_8hrs_uncomp_10MHz_000_05.003111-05.033750.uvfits')
    uvd = UVData()
    uvd.baseline_array = np.array([1, 2])
    uvd.uvw_array = np.array([[0, 1, 0], [1, 0, 0]])
    # uvd.read(testfile)
    # uvd.unphase_to_drift(use_ant_pos=True)
    nt.assert_raises(ValueError, DelaySpectrum, uv=uvd)


def test_error_if_uv_not_uvdata():
    """Test error is raised when input uv is not a UVData object."""
    bad_input = TestClass()
    nt.assert_raises(ValueError, DelaySpectrum, uv=bad_input)


def test_custom_taper():
    """Test setting custom taper."""
    test_win = windows.blackman
    dspec = DelaySpectrum(taper=test_win)
    nt.assert_equal(test_win, dspec.taper)


class TestBasicFunctions(object):
    """Test basic equality functions."""

    def setUp(self):
        """Setup for tests of basic methods."""
        self.uvdata_object = UVData()
        self.testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
        self.uvdata_object.read(self.testfile)
        self.dspec_object = DelaySpectrum(uv=self.uvdata_object)
        self.dspec_object2 = copy.deepcopy(self.dspec_object)

    def teardown(self):
        """Test teardown: delete objects."""
        del(self.dspec_object)
        del(self.dspec_object2)
        del(self.uvdata_object)

    def test_equality(self):
        """Basic equality test."""
        nt.assert_equal(self.dspec_object, self.dspec_object2)

    def test_check(self):
        """Test that check function operates as expected."""
        nt.assert_true(self.dspec_object.check())

        # test that it fails if we change values
        self.dspec_object.Ntimes += 1
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Ntimes -= 1

        self.dspec_object.Nbls += 1
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nbls -= 1

        self.dspec_object.Nfreqs += 1
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs -= 1

        self.dspec_object.Npols += 1
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Npols -= 1

        self.dspec_object.Ndelays += 1
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Ndelays -= 1

        self.dspec_object.Ndelays = np.float(self.dspec_object.Ndelays)
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Ndelays = np.int(self.dspec_object.Ndelays)

        self.dspec_object.polarization_array = self.dspec_object.polarization_array.astype(np.float)
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.polarization_array = self.dspec_object.polarization_array.astype(np.int)

        Nfreqs = copy.deepcopy(self.dspec_object.Nfreqs)
        self.dspec_object.Nfreqs = None
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs = Nfreqs

        self.dspec_object.vis_units = 2
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.vis_units = 'Jy'

        self.dspec_object.Nfreqs = (2, 1, 2)
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs = Nfreqs

        self.dspec_object.Nfreqs = np.complex(2)
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs = Nfreqs

        freq_back = copy.deepcopy(self.dspec_object.freq_array)
        self.dspec_object.freq_array = np.arange(self.dspec_object.Nfreqs).reshape(1, Nfreqs).tolist()
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.freq_array = freq_back

        self.dspec_object.freq_array = freq_back.value.astype(np.complex) * freq_back.unit
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.freq_array = freq_back

        integration_time_back = copy.deepcopy(self.dspec_object.integration_time)
        self.dspec_object.integration_time = integration_time_back.astype(complex)
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.integration_time = integration_time_back

        Nuv = self.dspec_object.Nuv
        self.dspec_object.Nuv = 10
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nuv = Nuv

        self.dspec_object.data_type = 'delay'
        nt.assert_raises(ValueError, self.dspec_object.check)
        self.dspec_object.data_type = 'frequency'

        self.dspec_object.data_array = self.dspec_object.data_array * units.Hz
        nt.assert_raises(ValueError, self.dspec_object.check)

        self.dspec_object.data_type = 'delay'
        nt.assert_true(self.dspec_object.check())
        self.dspec_object.data_type = 'frequency'
        self.dspec_object.data_array = self.dspec_object.data_array.value * units.Jy
        nt.assert_true(self.dspec_object.check())

    def test_add_wrong_units(self):
        """Test error is raised when adding a uvdata_object with the wrong units."""
        uvd = UVData()
        uvd.read(self.testfile)
        uvd.vis_units = 'K str'
        nt.assert_raises(units.UnitConversionError, self.dspec_object.add_uvdata_object, uvd)
        uvd.vis_units = 'uncalib'
        warn_message = ['Data is uncalibrated. Unable to covert '
                        'noise array to unicalibrated units.']

        nt.assert_raises(units.UnitConversionError, uvtest.checkWarnings,
                         self.dspec_object.add_uvdata_object, func_args=[uvd],
                         category=UserWarning,
                         nwarnings=len(warn_message),
                         message=warn_message)

    def test_add_too_many_UVData(self):
        """Test error is raised when adding too many UVData objects."""
        uvd = UVData()
        uvd.read(self.testfile)
        self.dspec_object.Nuv = 2
        nt.assert_raises(ValueError, self.dspec_object.add_uvdata_object, uvd)

    def test_incompatible_parameters(self):
        """Test UVData objects with incompatible paramters are rejected."""
        uvd = UVData()
        uvd.read(self.testfile)
        uvd.select(freq_chans=np.arange(12))
        nt.assert_raises(ValueError, self.dspec_object.add_uvdata_object, uvd)

    def test_adding_spectral_windows_different_tuple_shape(self):
        """Test error is raised if spectral windows have different shape input."""
        nt.assert_raises(ValueError, self.dspec_object.select_spectral_windows,
                         spectral_windows=((2, 3), (1, 2, 4)))

    def test_adding_spectral_windows_different_lengths(self):
        """Test error is raised if spectral windows have different shape input."""
        nt.assert_raises(ValueError, self.dspec_object.select_spectral_windows,
                         spectral_windows=((2, 3), (2, 6)))

    def test_add_second_uvdata_object(self):
        """Test a second UVdata object can be added correctly."""
        uvd = UVData()
        uvd.read(self.testfile)
        # multiply by a scalar here to track if it gets set in the correct slot
        uvd.data_array *= np.sqrt(2)
        self.dspec_object.add_uvdata_object(uvd)
        nt.assert_equal(self.dspec_object.Nuv, 2)
        nt.assert_true(np.allclose(self.dspec_object.data_array[:, 0].value,
                                   self.dspec_object.data_array[:, 1].value / np.sqrt(2)))


def test_adding_spectral_window_one_tuple():
    """Test spectral window can be added when only one tuple given."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=(3, 12))
    nt.assert_equal(dspec_object.Nfreqs, 10)
    nt.assert_equal(dspec_object.Ndelays, 10)
    nt.assert_true(np.allclose(dspec_object.freq_array.to('Hz').value,
                               uvd.freq_array[:, 3:13]))


def test_adding_spectral_window_between_uvdata():
    """Test that adding a spectral window between uvdata objects is handled."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=[(3, 12)])
    uvd1 = copy.deepcopy(uvd)
    dspec_object.add_uvdata_object(uvd1)
    nt.assert_true(dspec_object.check())


def test_adding_new_uvdata_with_different_freqs():
    """Test error is raised when trying to add a uvdata object with the same freqs."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=[(3, 12)])
    uvd1 = copy.deepcopy(uvd)
    uvd1.freq_array *= 11.1
    nt.assert_raises(ValueError, dspec_object.add_uvdata_object, uvd1)


def test_select_spectral_window_not_inplace():
    """Test it is possible to return a different object from select spectral window."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    new_dspec = dspec_object.select_spectral_windows(spectral_windows=[(3, 12)],
                                                     inplace=False)
    nt.assert_not_equal(dspec_object, new_dspec)


def test_loading_different_arrays():
    """Test error is raised trying to combine different arrays."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    bls = np.unique(uvd.baseline_array)[:-1]
    ants = [uvd.baseline_to_antnums(bl) for bl in bls]
    ants = [(a1, a2) for a1, a2 in ants]
    uvd.select(bls=ants)
    nt.assert_raises(ValueError, dspec_object.add_uvdata_object, uvd)


def test_loading_uvb_object():
    """Test a uvb object can have the beam_area and beam_sq_area read."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uv_beam(uvb=uvb)
    uvb.select(frequencies=uvd.freq_array[0])
    nt.assert_true(np.allclose(uvb.get_beam_area(pol='pI'),
                               dspec_object.beam_area.to('sr').value))


def test_noise_shape():
    """Test the generate noise and calculate_noise_power produce correct shape."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.trcvr = np.zeros_like(dspec_object.trcvr)
    dspec_object.beam_area = np.ones_like(dspec_object.beam_area)
    dspec_object.generate_noise()
    nt.assert_equal(dspec_object._noise_array.expected_shape(dspec_object),
                    dspec_object.noise_array.shape)


def test_noise_shape():
    """Test the generate noise and calculate_noise_power produce correct units."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.trcvr = np.zeros_like(dspec_object.trcvr)
    dspec_object.beam_area = np.ones_like(dspec_object.beam_area)
    dspec_object.generate_noise()
    nt.assert_equal(dspec_object.noise_array.unit, units.Jy)


def test_noise_amplitude():
    """Test noise amplitude with a fixed seed."""
    np.random.seed(0)
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.trcvr = np.zeros_like(dspec_object.trcvr)
    dspec_object.beam_area = np.ones_like(dspec_object.beam_area)
    dspec_object.nsample_array = np.ones_like(dspec_object.nsample_array)
    dspec_object.integration_time = np.ones_like(dspec_object.integration_time)
    dspec_object.polarization_array = np.array([-5])

    dspec_object.generate_noise()
    var = np.var(dspec_object.noise_array, axis=(0, 1, 2, 3)).mean(0)
    test_amplitude = (180 * units.K * np.power((dspec_object.freq_array.to('GHz') / (.18 * units.GHz)), -2.55)
                      / np.sqrt(np.diff(dspec_object.freq_array[0])[0].value)).reshape(1, 1, dspec_object.Nfreqs)
    test_amplitude *= dspec_object.beam_area / utils.jy_to_mk(dspec_object.freq_array)
    test_var = test_amplitude.to('Jy')**2
    # this was from running this test by hand
    ratio = np.array([[1.07735447, 1.07082788, 1.07919504, 1.04992591, 1.02254714,
                       0.99884931, 0.94861011, 1.01908474, 1.03877442, 1.00549461,
                       1.09642801, 1.01100747, 1.0201933, 1.05762868, 0.95156612,
                       1.00190002, 1.00046522, 1.02796162, 1.04277506, 0.98373618,
                       1.01235802]])
    nt.assert_true(np.allclose(ratio, (test_var / var).value))


def test_delay_transform_units():
    """Test units after calling delay_transform are correct."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    dspec_object.delay_transform()
    nt.assert_true(dspec_object.data_array.unit.is_equivalent(units.Jy * units.Hz))
    dspec_object.delay_transform()
    nt.assert_true(dspec_object.data_array.unit.is_equivalent(units.Jy))


def test_warning_from_uncalibrated_data():
    """Test scaling warning is raised when delay transforming uncalibrated data."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    uvd.vis_units = 'uncalib'
    warn_message = ["Data is uncalibrated. Unable to covert noise array "
                    "to unicalibrated units."]
    dspec_object = uvtest.checkWarnings(DelaySpectrum, func_args=[uvd],
                                        category=[UserWarning],
                                        nwarnings=1,
                                        message=warn_message)
    warn_message = ["Fourier Transforming uncalibrated data. Units will "
                    "not have physical meaning. "
                    "Data will be arbitrarily scaled."]
    uvtest.checkWarnings(dspec_object.delay_transform,
                         category=[UserWarning],
                         nwarnings=1,
                         message=warn_message)


def test_delay_transform_bad_data_type():
    """Test error is raised in delay_transform if data_type is bad."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)

    dspec_object = DelaySpectrum(uvd)
    dspec_object.data_type = 'test'
    nt.assert_raises(ValueError, dspec_object.delay_transform)


def test_delay_spectrum_power_units():
    """Test the units on the output power are correct."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uv_beam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    nt.assert_equal(units.mK**2 * units.Mpc**3, dspec_object.power_array.unit)


def test_delay_spectrum_power_shape():
    """Test the shape of the output power are correct."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uv_beam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    power_shape = (dspec_object.Nspws, dspec_object.Npols, dspec_object.Nbls,
                   dspec_object.Nbls, dspec_object.Ntimes, dspec_object.Ndelays)
    nt.assert_equal(power_shape, dspec_object.power_array.shape)


def test_delay_spectrum_power_shape_two_uvdata_objects_read():
    """Test the shape of the output power are correct."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvh5')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd] * 2)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uv_beam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    power_shape = (dspec_object.Nspws, dspec_object.Npols, dspec_object.Nbls,
                   dspec_object.Nbls, dspec_object.Ntimes, dspec_object.Ndelays)
    nt.assert_equal(power_shape, dspec_object.power_array.shape)



@unittest.skip('Skipping some of detailed tests during conversion')
def test_delay_spectrum_power_units_input_kelvin_str():
    """Test the units on the output power are correct when input kelvin*str."""
    test_miriad = os.path.join(DATA_PATH, 'paper_testfile_k_units.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.']
    pend_dep_message = ['antenna_positions are not defined. '
                        'antenna_positions will be a required parameter in '
                        'future versions.']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [PendingDeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
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


@unittest.skip('Skipping some of detailed tests during conversion')
def test_delay_spectrum_power_units_input_uncalib():
    """Test the units on the output power are correct if input uncalib."""
    test_miriad = os.path.join(DATA_PATH, 'paper_testfile_uncalib_units.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.']
    pend_dep_message = ['antenna_positions are not defined. '
                        'antenna_positions will be a required parameter in '
                        'future versions.']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [PendingDeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
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
    nt.assert_equal(units.Mpc**3, delay_power.unit)


@unittest.skip('Skipping some of detailed tests during conversion')
def test_delay_spectrum_noise_power_units():
    """Test the units on the output noise power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_testfile.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']
    pend_dep_message = ['antenna_positions are not defined. '
                        'antenna_positions will be a required parameter in '
                        'future versions.']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [PendingDeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    warn_message = ['Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']
    uvtest.checkWarnings(test_uv_1.select, func_args=[],
                         func_kwargs={'freq_chans': np.arange(95, 116)},
                         category=UserWarning,
                         nwarnings=len(warn_message),
                         message=warn_message)

    uvtest.checkWarnings(test_uv_2.select, func_args=[],
                         func_kwargs={'freq_chans': np.arange(95, 116)},
                         category=UserWarning,
                         nwarnings=len(warn_message),
                         message=warn_message)

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(units.mK**2 * units.Mpc**3, noise_power.unit)


@unittest.skip('Skipping some of detailed tests during conversion')
def test_delay_spectrum_thermal_power_units():
    """Test the units on the output thermal power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_testfile.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']
    pend_dep_message = ['antenna_positions are not defined. '
                        'antenna_positions will be a required parameter in '
                        'future versions.']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [PendingDeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv_1)
    reds = np.array(list(set(test_uv_2.baseline_array)))

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    warn_message = ['Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']
    uvtest.checkWarnings(test_uv_1.select, func_args=[],
                         func_kwargs={'freq_chans': np.arange(95, 116)},
                         category=UserWarning,
                         nwarnings=len(warn_message),
                         message=warn_message)

    uvtest.checkWarnings(test_uv_2.select, func_args=[],
                         func_kwargs={'freq_chans': np.arange(95, 116)},
                         category=UserWarning,
                         nwarnings=len(warn_message),
                         message=warn_message)

    output_array = dspec.calculate_delay_spectrum(uv_even=test_uv_1,
                                                  uv_odd=test_uv_2, uvb=uvb,
                                                  trcvr=144 * units.K,
                                                  reds=reds)
    delays, delay_power, noise_power, thermal_power = output_array
    nt.assert_equal(units.mK**2 * units.Mpc**3, thermal_power.unit)
