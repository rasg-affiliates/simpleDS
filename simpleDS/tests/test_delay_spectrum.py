# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Test Delay Spectrum calculations."""
from __future__ import print_function

import os
import numpy as np
import copy
import pytest
import unittest

from pyuvdata import UVBeam, UVData
import pyuvdata.tests as uvtest
from astropy import units
from astropy.cosmology import Planck15, WMAP9
from scipy.signal import windows

from simpleDS import DelaySpectrum
from simpleDS import utils
from simpleDS.data import DATA_PATH
from pyuvdata.data import DATA_PATH as UVDATA_PATH
import simpleDS.tests as sdstest


class DummyClass(object):
    """A Dummy object for comparison."""

    def __init__(self):
        """Do Nothing."""
        pass


class TestDelaySpectrumInit(unittest.TestCase):
    """A test class to check DelaySpectrum objects."""

    def setUp(self):
        """Initialize basic parameter, property and iterator tests."""
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
            assert a in required, ('expected attribute ' + a
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
                assert rand_num == this_param.value
            except(AssertionError):
                print('setting {prop_name} to a random number failed'.format(prop_name=k))
                raise(AssertionError)


def test_errors_when_taper_not_function():
    """Test that init errors if taper not a function."""
    pytest.raises(ValueError, DelaySpectrum, taper='test')


def test_error_for_multiple_baselines():
    """Test an error is raised if there are more than one unique baseline in input UVData."""
    # testfile = os.path.join(UVDATA_PATH, 'hera19_8hrs_uncomp_10MHz_000_05.003111-05.033750.uvfits')
    uvd = UVData()
    uvd.baseline_array = np.array([1, 2])
    uvd.uvw_array = np.array([[0, 1, 0], [1, 0, 0]])
    # uvd.read(testfile)
    # uvd.unphase_to_drift(use_ant_pos=True)
    pytest.raises(ValueError, DelaySpectrum, uv=uvd)


def test_error_if_uv_not_uvdata():
    """Test error is raised when input uv is not a UVData object."""
    bad_input = DummyClass()
    pytest.raises(ValueError, DelaySpectrum, uv=bad_input)


def test_custom_taper():
    """Test setting custom taper."""
    test_win = windows.blackman
    dspec = DelaySpectrum(taper=test_win)
    assert test_win == dspec.taper


class TestBasicFunctions(unittest.TestCase):
    """Test basic equality functions."""

    def setUp(self):
        """Initialize tests of basic methods."""
        self.uvdata_object = UVData()
        self.testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
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
        assert self.dspec_object == self.dspec_object2

    def test_check(self):
        """Test that check function operates as expected."""
        assert self.dspec_object.check()

        # test that it fails if we change values
        self.dspec_object.Ntimes += 1
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Ntimes -= 1

        self.dspec_object.Nbls += 1
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nbls -= 1

        self.dspec_object.Nfreqs += 1
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs -= 1

        self.dspec_object.Npols += 1
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Npols -= 1

        self.dspec_object.Ndelays += 1
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Ndelays -= 1

        self.dspec_object.Ndelays = np.float(self.dspec_object.Ndelays)
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Ndelays = np.int(self.dspec_object.Ndelays)

        self.dspec_object.polarization_array = self.dspec_object.polarization_array.astype(np.float)
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.polarization_array = self.dspec_object.polarization_array.astype(np.int)

        Nfreqs = copy.deepcopy(self.dspec_object.Nfreqs)
        self.dspec_object.Nfreqs = None
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs = Nfreqs

        self.dspec_object.vis_units = 2
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.vis_units = 'Jy'

        self.dspec_object.Nfreqs = (2, 1, 2)
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs = Nfreqs

        self.dspec_object.Nfreqs = np.complex(2)
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs = Nfreqs

        freq_back = copy.deepcopy(self.dspec_object.freq_array)
        self.dspec_object.freq_array = np.arange(self.dspec_object.Nfreqs).reshape(1, Nfreqs).tolist()
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.freq_array = freq_back

        freq_back = copy.deepcopy(self.dspec_object.freq_array)
        self.dspec_object.freq_array = freq_back.value.copy() * units.m
        pytest.raises(units.UnitConversionError, self.dspec_object.check)
        self.dspec_object.freq_array = freq_back

        self.dspec_object.freq_array = freq_back.value.astype(np.complex) * freq_back.unit
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.freq_array = freq_back

        integration_time_back = copy.deepcopy(self.dspec_object.integration_time)
        self.dspec_object.integration_time = integration_time_back.astype(complex)
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.integration_time = integration_time_back

        Nuv = self.dspec_object.Nuv
        self.dspec_object.Nuv = 10
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nuv = Nuv

        self.dspec_object.data_type = 'delay'
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.data_type = 'frequency'

        self.dspec_object.data_array = self.dspec_object.data_array * units.Hz
        pytest.raises(ValueError, self.dspec_object.check)

        self.dspec_object.data_type = 'delay'
        assert self.dspec_object.check()
        self.dspec_object.data_type = 'frequency'
        self.dspec_object.data_array = self.dspec_object.data_array.value * units.Jy
        assert self.dspec_object.check()

    def test_add_wrong_units(self):
        """Test error is raised when adding a uvdata_object with the wrong units."""
        uvd = UVData()
        uvd.read(self.testfile)
        uvd.vis_units = 'K str'
        pytest.raises(units.UnitConversionError, self.dspec_object.add_uvdata, uvd)
        uvd.vis_units = 'uncalib'
        warn_message = ['Data is uncalibrated. Unable to covert '
                        'noise array to unicalibrated units.']

        pytest.raises(units.UnitConversionError, uvtest.checkWarnings,
                      self.dspec_object.add_uvdata, func_args=[uvd],
                      category=UserWarning,
                      nwarnings=len(warn_message),
                      message=warn_message)

    def test_add_too_many_UVData(self):
        """Test error is raised when adding too many UVData objects."""
        uvd = UVData()
        uvd.read(self.testfile)
        self.dspec_object.Nuv = 2
        pytest.raises(ValueError, self.dspec_object.add_uvdata, uvd)

    def test_incompatible_parameters(self):
        """Test UVData objects with incompatible paramters are rejected."""
        uvd = UVData()
        uvd.read(self.testfile)
        uvd.select(freq_chans=np.arange(12))
        pytest.raises(ValueError, self.dspec_object.add_uvdata, uvd)

    def test_adding_spectral_windows_different_tuple_shape(self):
        """Test error is raised if spectral windows have different shape input."""
        pytest.raises(ValueError, self.dspec_object.select_spectral_windows,
                      spectral_windows=((2, 3), (1, 2, 4)))

    def test_adding_spectral_windows_different_lengths(self):
        """Test error is raised if spectral windows have different shape input."""
        pytest.raises(ValueError, self.dspec_object.select_spectral_windows,
                      spectral_windows=((2, 3), (2, 6)))

    def test_adding_multiple_spectral_windows(self):
        """Test multiple spectral windows are added correctly."""
        self.dspec_object.select_spectral_windows([(3, 5), (7, 9)])
        expected_shape = (2, 1, self.dspec_object.Npols, self.dspec_object.Nbls,
                          self.dspec_object.Ntimes, 3)
        assert expected_shape == self.dspec_object.data_array.shape
        assert self.dspec_object.check()

    def test_add_second_uvdata_object(self):
        """Test a second UVdata object can be added correctly."""
        uvd = UVData()
        uvd.read(self.testfile)
        # multiply by a scalar here to track if it gets set in the correct slot
        uvd.data_array *= np.sqrt(2)
        self.dspec_object.add_uvdata(uvd)
        assert self.dspec_object.Nuv == 2
        assert (np.allclose(self.dspec_object.data_array[:, 0].value,
                            self.dspec_object.data_array[:, 1].value / np.sqrt(2)))


def test_adding_spectral_window_one_tuple():
    """Test spectral window can be added when only one tuple given."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=(3, 12))
    assert dspec_object.Nfreqs == 10
    assert dspec_object.Ndelays == 10
    assert (np.allclose(dspec_object.freq_array.to('Hz').value,
                        uvd.freq_array[:, 3:13]))


def test_adding_spectral_window_between_uvdata():
    """Test that adding a spectral window between uvdata objects is handled."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=[(3, 12)])
    uvd1 = copy.deepcopy(uvd)
    dspec_object.add_uvdata(uvd1)
    assert dspec_object.check()


def test_adding_new_uvdata_with_different_freqs():
    """Test error is raised when trying to add a uvdata object with different freqs."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=[(3, 12)])
    uvd1 = copy.deepcopy(uvd)
    uvd1.freq_array *= 11.1
    pytest.raises(ValueError, dspec_object.add_uvdata, uvd1)


def test_adding_new_uvdata_with_different_lsts():
    """Test error is raised when trying to add a uvdata object with different LSTS."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=[(3, 12)])
    uvd1 = copy.deepcopy(uvd)
    uvd1.lst_array += (3 * units.min * np.pi / (12 * units.h).to('min')).value
    # the actual output of this warning depends on the time difference of the
    #  arrays so we'll cheat on the check.
    warn_message = ["Input LST arrays differ on average by"]
    uvtest.checkWarnings(dspec_object.add_uvdata, func_args=[uvd1],
                         message=warn_message,
                         nwarnings=len(warn_message),
                         category=UserWarning)


def test_select_spectral_window_not_inplace():
    """Test it is possible to return a different object from select spectral window."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    new_dspec = dspec_object.select_spectral_windows(spectral_windows=[(3, 12)],
                                                     inplace=False)
    assert dspec_object != new_dspec


def test_loading_different_arrays():
    """Test error is raised trying to combine different arrays."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    bls = np.unique(uvd.baseline_array)[:-1]
    ants = [uvd.baseline_to_antnums(bl) for bl in bls]
    ants = [(a1, a2) for a1, a2 in ants]
    uvd.select(bls=ants)
    pytest.raises(ValueError, dspec_object.add_uvdata, uvd)


def test_loading_uvb_object():
    """Test a uvb object can have the beam_area and beam_sq_area read."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    uvb.select(frequencies=uvd.freq_array[0])
    assert np.allclose(uvb.get_beam_area(pol='pI'),
                       dspec_object.beam_area.to('sr').value)


def test_loading_uvb_object_no_data():
    """Test error is raised if adding a UVBeam object but no data."""
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    pytest.raises(ValueError, DelaySpectrum, uvb=uvb)


def test_loading_uvb_object_with_data():
    """Test uvbeam can be added in init."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')

    uv = UVData()
    uv.read(testfile)
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object = DelaySpectrum(uv=uv, uvb=uvb)

    assert dspec_object.check()
    assert np.allclose(uvb.get_beam_area(pol='pI'),
                       dspec_object.beam_area.to('sr').value)


def test_loading_uvb_object_with_trcvr():
    """Test a uvb object with trcvr gets added properly."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    uvb.receiver_temperature_array = np.ones((1, uvb.Nfreqs)) * 144
    dspec_object.add_uvbeam(uvb=uvb)
    uvb.select(frequencies=uvd.freq_array[0])
    assert np.allclose(uvb.receiver_temperature_array[0],
                       dspec_object.trcvr.to('K')[0].value)


def test_add_trcvr_scalar():
    """Test a scalar trcvr quantity is broadcast to the correct shape."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.add_trcvr(9 * units.K)
    expected_shape = (dspec_object.Nspws, dspec_object.Nfreqs)
    assert expected_shape == dspec_object.trcvr.shape


def test_add_trcvr_bad_number_of_spectral_windows():
    """Test error is raised if the number of spectral windows do not match with input trcvr."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    bad_temp = np.ones((4, 21)) * units.K
    pytest.raises(ValueError, dspec_object.add_trcvr, bad_temp)


def test_add_trcvr_bad_number_of_freqs():
    """Test error is raised if number of frequencies does not match input trcvr."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    bad_temp = np.ones((1, 51)) * units.K
    pytest.raises(ValueError, dspec_object.add_trcvr, bad_temp)


def test_add_trcvr_vector():
    """Test an arry of trcvr quantity."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    good_temp = np.ones((1, 21)) * 9 * units.K
    dspec_object.add_trcvr(good_temp)
    expected_shape = (dspec_object.Nspws, dspec_object.Nfreqs)
    assert expected_shape == dspec_object.trcvr.shape


def test_add_trcvr_init():
    """Test a scalar trcvr quantity is broadcast to the correct shape during init."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd, trcvr=9 * units.K)
    expected_shape = (dspec_object.Nspws, dspec_object.Nfreqs)
    assert expected_shape == dspec_object.trcvr.shape


def test_add_trcvr_init_error():
    """Test error is raised if trcvr is the only input to init."""
    pytest.raises(ValueError, DelaySpectrum, trcvr=9 * units.K)


def test_spectrum_on_no_data():
    """Test error is raised if spectrum attempted to be taken with no data."""
    dspec_object = DelaySpectrum()
    pytest.raises(ValueError, dspec_object.calculate_delay_spectrum)


def test_noise_shape():
    """Test the generate noise and calculate_noise_power produce correct shape."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.trcvr = np.zeros_like(dspec_object.trcvr)
    dspec_object.beam_area = np.ones_like(dspec_object.beam_area)
    dspec_object.generate_noise()
    assert (dspec_object._noise_array.expected_shape(dspec_object)
            == dspec_object.noise_array.shape)


def test_noise_unit():
    """Test the generate noise and calculate_noise_power produce correct units."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.trcvr = np.zeros_like(dspec_object.trcvr)
    dspec_object.beam_area = np.ones_like(dspec_object.beam_area)
    dspec_object.generate_noise()
    assert dspec_object.noise_array.unit == units.Jy


def test_noise_amplitude():
    """Test noise amplitude with a fixed seed."""
    np.random.seed(0)
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
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
    assert np.allclose(ratio, (test_var / var).value)


def test_delay_transform_units():
    """Test units after calling delay_transform are correct."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    dspec_object.delay_transform()
    assert dspec_object.data_array.unit.is_equivalent(units.Jy * units.Hz)
    dspec_object.delay_transform()
    assert dspec_object.data_array.unit.is_equivalent(units.Jy)


def test_warning_from_uncalibrated_data():
    """Test scaling warning is raised when delay transforming uncalibrated data."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
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
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    uvd = UVData()
    uvd.read(testfile)

    dspec_object = DelaySpectrum(uvd)
    dspec_object.data_type = 'test'
    pytest.raises(ValueError, dspec_object.delay_transform)


def test_delay_spectrum_power_units():
    """Test the units on the output power are correct."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    assert (units.mK**2 * units.Mpc**3).is_equivalent(dspec_object.power_array.unit)


def test_delay_spectrum_power_shape():
    """Test the shape of the output power is correct."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    power_shape = (dspec_object.Nspws, dspec_object.Npols, dspec_object.Nbls,
                   dspec_object.Nbls, dspec_object.Ntimes, dspec_object.Ndelays)
    assert power_shape == dspec_object.power_array.shape


def test_delay_spectrum_power_shape_two_uvdata_objects_read():
    """Test the shape of the output power is correct when two uvdata objects read."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd] * 2)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    power_shape = (dspec_object.Nspws, dspec_object.Npols, dspec_object.Nbls,
                   dspec_object.Nbls, dspec_object.Ntimes, dspec_object.Ndelays)
    assert power_shape == dspec_object.power_array.shape


def test_delay_spectrum_power_shape_two_spectral_windows():
    """Test the shape of the output power when multiple spectral windows given."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    power_shape = (dspec_object.Nspws, dspec_object.Npols, dspec_object.Nbls,
                   dspec_object.Nbls, dspec_object.Ntimes, dspec_object.Ndelays)
    assert power_shape == dspec_object.power_array.shape


def test_cosmological_units():
    """Test the units on cosmological parameters."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    assert dspec_object.k_perpendicular.unit.is_equivalent(1. / units.Mpc)
    assert dspec_object.k_parallel.unit.is_equivalent(1. / units.Mpc)


def test_delay_spectrum_power_units_input_kelvin_str():
    """Test the units on the output power are correct when input kelvin*str."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file_k_units.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.']
    pend_dep_message = ['antenna_positions are not defined. '
                        'antenna_positions will be a required parameter in '
                        'version 1.5']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [DeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    dspec_object = DelaySpectrum(uv=[test_uv_1, test_uv_2])

    dspec_object.calculate_delay_spectrum()
    dspec_object.add_trcvr(144 * units.K)

    assert (units.mK**2 * units.Mpc**3).is_equivalent(dspec_object.power_array.unit)


def test_delay_spectrum_power_units_input_uncalib():
    """Test the units on the output power are correct if input uncalib."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file_uncalib_units.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.']
    pend_dep_message = ['antenna_positions are not defined. '
                        'antenna_positions will be a required parameter in '
                        'version 1.5']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [DeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    warn_message = ['Data is uncalibrated. Unable to covert noise '
                    'array to unicalibrated units.',
                    'Data is uncalibrated. Unable to covert noise '
                    'array to unicalibrated units.']
    dspec_object = uvtest.checkWarnings(DelaySpectrum,
                                        func_kwargs={'uv': [test_uv_1, test_uv_2]},
                                        message=warn_message,
                                        nwarnings=len(warn_message),
                                        category=UserWarning)

    dspec_object.add_trcvr(144 * units.K)
    warn_message = ['Fourier Transforming uncalibrated data. '
                    'Units will not have physical meaning. '
                    'Data will be arbitrarily scaled.']
    uvtest.checkWarnings(dspec_object.calculate_delay_spectrum,
                         message=warn_message,
                         nwarnings=len(warn_message),
                         category=UserWarning)

    assert (units.Hz**2).is_equivalent(dspec_object.power_array.unit)


def test_delay_spectrum_noise_power_units():
    """Test the units on the output noise power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
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
                        'version 1.5']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [DeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv_1)

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

    dspec_object = DelaySpectrum(uv=[test_uv_1, test_uv_2])

    dspec_object.calculate_delay_spectrum()
    dspec_object.add_trcvr(144 * units.K)
    assert (units.mK**2 * units.Mpc**3).is_equivalent(dspec_object.noise_power.unit)


def test_delay_spectrum_thermal_power_units():
    """Test the units on the output thermal power are correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
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
                        'version 1.5']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [DeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv_1)

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

    dspec_object = DelaySpectrum(uv=[test_uv_1, test_uv_2])

    dspec_object.calculate_delay_spectrum()
    dspec_object.add_trcvr(144 * units.K)
    assert (units.mK**2 * units.Mpc**3).is_equivalent(dspec_object.thermal_power.unit)


def test_delay_spectrum_thermal_power_shape():
    """Test the shape of the output thermal power is correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
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
                        'version 1.5']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [DeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv_1)

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

    dspec_object = DelaySpectrum(uv=[test_uv_1, test_uv_2])

    dspec_object.calculate_delay_spectrum()
    dspec_object.add_trcvr(144 * units.K)
    assert dspec_object._thermal_power.expected_shape(dspec_object) == dspec_object.thermal_power.shape


def test_multiple_polarization_file():
    """Test the units on cosmological parameters."""
    testfile = os.path.join(DATA_PATH, 'test_two_pol_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_multiple_pol.beamfits')
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    assert dspec_object.check()
    dspec_object.calculate_delay_spectrum()
    assert dspec_object.check()


def test_update_cosmology_units_and_shapes():
    """Test the check function on DelaySpectrum after changing cosmologies."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    test_cosmo = Planck15
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()

    assert dspec_object.check()

    dspec_object.update_cosmology(cosmology=test_cosmo)
    assert dspec_object.check()


def test_update_cosmology_error_if_not_cosmology_object():
    """Test update cosmology function errors if new cosmology is not a Cosmology object."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    bad_input = DummyClass()

    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    assert dspec_object.check()

    pytest.raises(ValueError, dspec_object.update_cosmology, cosmology=bad_input)


def test_update_cosmology_unit_and_shape_kelvin_sr():
    """Test the check function after changing cosmolgies, input visibility Kelvin * sr."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file_k_units.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    test_cosmo = Planck15

    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.']
    pend_dep_message = ['antenna_positions are not defined. '
                        'antenna_positions will be a required parameter in '
                        'version 1.5']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [DeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    dspec_object = DelaySpectrum(uv=[test_uv_1, test_uv_2])

    dspec_object.calculate_delay_spectrum()
    dspec_object.add_trcvr(144 * units.K)
    assert dspec_object.check()

    dspec_object.update_cosmology(cosmology=test_cosmo)
    assert dspec_object.check()


def test_update_cosmology_unit_and_shape_uncalib():
    """Test the check function after changing cosmolgies, input visibility uncalibrated."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file_uncalib_units.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    test_cosmo = Planck15

    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.']
    pend_dep_message = ['antenna_positions are not defined. '
                        'antenna_positions will be a required parameter in '
                        'version 1.5']

    test_uv_1 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [DeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, 'test_paper_pI.beamfits')

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    warn_message = ['Data is uncalibrated. Unable to covert noise '
                    'array to unicalibrated units.',
                    'Data is uncalibrated. Unable to covert noise '
                    'array to unicalibrated units.']
    dspec_object = uvtest.checkWarnings(DelaySpectrum,
                                        func_kwargs={'uv': [test_uv_1, test_uv_2]},
                                        message=warn_message,
                                        nwarnings=len(warn_message),
                                        category=UserWarning)

    dspec_object.add_trcvr(144 * units.K)
    warn_message = ['Fourier Transforming uncalibrated data. '
                    'Units will not have physical meaning. '
                    'Data will be arbitrarily scaled.']
    uvtest.checkWarnings(dspec_object.calculate_delay_spectrum,
                         message=warn_message,
                         nwarnings=len(warn_message),
                         category=UserWarning)

    assert dspec_object.check()

    dspec_object.update_cosmology(cosmology=test_cosmo)
    assert dspec_object.check()


@sdstest.skipIf_py2
def test_update_cosmology_littleh_units():
    """Test the units can convert to 'littleh' units in python 3."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    test_cosmo = Planck15
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    assert dspec_object.check()

    dspec_object.update_cosmology(cosmology=test_cosmo, littleh_units=True)

    assert dspec_object.check()
    test_unit = (units.mK**2) / (units.littleh / units.Mpc)**3
    assert dspec_object.power_array.unit, test_unit


@sdstest.skipIf_py2
def test_update_cosmology_littleh_units_from_calc_delay_spectr():
    """Test the units can convert to 'littleh' units in python 3 passed through calculate_delay_spectrum."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    test_cosmo = Planck15
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum(cosmology=test_cosmo, littleh_units=True)

    assert dspec_object.check()

    test_unit = (units.mK**2) / (units.littleh / units.Mpc)**3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == 'Planck15'


@sdstest.skipIf_py2
def test_call_update_cosmology_twice():
    """Test cosmology can be updated at least twice in a row with littleh_units."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    test_cosmo1 = WMAP9
    test_cosmo2 = Planck15
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum(cosmology=test_cosmo1, littleh_units=True)

    assert dspec_object.check()
    assert dspec_object.cosmology.name == 'WMAP9'

    dspec_object.update_cosmology(test_cosmo2, littleh_units=True)
    test_unit = (units.mK**2) / (units.littleh / units.Mpc)**3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == 'Planck15'
    assert dspec_object.check()


def test_call_update_cosmology_twice_no_littleh():
    """Test cosmology can be updated at least twice in a row without littleh_units."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    test_cosmo1 = WMAP9
    test_cosmo2 = Planck15
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum(cosmology=test_cosmo1, littleh_units=False)

    assert dspec_object.check()
    assert dspec_object.cosmology.name == 'WMAP9'

    dspec_object.update_cosmology(test_cosmo2, littleh_units=False)
    test_unit = units.mK**2 * units.Mpc**3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == 'Planck15'
    assert dspec_object.check()


def test_call_delay_spectrum_twice_no_littleh():
    """Test calculate_delay_spectrum can be called at least twice in a row without littleh_units."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    test_cosmo1 = WMAP9
    test_cosmo2 = Planck15
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum(cosmology=test_cosmo1, littleh_units=False)

    assert dspec_object.check()
    assert dspec_object.cosmology.name == 'WMAP9'

    dspec_object.calculate_delay_spectrum(cosmology=test_cosmo2, littleh_units=False)
    test_unit = units.mK**2 * units.Mpc**3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == 'Planck15'
    assert dspec_object.check()


@sdstest.skipIf_py2
def test_call_delay_spectrum_twice():
    """Test calculate_delay_spectrum can be called at least twice in a row."""
    testfile = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
    test_uvb_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
    test_cosmo1 = WMAP9
    test_cosmo2 = Planck15
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum(cosmology=test_cosmo1, littleh_units=True)

    assert dspec_object.check()
    assert dspec_object.cosmology.name == 'WMAP9'

    dspec_object.calculate_delay_spectrum(cosmology=test_cosmo2, littleh_units=True)
    test_unit = units.mK**2 * units.Mpc**3 / units.littleh**3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == 'Planck15'
    assert dspec_object.check()
