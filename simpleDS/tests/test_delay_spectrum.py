# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Test Delay Spectrum calculations."""
from __future__ import print_function

import os
import numpy as np
import copy
import pytest
import unittest
from itertools import chain

from pyuvdata import UVBeam, UVData
import pyuvdata.tests as uvtest
from astropy import units
from astropy.cosmology import Planck15, WMAP9
from astropy.cosmology.units import littleh

from scipy.signal import windows

from simpleDS import DelaySpectrum
from simpleDS import utils
from simpleDS.data import DATA_PATH
from pyuvdata.data import DATA_PATH as UVDATA_PATH


@pytest.fixture()
def ds_from_uvfits():
    """Fixture to initialize a DS object."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")

    uvd = UVData()
    uvd.read(testfile)
    ds = DelaySpectrum(uv=[uvd])
    ds.select_spectral_windows([(0, 10), (10, 20)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    ds.add_uvbeam(uvb=uvb)

    yield ds

    del ds, uvd, uvb


@pytest.fixture()
def ds_uvfits_and_uvb():
    """Fixture to also return the UVBeam object."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")

    uvd = UVData()
    uvd.read(testfile)
    ds = DelaySpectrum(uv=[uvd])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    ds.add_uvbeam(uvb=uvb)

    yield ds, uvd, uvb

    del ds, uvd, uvb


@pytest.fixture()
def ds_from_mwa():
    """Fixture to initialize a DS object."""
    testfile = os.path.join(DATA_PATH, "mwa_full_poll.uvh5")

    uvd = UVData()
    uvd.read(testfile)
    uvd.x_orientation = "east"
    ds = DelaySpectrum(uv=[uvd])

    yield ds

    del ds, uvd


@pytest.fixture()
def ds_with_two_uvd():
    """Fixture to return DS object."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")

    uvd = UVData()
    uvd.read(testfile)
    uvd.x_orientation = "east"

    ds = DelaySpectrum(uv=[uvd])
    uvd.data_array += 1e3

    ds.add_uvdata(uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    ds.add_uvbeam(uvb=uvb)

    yield ds

    del ds, uvd, uvb


class DummyClass(object):
    """A Dummy object for comparison."""

    def __init__(self):
        """Do Nothing."""
        pass


class TestDelaySpectrumInit(unittest.TestCase):
    """A test class to check DelaySpectrum objects."""

    def setUp(self):
        """Initialize basic parameter, property and iterator tests."""
        self.required_parameters = [
            "_Ntimes",
            "_Nbls",
            "_Nfreqs",
            "_Npols",
            "_vis_units",
            "_Ndelays",
            "_freq_array",
            "_delay_array",
            "_data_array",
            "_nsample_array",
            "_flag_array",
            "_lst_array",
            "_ant_1_array",
            "_ant_2_array",
            "_baseline_array",
            "_polarization_array",
            "_uvw",
            "_trcvr",
            "_redshift",
            "_k_perpendicular",
            "_k_parallel",
            "_beam_area",
            "_beam_sq_area",
            "_taper",
            "_Nants_telescope",
            "_Nants_data",
        ]

        self.required_properties = [
            "Ntimes",
            "Nbls",
            "Nfreqs",
            "Npols",
            "vis_units",
            "Ndelays",
            "freq_array",
            "delay_array",
            "data_array",
            "nsample_array",
            "flag_array",
            "lst_array",
            "ant_1_array",
            "ant_2_array",
            "baseline_array",
            "polarization_array",
            "uvw",
            "trcvr",
            "redshift",
            "k_perpendicular",
            "k_parallel",
            "beam_area",
            "beam_sq_area",
            "taper",
            "Nants_telescope",
            "Nants_data",
        ]
        self.extra_parameters = ["_power_array"]
        self.extra_properties = ["power_array"]
        self.dspec_object = DelaySpectrum()

    def teardown(self):
        """Test teardown: delete object."""
        del self.dspec_object

    def test_required_parameter_iter_metadata_only(self):
        """Test expected required parameters."""
        required = []
        for prop in self.dspec_object.required():
            required.append(prop)
        for a in self.required_parameters:
            if a.lstrip("_") not in chain(
                self.dspec_object._visdata_params,
                self.dspec_object._power_params,
                self.dspec_object._thermal_params,
            ):
                assert a in required, (
                    "expected attribute " + a + " not returned in required iterator"
                )

    def test_required_parameter(self):
        """Test expected required parameters with data."""
        testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")

        uvd = UVData()
        uvd.read(testfile)
        self.dspec_object = DelaySpectrum(uv=[uvd])

        required = []
        for prop in self.dspec_object.required():
            required.append(prop)
        for a in self.required_parameters:
            assert a in required, (
                "expected attribute " + a + " not returned in required iterator"
            )

    def test_properties(self):
        """Test that properties can be get and set properly."""
        prop_dict = dict(list(zip(self.required_properties, self.required_parameters)))
        for k, v in prop_dict.items():
            rand_num = np.random.rand()
            setattr(self.dspec_object, k, rand_num)
            this_param = getattr(self.dspec_object, v)
            try:
                assert rand_num == this_param.value
            except (AssertionError):
                print(
                    "setting {prop_name} to a random number failed".format(prop_name=k)
                )
                raise (AssertionError)


def test_errors_when_taper_not_function():
    """Test that init errors if taper not a function."""
    pytest.raises(ValueError, DelaySpectrum, taper="test")


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


def test_no_taper():
    """Test default setting of set_taper."""
    dspec = DelaySpectrum()
    dspec.set_taper()
    assert dspec.taper == windows.blackmanharris


class TestBasicFunctions(unittest.TestCase):
    """Test basic equality functions."""

    def setUp(self):
        """Initialize tests of basic methods."""
        self.uvdata_object = UVData()
        self.testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
        self.uvdata_object.read(self.testfile)
        self.dspec_object = DelaySpectrum(uv=self.uvdata_object)
        self.dspec_object2 = copy.deepcopy(self.dspec_object)

    def teardown(self):
        """Test teardown: delete objects."""
        del self.dspec_object
        del self.dspec_object2
        del self.uvdata_object

    def test_equality(self):
        """Basic equality test."""
        print(np.allclose(self.dspec_object.flag_array, self.dspec_object2.flag_array))
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

        self.dspec_object.Ndelays = np.float64(self.dspec_object.Ndelays)
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Ndelays = np.int32(self.dspec_object.Ndelays)

        self.dspec_object.polarization_array = (
            self.dspec_object.polarization_array.astype(np.float32)
        )
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.polarization_array = (
            self.dspec_object.polarization_array.astype(np.int64)
        )

        Nfreqs = copy.deepcopy(self.dspec_object.Nfreqs)
        self.dspec_object.Nfreqs = None
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs = Nfreqs

        self.dspec_object.vis_units = 2
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.vis_units = "Jy"

        self.dspec_object.Nfreqs = (2, 1, 2)
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs = Nfreqs

        self.dspec_object.Nfreqs = np.complex64(2)
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.Nfreqs = Nfreqs

        freq_back = copy.deepcopy(self.dspec_object.freq_array)
        self.dspec_object.freq_array = (
            np.arange(self.dspec_object.Nfreqs).reshape(1, Nfreqs).tolist()
        )
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.freq_array = freq_back

        freq_back = copy.deepcopy(self.dspec_object.freq_array)
        self.dspec_object.freq_array = freq_back.value.copy() * units.m
        pytest.raises(units.UnitConversionError, self.dspec_object.check)
        self.dspec_object.freq_array = freq_back

        self.dspec_object.freq_array = (
            freq_back.value.astype(np.complex64) * freq_back.unit
        )
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

        self.dspec_object.data_type = "delay"
        pytest.raises(ValueError, self.dspec_object.check)
        self.dspec_object.data_type = "frequency"

        self.dspec_object.data_array = self.dspec_object.data_array * units.Hz
        pytest.raises(ValueError, self.dspec_object.check)

        self.dspec_object.data_type = "delay"
        assert self.dspec_object.check()
        self.dspec_object.data_type = "frequency"
        self.dspec_object.data_array = self.dspec_object.data_array.value * units.Jy
        assert self.dspec_object.check()

    def test_add_wrong_units(self):
        """Test error is raised when adding a uvdata_object with the wrong units."""
        uvd = UVData()
        uvd.read(self.testfile)
        uvd.vis_units = "K str"
        pytest.raises(units.UnitConversionError, self.dspec_object.add_uvdata, uvd)
        uvd.vis_units = "uncalib"
        warn_message = [
            "Data is uncalibrated. Unable to covert "
            "noise array to unicalibrated units."
        ]

        with pytest.raises(
            units.UnitConversionError,
            match="Input data object is in units incompatible",
        ):
            with uvtest.check_warnings(UserWarning, warn_message):
                self.dspec_object.add_uvdata(uvd)

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
        pytest.raises(
            ValueError,
            self.dspec_object.select_spectral_windows,
            spectral_windows=((2, 3), (1, 2, 4)),
        )

    def test_adding_spectral_windows_different_lengths(self):
        """Test error is raised if spectral windows have different shape input."""
        pytest.raises(
            ValueError,
            self.dspec_object.select_spectral_windows,
            spectral_windows=((2, 3), (2, 6)),
        )

    def test_adding_multiple_spectral_windows(self):
        """Test multiple spectral windows are added correctly."""
        self.dspec_object.select_spectral_windows([(3, 5), (7, 9)])
        expected_shape = (
            2,
            1,
            self.dspec_object.Npols,
            self.dspec_object.Nbls,
            self.dspec_object.Ntimes,
            3,
        )
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
        assert np.allclose(
            self.dspec_object.data_array[:, 0].value,
            self.dspec_object.data_array[:, 1].value / np.sqrt(2),
        )


def test_adding_spectral_window_one_tuple():
    """Test spectral window can be added when only one tuple given."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=(3, 12))
    assert dspec_object.Nfreqs == 10
    assert dspec_object.Ndelays == 10
    assert np.allclose(dspec_object.freq_array.to("Hz").value, uvd.freq_array[:, 3:13])


def test_adding_spectral_window_between_uvdata():
    """Test that adding a spectral window between uvdata objects is handled."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=[(3, 12)])
    uvd1 = copy.deepcopy(uvd)
    dspec_object.add_uvdata(uvd1)
    assert dspec_object.check()


def test_adding_new_uvdata_with_different_freqs():
    """Test error is raised when trying to add a uvdata object with different freqs."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=[(3, 12)])
    uvd1 = copy.deepcopy(uvd)
    uvd1.freq_array *= 11.1
    pytest.raises(ValueError, dspec_object.add_uvdata, uvd1)


def test_adding_new_uvdata_with_different_lsts():
    """Test error is raised when trying to add a uvdata object with different LSTS."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.select_spectral_windows(spectral_windows=[(3, 12)])
    uvd1 = copy.deepcopy(uvd)
    uvd1.lst_array += (3 * units.min * np.pi / (12 * units.h).to("min")).value
    # the actual output of this warning depends on the time difference of the
    #  arrays so we'll cheat on the check.
    warn_message = ["Input LST arrays differ on average by"]
    with uvtest.check_warnings(UserWarning, warn_message):
        dspec_object.add_uvdata(uvd1)


def test_select_spectral_window_not_inplace():
    """Test it is possible to return a different object from select spectral window."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    new_dspec = dspec_object.select_spectral_windows(
        spectral_windows=[(3, 12)], inplace=False
    )
    assert dspec_object != new_dspec


def test_loading_different_arrays():
    """Test error is raised trying to combine different arrays."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
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
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb, use_exact=True)
    uvb.select(frequencies=uvd.freq_array[0])
    assert np.allclose(
        uvb.get_beam_area(pol="pI"), dspec_object.beam_area.to("sr").value
    )


def test_add_uvb_interp_areas():
    """Test that the returned interped uvb areas match the exact ones."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object2 = copy.deepcopy(dspec_object)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb, use_exact=True)
    uvb.freq_array += 1e6  # Add 1 MHz to force interpolation
    assert uvb.freq_array != dspec_object.freq_array
    dspec_object2.add_uvbeam(uvb=uvb)

    assert np.allclose(
        dspec_object.beam_area.to_value("sr"), dspec_object2.beam_area.to_value("sr")
    )
    assert np.allclose(
        dspec_object.beam_sq_area.to_value("sr"),
        dspec_object2.beam_sq_area.to_value("sr"),
    )
    assert np.allclose(
        dspec_object.trcvr.to_value("K"), dspec_object2.trcvr.to_value("K")
    )


def test_add_uvb_interp_missing_freqs():
    """Test that the built in UVBeam interps match the interped beam areas."""
    pytest.importorskip("astropy_healpix")
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object2 = copy.deepcopy(dspec_object)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object2.add_uvbeam(uvb=uvb)
    uvb.select(frequencies=uvb.freq_array.squeeze()[::2])
    uvb.interpolation_function = "healpix_simple"
    dspec_object.add_uvbeam(uvb=uvb, use_exact=True)
    assert np.allclose(
        dspec_object.beam_area.to_value("sr"), dspec_object2.beam_area.to_value("sr")
    )
    assert np.allclose(
        dspec_object.beam_sq_area.to_value("sr"),
        dspec_object2.beam_sq_area.to_value("sr"),
    )
    assert np.allclose(
        dspec_object.trcvr.to_value("K"), dspec_object2.trcvr.to_value("K")
    )


def test_add_uvdata_uvbeam_uvdata():
    """Test that a uvb can be added in between two uvdata objects."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb, use_exact=True)
    dspec_object.add_uvdata(uvd)
    assert dspec_object.check()


def test_loading_uvb_object_no_data():
    """Test error is raised if adding a UVBeam object but no data."""
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    pytest.raises(ValueError, DelaySpectrum, uvb=uvb)


def test_loading_uvb_object_with_data():
    """Test uvbeam can be added in init."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")

    uv = UVData()
    uv.read(testfile)
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object = DelaySpectrum(uv=uv, uvb=uvb)

    assert dspec_object.check()
    assert np.allclose(
        uvb.get_beam_area(pol="pI"), dspec_object.beam_area.to("sr").value
    )


def test_loading_uvb_object_with_trcvr():
    """Test a uvb object with trcvr gets added properly."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    uvb.receiver_temperature_array = np.ones((1, uvb.Nfreqs)) * 144
    dspec_object.add_uvbeam(uvb=uvb, use_exact=True)
    uvb.select(frequencies=uvd.freq_array[0])
    assert np.allclose(
        uvb.receiver_temperature_array[0], dspec_object.trcvr.to("K")[0].value
    )


def test_loading_uvb_object_with_trcvr_interp():
    """Test a uvb object with trcvr gets added properly."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    uvb.receiver_temperature_array = np.ones((1, uvb.Nfreqs)) * 144
    uvb.select(frequencies=uvb.freq_array.squeeze()[::2])
    dspec_object.add_uvbeam(uvb=uvb, use_exact=False)
    assert np.allclose(144, dspec_object.trcvr.to("K")[0].value)


def test_add_trcvr_scalar():
    """Test a scalar trcvr quantity is broadcast to the correct shape."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.add_trcvr(9 * units.K)
    expected_shape = (dspec_object.Nspws, dspec_object.Nfreqs)
    assert expected_shape == dspec_object.trcvr.shape


def test_add_trcvr_bad_number_of_spectral_windows():
    """Test error is raised if the number of spectral windows do not match with input trcvr."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    bad_temp = np.ones((4, 21)) * units.K
    pytest.raises(ValueError, dspec_object.add_trcvr, bad_temp)


def test_add_trcvr_bad_number_of_freqs():
    """Test error is raised if number of frequencies does not match input trcvr."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    bad_temp = np.ones((1, 51)) * units.K
    pytest.raises(ValueError, dspec_object.add_trcvr, bad_temp)


def test_add_trcvr_vector():
    """Test an arry of trcvr quantity."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    good_temp = np.ones((1, 21)) * 9 * units.K
    dspec_object.add_trcvr(good_temp)
    expected_shape = (dspec_object.Nspws, dspec_object.Nfreqs)
    assert expected_shape == dspec_object.trcvr.shape


def test_add_trcvr_init():
    """Test a scalar trcvr quantity is broadcast to the correct shape during init."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
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
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)
    dspec_object.trcvr = np.zeros_like(dspec_object.trcvr)
    dspec_object.beam_area = np.ones_like(dspec_object.beam_area)
    dspec_object.generate_noise()
    assert (
        dspec_object._noise_array.expected_shape(dspec_object)
        == dspec_object.noise_array.shape
    )


def test_noise_unit():
    """Test the generate noise and calculate_noise_power produce correct units."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
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
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
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
    test_amplitude = (
        180
        * units.K
        * np.power((dspec_object.freq_array.to("GHz") / (0.18 * units.GHz)), -2.55)
        / np.sqrt(np.diff(dspec_object.freq_array[0])[0].value)
    ).reshape(1, 1, dspec_object.Nfreqs)
    test_amplitude *= dspec_object.beam_area / utils.jy_to_mk(dspec_object.freq_array)
    test_var = test_amplitude.to("Jy") ** 2
    # this was from running this test by hand
    ratio = np.array(
        [
            [
                1.07735447,
                1.07082788,
                1.07919504,
                1.04992591,
                1.02254714,
                0.99884931,
                0.94861011,
                1.01908474,
                1.03877442,
                1.00549461,
                1.09642801,
                1.01100747,
                1.0201933,
                1.05762868,
                0.95156612,
                1.00190002,
                1.00046522,
                1.02796162,
                1.04277506,
                0.98373618,
                1.01235802,
            ]
        ]
    )
    assert np.allclose(ratio, (test_var / var).value)


def test_delay_transform_units():
    """Test units after calling delay_transform are correct."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    dspec_object.delay_transform()
    assert dspec_object.data_array.unit.is_equivalent(units.Jy * units.Hz)
    dspec_object.delay_transform()
    assert dspec_object.data_array.unit.is_equivalent(units.Jy)


def test_warning_from_uncalibrated_data():
    """Test scaling warning is raised when delay transforming uncalibrated data."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    uvd.vis_units = "uncalib"
    warn_message = [
        "Data is uncalibrated. Unable to covert noise array to unicalibrated units."
    ]
    with uvtest.check_warnings(UserWarning, warn_message):
        dspec_object = DelaySpectrum(uvd)

    warn_message = [
        "Fourier Transforming uncalibrated data. Units will "
        "not have physical meaning. "
        "Data will be arbitrarily scaled."
    ]
    with uvtest.check_warnings(UserWarning, warn_message):
        dspec_object.delay_transform()


def test_delay_transform_bad_data_type():
    """Test error is raised in delay_transform if data_type is bad."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)

    dspec_object = DelaySpectrum(uvd)
    dspec_object.data_type = "test"
    pytest.raises(ValueError, dspec_object.delay_transform)


def test_delay_spectrum_power_units():
    """Test the units on the output power are correct."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb, use_exact=True)
    dspec_object.calculate_delay_spectrum()
    assert (units.mK**2 * units.Mpc**3).is_equivalent(dspec_object.power_array.unit)


def test_delay_spectrum_power_shape():
    """Test the shape of the output power is correct."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=uvd)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    power_shape = (
        dspec_object.Nspws,
        dspec_object.Npols,
        dspec_object.Nbls,
        dspec_object.Nbls,
        dspec_object.Ntimes,
        dspec_object.Ndelays,
    )
    assert power_shape == dspec_object.power_array.shape


def test_delay_spectrum_power_shape_two_uvdata_objects_read():
    """Test the shape of the output power is correct when two uvdata objects read."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd] * 2)

    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    power_shape = (
        dspec_object.Nspws,
        dspec_object.Npols,
        dspec_object.Nbls,
        dspec_object.Nbls,
        dspec_object.Ntimes,
        dspec_object.Ndelays,
    )
    assert power_shape == dspec_object.power_array.shape


def test_delay_spectrum_power_shape_two_spectral_windows():
    """Test the shape of the output power when multiple spectral windows given."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.calculate_delay_spectrum()
    power_shape = (
        dspec_object.Nspws,
        dspec_object.Npols,
        dspec_object.Nbls,
        dspec_object.Nbls,
        dspec_object.Ntimes,
        dspec_object.Ndelays,
    )
    assert power_shape == dspec_object.power_array.shape


def test_cosmological_units():
    """Test the units on cosmological parameters."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    assert dspec_object.k_perpendicular.unit.is_equivalent(1.0 / units.Mpc)
    assert dspec_object.k_parallel.unit.is_equivalent(1.0 / units.Mpc)


def test_delay_spectrum_power_units_input_kelvin_str():
    """Test the units on the output power are correct when input kelvin*str."""
    test_file = os.path.join(DATA_PATH, "paper_test_file_k_units.uvh5")
    test_uv_1 = UVData()
    test_uv_1.read(test_file)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, "test_paper_pI.beamfits")

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
    test_file = os.path.join(DATA_PATH, "paper_test_file_uncalib_units.uvh5")

    test_uv_1 = UVData()
    test_uv_1.read(test_file)

    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, "test_paper_pI.beamfits")

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    warn_message = [
        "Data is uncalibrated. Unable to covert noise array to unicalibrated units.",
        "Data is uncalibrated. Unable to covert noise array to unicalibrated units.",
    ]
    with uvtest.check_warnings(UserWarning, warn_message):
        dspec_object = DelaySpectrum([test_uv_1, test_uv_2])

    dspec_object.add_trcvr(144 * units.K)
    warn_message = [
        "Fourier Transforming uncalibrated data. "
        "Units will not have physical meaning. "
        "Data will be arbitrarily scaled."
    ]
    with uvtest.check_warnings(UserWarning, match=warn_message):
        dspec_object.calculate_delay_spectrum()

    assert (units.Hz**2).is_equivalent(dspec_object.power_array.unit)


def test_delay_spectrum_noise_power_units():
    """Test the units on the output noise power are correct."""
    test_file = os.path.join(DATA_PATH, "paper_test_file.uvh5")

    test_uv_1 = UVData()
    test_uv_1.read(test_file)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, "test_paper_pI.beamfits")

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    dspec_object = DelaySpectrum(uv=[test_uv_1, test_uv_2])

    dspec_object.calculate_delay_spectrum()
    dspec_object.add_trcvr(144 * units.K)
    assert (units.mK**2 * units.Mpc**3).is_equivalent(dspec_object.noise_power.unit)


def test_delay_spectrum_thermal_power_units():
    """Test the units on the output thermal power are correct."""
    test_file = os.path.join(DATA_PATH, "paper_test_file.uvh5")

    test_uv_1 = UVData()
    test_uv_1.read(test_file)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, "test_paper_pI.beamfits")

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    dspec_object = DelaySpectrum(uv=[test_uv_1, test_uv_2])

    dspec_object.calculate_delay_spectrum()
    dspec_object.add_trcvr(144 * units.K)
    assert (units.mK**2 * units.Mpc**3).is_equivalent(
        dspec_object.thermal_power.unit
    )


def test_delay_spectrum_thermal_power_shape():
    """Test the shape of the output thermal power is correct."""
    test_file = os.path.join(DATA_PATH, "paper_test_file.uvh5")

    test_uv_1 = UVData()
    test_uv_1.read(test_file)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, "test_paper_pI.beamfits")

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    dspec_object = DelaySpectrum(uv=[test_uv_1, test_uv_2])

    dspec_object.calculate_delay_spectrum()
    dspec_object.add_trcvr(144 * units.K)
    assert (
        dspec_object._thermal_power.expected_shape(dspec_object)
        == dspec_object.thermal_power.shape
    )


def test_multiple_polarization_file():
    """Test the units on cosmological parameters."""
    testfile = os.path.join(DATA_PATH, "test_two_pol_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_multiple_pol.beamfits")
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


def test_remove_cosmology():
    """Test removing cosmology does not alter data from before cosmology is applied."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)

    dspec_object2 = copy.deepcopy(dspec_object)

    dspec_object.calculate_delay_spectrum(littleh_units=True)

    dspec_object.remove_cosmology()

    assert dspec_object.power_array.unit.is_equivalent(units.Jy**2 * units.Hz**2)

    dspec_object2.delay_transform()
    dspec_object2.power_array = utils.cross_multiply_array(
        array_1=dspec_object2.data_array[:, 0], axis=2
    )

    assert units.allclose(dspec_object2.power_array, dspec_object.power_array)


def test_remove_cosmology_no_cosmo():
    """Test removing cosmology does not alter data from before cosmology is applied."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.select_spectral_windows([(1, 3), (4, 6)])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    dspec_object.add_uvbeam(uvb=uvb)
    dspec_object.delay_transform()
    dspec_object.power_array = utils.cross_multiply_array(
        array_1=dspec_object.data_array[:, 0], axis=2
    )
    dspec_object.noise_power = utils.cross_multiply_array(
        array_1=dspec_object.noise_array[:, 0], axis=2
    )

    dspec_object2 = copy.deepcopy(dspec_object)

    dspec_object.remove_cosmology()

    assert dspec_object.power_array.unit.is_equivalent(units.Jy**2 * units.Hz**2)

    assert units.allclose(dspec_object2.power_array, dspec_object.power_array)


def test_remove_cosmology_cosmo_none():
    """Test removing cosmology does not alter data from before cosmology is applied."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    uvd = UVData()
    uvd.read(testfile)
    dspec_object = DelaySpectrum(uv=[uvd])
    dspec_object.cosmology = None

    with pytest.raises(ValueError) as cm:
        dspec_object.remove_cosmology()
    assert str(cm.value).startswith("Cannot remove cosmology of type")


def test_update_cosmology_units_and_shapes():
    """Test the check function on DelaySpectrum after changing cosmologies."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
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
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
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
    test_file = os.path.join(DATA_PATH, "paper_test_file_k_units.uvh5")
    test_cosmo = Planck15

    test_uv_1 = UVData()
    test_uv_1.read(test_file)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, "test_paper_pI.beamfits")

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
    test_file = os.path.join(DATA_PATH, "paper_test_file_uncalib_units.uvh5")
    test_cosmo = Planck15

    test_uv_1 = UVData()
    test_uv_1.read(test_file)
    test_uv_2 = copy.deepcopy(test_uv_1)

    beam_file = os.path.join(DATA_PATH, "test_paper_pI.beamfits")

    uvb = UVBeam()
    uvb.read_beamfits(beam_file)

    test_uv_1.select(freq_chans=np.arange(95, 116))
    test_uv_2.select(freq_chans=np.arange(95, 116))

    warn_message = [
        "Data is uncalibrated. Unable to covert noise array to unicalibrated units.",
        "Data is uncalibrated. Unable to covert noise array to unicalibrated units.",
    ]
    with uvtest.check_warnings(UserWarning, warn_message):
        dspec_object = DelaySpectrum([test_uv_1, test_uv_2])

    dspec_object.add_trcvr(144 * units.K)
    warn_message = [
        "Fourier Transforming uncalibrated data. "
        "Units will not have physical meaning. "
        "Data will be arbitrarily scaled."
    ]
    with uvtest.check_warnings(UserWarning, warn_message):
        dspec_object.calculate_delay_spectrum()

    assert dspec_object.check()

    dspec_object.update_cosmology(cosmology=test_cosmo)
    assert dspec_object.check()


def test_update_cosmology_littleh_units():
    """Test the units can convert to 'littleh' units in python 3."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
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
    test_unit = (units.mK**2) / (littleh / units.Mpc) ** 3
    assert dspec_object.power_array.unit, test_unit


def test_update_cosmology_littleh_units_from_calc_delay_spectr():
    """Test the units can convert to 'littleh' units in python 3 passed through calculate_delay_spectrum."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
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

    test_unit = (units.mK**2) / (littleh / units.Mpc) ** 3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == "Planck15"


def test_call_update_cosmology_twice():
    """Test cosmology can be updated at least twice in a row with littleh_units."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
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
    assert dspec_object.cosmology.name == "WMAP9"

    dspec_object.update_cosmology(test_cosmo2, littleh_units=True)
    test_unit = (units.mK**2) / (littleh / units.Mpc) ** 3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == "Planck15"
    assert dspec_object.check()


def test_call_update_cosmology_twice_no_littleh():
    """Test cosmology can be updated at least twice in a row without littleh_units."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
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
    assert dspec_object.cosmology.name == "WMAP9"

    dspec_object.update_cosmology(test_cosmo2, littleh_units=False)
    test_unit = units.mK**2 * units.Mpc**3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == "Planck15"
    assert dspec_object.check()


def test_call_delay_spectrum_twice_no_littleh():
    """Test calculate_delay_spectrum can be called at least twice in a row without littleh_units."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
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
    assert dspec_object.cosmology.name == "WMAP9"

    dspec_object.calculate_delay_spectrum(cosmology=test_cosmo2, littleh_units=False)
    test_unit = units.mK**2 * units.Mpc**3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == "Planck15"
    assert dspec_object.check()


def test_call_delay_spectrum_twice():
    """Test calculate_delay_spectrum can be called at least twice in a row."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")
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
    assert dspec_object.cosmology.name == "WMAP9"

    dspec_object.calculate_delay_spectrum(cosmology=test_cosmo2, littleh_units=True)
    test_unit = units.mK**2 * units.Mpc**3 / littleh**3
    assert dspec_object.power_array.unit == test_unit
    assert dspec_object.cosmology.name == "Planck15"
    assert dspec_object.check()


@pytest.mark.parametrize(
    "input,err_type,err_message",
    [
        ({"antenna_nums": [-1]}, ValueError, "Antenna -1 is not present in either"),
        ({"bls": []}, ValueError, "bls must be a list of tuples of antenna numbers"),
        (
            {"bls": [(0, 44), "test"]},
            ValueError,
            "bls must be a list of tuples of antenna numbers",
        ),
        (
            {"bls": [(1, "2")]},
            ValueError,
            "bls must be a list of tuples of antenna numbers",
        ),
        (
            {"bls": [("1", 2)]},
            ValueError,
            "bls must be a list of tuples of antenna numbers",
        ),
        (
            {"bls": [(1, 2, "xx")], "polarizations": "yy"},
            ValueError,
            "Cannot provide length-3 tuples and also",
        ),
        (
            {"bls": [(1, 2, 3)]},
            ValueError,
            "The third element in each bl must be a polarization",
        ),
        ({"bls": [(2, 3)]}, ValueError, "Baseline (2, 3) has no data associate"),
        ({"spws": ["pi"]}, ValueError, "Input spws must be an array_like of integers"),
        ({"spws": [5]}, ValueError, "Input spectral window values must be less"),
        (
            {"frequencies": [12] * units.Hz},
            ValueError,
            "Frequency 12.0 Hz not present in the frequency array.",
        ),
        (
            {"frequencies": [146798030.15625, 147290641.0, 151724138.59375] * units.Hz},
            ValueError,
            "Frequencies provided for selection will result in a non-rectangular",
        ),
        (
            {"delays": [12] * units.ns},
            ValueError,
            "The input delay 12.0 ns is not present in the delay_array.",
        ),
        (
            {"lsts": [7] * units.rad},
            ValueError,
            "The input lst 7.0 rad is not present in the lst_array.",
        ),
        (
            {"lst_range": [0, 2, 3] * units.rad},
            ValueError,
            "Parameter lst_range must be an Astropy Quantity object with size 2 ",
        ),
        (
            {"polarizations": ["pU"]},
            ValueError,
            "Polarization 3 not present in polarization_array.",
        ),
        (
            {"delay_chans": np.arange(11).tolist(), "delays": -96.66666543 * units.ns},
            ValueError,
            "The intersection of the input delays and delay_chans ",
        ),
        (
            {"uv_index": np.arange(5).tolist()},
            ValueError,
            "The number of UVData objects in this DelaySpectrum object",
        ),
    ],
)
def test_select_preprocess_errors(ds_from_uvfits, input, err_type, err_message):
    """Test Errors raised by _select_preprocess."""
    ds = ds_from_uvfits

    ds.delay_transform()

    with pytest.raises(err_type) as cm:
        ds.select(**input)
    assert str(cm.value).startswith(err_message)


@pytest.mark.parametrize(
    "input",
    [
        {"antenna_nums": [0, 44]},
        {"bls": (0, 26)},  # if statement looking for just one input that is a tuple
        {"bls": (26, 0)},  # reverse the baseline to see if it is taken
        {"bls": [(0, 26), (1, 4)]},
        {"bls": [(0, 26), 69637]},
        {"bls": [(0, 26, "pI"), (1, 4, "pI")]},
        {"antenna_nums": [0, 44], "bls": [157697]},  # Mix bls and antenna_nums
        {"freq_chans": np.arange(11).tolist()},
        {"freq_chans": [np.arange(11).tolist()]},
        {"frequencies": [146798030.15625, 147290641.0, 147783251.84375] * units.Hz},
        {
            "freq_chans": np.arange(3).tolist(),
            "frequencies": [146798030.15625, 147290641.0, 147783251.84375] * units.Hz,
        },
        {"polarizations": ["pI"]},
        {"polarizations": [[1]]},
    ],
)
@pytest.mark.filterwarnings(
    "ignore:This object has already been converted to a power spectrum"
)
def test_select(ds_uvfits_and_uvb, input):
    """Test select befoe or after making a delay spectrum object is consistent."""
    ds, uvd, uvb = ds_uvfits_and_uvb
    uvd_input = {
        key: val.value if isinstance(val, units.Quantity) else val
        for key, val in input.items()
    }
    if "bls" in uvd_input and not isinstance(uvd_input["bls"], tuple):
        uvd_input["bls"] = [
            uvd.baseline_to_antnums(bl)
            if isinstance(bl, (int, np.int_, np.intc))
            else bl
            for bl in uvd_input["bls"]
        ]

    uvd.select(**uvd_input)
    ds.calculate_delay_spectrum()

    if (
        "bls" in input
        and isinstance(input["bls"], list)
        and any(isinstance(b, (int, np.intc, np.int_)) for b in input["bls"])
    ):
        with pytest.warns(UserWarning) as cm:
            ds.select(**input, inplace=True)
        assert len(cm) == 1
        assert str(cm[0].message).startswith(
            "Input baseline array is a mix of integers and tuples of integers."
        )
    else:
        ds.select(**input, inplace=True)

    ds2 = DelaySpectrum(uvd, uvb=uvb)
    ds2.calculate_delay_spectrum()
    # the uvw and k_perp will be different because of how the redundant group
    # is calculated
    if not units.allclose(ds2.uvw, ds.uvw):
        ds2.uvw = ds.uvw
    if ds2.k_perpendicular != ds.k_perpendicular:
        ds2.k_perpendicular = ds.k_perpendicular

    if "freq_chans" in input or "frequencies" in input:
        # changing frequencies will alter how the data is tapered
        for param1, param2 in zip(ds.power_like_parameters, ds2.power_like_parameters):
            assert param1.shape != param2.shape

        assert ds._data_array != ds2._data_array
        assert ds.Ndelays != ds2.Ndelays
        assert ds._delay_array != ds2._delay_array
        assert ds._k_parallel != ds2._k_parallel

        # force equality though for the rest of the check
        for name in ds._power_params:
            setattr(ds2, name, getattr(ds, name))
        ds2.Ndelays = ds.Ndelays
        ds2.k_parallel = ds.k_parallel
        ds2.delay_array = ds.delay_array
        ds2.data_array = ds.data_array

    assert ds == ds2


@pytest.mark.parametrize(
    "input,ndelays",
    [
        ({"delay_chans": np.arange(11).tolist()}, 11),
        ({"delay_chans": [np.arange(11).tolist()]}, 11),
        (
            {
                "delay_chans": np.arange(11).tolist(),
                "delays": -483.333327140625 * units.ns,
            },
            1,
        ),
    ],
)
def test_select_delay_chans(ds_uvfits_and_uvb, input, ndelays):
    """Test selection along delay axis."""
    ds, uvd, uvb = ds_uvfits_and_uvb
    ds.calculate_delay_spectrum()

    ds_out = ds.select(**input)
    assert ds_out.Ndelays == ndelays
    if "delay_chans" in input and "delays" not in input:
        print("ds_out", ds_out.delay_array)
        print("original", ds.delay_array[np.squeeze(input["delay_chans"])])
        expected = ds.delay_array[np.squeeze(input["delay_chans"])]
    elif "delays" in input and "delay_chans" not in input:
        expected = input["delays"]
    else:
        expected = units.Quantity(
            sorted(
                set(input["delays"].flatten()).intersection(
                    ds.delay_array[np.squeeze(input["delay_chans"])]
                )
            )
        )
    assert units.allclose(ds_out.delay_array, expected)


@pytest.mark.parametrize(
    "input,ntimes",
    [
        ({"lsts": [0, 1]}, 2),
        ({"lst_range": [2, 3] * units.rad}, 9),
        (
            {
                "lsts": [0, 1],
                "lst_range": [1.9, 3] * units.rad,
            },
            2,
        ),
    ],
)
def test_select_lsts(ds_uvfits_and_uvb, input, ntimes):
    """Test time/lst selection."""
    ds, uvd, uvb = ds_uvfits_and_uvb
    ds.calculate_delay_spectrum()

    if "lsts" in input:
        input["lsts"] = units.Quantity([ds.lst_array[ind] for ind in input["lsts"]])

    ds_out = ds.select(**input)
    assert ds_out.Ntimes == ntimes
    if "lst_range" in input and "lsts" not in input:
        inds = np.logical_and(
            input["lst_range"][0] <= ds.lst_array, ds.lst_array <= input["lst_range"][1]
        )
        expected = ds.lst_array[inds]
    elif "lsts" in input and "lst_range" not in input:
        expected = input["lsts"]
    else:
        inds = np.logical_and(
            input["lst_range"][0] <= ds.lst_array, ds.lst_array <= input["lst_range"][1]
        )
        expected = units.Quantity(
            sorted(set(input["lsts"].flatten()).intersection(ds.lst_array[inds]))
        )
    assert units.allclose(ds_out.lst_array, expected)


def test_select_spws(ds_from_uvfits):
    """Test various selections with spectral windows input."""
    ds = ds_from_uvfits
    ds_copy = copy.deepcopy(ds)

    ds.select_spectral_windows([(0, 10), (5, 15), (10, 20)])
    ds_out1 = ds.select(spws=0, inplace=False)

    ds_expected = copy.deepcopy(ds_copy)
    ds_expected.select_spectral_windows((0, 10))
    assert ds_out1 == ds_expected

    ds_expected = copy.deepcopy(ds_copy)
    ds_expected.select_spectral_windows([(5, 15), (10, 20)])
    ds_out2 = ds.select(spws=[1, 2], inplace=False)
    assert ds_out2 == ds_expected


def test_select_spws_and_freqs(ds_from_uvfits):
    """Test selection of spectral windows and frequencies."""
    ds = ds_from_uvfits
    freqs = copy.deepcopy(ds.freq_array)
    expected = freqs[0, 0:11]
    ds.select_spectral_windows([(0, 10), (5, 15), (10, 20)])

    ds_out = ds.select(spws=0, frequencies=freqs[0, 0:20])
    assert units.allclose(ds_out.freq_array.flatten(), expected)


def test_select_spws_with_power_spectrum(ds_from_uvfits):
    """Test select after power spectrum estimation."""
    ds = ds_from_uvfits
    freqs = copy.deepcopy(ds.freq_array)
    expected = freqs[0, 0:11]
    ds.select_spectral_windows([(0, 10), (5, 15), (10, 20)])
    ds.calculate_delay_spectrum()

    ds_out = ds.select(spws=0, frequencies=freqs[0, 0:20])

    assert units.allclose(ds_out.freq_array.flatten(), expected)
    assert ds_out.Nspws == 1
    assert ds_out.redshift.size == 1


def test_select_full_array(ds_from_uvfits):
    """Test select using the full array for each input."""
    ds = ds_from_uvfits
    output_inds = ds._select_preprocess(
        antenna_nums=list(set(ds.ant_1_array).union(ds.ant_2_array)),
        spws=list(range(ds.Nspws)),
        frequencies=ds.freq_array.flatten(),
        delays=ds.delay_array.flatten(),
        lsts=ds.lst_array,
        polarizations=ds.polarization_array,
        uv_index=list(range(ds.Nuv)),
    )
    for ind in output_inds:
        assert ind is None


def test_select_pol(ds_from_mwa):
    """Test selectino using polarization dimension."""
    ds = ds_from_mwa
    expected = [-5, -6]
    ds.select(polarizations=expected, inplace=True)
    assert np.array_equal(ds.polarization_array, expected)


def test_select_uv_index(ds_with_two_uvd):
    """Test you can select separate uvdata objects out of a DS object."""
    ds = ds_with_two_uvd
    ds1 = ds.select(uv_index=[[0]])
    ds2 = ds.select(uv_index=1)
    ds2.data_array -= 1e3 * ds2.data_array.unit
    assert ds1 == ds2
