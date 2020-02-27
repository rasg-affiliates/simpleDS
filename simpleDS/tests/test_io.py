# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2020 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Test Delay Spectrum I/O Operations."""
from __future__ import print_function

import os
import copy
import h5py
import pytest
import numpy as np

from astropy import units
import pyuvdata.tests as uvtest
from scipy.signal import windows
from pyuvdata import UVBeam, UVData

from simpleDS import DelaySpectrum
from simpleDS.data import DATA_PATH
from pyuvdata.data import DATA_PATH as UVDATA_PATH


@pytest.fixture()
def ds_from_uvfits():
    """Fixture to return DS object."""
    testfile = os.path.join(UVDATA_PATH, "test_redundant_array.uvfits")
    test_uvb_file = os.path.join(DATA_PATH, "test_redundant_array.beamfits")

    uvd = UVData()
    uvd.read(testfile)
    uvd.x_orientation = "east"

    ds = DelaySpectrum(uv=[uvd])
    uvb = UVBeam()
    uvb.read_beamfits(test_uvb_file)
    ds.add_uvbeam(uvb=uvb)

    yield ds

    del ds, uvd, uvb


@pytest.fixture()
def test_outfile():
    """Fixture to get the outfile."""
    filename = os.path.join(pytest.testdir, "test_out.h5")

    yield filename

    if os.path.exists(filename):
        os.remove(filename)


def test_readwrite_ds_object(ds_from_uvfits, test_outfile):
    """Test a ds object can be written and read without chaning the object."""
    ds = ds_from_uvfits
    ds.select_spectral_windows([(1, 3), (4, 6)])
    ds.add_trcvr(9 * units.K)

    ds.write(test_outfile)

    ds_in = DelaySpectrum()
    ds_in.read(test_outfile)

    assert ds == ds_in


def test_read_metadata_only(ds_from_uvfits, test_outfile):
    """Test reading only metadata from file."""
    ds = ds_from_uvfits
    ds.select_spectral_windows([(1, 3), (4, 6)])
    ds.add_trcvr(9 * units.K)

    ds.write(test_outfile)
    ds_in = DelaySpectrum()
    ds_in.read(test_outfile, read_data=False)
    assert ds_in.metadata_only
    for param in ds_in.required():
        assert getattr(ds, param) == getattr(ds_in, param)


def test_read_no_Nants_info(ds_from_uvfits, test_outfile):
    """Test reading when Nants data and Nants Telescope are missing."""
    ds = ds_from_uvfits
    ds.Nants_telescope = 64
    ds.select_spectral_windows([(1, 3), (4, 6)])
    ds.add_trcvr(9 * units.K)

    ds.write(test_outfile)
    with h5py.File(test_outfile) as h5:
        del h5["/Header/Nants_data"], h5["Header/Nants_telescope"]

    ds_in = DelaySpectrum()
    with pytest.warns(UserWarning) as record:
        ds_in.read(test_outfile)
    assert len(record) == 1
    assert str(record[0].message).startswith(
        "Nants_telescope is not present in the header of this save file."
    )

    assert ds_in.Nants_telescope != ds.Nants_telescope
    ds_in.Nants_telescope = ds.Nants_telescope
    assert ds_in == ds


def test_readwrite_ds_object_power_units(ds_from_uvfits, test_outfile):
    """Test a ds object can be written and read without chaning the object."""
    ds = ds_from_uvfits
    ds.select_spectral_windows([(1, 3), (4, 6)])

    ds.calculate_delay_spectrum()

    uvtest.checkWarnings(
        ds.write,
        func_args=[test_outfile],
        message="Cannot write DelaySpectrum objects to file when power ",
        nwarnings=1,
        category=UserWarning,
    )

    ds_in = DelaySpectrum()
    ds_in.read(test_outfile)

    assert ds == ds_in


def test_readwrite_custom_taper(ds_from_uvfits, test_outfile):
    """Test a ds object can be written and read without chaning the object."""
    ds = ds_from_uvfits
    ds.select_spectral_windows([(1, 3), (4, 6)])

    ds.set_taper(windows.cosine)

    ds.delay_transform()

    uvtest.checkWarnings(
        ds.write,
        func_args=[test_outfile],
        message="The given taper funtion has a different name than ",
        nwarnings=1,
        category=UserWarning,
    )

    ds_in = DelaySpectrum()
    uvtest.checkWarnings(
        ds_in.read,
        func_args=[test_outfile],
        func_kwargs={"run_check": False},
        message="Saved taper function has different name than",
        nwarnings=1,
        category=UserWarning,
    )

    assert ds != ds_in
    ds_in.set_taper(windows.cosine)
    assert ds == ds_in


def test_read_bad_file():
    """Test error raised when reading non-existant file."""
    ds = DelaySpectrum()
    with pytest.raises(IOError) as cm:
        ds.read("bad_file")
    assert str(cm.value).startswith("bad_file not found")


def test_file_exists(ds_from_uvfits, test_outfile):
    """Test error when file exists."""
    ds = ds_from_uvfits
    ds.select_spectral_windows([(1, 3), (4, 6)])

    ds.write(test_outfile)
    with pytest.raises(IOError) as cm:
        ds.write(test_outfile)
    assert str(cm.value).startswith("File exists")


def test_overwrite_file(ds_from_uvfits, test_outfile):
    """Test file can be overwritten."""
    with open(test_outfile, "w") as f:
        f.write("")

    ds = ds_from_uvfits
    ds.x_orientation = None
    ds.select_spectral_windows([(1, 3), (4, 6)])

    ds.delay_transform()
    ds.write(test_outfile, overwrite=True)

    ds_in = DelaySpectrum()
    with pytest.warns(UserWarning) as record:
        ds_in.read(test_outfile)
    assert len(record) == 1
    assert str(record[0].message).startswith(
        "The parameter x_orientation is not present in the header of this save file."
    )
    assert ds == ds_in


def test_partial_read_spectral_windows(ds_from_uvfits, test_outfile):
    """Test making spectral window selection on read."""
    ds = ds_from_uvfits
    ds.select_spectral_windows([(0, 10), (5, 15), (10, 20)])

    ds.write(test_outfile)

    ds_in1 = DelaySpectrum()
    ds_in2 = DelaySpectrum()
    ds_in1.read(test_outfile, spws=0)
    ds_in2.read(test_outfile, spws=[1, 2])

    ds_test = ds.select(spws=[0], inplace=False)
    assert ds_in1 == ds_test

    ds_test = ds.select(spws=[1, 2], inplace=False)
    assert ds_in2 == ds_test


def test_partial_read_frequencies_shortest(ds_from_uvfits, test_outfile):
    """Test partial frequency read."""
    ds = ds_from_uvfits
    freqs = copy.deepcopy(ds.freq_array.flatten())
    expected = freqs[0:2]
    ds.select_spectral_windows([(0, 17), (3, 20)])
    assert units.allclose(ds.freq_array[0, 0:2], expected)
    ds.write(test_outfile)
    ds.select(frequencies=freqs[0:2], inplace=True)

    ds_in = DelaySpectrum()
    ds_in.read(test_outfile, frequencies=freqs[0:2])
    assert ds_in.check()
    assert units.allclose(ds_in.freq_array.flatten(), expected)
    assert ds_in == ds


def test_partial_read_frequencies_longest(ds_from_uvfits, test_outfile):
    """Test partial frequency read."""
    ds = ds_from_uvfits
    freqs = copy.deepcopy(ds.freq_array.flatten())
    expected = freqs[0:7]
    ds.select_spectral_windows([(0, 9), (10, 19)])
    assert units.allclose(ds.freq_array[0, 0:7], expected)
    ds.write(test_outfile)
    ds.select(frequencies=freqs[0:7], inplace=True)

    ds_in = DelaySpectrum()
    ds_in.read(test_outfile, frequencies=freqs[0:7])
    assert ds_in.check()
    assert units.allclose(ds_in.freq_array.flatten(), expected)
    assert ds_in == ds


@pytest.mark.parametrize(
    "select_kwargs",
    [
        {"delays": -405.999994798125 * units.ns},
        {"delay_chans": np.arange(8).tolist()},
        {"bls": [(0, 26), 69637]},
        {"lsts": [1.9681493255346292, 1.9712812654619116] * units.rad},
        {"lst_range": [2, 3] * units.rad},
        {"polarizations": 1},
    ],
)
@pytest.mark.filterwarnings(
    "ignore:This object has already been converted to a power spectrum"
)
@pytest.mark.filterwarnings(
    "ignore:Input baseline array is a mix of integers and tuples of integers"
)
@pytest.mark.filterwarnings(
    "ignore:Cannot write DelaySpectrum objects to file when power is in cosmological units"
)
def test_partial_reads(ds_from_uvfits, test_outfile, select_kwargs):
    """Test various partial reads."""
    ds = ds_from_uvfits
    ds.select_spectral_windows([(0, 9), (10, 19)])
    ds.calculate_delay_spectrum()
    ds.write(test_outfile)

    ds.select(**select_kwargs, inplace=True)

    ds_in = DelaySpectrum()
    ds_in.read(test_outfile, **select_kwargs)
    assert ds_in == ds
