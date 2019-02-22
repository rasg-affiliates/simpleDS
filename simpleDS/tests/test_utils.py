# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Matthew Kolopanis
# Licensed under the 3-clause BSD License
"""Test utils."""
from __future__ import print_function

import os
import sys
import copy
import numpy as np
import nose.tools as nt
from scipy.signal import windows
import pyuvdata
from pyuvdata import UVData, utils as uvutils
import pyuvdata.tests as uvtest
from simpleDS import utils
from simpleDS.data import DATA_PATH
from astropy import constants as const
from astropy import units


def test_read_no_file():
    """Test no file given."""
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']

    nt.assert_raises(TypeError, uvtest.checkWarnings,
                     utils.read_paper_miriad, func_args=[None, None],
                     func_kwargs={'skip_header': 3, 'usecols': [1, 2, 3]},
                     category=UserWarning,
                     nwarnings=len(warn_message),
                     message=warn_message)


def test_load_uv_no_antpos():
    """Test an Exception is raised when no antpos is provided."""
    test_file = os.path.join(DATA_PATH, 'paper_test_file.uv')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']

    nt.assert_raises(ValueError, uvtest.checkWarnings,
                     utils.read_paper_miriad,
                     func_args=[test_file, None],
                     func_kwargs={'skip_header': 3, 'usecols': [1, 2, 3]},
                     category=UserWarning,
                     nwarnings=len(warn_message),
                     message=warn_message)


def test_no_antpos_file():
    """Test an Exception is raised if antpos file does not exist."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    dne_file = os.path.join(DATA_PATH, 'not_real_file.txt')

    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']

    nt.assert_raises(IOError, uvtest.checkWarnings,
                     utils.read_paper_miriad,
                     func_args=[test_miriad, dne_file],
                     category=UserWarning,
                     nwarnings=len(warn_message),
                     message=warn_message)


def test_antpos_from_file():
    """Test antpos created from input file."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    read_antpos = np.genfromtxt(test_antpos_file, skip_header=3,
                                usecols=[1, 2, 3])

    test_uvws = np.zeros_like(test_uv.uvw_array)
    for bl in list(set(test_uv.baseline_array)):
        baseline_inds = np.where(test_uv.baseline_array == bl)[0]
        ant_1, ant_2 = test_uv.baseline_to_antnums(bl)
        uvw = read_antpos[ant_2] - read_antpos[ant_1]
        test_uvws[baseline_inds, :] = uvw

    nt.assert_true(np.allclose(test_uvws, test_uv.uvw_array))


def test_setting_frf_nebw_as_inttime():
    """Test integration time is set if FRF_NEBW is in extra_keywords."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)
    frf_nebw_array = np.ones_like(test_uv.integration_time)
    frf_nebw_array *= test_uv.extra_keywords['FRF_NEBW']
    nt.assert_true(np.allclose(frf_nebw_array, test_uv.integration_time))


def test_paper_test_file_requires_bl_conjugation():
    """Test the input paper test file has baselines which require conjugation."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
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

    uv1 = UVData()
    _ = uvtest.checkWarnings(uv1.read_miriad, func_args=[test_miriad],
                             category=[UserWarning] * len(warn_message)
                             + [PendingDeprecationWarning],
                             nwarnings=len(warn_message) + 1,
                             message=warn_message + pend_dep_message)
    nt.assert_true(np.logical_not(np.all(uv1.uvw_array[:, 0] > 0)))


def test_conjugating_baselines_in_paper_read():
    """Test that baslines are properly cojugated when read with utils."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)
    nt.assert_true(np.all(test_uv.uvw_array[:, 0] > 0))


def test_get_data_array():
    """Test data is stored into the array the same."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    data_array = utils.get_data_array(test_uv, reds=baseline_array)

    compare_data = np.zeros((test_uv.Npols, test_uv.Nbls,
                             test_uv.Ntimes, test_uv.Nfreqs), dtype=np.complex)

    pol_array = uvutils.polnum2str(test_uv.polarization_array)
    for pol_cnt, pol in enumerate(pol_array):
        for cnt, baseline in enumerate(list(set(test_uv.baseline_array))):
            ant_1, ant_2 = test_uv.baseline_to_antnums(baseline)
            compare_data[pol_cnt, cnt] = test_uv.get_data(ant_1, ant_2, pol)

    compare_data = compare_data.squeeze(axis=0)
    nt.assert_true(np.allclose(compare_data, data_array))


def test_get_data_no_squeeze():
    """Test data is stored into the array the same with no squeezing."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    data_array = utils.get_data_array(test_uv, reds=baseline_array,
                                      squeeze=False)

    compare_data = np.zeros((test_uv.Npols, test_uv.Nbls,
                             test_uv.Ntimes, test_uv.Nfreqs), dtype=np.complex)

    pol_array = uvutils.polnum2str(test_uv.polarization_array)
    for pol_cnt, pol in enumerate(pol_array):
        for cnt, baseline in enumerate(list(set(test_uv.baseline_array))):
            ant_1, ant_2 = test_uv.baseline_to_antnums(baseline)
            compare_data[pol_cnt, cnt] = test_uv.get_data(ant_1, ant_2, pol)

    nt.assert_true(np.allclose(compare_data, data_array))


def test_get_nsamples_array():
    """Test nsamples is returned the same."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    nsample_array = utils.get_nsample_array(test_uv, reds=baseline_array)

    test_samples = np.zeros((test_uv.Npols, test_uv.Nbls,
                             test_uv.Ntimes, test_uv.Nfreqs), dtype=np.float)

    pol_array = uvutils.polnum2str(test_uv.polarization_array)
    for pol_cnt, pol in enumerate(pol_array):
        for cnt, baseline in enumerate(list(set(test_uv.baseline_array))):
            ant_1, ant_2 = test_uv.baseline_to_antnums(baseline)
            test_samples[pol_cnt, cnt] = test_uv.get_nsamples(ant_1, ant_2)

    test_samples = np.squeeze(test_samples, axis=0)
    nt.assert_true(np.all(test_samples == nsample_array))


def test_get_nsamples_no_squeeze():
    """Test nsamples is returned the same with no squeeze."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    nsample_array = utils.get_nsample_array(test_uv, reds=baseline_array,
                                            squeeze=False)

    test_samples = np.zeros((test_uv.Npols, test_uv.Nbls,
                             test_uv.Ntimes, test_uv.Nfreqs), dtype=np.float)

    pol_array = uvutils.polnum2str(test_uv.polarization_array)
    for pol_cnt, pol in enumerate(pol_array):
        for cnt, baseline in enumerate(list(set(test_uv.baseline_array))):
            ant_1, ant_2 = test_uv.baseline_to_antnums(baseline)
            test_samples[pol_cnt, cnt] = test_uv.get_nsamples(ant_1, ant_2)

    nt.assert_true(np.all(test_samples == nsample_array))


def test_get_flag_array():
    """Test nsamples is returned the same."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    flag_array = utils.get_flag_array(test_uv, reds=baseline_array)

    test_flags = np.zeros((test_uv.Npols, test_uv.Nbls,
                           test_uv.Ntimes, test_uv.Nfreqs), dtype=np.float)

    pol_array = uvutils.polnum2str(test_uv.polarization_array)
    for pol_cnt, pol in enumerate(pol_array):
        for cnt, baseline in enumerate(list(set(test_uv.baseline_array))):
            ant_1, ant_2 = test_uv.baseline_to_antnums(baseline)
            test_flags[pol_cnt, cnt] = test_uv.get_flags(ant_1, ant_2)

    test_flags = np.squeeze(test_flags, axis=0)
    nt.assert_true(np.all(test_flags == flag_array))


def test_get_flag_array_no_squeeze():
    """Test nsamples is returned the same with no squeeze."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    flag_array = utils.get_flag_array(test_uv, reds=baseline_array,
                                      squeeze=False)

    test_flags = np.zeros((test_uv.Npols, test_uv.Nbls,
                           test_uv.Ntimes, test_uv.Nfreqs), dtype=np.float)

    pol_array = uvutils.polnum2str(test_uv.polarization_array)
    for pol_cnt, pol in enumerate(pol_array):
        for cnt, baseline in enumerate(list(set(test_uv.baseline_array))):
            ant_1, ant_2 = test_uv.baseline_to_antnums(baseline)
            test_flags[pol_cnt, cnt] = test_uv.get_flags(ant_1, ant_2)

    nt.assert_true(np.all(test_flags == flag_array))


def test_bootstrap_array_invalid_axis():
    """Test Exception is raised if axis is larger than size of array."""
    test_array = np.zeros((3, 4))
    test_axis = 2
    nboot = 5
    nt.assert_raises(ValueError, utils.bootstrap_array, test_array,
                     nboot=nboot, axis=test_axis)


def test_bootstrap_array_shape():
    """Test returned array is the correct shape."""
    test_array = np.zeros((3, 4))
    test_axis = 1
    nboot = 5
    new_array = utils.bootstrap_array(test_array, nboot=nboot, axis=test_axis)
    shape = (3, 4, 5)
    nt.assert_equal(shape, new_array.shape)


def test_noise_equiv_bandwidth():
    """Test that relative noise equivalent bandwidth calculation converges."""
    win = windows.blackmanharris(2000)
    nt.assert_true(np.isclose(2, 1. / utils.noise_equivalent_bandwidth(win),
                   rtol=1e-2))


def test_noise_equiv_bandwidth_boxcar():
    """Test that relative noise equivalent bandwidth is unity on boxcar."""
    win = windows.boxcar(2000)
    nt.assert_equal(1, 1. / utils.noise_equivalent_bandwidth(win))


def test_cross_multiply_array_different_shapes():
    """Test Exception is raised if both arrays have different shapes."""
    array_1 = np.zeros((1, 2, 3))
    array_2 = np.zeros((2, 3, 4))
    axis = 2
    nt.assert_raises(ValueError, utils.cross_multiply_array,
                     array_1, array_2, axis)


def test_cross_multiply_shape():
    """Test that the shape of the array is correct size."""
    array_1 = np.ones((1, 3))
    axis = 1
    array_out = utils.cross_multiply_array(array_1, axis=1)
    nt.assert_equal((1, 3, 3), array_out.shape)


def test_cross_multiply_from_list():
    """Test that conversion to array occurs correctly from list."""
    array_1 = np.ones((1, 3)).tolist()
    axis = 1
    array_out = utils.cross_multiply_array(array_1, axis=1)
    nt.assert_equal((1, 3, 3), array_out.shape)


def test_cross_multiply_array_2_list():
    """Test array_2 behaves properly if originally a list."""
    array_1 = np.ones((1, 3))
    array_2 = np.ones((1, 3)).tolist()
    axis = 1
    array_out = utils.cross_multiply_array(array_1, array_2, axis=1)
    nt.assert_equal((1, 3, 3), array_out.shape)


def test_cross_multiply_quantity():
    """Test that cross mulitplying quantities behaves well."""
    array_1 = np.ones((1, 3)) * units.Hz
    axis = 1
    array_out = utils.cross_multiply_array(array_1, axis=1)
    nt.assert_equal((1, 3, 3), array_out.shape)


def test_cross_multiply_quantity_units():
    """Test that cross mulitplying quantities have the right units."""
    array_1 = np.ones((1, 3)) * units.Hz
    axis = 1
    array_out = utils.cross_multiply_array(array_1, axis=1)
    nt.assert_equal(units.Hz**2, array_out.unit)


def test_noise_shape():
    """Test shape of generate_noise matches nsample array."""
    test_sample = np.ones((2, 13, 21)) * 3
    test_noise = utils.generate_noise(test_sample)
    nt.assert_equal(test_sample.shape, test_noise.shape)


def test_noise_amplitude():
    """Ensure noise amplitude is reasonable within 1 percent."""
    rtol = 1e-2
    test_sample = np.ones((100, 1000)) * 3
    test_noise = utils.generate_noise(test_sample)
    noise_power = test_noise.std(1)
    noise_power_uncertainty = noise_power.std()
    nt.assert_true(np.isclose(test_noise.std(), 3,
                              atol=noise_power_uncertainty))


def test_align_lst_error():
    """Test lst_align enforces same sampling rate."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv)
    test_uv_2.time_array = 2 * test_uv_2.time_array

    nt.assert_raises(ValueError, utils.lst_align, test_uv, test_uv_2,
                     ra_range=[0, 24])


def test_align_lst_shapes_equal():
    """Test the shape of the time_arrays are equal after lst_align."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)
    test_uv_2 = copy.deepcopy(test_uv)
    ra_range = [0, 12]

    warn_message = ['Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types'] * 2

    test_uv_out, test_uv_2_out = uvtest.checkWarnings(utils.lst_align,
                                                      func_args=[test_uv, test_uv_2],
                                                      func_kwargs={'ra_range': ra_range,
                                                                   'inplace': False},
                                                      category=UserWarning,
                                                      nwarnings=len(warn_message),
                                                      message=warn_message)
    nt.assert_equal(test_uv_out.time_array.shape, test_uv_out.time_array.shape)


def test_align_lst_shapes_equal_uv_2_longer():
    """Test shape of time_array are equal after lst_align: 2nd uv longer."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_miriad_2 = os.path.join(DATA_PATH, 'paper_test_file_2nd_time.uv')
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
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.',
                    'Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.']
    warn_category = [UserWarning, UserWarning, UserWarning, UserWarning,
                     PendingDeprecationWarning] * 2
    warn_category += [UserWarning, PendingDeprecationWarning] * 4
    test_uv_2 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[[test_miriad, test_miriad_2],
                                                test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=warn_category,
                                     nwarnings=len(warn_message),
                                     message=warn_message)
    ra_range = [0, 12]

    warn_message = ['Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types'] * 2

    test_uv_out, test_uv_2_out = uvtest.checkWarnings(utils.lst_align,
                                                      func_args=[test_uv, test_uv_2],
                                                      func_kwargs={'ra_range': ra_range,
                                                                   'inplace': False},
                                                      category=UserWarning,
                                                      nwarnings=len(warn_message),
                                                      message=warn_message)

    nt.assert_equal(test_uv_out.time_array.shape, test_uv_out.time_array.shape)


def test_align_lst_shapes_equal_uv_1_longer():
    """Test shape of time_array are equal after lst_align: 1st uv longer."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_miriad_2 = os.path.join(DATA_PATH, 'paper_test_file_2nd_time.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.',
                    'Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'antenna_positions are not defined. '
                    'antenna_positions will be a required parameter in '
                    'future versions.']
    warn_category = [UserWarning, UserWarning, UserWarning, UserWarning,
                     PendingDeprecationWarning] * 2
    warn_category += [UserWarning, PendingDeprecationWarning] * 4
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[[test_miriad, test_miriad_2],
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=warn_category,
                                   nwarnings=len(warn_message),
                                   message=warn_message)
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
    test_uv_2 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=[UserWarning] * len(warn_message)
                                     + [PendingDeprecationWarning],
                                     nwarnings=len(warn_message) + 1,
                                     message=warn_message + pend_dep_message)
    ra_range = [0, 12]

    warn_message = ['Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types'] * 2

    test_uv_out, test_uv_2_out = uvtest.checkWarnings(utils.lst_align,
                                                      func_args=[test_uv, test_uv_2],
                                                      func_kwargs={'ra_range': ra_range,
                                                                   'inplace': False},
                                                      category=UserWarning,
                                                      nwarnings=len(warn_message),
                                                      message=warn_message)
    nt.assert_equal(test_uv_out.time_array.shape, test_uv_out.time_array.shape)


def test_get_integration_time_shape():
    """Test the shape of the integration_time array is correct."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    inttime_array = utils.get_integration_time(test_uv, reds=baseline_array)
    test_shape = (test_uv.Nbls, test_uv.Ntimes)
    nt.assert_equal(test_shape, inttime_array.shape)


def test_get_integration_time_vals():
    """Test the values of the integration_time array is correct."""
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
                        'future versions.']
    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=[UserWarning] * len(warn_message)
                                   + [PendingDeprecationWarning],
                                   nwarnings=len(warn_message) + 1,
                                   message=warn_message + pend_dep_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    inttime_array = utils.get_integration_time(test_uv, reds=baseline_array)
    test_shape = (test_uv.Nbls, test_uv.Ntimes)
    test_array = test_uv.integration_time.copy()
    test_array = test_array.reshape(test_shape)
    nt.assert_true(np.allclose(test_array, inttime_array))


def test_jy_to_mk_value():
    """Test the Jy to mK conversion factor."""
    test_fq = np.array([.1]) * units.GHz
    jy_to_mk = utils.jy_to_mk(test_fq)
    test_conversion = const.c**2 / (2 * test_fq.to('1/s')**2 * const.k_B)
    test_conversion = test_conversion.to('mK/Jy')
    nt.assert_true(np.allclose(test_conversion.value, jy_to_mk.value))


def test_jy_to_mk_units():
    """Test the Jy to mK conversion factor."""
    test_fq = np.array([.1]) * units.GHz
    jy_to_mk = utils.jy_to_mk(test_fq)
    test_conversion = const.c**2 / (2 * test_fq.to('1/s')**2 * const.k_B) * units.sr
    test_conversion = test_conversion.to('mK*sr/Jy')
    nt.assert_equal(test_conversion.unit.to_string(),
                    jy_to_mk.unit.to_string())


def test_jy_to_mk_freq_unitless():
    """Test the Jy to mK conversion factor."""
    test_fq = np.array([.1])
    nt.assert_raises(TypeError, utils.jy_to_mk, test_fq)


def test_jy_to_mk_freq_wrong_units():
    """Test the Jy to mK conversion factor."""
    test_fq = np.array([.1]) * units.m
    nt.assert_raises(units.UnitsError, utils.jy_to_mk, test_fq)


def test_normalized_fourier_transform():
    """Test the delay transform and cross-multiplication function."""
    fake_data = np.zeros((1, 13, 21))
    fake_data[0, 7, 11] += 1
    fake_corr = utils.normalized_fourier_transform(fake_data,
                                                   1 * units.dimensionless_unscaled,
                                                   taper=windows.boxcar,
                                                   axis=2)
    test_corr = np.fft.fft(fake_data, axis=-1)
    test_corr = np.fft.fftshift(test_corr, axes=-1)
    fake_corr = fake_corr.value
    nt.assert_true(np.allclose(test_corr, fake_corr))


def test_ft_with_pols():
    """Test fourier transform is correct shape when pols are present."""
    fake_data = np.zeros((3, 2, 13, 31))
    fake_data[:, 0, 7, 11] += 1.
    fake_corr = utils.normalized_fourier_transform(fake_data,
                                                   1 * units.dimensionless_unscaled,
                                                   taper=windows.boxcar,
                                                   axis=3)
    nt.assert_equal((3, 2, 13, 31), fake_corr.shape)


def test_delay_vals_with_pols():
    """Test values in normalized_fourier_transform when pols present."""
    fake_data = np.zeros((3, 2, 13, 31))
    fake_data[:, 0, 7, 11] += 1.
    fake_corr = utils.normalized_fourier_transform(fake_data,
                                                   1 * units.dimensionless_unscaled,
                                                   taper=windows.boxcar,
                                                   axis=3)
    test_corr = np.fft.fft(fake_data, axis=-1)
    test_corr = np.fft.fftshift(test_corr, axes=-1)
    fake_corr = fake_corr.value
    nt.assert_true(np.allclose(test_corr, fake_corr))


def test_units_normalized_fourier_transform():
    """Test units are returned from normalized_fourier_transform."""
    fake_data = np.zeros((1, 13, 21)) * units.m
    fake_data[0, 7, 11] += 1 * units.m
    fake_corr = utils.normalized_fourier_transform(fake_data,
                                                   1 * units.Hz,
                                                   taper=windows.boxcar,
                                                   axis=2)
    test_units = units.m * units.Hz
    nt.assert_equal(test_units, fake_corr.unit)


def test_units_normalized_inverse_fourier_transform():
    """Test units are returned from normalized_fourier_transform."""
    fake_data = np.zeros((1, 13, 21)) * units.Jy * units.Hz
    fake_data[0, 7, 11] += 1 * units.Jy * units.Hz
    fake_corr = utils.normalized_fourier_transform(fake_data,
                                                   1 * units.s,
                                                   taper=windows.boxcar,
                                                   axis=2, inverse=True)
    test_units = units.Jy
    nt.assert_equal(test_units, fake_corr.unit)


def test_delta_x_unitless():
    """Test delta_x is unitless raises exception."""
    fake_data = np.zeros((1, 13, 21)) * units.m
    fake_data[0, 7, 11] += 1 * units.m
    nt.assert_raises(ValueError, utils.normalized_fourier_transform, fake_data,
                     delta_x=2., axis=2)


def test_combine_nsamples_different_shapes():
    """Test an error is raised if nsample_arrays have different shapes."""
    test_sample_1 = np.ones((2, 13, 21))
    test_sample_2 = np.ones((3, 13, 21))
    nt.assert_raises(ValueError, utils.combine_nsamples,
                     test_sample_1, test_sample_2)


def test_combine_nsamples_one_array():
    """Test that if only one array is given the samples are the same."""
    test_samples = np.ones((2, 13, 21)) * 3
    samples_out = utils.combine_nsamples(test_samples, axis=0)
    test_full_samples = np.ones((2, 2, 13, 21)) * 3
    nt.assert_true(np.allclose(test_full_samples, samples_out))


def test_combine_nsamples_with_pols():
    """Test that if only one array is given the samples are the same."""
    test_samples_1 = np.ones((3, 2, 13, 21)) * 3
    test_samples_2 = np.ones((3, 2, 13, 21)) * 2
    samples_out = utils.combine_nsamples(test_samples_1, test_samples_2, axis=1)
    test_full_samples = np.ones((3, 2, 2, 13, 21)) * np.sqrt(6)
    nt.assert_true(np.all(test_full_samples == samples_out))


def test_remove_autos_bad_axis_shape():
    """Test error raised if axes keyword is bad shape."""
    test_array = np.ones((3, 3, 11, 21))
    axes = (1, 2, 3)
    nt.assert_raises(ValueError, utils.remove_auto_correlations, test_array,
                     axes=axes)


def test_remove_autos_axes_different_shapes():
    """Test error raised if axes have different shapes."""
    test_array = np.ones((3, 5, 11, 21))
    axes = (0, 1)
    nt.assert_raises(ValueError, utils.remove_auto_correlations, test_array,
                     axes=axes)


def test_remove_autos_axes_not_adjacent():
    """Test error raised if axes are not adjacent."""
    test_array = np.ones((3, 3, 11, 21))
    axes = (0, 2)
    nt.assert_raises(ValueError, utils.remove_auto_correlations, test_array,
                     axes=axes)


def test_remove_autos():
    """Test that the remove auto_correlations function returns right shape."""
    test_array = np.ones((3, 3, 11, 21))
    out_array = utils.remove_auto_correlations(test_array, axes=(0, 1))
    nt.assert_equal((6, 11, 21), out_array.shape)


def test_remove_middle_axis():
    """Test that the remove autos function returns right shape axes in middle of array."""
    test_array = np.ones((13, 17, 19, 3, 3, 11, 21))
    out_array = utils.remove_auto_correlations(test_array, axes=(3, 4))
    nt.assert_equal((13, 17, 19, 6, 11, 21), out_array.shape)


def test_remove_autos_with_pols():
    """Test remove auto_correlations function returns right shape with pols."""
    test_array = np.ones((4, 3, 3, 11, 21))
    out_array = utils.remove_auto_correlations(test_array, axes=(1, 2))
    nt.assert_equal((4, 6, 11, 21), out_array.shape)


def test_weighted_average_unit_mismatch():
    """Test error is raised if array and uncertainty have incompatible units."""
    test_array = np.ones((3, 3, 2)) * units.m
    test_error = np.ones_like(test_array.value) * units.Hz
    nt.assert_raises(units.UnitConversionError, utils.weighted_average,
                     test_array, test_error)


def test_weighted_average_array_not_quantity_error_is_quantity():
    """Test error is raised if array and uncertainty have incompatible units."""
    test_array = np.ones((3, 3, 2))
    test_error = np.ones_like(test_array) * units.m
    nt.assert_raises(ValueError, utils.weighted_average, test_array, test_error)


def test_weighted_average_array_is_quantity_error_not_quantity():
    """Test error is raised if array and uncertainty have incompatible units."""
    test_array = np.ones((3, 3, 2)) * units.m
    test_error = np.ones_like(test_array.value)
    nt.assert_raises(ValueError, utils.weighted_average, test_array, test_error)


def test_weighted_average_shape_mismatch():
    """Test error is raised is shape of array and uncertainty differ."""
    test_array = np.ones((3, 3, 2)) * units.m
    test_error = np.ones((4, 3, 2)) * units.m
    nt.assert_raises(ValueError, utils.weighted_average, test_array, test_error)


def test_weighted_average_weights_bad_shape_one_dimensional():
    """Test error is raised if the weights have a bad shape if in 1-D."""
    test_array = np.ones((3, 3, 2)) * units.m
    test_error = np.ones((3, 3, 2)) * units.m
    weights = np.ones(4)
    nt.assert_raises(ValueError, utils.weighted_average, test_array,
                     test_error, weights=weights)


def test_weighted_average_weights_bad_shape():
    """Test error is raised if the weights have a bad shape."""
    test_array = np.ones((3, 3, 2)) * units.m
    test_error = np.ones((3, 3, 2)) * units.m
    weights = np.ones((5, 3, 2))
    nt.assert_raises(ValueError, utils.weighted_average, test_array,
                     test_error, weights=weights)


def test_weighted_average_uniform_weights_value():
    """Test when weights are one a uniform average is performed."""
    test_array = np.arange(20).reshape(4, 5) * units.m
    test_error = np.ones(20).reshape(4, 5) * units.m
    weights = np.ones_like(test_array.value)
    array_out, error_out = utils.weighted_average(test_array, test_error,
                                                  weights=weights, axis=0)
    nt.assert_true(np.array_equal(np.array([7.5, 8.5, 9.5, 10.5, 11.5]), array_out.value))
    nt.assert_equal(units.m, array_out.unit)
    print(error_out**2)
    nt.assert_true(np.array_equal(np.array([1, 1, 1, 1, 1]) / np.sqrt(test_array.shape[0]),
                                  error_out.value))
    nt.assert_equal(units.m, error_out.unit)


def test_weighted_average_inverse_variance_weights():
    """Test the output of weighted average with inverse variance weights."""
    test_array = np.array([1, 3, 10])
    test_error = np.array([1, 3, 10])
    array_out, error_out = utils.weighted_average(test_array, test_error)
    print(error_out)
    nt.assert_true(np.isclose(array_out, 1.2784935579781962))
    nt.assert_true(np.isclose(error_out, 0.9444428250308379))


def test_fold_along_delay_mismatched_sizes():
    """Test fold_along_delay errors if inputs are a different sizes."""
    delays = np.arange(20) * units.s
    array = np.ones((1, 10, 3)) * units.mK**2 * units.Mpc**3
    errs = np.ones((1, 10, 3)) * units.mK**2 * units.Mpc**3
    axis = 2
    nt.assert_raises(ValueError, utils.fold_along_delay,
                     delays, array, errs, axis=axis)


def test_fold_along_delay_mismatched_uncertainty_shape():
    """Test fold_along_delay errors if inputs are a different sizes."""
    delays = np.arange(20) * units.s
    array = np.ones((1, 10, 20)) * units.mK**2 * units.Mpc**3
    errs = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    axis = 2
    nt.assert_raises(ValueError, utils.fold_along_delay,
                     delays, array, errs, axis=axis)


def test_fold_along_delay_delays_no_zero_bin():
    """Test fold_along_delay errors if inputs are a different sizes."""
    delays = (np.arange(21) + 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    errs = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    axis = -1
    nt.assert_raises(ValueError, utils.fold_along_delay,
                     delays, array, errs, axis=axis)


def test_fold_along_delay_odd_length_ones_unchanged():
    """Test fold_along_delay returns all ones if  odd shaped input is ones."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    errs = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    axis = -1
    array_out, errs_out = utils.fold_along_delay(delays, array, errs, axis=axis)
    nt.assert_true(np.allclose(np.ones((1, 10, 11)), array_out.value))


def test_fold_along_delay_odd_length_units_unchanged():
    """Test fold_along_delay returns the same unit as odd shaped input."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    errs = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    axis = -1
    array_out, errs_out = utils.fold_along_delay(delays, array, errs, axis=axis)
    nt.assert_equal(units.mK**2 * units.Mpc**3, array_out.unit)


def test_fold_along_delay_even_length_ones_unchanged():
    """Test fold_along_delay returns all ones if even shaped input is ones."""
    delays = (np.arange(-10, 10) + .5) * units.s
    array = np.ones((1, 10, 20)) * units.mK**2 * units.Mpc**3
    errs = np.ones((1, 10, 20)) * units.mK**2 * units.Mpc**3
    axis = -1
    array_out, errs_out = utils.fold_along_delay(delays, array, errs, axis=axis)
    nt.assert_true(np.allclose(np.ones((1, 10, 10)), array_out.value))


def test_fold_along_delay_even_length_units_unchanged():
    """Test fold_along_delay returns the same unit as the even shaped input."""
    delays = (np.arange(-10, 10) + .5) * units.s
    array = np.ones((1, 10, 20)) * units.mK**2 * units.Mpc**3
    errs = np.ones((1, 10, 20)) * units.mK**2 * units.Mpc**3
    axis = -1
    array_out, errs_out = utils.fold_along_delay(delays, array, errs, axis=axis)
    nt.assert_equal(units.mK**2 * units.Mpc**3, array_out.unit)


def test_fold_along_delay_amplitude_check():
    """Test fold_along_delay returns correct amplitude during average."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    array[:, :, 11:] *= np.sqrt(2)
    array[:, :, 10] *= 3
    axis = -1
    errs = np.ones_like(array)
    array_out, errs_out = utils.fold_along_delay(delays, array, errs, axis=axis)
    test_value_array = np.ones((1, 10, 11)) * np.mean([np.sqrt(2), 1])
    test_value_array[:, :, 0] = 3
    nt.assert_true(np.allclose(test_value_array, array_out.value))


def test_fold_along_delay_amplitude_check_with_weights():
    """Test fold_along_delay returns correct amplitude of weighted average."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    array[:, :, 11:] *= np.sqrt(2)
    array[:, :, 10] *= 3
    errs = np.ones_like(array)
    errs[:, :, 11:] *= 2
    axis = -1
    array_out, errs_out = utils.fold_along_delay(delays, array, errs,
                                                 weights=1. / errs**2, axis=axis)
    test_value_array = np.ones((1, 10, 11))
    test_value_array[:, :, 1:] *= np.average([np.sqrt(2), 1],
                                             weights=1. / np.array([2., 1.])**2)
    test_value_array[:, :, 0] = 3
    nt.assert_true(np.allclose(test_value_array, array_out.value))


def test_fold_along_delay_weight_check():
    """Test fold_along_delay returns correct weighted weights."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    array[:, :, 11:] *= np.sqrt(2)
    array[:, :, 10] *= 3
    errs = np.ones_like(array)
    errs[:, :, 11:] *= 2
    axis = -1
    array_out, errs_out = utils.fold_along_delay(delays, array, errs,
                                                 weights=1. / errs**2, axis=axis)
    test_errs_array = np.ones((1, 10, 11))
    test_errs_array[:, :, 1:] *= np.sqrt(1. / np.sum(1. / np.array([2., 1.])**2))
    test_errs_array[:, :, 0] = 1
    # print('known:', test_weight_array)
    # print('calc', weights_out)
    nt.assert_true(np.allclose(test_errs_array, errs_out.value))


def test_fold_along_delay_amplitude_check_complex():
    """Test fold_along_delay returns expected complex values."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * (1 + 1j) * units.mK**2 * units.Mpc**3
    array[:, :, 11:].real *= np.sqrt(2)
    array[:, :, 10].real *= 5
    array[:, :, 11:].imag *= np.sqrt(3)
    array[:, :, 10].imag *= 6
    axis = -1
    errs = np.ones_like(array)
    array_out, errs_out = utils.fold_along_delay(delays, array, errs, axis=axis)
    test_value_array = np.ones((1, 10, 11)).astype(np.complex)
    test_value_array[:, :, 1:] *= (np.mean([np.sqrt(2), 1])
                                   + 1j * np.mean([np.sqrt(3), 1]))
    test_value_array[:, :, 0] = 5 + 6j
    nt.assert_true(np.allclose(test_value_array, array_out.value))


def test_fold_along_delay_amplitude_check_complex_mismatched_weights():
    """Test fold_along_delay returns expected values if weights are not complex."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * (1 + 1j) * units.mK**2 * units.Mpc**3
    array[:, :, 11:].real *= np.sqrt(2)
    array[:, :, 10].real *= 5
    array[:, :, 11:].imag *= np.sqrt(3)
    array[:, :, 10].imag *= 6
    errs = np.ones_like(array).real
    errs[:, :, 11:] *= 2
    axis = -1
    array_out, errs_out = utils.fold_along_delay(delays, array, errs,
                                                 weights=1. / errs**2, axis=axis)
    test_errs_array = np.ones((1, 10, 11)).astype(np.complex)
    test_errs_array[:, :, 1:].real *= 1. / np.sqrt(np.sum(1. / np.array([2., 1.])**2))
    test_errs_array[:, :, 0] = 1 + 0j
    nt.assert_true(np.allclose(test_errs_array, errs_out.value))


def test_fold_along_delay_amplitude_check_mismatched_complex_weights():
    """Test fold_along_delay returns expected values if weights are complex with all zeros."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * (1 + 1j) * units.mK**2 * units.Mpc**3
    array[:, :, 11:].real *= np.sqrt(2)
    array[:, :, 10].real *= 5
    array[:, :, 11:].imag *= np.sqrt(3)
    array[:, :, 10].imag *= 6
    errs = np.ones_like(array).astype(np.complex)
    errs[:, :, 11:].real *= 2
    axis = -1
    array_out, errs_out = utils.fold_along_delay(delays, array, errs,
                                                 weights=1. / errs**2, axis=axis)
    test_errs_array = np.ones((1, 10, 11)).astype(np.complex)
    test_errs_array[:, :, 1:].real *= 1. / np.sqrt(np.sum(1. / np.array([2., 1.])**2))
    test_errs_array[:, :, 0] = 1 + 0j
    nt.assert_true(np.allclose(test_errs_array, errs_out.value))
