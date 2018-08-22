"""Test PAPER miriad io."""
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


def test_read_no_calfile():
    """Test ValueError when no file given."""
    nt.assert_raises(ValueError, utils.import_calfile, None)


def test_read_blank_calfile_():
    """Test IOError when no file given exists."""
    nt.assert_raises(IOError, utils.import_calfile, '')


def test_calfile():
    """Test the input calfile is read has same attributes."""
    test_cal_dir = DATA_PATH
    test_cal_base = 'paper_cal'
    test_cal = os.path.join(test_cal_dir, test_cal_base)
    cal_out = utils.import_calfile(test_cal)
    aa_out = cal_out.get_aa(np.array([.1]))

    sys.path.append(test_cal_dir)
    exec('import {0} as test_cal'.format(test_cal_base))
    test_aa = test_cal.get_aa(np.array([.1]))
    nt.assert_dict_equal(test_aa.array_params, aa_out.array_params)


def test_calfile_with_py():
    """Test input calfile with appelation py is read has same attributes."""
    test_cal_dir = DATA_PATH
    test_cal_base = 'paper_cal'
    test_cal = os.path.join(test_cal_dir, test_cal_base)
    cal_out = utils.import_calfile(test_cal+'.py')
    aa_out = cal_out.get_aa(np.array([.1]))

    sys.path.append(test_cal_dir)
    exec('import {0} as test_cal'.format(test_cal_base))
    test_aa = test_cal.get_aa(np.array([.1]))
    nt.assert_dict_equal(test_aa.array_params, aa_out.array_params)


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
                     utils.read_paper_miriad, func_args=[None, None, None],
                     func_kwargs={'skip_header': 3, 'usecols': [1, 2, 3]},
                     category=UserWarning, nwarnings=4,
                     message=warn_message)


def test_load_uv_no_antpos():
    """Test an Exception is raised when no antpos or calfile is provided."""
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
                     func_args=[test_file, None, None],
                     func_kwargs={'skip_header': 3, 'usecols': [1, 2, 3]},
                     category=UserWarning, nwarnings=4,
                     message=warn_message)


def test_antpos_from_cal():
    """Test antpos and uvw from calfile."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_cal_dir = DATA_PATH
    test_cal_base = 'paper_cal'
    test_cal = os.path.join(test_cal_dir, test_cal_base)
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, test_cal, None],
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)
    # test_uv = utils.read_paper_miriad(test_miriad, calfile=test_cal)

    sys.path.append(test_cal_dir)
    exec('import {0} as test_cal'.format(test_cal_base))
    test_aa = test_cal.get_aa(np.array([.1]))
    test_uvws = np.zeros_like(test_uv.uvw_array)
    for bl in list(set(test_uv.baseline_array)):
        baseline_inds = np.where(test_uv.baseline_array == bl)[0]
        ant_1, ant_2 = test_uv.baseline_to_antnums(bl)
        uvw = test_aa.get_baseline(ant_1, ant_2, src='z')
        uvw *= const.c.to('m/ns').value
        test_uvws[baseline_inds, :] = uvw
    test_uvws = np.array(test_uvws)
    nt.assert_true(np.allclose(test_uvws, test_uv.uvw_array))


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
                     func_args=[test_miriad, None, dne_file],
                     category=UserWarning, nwarnings=4,
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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)
    # test_uv = utils.read_paper_miriad(test_miriad,
    #                                   antpos_file=test_antpos_file,
    #                                   skip_header=3, usecols=[1, 2, 3])
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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)
    frf_nebw_array = np.ones_like(test_uv.integration_time)
    frf_nebw_array *= test_uv.extra_keywords['FRF_NEBW']
    nt.assert_true(np.allclose(frf_nebw_array, test_uv.integration_time))


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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

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
    nt.assert_true(np.isclose(2, 1./utils.noise_equivalent_bandwidth(win),
                   rtol=1e-2))


def test_noise_equiv_bandwidth_boxcar():
    """Test that relative noise equivalent bandwidth is unity on boxcar."""
    win = windows.boxcar(2000)
    nt.assert_equal(1, 1./utils.noise_equivalent_bandwidth(win))


def test_cross_multiply_array_different_shapes():
    """Test Exception is raised if both arrays have different shapes."""
    array_1 = np.zeros((1, 2, 3))
    array_2 = np.zeros((2, 3, 4))
    axis = 2
    nt.assert_raises(ValueError, utils.cross_multipy_array,
                     array_1, array_2, axis)


def test_cross_multiply_shape():
    """Test that the shape of the array is correct size."""
    array_1 = np.ones((1, 3))
    axis = 1
    array_out = utils.cross_multipy_array(array_1, axis=1)
    nt.assert_equal((1, 3, 3), array_out.shape)


def test_cross_multiply_from_list():
    """Test that conversion to array occurs correctly from list."""
    array_1 = np.ones((1, 3)).tolist()
    axis = 1
    array_out = utils.cross_multipy_array(array_1, axis=1)
    nt.assert_equal((1, 3, 3), array_out.shape)


def test_cross_multiply_array_2_list():
    """Test array_2 behaves properly if originally a list."""
    array_1 = np.ones((1, 3))
    array_2 = np.ones((1, 3)).tolist()
    axis = 1
    array_out = utils.cross_multipy_array(array_1, array_2, axis=1)
    nt.assert_equal((1, 3, 3), array_out.shape)


def test_cross_multiply_quantity():
    """Test that cross mulitplying quantities behaves well."""
    array_1 = np.ones((1, 3)) * units.Hz
    axis = 1
    array_out = utils.cross_multipy_array(array_1, axis=1)
    nt.assert_equal((1, 3, 3), array_out.shape)


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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)
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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)
    test_uv_2 = copy.deepcopy(test_uv)
    ra_range = [0, 12]

    warn_message = ['Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']*2

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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']

    test_uv_2 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[[test_miriad, test_miriad_2],
                                                None,
                                                test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=UserWarning,
                                     nwarnings=len(warn_message),
                                     message=warn_message)
    ra_range = [0, 12]

    warn_message = ['Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']*2

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
                    'Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[[test_miriad, test_miriad_2],
                                              None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=len(warn_message),
                                   message=warn_message)
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']

    test_uv_2 = uvtest.checkWarnings(utils.read_paper_miriad,
                                     func_args=[test_miriad, None,
                                                test_antpos_file],
                                     func_kwargs={'skip_header': 3,
                                                  'usecols': [1, 2, 3]},
                                     category=UserWarning,
                                     nwarnings=4,
                                     message=warn_message)
    ra_range = [0, 12]

    warn_message = ['Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']*2

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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    inttime_array = utils.get_integration_time(test_uv, reds=baseline_array)
    test_shape = (test_uv.Nbls, test_uv.Ntimes, 1)
    nt.assert_equal(test_shape, inttime_array.shape)


def test_get_integration_time_shape_with_pol():
    """Test the shape of the integration_time array is correct with pols."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    inttime_array = utils.get_integration_time(test_uv, reds=baseline_array,
                                               squeeze=False)
    test_shape = (1, test_uv.Nbls, test_uv.Ntimes, 1)
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

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    inttime_array = utils.get_integration_time(test_uv, reds=baseline_array)
    test_shape = (test_uv.Nbls, test_uv.Ntimes, 1)
    test_array = test_uv.integration_time.copy()
    test_array = test_array.reshape(test_shape)
    nt.assert_true(np.allclose(test_array, inttime_array))


def test_get_integration_time_vals_with_pol():
    """Test the values of the integration_time array is correct with pols."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')
    warn_message = ['Antenna positions are not present in the file.',
                    'Antenna positions are not present in the file.',
                    'Ntimes does not match the number of unique '
                    'times in the data',
                    'Xantpos in extra_keywords is a list, array or dict, '
                    'which will raise an error when writing uvfits '
                    'or miriad file types']

    test_uv = uvtest.checkWarnings(utils.read_paper_miriad,
                                   func_args=[test_miriad, None,
                                              test_antpos_file],
                                   func_kwargs={'skip_header': 3,
                                                'usecols': [1, 2, 3]},
                                   category=UserWarning,
                                   nwarnings=4,
                                   message=warn_message)

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    inttime_array = utils.get_integration_time(test_uv, reds=baseline_array,
                                               squeeze=False)
    test_shape = (1, test_uv.Nbls, test_uv.Ntimes, 1)
    test_array = test_uv.integration_time.copy()
    test_array = test_array.reshape(test_shape)
    nt.assert_true(np.allclose(test_array, inttime_array))


def test_fold_along_delay_mismatched_sizes():
    """Test fold_along_delay errors if inputs are a different sizes."""
    delays = np.arange(20) * units.s
    array = np.ones((1, 10, 3)) * units.mK**2 * units.Mpc**3
    axis = 2
    nt.assert_raises(ValueError, utils.fold_along_delay,
                     array, delays, axis=axis)


def test_fold_along_delay_delays_no_zero_bin():
    """Test fold_along_delay errors if inputs are a different sizes."""
    delays = (np.arange(21) + 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    axis = -1
    nt.assert_raises(ValueError, utils.fold_along_delay,
                     array, delays, axis=axis)


def test_fold_along_delay_odd_length_ones_unchanged():
    """Test fold_along_delay returns all ones if  odd shaped input is ones."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays, axis=axis)
    nt.assert_true(np.allclose(np.ones((1, 10, 11)), array_out.value))


def test_fold_along_delay_odd_length_units_unchanged():
    """Test fold_along_delay returns the same unit as odd shaped input."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays, axis=axis)
    nt.assert_equal(units.mK**2 * units.Mpc**3, array_out.unit)


def test_fold_along_delay_even_length_ones_unchanged():
    """Test fold_along_delay returns all ones if even shaped input is ones."""
    delays = (np.arange(-10, 10) + .5) * units.s
    array = np.ones((1, 10, 20)) * units.mK**2 * units.Mpc**3
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays, axis=axis)
    nt.assert_true(np.allclose(np.ones((1, 10, 10)), array_out.value))


def test_fold_along_delay_even_length_units_unchanged():
    """Test fold_along_delay returns the same unit as the even shaped input."""
    delays = (np.arange(-10, 10) + .5) * units.s
    array = np.ones((1, 10, 20)) * units.mK**2 * units.Mpc**3
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays, axis=axis)
    nt.assert_equal(units.mK**2 * units.Mpc**3, array_out.unit)


def test_fold_along_delay_amplitude_check():
    """Test fold_along_delay returns correct amplitude during average."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    array[:, :, 11:] *= np.sqrt(2)
    array[:, :, 10] *= 3
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays, axis=axis)
    test_value_array = np.ones((1, 10, 11)) * np.mean([np.sqrt(2), 1])
    test_value_array[:, :, 0] = 3
    nt.assert_true(np.allclose(test_value_array, array_out.value))


def test_fold_along_delay_amplitude_check_with_weights():
    """Test fold_along_delay returns correct amplitude of weighted average."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    array[:, :, 11:] *= np.sqrt(2)
    array[:, :, 10] *= 3
    weights = np.ones_like(array)
    weights[:, :, 11:] *= 2
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays,
                                                    weights=weights, axis=axis)
    test_value_array = np.ones((1, 10, 11))
    test_value_array[:, :, 1:] *= np.average([np.sqrt(2), 1],
                                             weights=1./np.array([2., 1.])**2)
    test_value_array[:, :, 0] = 3
    nt.assert_true(np.allclose(test_value_array, array_out.value))


def test_fold_along_delay_weight_check():
    """Test fold_along_delay returns correct weighted weights."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * units.mK**2 * units.Mpc**3
    array[:, :, 11:] *= np.sqrt(2)
    array[:, :, 10] *= 3
    weights = np.ones_like(array)
    weights[:, :, 11:] *= 2
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays,
                                                    weights=weights, axis=axis)
    test_weight_array = np.ones((1, 10, 11))
    test_weight_array[:, :, 1:] *= np.sqrt(np.average(np.array([2., 1.])**2,
                                                      weights=1./np.array([2., 1.])**2))
    test_weight_array[:, :, 0] = 1
    nt.assert_true(np.allclose(test_weight_array, weights_out.value))


def test_fold_along_delay_amplitude_check_complex():
    """Test fold_along_delay returns expected complex values."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * (1 + 1j) * units.mK**2 * units.Mpc**3
    array[:, :, 11:].real *= np.sqrt(2)
    array[:, :, 10].real *= 5
    array[:, :, 11:].imag *= np.sqrt(3)
    array[:, :, 10].imag *= 6
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays, axis=axis)
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
    weights = np.ones_like(array).real
    weights[:, :, 11:] *= 2
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays,
                                                    weights=weights, axis=axis)
    test_weight_array = np.ones((1, 10, 11)).astype(np.complex)
    test_weight_array[:, :, 1:].real *= np.sqrt(np.average(np.array([2., 1.])**2, weights=1./np.array([2., 1.])**2))
    test_weight_array[:, :, 0] = 1 + 0j
    nt.assert_true(np.allclose(test_weight_array, weights_out.value))


def test_fold_along_delay_amplitude_check_mismatched_complex_weights():
    """Test fold_along_delay returns expected values if weights are complex with all zeros."""
    delays = np.arange(-10, 11) * units.s
    array = np.ones((1, 10, 21)) * (1 + 1j) * units.mK**2 * units.Mpc**3
    array[:, :, 11:].real *= np.sqrt(2)
    array[:, :, 10].real *= 5
    array[:, :, 11:].imag *= np.sqrt(3)
    array[:, :, 10].imag *= 6
    weights = np.ones_like(array).astype(np.complex)
    weights[:, :, 11:].real *= 2
    axis = -1
    array_out, weights_out = utils.fold_along_delay(array, delays,
                                                    weights=weights, axis=axis)
    test_weight_array = np.ones((1, 10, 11)).astype(np.complex)
    test_weight_array[:, :, 1:].real *= np.sqrt(np.average(np.array([2., 1.])**2, weights=1./np.array([2., 1.])**2))
    test_weight_array[:, :, 0] = 1 + 0j
    nt.assert_true(np.allclose(test_weight_array, weights_out.value))
