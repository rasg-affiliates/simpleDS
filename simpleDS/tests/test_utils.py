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
    nt.assert_raises(TypeError, utils.read_paper_miriad, None)


def test_load_uv_no_antpos():
    """Test an Exception is raised when no antpos or calfile is provided."""
    test_file = os.path.join(DATA_PATH, 'paper_test_file.uv')
    nt.assert_raises(ValueError, utils.read_paper_miriad, test_file)


def test_antpos_from_cal():
    """Test antpos and uvw from calfile."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_cal_dir = DATA_PATH
    test_cal_base = 'paper_cal'
    test_cal = os.path.join(test_cal_dir, test_cal_base)
    test_uv = utils.read_paper_miriad(test_miriad, calfile=test_cal)

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
    nt.assert_raises(IOError, utils.read_paper_miriad, test_miriad,
                     antpos_file=dne_file)


def test_antpos_from_file():
    """Test antpos created from input file."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])
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

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])
    frf_nebw_array = np.ones_like(test_uv.integration_time)
    frf_nebw_array *= test_uv.extra_keywords['FRF_NEBW']
    nt.assert_true(np.allclose(frf_nebw_array, test_uv.integration_time))


def test_get_data_array():
    """Test data is stored into the array the same."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

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

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

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

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

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

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

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

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

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

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

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

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv)
    test_uv_2.time_array = 2 * test_uv_2.time_array

    nt.assert_raises(ValueError, utils.lst_align, test_uv, test_uv_2,
                     ra_range=[0, 24])


def test_align_lst_shapes_equal():
    """Test the shape of the time_arrays are equal after lst_align."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = copy.deepcopy(test_uv)
    ra_range = [0, 12]

    test_uv_out, test_uv_2_out = utils.lst_align(test_uv, test_uv_2,
                                                 ra_range=ra_range,
                                                 inplace=False)
    nt.assert_equal(test_uv_out.time_array.shape, test_uv_out.time_array.shape)


def test_align_lst_shapes_equal_uv_2_longer():
    """Test shape of time_array are equal after lst_align: 2nd uv longer."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_miriad_2 = os.path.join(DATA_PATH, 'paper_test_file_2nd_time.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = utils.read_paper_miriad([test_miriad, test_miriad_2],
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    ra_range = [0, 12]

    test_uv_out, test_uv_2_out = utils.lst_align(test_uv, test_uv_2,
                                                 ra_range=ra_range,
                                                 inplace=False)
    nt.assert_equal(test_uv_out.time_array.shape, test_uv_out.time_array.shape)


def test_align_lst_shapes_equal_uv_1_longer():
    """Test shape of time_array are equal after lst_align: 1st uv longer."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_miriad_2 = os.path.join(DATA_PATH, 'paper_test_file_2nd_time.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv = utils.read_paper_miriad([test_miriad, test_miriad_2],
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])
    test_uv_2 = utils.read_paper_miriad(test_miriad,
                                        antpos_file=test_antpos_file,
                                        skip_header=3, usecols=[1, 2, 3])
    ra_range = [0, 12]

    test_uv_out, test_uv_2_out = utils.lst_align(test_uv, test_uv_2,
                                                 ra_range=ra_range,
                                                 inplace=False)
    nt.assert_equal(test_uv_out.time_array.shape, test_uv_out.time_array.shape)


def test_get_integration_time_shape():
    """Test the shape of the integration_time array is correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    inttime_array = utils.get_integration_time(test_uv, reds=baseline_array)
    test_shape = (test_uv.Nbls, test_uv.Ntimes, 1)
    nt.assert_equal(test_shape, inttime_array.shape)


def test_get_integration_time_shape_with_pol():
    """Test the shape of the integration_time array is correct with pols."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    inttime_array = utils.get_integration_time(test_uv, reds=baseline_array,
                                               squeeze=False)
    test_shape = (1, test_uv.Nbls, test_uv.Ntimes, 1)
    nt.assert_equal(test_shape, inttime_array.shape)


def test_get_integration_time_vals():
    """Test the values of the integration_time array is correct."""
    test_miriad = os.path.join(DATA_PATH, 'paper_test_file.uv')
    test_antpos_file = os.path.join(DATA_PATH, 'paper_antpos.txt')

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

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

    test_uv = utils.read_paper_miriad(test_miriad,
                                      antpos_file=test_antpos_file,
                                      skip_header=3, usecols=[1, 2, 3])

    baseline_array = np.array(list(set(test_uv.baseline_array)))
    inttime_array = utils.get_integration_time(test_uv, reds=baseline_array,
                                               squeeze=False)
    test_shape = (1, test_uv.Nbls, test_uv.Ntimes, 1)
    test_array = test_uv.integration_time.copy()
    test_array = test_array.reshape(test_shape)
    nt.assert_true(np.allclose(test_array, inttime_array))
