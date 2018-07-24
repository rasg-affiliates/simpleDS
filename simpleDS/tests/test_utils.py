"""Test PAPER miriad io."""
from __future__ import print_function

import os
import sys
import numpy as np
import nose.tools as nt
import pyuvdata
from pyuvdata import UVData, utils as uvutils
from simpleDS import utils
from simpleDS.data import DATA_PATH
from builtins import range, zip
from astropy import constants as const

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
