"""Read and Write support for PAPER miriad files with pyuvdata."""

import os
import sys
import numpy as np
from pyuvdata import UVData, utils as uvutils
from builtins import range, map
from astropy import constants as const


def import_calfile(filename=None):
    """Import the calfile from the PAPER array."""
    if filename is None:
        raise ValueError("Must Supply calfile.")
    if filename.split('.')[-1] == 'py':
        # If the input name has a .py attached it needs to be removed
        # before it can be imported
        filename = '.'.join(filename.split('.')[:-1])
    if not os.path.exists(filename + '.py'):
        raise IOError(filename + '.py not found')
    cal_dir_name = os.path.dirname(filename)
    cal_base_name = os.path.basename(filename)
    sys.path.append(cal_dir_name)
    exec("import {0} as calfile".format(cal_base_name))
    return calfile


def read_paper_miriad(filename, calfile=None, antpos_file=None, **kwargs):
    """Read PAPER miriad files and return pyuvdata object.

    One of a calfile or an antpos_file is required to generate uvws.
    kwargs passed to numpy.genfromtxt if antpos file is provided.
    """
    if not isinstance(filename, (list, np.ndarray)):
        filename = [filename]
    uv = UVData()
    uv.read_miriad(filename)

    if all(x is None for x in [calfile, antpos_file]):
        raise ValueError("Either an antpos_file file or a calfile "
                         "is required to generate uvw array.")
    elif calfile:
        calfile = import_calfile(calfile)
        # load an AtennaArray object from calfile with a dummy Frequency
        aa = calfile.get_aa(np.array([.1]))
        antpos = np.array([aa.get_baseline(0, i, src='z')
                           for i in range(len(aa.ants))])
        # Convert light-nanoseconds to meters
        antpos *= const.c.to('m/ns').value
        antpos -= antpos.mean(0)
    else:
        if not os.path.exists(antpos_file):
            raise IOError("{0} not found.".format(antpos_file))
        antpos = np.genfromtxt(antpos_file, **kwargs)
    antpos_ecef = uvutils.ECEF_from_ENU(antpos,
                                        *uv.telescope_location_lat_lon_alt)
    antpos_itrf = antpos_ecef - uv.telescope_location
    good_ants = list(map(int, uv.antenna_names))
    antpos_itrf = np.take(antpos_itrf, good_ants, axis=0)
    setattr(uv, 'antenna_positions', antpos_itrf)
    uv.set_uvws_from_antenna_positions()

    return uv


def get_data_array(uv, reds, squeeze=True):
    """Remove data from pyuvdata object and store in numpy array.

    Duplicates data in redundant group as an array for faster calculations.
    Arguments:
        uv : data object which can support uv.get_data(ant_1, ant_2, pol)
        reds: list of all redundant baselines of interest as baseline numbers
    keywords:
        squeeze: set true to squeeze the polarization dimension.
                 This has no effect for data with Npols > 1.

    Returns:
        data_array : (Nbls , Ntimes, Nfreqs) numpy array or
                     (Npols, Nbls, Ntimes, Nfreqs)
    """
    data_shape = (uv.Npols, uv.Nbls, uv.Ntimes, uv.Nfreqs)
    data_array = np.zeros(data_shape, dtype=np.complex)

    for count, baseline in enumerate(reds):
        tmp_data = uv.get_data(baseline, squeeze='none')
        # Keep the polarization dimenions and squeeze out spw
        tmp_data = np.squeeze(tmp_data, axis=1)
        # Reorder to: Npols, Ntimes, Nfreqs

        data_array[:, count] = np.transpose(tmp_data, [2, 0, 1])

    if squeeze:
        if data_array.shape[0] == 1:
            data_array = np.squeeze(data_array, axis=0)

    return data_array


def get_nsample_array(uv, reds, squeeze=True):
    """Remove nsamples from pyuvdata object and store in numpy array.

    Duplicates nsamples in redundant group as an array for faster calculations.
    Arguments:
        uv : data object which can support uv.get_nsamples(ant_1, ant_2, pol)
        reds: list of all redundant baselines of interest as baseline numbers
    keywords:
        squeeze: set true to squeeze the polarization dimension.
                 This has no effect for data with Npols > 1.

    Returns:
        nsample_array - Nbls x Ntimes numpy array
    """
    nsample_shape = (uv.Npols, uv.Nbls, uv.Ntimes, uv.Nfreqs)
    nsample_array = np.zeros(nsample_shape, dtype=np.complex)

    for count, baseline in enumerate(reds):
        tmp_data = uv.get_nsamples(baseline, squeeze='none')
        # Keep the polarization dimenions and squeeze out spw
        tmp_data = np.squeeze(tmp_data, axis=1)
        # Reorder to: Npols, Ntimes, Nfreqs
        nsample_array[:, count] = np.transpose(tmp_data, [2, 0, 1])

    if squeeze:
        if nsample_array.shape[0] == 1:
            nsample_array = np.squeeze(nsample_array, axis=0)

    return nsample_array



def get_flag_array(uv, reds, squeeze=True):
    """Remove nsamples from pyuvdata object and store in numpy array.

    Duplicates nsamples in redundant group as an array for faster calculations.
    Arguments:
        uv : data object which can support uv.get_nsamples(ant_1, ant_2, pol)
        reds: list of all redundant baselines of interest as baseline numbers
    keywords:
        squeeze: set true to squeeze the polarization dimension.
                 This has no effect for data with Npols > 1.

    Returns:
        nsample_array - Nbls x Ntimes numpy array
    """
    flag_shape = (uv.Npols, uv.Nbls, uv.Ntimes, uv.Nfreqs)
    flag_array = np.zeros(flag_shape, dtype=np.complex)
    reds = np.array(reds)

    for count, baseline in enumerate(reds):
        tmp_data = uv.get_flags(baseline, squeeze='none')
        # Keep the polarization dimenions and squeeze out spw
        tmp_data = np.squeeze(tmp_data, axis=1)
        # Reorder to: Npols, Ntimes, Nfreqs
        flag_array[:, count] = np.transpose(tmp_data, [2, 0, 1])

    if squeeze:
        if flag_array.shape[0] == 1:
            flag_array = np.squeeze(flag_array, axis=0)

    return flag_array
