"""Read and Write support for PAPER miriad files with pyuvdata."""

import os
import sys
import numpy as np
from pyuvdata import UVData, utils as uvutils
import six
from six.moves import range, map
from astropy import constants as const
from astropy import units
import copy


def read_paper_miriad(filename, antpos_file=None, **kwargs):
    """Read PAPER miriad files and return pyuvdata object.

    Arguments
        filename: pyuvdata compatible PAPER file (Default miriad)
        antpols_file: A file of antenna postions. Required to generate uvws.
        kwargs: passed to:
                    numpy.genfromtxt
                    UVData.read
    Returns
        uv: Correctly formatted pyuvdata object from input PAPER data
    """
    uv = UVData()
    if six.PY2:
        kwargs_uvdata = {key: kwargs[key] for key in kwargs
                         if key in uv.read.func_code.co_varnames}
    else:
        kwargs_uvdata = {key: kwargs[key] for key in kwargs
                         if key in uv.read.__code__.co_varnames}
    uv.read(filename, **kwargs_uvdata)

    if antpos_file is None:
        raise ValueError("An antpos_file file "
                         "is required to generate uvw array.")

    if not os.path.exists(antpos_file):
        raise IOError("{0} not found.".format(antpos_file))
    if six.PY2:
        kwargs_genfromtxt = {key: kwargs[key] for key in kwargs
                             if key in np.genfromtxt.func_code.co_varnames}
    else:
        kwargs_genfromtxt = {key: kwargs[key] for key in kwargs
                             if key in np.genfromtxt.__code__.co_varnames}
    antpos = np.genfromtxt(antpos_file, **kwargs_genfromtxt)

    antpos_ecef = uvutils.ECEF_from_ENU(antpos,
                                        *uv.telescope_location_lat_lon_alt)
    antpos_itrf = antpos_ecef - uv.telescope_location
    good_ants = list(map(int, uv.antenna_names))
    antpos_itrf = np.take(antpos_itrf, good_ants, axis=0)
    setattr(uv, 'antenna_positions', antpos_itrf)
    uv.set_uvws_from_antenna_positions()

    if 'FRF_NEBW' in uv.extra_keywords:
        uv.integration_time = np.ones_like(uv.integration_time)
        uv.integration_time *= uv.extra_keywords['FRF_NEBW']

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
                     (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False
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
        nsample_array : (Nbls, Ntimes, Nfreqs) numpy array
                        (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False
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
        flag_array : (Nbls, Ntimes, Nfreqs) numpy array
                     (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False
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


def get_integration_time(uv, reds, squeeze=True):
    """Extract the integration_time array from pyuvdata objectself.

    Duplicates integration time for redundant group for faster calculations.

    Arguments:
        uv: pyuvdata data object from which to get integration time array.
        reds: list of all redundant baselines of interest as baseline numbers
    keywords:
        squeeze: set true to squeeze the polarization dimension.
                 This has no effect for data with Npols > 1.

    Returns:
        integration_time : (Nbls, Ntimes, Nfreqs) numpy array
                           (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False
    """
    shape = (uv.Nbls, uv.Ntimes)
    integration_time = np.zeros(shape, dtype=np.complex)
    reds = np.array(reds)

    for count, baseline in enumerate(reds):
        blt_inds, conj_inds, pol_inds = uv._key2inds(baseline)
        # The integration doesn't care about conjugation, just need all the
        # times associated with this baseline
        inds = np.concatenate([blt_inds, conj_inds])
        inds.sort()
        integration_time[count] = uv.integration_time[inds]

    # tile to Npols, Nbls, Ntimes, Nfreqs to be broadcastable with other arrays
    integration_time = np.tile(integration_time.reshape(uv.Nbls, uv.Ntimes, 1),
                               (uv.Npols, 1, 1, 1))
    if squeeze:
        if integration_time.shape[0] == 1:
            integration_time = np.squeeze(integration_time, axis=0)

    return integration_time


def bootstrap_array(array, nboot=100, axis=0):
    """Bootstrap resample the input array along the given axis.

    Arguments:
        array: N-dimensional array to bootstrap resample.
        nboot: Number of resamples to draw.
        axis: Axis along which resampling is performed
    Returns:
        array: The resampled array, if input is N-D output is N+1-D,
               extra dimension is added imediately suceeding the sampled axis.
    """
    if axis >= len(np.shape(array)):
        raise ValueError("Specified axis must be shorter than the lenght "
                         "of input array.\n"
                         "axis value was {0} but array has {} dimensions"
                         .format(axis, len(np.shape(array))))

    sample_inds = np.random.choice(array.shape[axis],
                                   size=(array.shape[axis], nboot),
                                   replace=True)
    return np.take(array, sample_inds, axis=axis)


def noise_equivalent_bandwidth(window):
    """Calculate the relative equivalent noise bandwidth of window function."""
    return np.sum(window)**2 / (np.sum(window**2) * len(window))


def cross_multipy_array(array_1, array_2=None, axis=0):
    """Cross multiply the arrays along the given axis.

    Cross multiplies along axis and computes array_1.conj() * array_2
    if axis has length M then a new axis of size M will be inserted
    directly succeeding the original.
    Arguments:
        array_1: N-dimensional numpy array
        array_2: N-dimenional array (copy of array_1 if None)
        axis: Axis along which to cross multiply

    Returns:
        cross_array : N+1 Dimensional array
    """
    if isinstance(array_1, list):
        array_1 = np.asarray(array_1)

    if array_2 is None:
        array_2 = copy.deepcopy(array_1)

    if isinstance(array_2, list):
        array_2 = np.asarray(array_2)

    unit_1, unit_2 = 1., 1.
    if isinstance(array_1, units.Quantity):
        unit_1 = array_1.unit

    if isinstance(array_2, units.Quantity):
        unit_2 = array_2.unit

    if array_2.shape != array_1.shape:
        raise ValueError("array_1 and array_2 must have the same shapes. "
                         "array_1 has shape {a1} but array_2 has shape {a2}"
                         .format(a1=np.shape(array_1), a2=np.shape(array_2)))

    cross_array = (np.expand_dims(array_1, axis=axis).conj()
                   * np.expand_dims(array_2, axis=axis + 1))

    return cross_array * unit_1 * unit_2


def lst_align(uv1, uv2, ra_range, inplace=True):
    """Align the LST values of two pyuvdata objects within the given range."""
    delta_t_1 = uv1._calc_single_integration_time()
    delta_t_2 = uv2._calc_single_integration_time()
    if not np.isclose(delta_t_1, delta_t_2):
        raise ValueError("The two UVData objects much have matching "
                         "time sample rates. "
                         "values were uv1: {0} and uv2:{1}"
                         .format(delta_t_1, delta_t_2))
    bl1 = uv1.baseline_array[0]
    bl2 = uv2.baseline_array[0]
    times_1 = uv1.get_times(bl1)
    times_2 = uv2.get_times(bl2)

    uv1_location = uv1.telescope_location_lat_lon_alt_degrees
    uv2_location = uv2.telescope_location_lat_lon_alt_degrees
    lsts_1 = uvutils.get_lst_for_time(times_1, *uv1_location) * 12. / np.pi
    lsts_2 = uvutils.get_lst_for_time(times_2, *uv2_location) * 12. / np.pi

    inds_1 = np.logical_and(lsts_1 >= ra_range[0], lsts_1 <= ra_range[-1])
    inds_2 = np.logical_and(lsts_2 >= ra_range[0], lsts_2 <= ra_range[-1])

    diff = inds_1.sum() - inds_2.sum()
    if diff > 0:
        last_ind = inds_1.size - inds_1[::-1].argmax() - 1
        inds_1[last_ind:last_ind - diff:-1] = False
    elif diff < 0:
        diff = np.abs(diff)
        last_ind = inds_2.size - inds_2[::-1].argmax() - 1
        inds_2[last_ind:last_ind - diff:-1] = False

    new_times_1 = times_1[inds_1]
    new_times_2 = times_2[inds_2]
    return (uv1.select(times=new_times_1, inplace=inplace),
            uv2.select(times=new_times_2, inplace=inplace))


@units.quantity_input(delays='time', array='mK^2*Mpc^3')
def fold_along_delay(array, delays, weights=None, axis=-1):
    """Fold input array over the delay axis.

    Arguments
        array: An N-Dimensional numpy array or nested lists.
        delays: A 1-Dimensional numpy array of interfometric delays.
        weights: Weights to use while averaging the input array.
                 Must have same shape as input array.
                 Default: np.ones_like(array)
        axis: The axis over which the input array is to be folded.
              Must have the same shape as the size of input delays.
    Returns
        array: The N-Dimensional input array folded over the axis specified
               give axis will have size np.shape(array)[axis]/2 if shape is even
               or (np.shape(array)[axis] + 1)/2 if shape is odd
        weights: The folded weights used corresponding to the input array.
    """
    # This function assumes your array is a block-square array,
    # e.g. all delays are the same.
    if array.shape[axis] != len(delays):
        raise ValueError(("Input array must have same length as the "
                          "delays along the specified axis."
                          "Axis given was {0}, the array has length {1} "
                          "but delays are length {2}".format(axis,
                                                             array.shape[axis],
                                                             len(delays))))
    if (len(delays) % 2 != 0 and np.abs(delays).min() != 0):
        raise ValueError(("Input delays must have either a delay=0 bin "
                          "as the central value or have an even size."))

    if weights is None:
        no_input_weights = True
        if not array.imag.value.any():
            weights = np.ones_like(array)
        else:
            weights = np.ones_like(array) * (1 + 1j)

    if np.abs(delays).min() == 0:
        split_index = np.argmin(np.abs(delays), axis=axis)
        split_inds = [split_index, split_index + 1]

        neg_vals, zero_bin, pos_vals = np.split(array, split_inds, axis=axis)
        neg_vals = np.flip(neg_vals, axis=axis)

        pos_vals = np.concatenate([zero_bin, pos_vals], axis=axis)
        neg_vals = np.concatenate([zero_bin, neg_vals], axis=axis)

        neg_weights, zero_weights, pos_weights = np.split(weights, split_inds,
                                                          axis=axis)
        neg_weights = np.flip(neg_weights, axis=axis)
        pos_weights = np.concatenate([zero_weights, pos_weights], axis=axis)
        neg_weights = np.concatenate([zero_weights, neg_weights], axis=axis)

    else:
        min_val_bool = np.abs(delays) == np.amin(np.abs(delays), axis=axis)
        split_index = np.where(np.logical_and(delays > 0, min_val_bool))
        split_inds = [np.squeeze(split_index)]

        neg_vals, pos_vals = np.split(array, split_inds, axis=axis)
        neg_vals = np.flip(neg_vals, axis=axis)

        neg_weights, pos_weights = np.split(weights, split_inds, axis=axis)
        neg_weights = np.flip(neg_weights, axis=axis)

    _array = np.stack([pos_vals, neg_vals], axis=0)
    _weights = np.stack([pos_weights, neg_weights], axis=0)

    if _array.unit is None:
        _array = _array.value * array.unit
    if _weights.unit is None:
        _weights = _weights.value * weights.unit

    if not _array.imag.value.any():
        out_array = np.average(_array.real, weights=1. / _weights.real**2,
                               axis=0)
        out_weights = np.sqrt(np.average(_weights.real**2,
                                         weights=1. / _weights.real**2, axis=0))
        return out_array, out_weights
    else:
        weight_check = _weights.imag.value.any()
        if not weight_check:
            try:
                _weights.imag = np.ones_like(_weights.real)
            except TypeError:
                _weights = _weights.astype(np.complex)
                _weights.imag = np.ones_like(_weights.real)
        out_array = (np.average(_array.real, weights=1. / _weights.real**2, axis=0)
                     + 1j * np.average(_array.imag, weights=1. / _weights.imag**2, axis=0))

        out_weights = (np.sqrt(np.average(_weights.real**2, weights=1. / _weights.real**2, axis=0))
                       + 1j * np.sqrt(np.average(_weights.imag**2, weights=1. / _weights.imag**2, axis=0)))

        if not weight_check:
            out_weights.imag = np.zeros_like(out_weights.real)

        return out_array, out_weights
