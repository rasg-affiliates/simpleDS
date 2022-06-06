# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Read and Write support for PAPER miriad files with pyuvdata."""

import copy
import numpy as np
from astropy import constants as const
from astropy import units
from astropy.cosmology.units import littleh
from pyuvdata import utils as uvutils
from scipy.signal import windows


def get_data_array(uv, reds, squeeze=True):
    """Remove data from pyuvdata object and store in numpy array.

    Uses UVData.get_data function to create a matrix of data of shape (Npols, Nbls, Ntimes, Nfreqs).
    Only valid to call on a set of redundant baselines with the same number of times.

    Parameters
    ----------
    uv : UVdata object, or subclass
        Data object which can support uv.get_data(ant_1, ant_2, pol)
    reds: list of ints
        list of all redundant baselines of interest as baseline numbers.
    squeeze : bool
        set true to squeeze the polarization dimension.
        This has no effect for data with Npols > 1.

    Returns
    -------
    data_array :complex arrary
        (Nbls , Ntimes, Nfreqs) numpy array or (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False

    """
    data_shape = (uv.Npols, uv.Nbls, uv.Ntimes, uv.Nfreqs)
    data_array = np.zeros(data_shape, dtype=np.complex64)

    for count, baseline in enumerate(reds):
        tmp_data = uv.get_data(baseline, squeeze="none")
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

    Uses UVData.get_nsamples function to create a matrix of data of shape (Npols, Nbls, Ntimes, Nfreqs).
    Only valid to call on a set of redundant baselines with the same number of times.

    Parameters
    ----------
    uv : UVdata object, or subclass
        Data object which can support uv.get_data(ant_1, ant_2, pol)
    reds: list of ints
        list of all redundant baselines of interest as baseline numbers.
    squeeze : bool
        set true to squeeze the polarization dimension.
        This has no effect for data with Npols > 1.

    Returns
    -------
    nsample_array : float array
        (Nbls, Ntimes, Nfreqs) numpy array or (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False

    """
    nsample_shape = (uv.Npols, uv.Nbls, uv.Ntimes, uv.Nfreqs)
    nsample_array = np.zeros(nsample_shape, dtype=np.float32)

    for count, baseline in enumerate(reds):
        tmp_data = uv.get_nsamples(baseline, squeeze="none")
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

    Uses UVData.get_flags function to create a matrix of data of shape (Npols, Nbls, Ntimes, Nfreqs).
    Only valid to call on a set of redundant baselines with the same number of times.

    Parameters
    ----------
    uv : UVdata object, or subclass
        Data object which can support uv.get_data(ant_1, ant_2, pol)
    reds: list of ints
        list of all redundant baselines of interest as baseline numbers.
    squeeze : bool
        set true to squeeze the polarization dimension.
        This has no effect for data with Npols > 1.

    Returns
    -------
    flag_array : bool array
        (Nbls, Ntimes, Nfreqs) numpy array  (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False

    """
    flag_shape = (uv.Npols, uv.Nbls, uv.Ntimes, uv.Nfreqs)
    flag_array = np.zeros(flag_shape, dtype=bool)
    reds = np.array(reds)

    for count, baseline in enumerate(reds):
        tmp_data = uv.get_flags(baseline, squeeze="none")
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

    Extracts the integration time array from a UVdata object to create a matrix of shape (Npols, Nbls, Ntimes, Nfreqs).
    Only valid to call on a set of redundant baselines with the same number of times.

    Parameters
    ----------
    uv : UVdata object, or subclass
        Data object which can support uv.get_data(ant_1, ant_2, pol)
    reds: list of ints
        list of all redundant baselines of interest as baseline numbers.
    squeeze : bool
        set true to squeeze the polarization dimension.
        This has no effect for data with Npols > 1.

    Returns
    -------
    integration_time : float array
        (Nbls, Ntimes) numpy array of integration times.

    """
    shape = (uv.Nbls, uv.Ntimes)
    integration_time = np.zeros(shape, dtype=np.float32)
    reds = np.array(reds)

    for count, baseline in enumerate(reds):
        blt_inds, conj_inds, pol_inds = uv._key2inds(baseline)
        # The integration doesn't care about conjugation, just need all the
        # times associated with this baseline
        inds = np.concatenate([blt_inds, conj_inds])
        inds.sort()
        integration_time[count] = uv.integration_time[inds]

    return integration_time


def bootstrap_array(array, nboot=100, axis=0):
    """Bootstrap resample the input array along the given axis.

    Randomly sample, with replacement along the given axis `nboot` times.
    Output array will always have N+1 dimensions with the extra axis immediately following the bootstrapped axis.

    Parameters
    ----------
    array : numpy array
        N-dimensional array to bootstrap resample.
    nboot : int
        Number of resamples to draw.
    axis : int
        Axis along which resampling is performed

    Returns
    -------
    array : numpy array
        The resampled array, if input is N-D output is N+1-D, extra dimension is added imediately suceeding the sampled axis.

    Raises
    ------
    ValueError
        If `axis` parameter is greater than the dimensions of the input array.

    """
    if axis >= len(np.shape(array)):
        raise ValueError(
            "Specified axis must be shorter than the length "
            "of input array.\n"
            "axis value was {0} but array has {1} dimensions".format(
                axis, len(np.shape(array))
            )
        )

    sample_inds = np.random.choice(
        array.shape[axis], size=(array.shape[axis], nboot), replace=True
    )
    return np.take(array, sample_inds, axis=axis)


def noise_equivalent_bandwidth(window):
    """Calculate the relative equivalent noise bandwidth of window function.

    Approximates the relative Noise Equivalent Bandwidth of a spectral taper function.

    Parameters
    ----------
    window : array_like
        A 1-Dimenaional array like.

    Returns
    -------
    float

    """
    return np.sum(window) ** 2 / (np.sum(window**2) * len(window))


def combine_nsamples(nsample_1, nsample_2=None, axis=-1):
    """Combine the nsample arrays for use in cross-multiplication.

    Uses numpy slicing to generate array of all sample cross-multiples.
    Used to find the combine samples for a the delay spectrum.
    The geometric mean is taken between nsamples_1 and nsamples_2 because
    nsmaples array is used to compute thermal variance in the delay spectrum.

    Parameters
    ----------
    nsample_1 : array of float
        (Nbls, Ntimes, Nfreqs) array from get_nsamples_array can also have shape (Npols, Nbls, Ntimes, Nfreqs)
    nsample_2 : array of float, optional
        Same type as nsample_1 if take cross-multiplication
        Defaults to copying nsample_1 for auto-correlation
    axis : int
        The axis over which the cross multiplication should occur.

    Returns
    -------
    samples_out : array of float
        (Nbls, Nbls, Nfreqs, Ntimes) array of geometric mean of the input sample arrays.
        Can also have shape (Npols, Nbls, Nbls, Ntimes, Nfreqs)

    Raises
    ------
    ValueError
        If both input arrays have different shapes.

    """
    if nsample_2 is None:
        nsample_2 = nsample_1.copy()

    if not nsample_1.shape == nsample_2.shape:
        raise ValueError(
            "nsample_1 and nsample_2 must have same shape, "
            "but nsample_1 has shape {d1_s} and "
            "nsample_2 has shape {d2_s}".format(
                d1_s=nsample_1.shape, d2_s=nsample_2.shape
            )
        )

    samples_out = np.sqrt(
        cross_multiply_array(array_1=nsample_1, array_2=nsample_2, axis=axis)
    )

    # The nsamples array is used to construct the thermal variance
    # Cross-correlation takes the geometric mean of thermal variance.
    return samples_out


def remove_auto_correlations(data_array, axes=(0, 1)):
    """Remove the auto-corrlation term from input array.

    Takes an N x N array, removes the diagonal components, and returns a flattened N(N-1) dimenion in place of the array.
    If uses on a M dimensional array, returns an M-1 array.

    Parameters
    ----------
    data_array : array
        Array shaped like (Nbls, Nbls, Ntimes, Nfreqs). Removes same baseline diagonal along the specifed axes.
    axes : tuple of int, length 2
        axes over which the diagonal will be removed.

    Returns
    -------
    data_out : array with the same type as `data_array`.
        (Nbls * (Nbls-1), Ntimes, Nfreqs) array.
        if input has pols: (Npols, Nbls * (Nbls -1), Ntimes, Nfreqs)

    Raises
    ------
    ValueError
        If axes is not a length 2 tuple.
        If axes are not adjecent (e.g. axes=(2,7)).
        If axes do not have the same shape.

    """
    if not np.shape(axes)[0] == 2:
        raise ValueError(
            "Shape must be a length 2 tuple/array/list of " "axis indices."
        )
    if axes[0] != axes[1] - 1:
        raise ValueError(
            "Axes over which diagonal components are to be " "remove must be adjacent."
        )
    if data_array.shape[axes[0]] != data_array.shape[axes[1]]:
        raise ValueError(
            "The axes over which diagonal components are to be "
            "removed must have the same shape."
        )
    n_inds = data_array.shape[axes[0]]
    # make a boolean index array with True off the diagonal and
    # False on the diagonal.
    indices = np.logical_not(np.diag(np.ones(n_inds, dtype=bool)))
    # move the axes so axes[0] is the 0th axis and axis 1 is the 1th
    data_array = np.moveaxis(data_array, axes[0], 0)
    data_array = np.moveaxis(data_array, axes[1], 1)
    data_out = data_array[indices]
    # put the compressed axis back in the original spot
    data_out = np.moveaxis(data_out, 0, axes[0])
    return data_out


def cross_multiply_array(array_1, array_2=None, axis=0):
    """Cross multiply the arrays along the given axis.

    Cross multiplies along axis and computes array_1.conj() * array_2
    if axis has length M then a new axis of size M will be inserted directly succeeding the original.

    Parameters
    ----------
    array_1 : array_like
        N-dimensional array_like
    array_2 : array_like, optional
        N-dimenional array.
        Defaults to copy of array_1
    axis : int
        Axis along which to cross multiply

    Returns
    -------
    cross_array : array_like
        N+1 Dimensional array

    Raises
    ------
    ValueError
        If input arrays have different shapes.

    """
    if isinstance(array_1, list):
        array_1 = np.asarray(array_1)

    if array_2 is None:
        array_2 = copy.deepcopy(array_1)

    if isinstance(array_2, list):
        array_2 = np.asarray(array_2)

    unit_1, unit_2 = 1, 1
    if isinstance(array_1, units.Quantity):
        unit_1 = array_1.unit
        array_1 = array_1.value

    if isinstance(array_2, units.Quantity):
        unit_2 = array_2.unit
        array_2 = array_2.value

    if array_2.shape != array_1.shape:
        raise ValueError(
            "array_1 and array_2 must have the same shapes. "
            "array_1 has shape {a1} but array_2 has shape {a2}".format(
                a1=np.shape(array_1), a2=np.shape(array_2)
            )
        )

    cross_array = np.expand_dims(array_1, axis=axis).conj() * np.expand_dims(
        array_2, axis=axis + 1
    )
    if isinstance(unit_1, units.UnitBase) or isinstance(unit_2, units.UnitBase):
        cross_array <<= unit_1 * unit_2

    return cross_array


def lst_align(uv1, uv2, ra_range, inplace=True, atol=1e-08, rtol=1e-05):
    """Align the LST values of two pyuvdata objects within the given range.

    Attempts to crudely align input UVData objects in LST by finding indices
    where the lsts fall within the input range.

    Parameters
    ----------
    uv1 : UVData object or subclass
        First object where data is desired to lie within the `ra_range`.
        Must have a time_array parameter to find lsts.
    uv2 : UVData object or subclass
        Must have the same properties as `uv1`
    ra_range : length 2 list of float
        The inclusive start and stop times in hours for the LSTs to align.
    inplace : bool
        If true performs UVData.select on `uv1` and `uv2` otherwise returns new objects.
    atol : float
        Absolute tolerance with which the integrations time should agree
    rtol : float
        Relative tolerance with which the integrations time should agree

    Returns
    -------
    (uv1, uv2) : tuple of UVData objects
        only returns when `inplace` is False.

    Raises
    ------
    ValueError
        if `uv1` and `uv2` have different integration times, or time cadences.

    """
    delta_t_1 = uv1._calc_single_integration_time()
    delta_t_2 = uv2._calc_single_integration_time()
    if not np.isclose(delta_t_1, delta_t_2, rtol=rtol, atol=atol):
        raise ValueError(
            "The two UVData objects much have matching "
            "time sample rates. "
            "values were uv1: {0} and uv2: {1}".format(delta_t_1, delta_t_2)
        )
    bl1 = uv1.baseline_array[0]
    bl2 = uv2.baseline_array[0]
    times_1 = uv1.get_times(bl1)
    times_2 = uv2.get_times(bl2)

    uv1_location = uv1.telescope_location_lat_lon_alt_degrees
    uv2_location = uv2.telescope_location_lat_lon_alt_degrees
    lsts_1 = uvutils.get_lst_for_time(times_1, *uv1_location) * 12.0 / np.pi
    lsts_2 = uvutils.get_lst_for_time(times_2, *uv2_location) * 12.0 / np.pi

    inds_1 = np.logical_and(lsts_1 >= ra_range[0], lsts_1 <= ra_range[-1])
    inds_2 = np.logical_and(lsts_2 >= ra_range[0], lsts_2 <= ra_range[-1])

    diff = inds_1.sum() - inds_2.sum()
    if diff > 0:
        last_ind = inds_1.size - inds_1[::-1].argmax() - 1
        inds_1[last_ind : last_ind - diff : -1] = False
    elif diff < 0:
        diff = np.abs(diff)
        last_ind = inds_2.size - inds_2[::-1].argmax() - 1
        inds_2[last_ind : last_ind - diff : -1] = False

    new_times_1 = times_1[inds_1]
    new_times_2 = times_2[inds_2]
    return (
        uv1.select(times=new_times_1, inplace=inplace),
        uv2.select(times=new_times_2, inplace=inplace),
    )


@units.quantity_input(freqs="frequency")
def jy_to_mk(freqs):
    """Calculate the Jy/sr to mK conversion lambda^2/(2 * K_boltzman).

    Parameters
    ----------
    freqs : Astropy Quantity with units equivalent to frequency
        frequencies where the conversion should be calculated.

    Returns
    -------
    Astropy Quantity
        The conversion factor from Jy to mK * sr at the given frequencies.

    """
    jy2t = units.sr * const.c.to("m/s") ** 2 / (2 * freqs.to("1/s") ** 2 * const.k_B)
    return jy2t << units.Unit("mK*sr/Jy")


def generate_noise(noise_power):
    """Generate noise given an input array of noise power.

    Parameters
    ----------
    noise_power : array_like of float
        N-dimensional array of noise power to generate white noise.

    Returns
    -------
    noise : array_like of complex
        Complex white noise drawn from a Gaussian distribution with width given by the value of the input noise_power array.

    """
    # divide by sqrt(2) to conserve total noise amplitude over real and imag
    noise = noise_power * (
        1 * np.random.normal(size=noise_power.shape)
        + 1j * np.random.normal(size=noise_power.shape)
    )
    noise /= np.sqrt(2)
    return noise


def normalized_fourier_transform(
    data_array, delta_x, axis=-1, taper=windows.blackmanharris, inverse=False
):
    """Perform the Fourier transform over specified axis.

    Perform the FFT over frequency using the specified taper function
    and normalizes by delta_x (the discrete of sampling rate along the axis).

    Parameters
    ----------
    data_array : array_like
        N-dimenaional array of data to Fourier Transform
    delta_x : Astropy Quantity
        The difference between channels in the data over the axis of transformation.
        This is used to properly normalize the Fourier Transform.
    taper : callable
        Spectral taper function used in Fourier transform.
        Default is scipy.signal.windows.blackmanharris
    inverse: (bool; Default False)
        Perform the inverse Fourier Transform with np.fft.ifft

    Returns
    -------
    fourier_arry : array_like complex
        N-Dimenaional array of the Fourier transform along
        specified axis, and normalized by the `provided delta_x`.

    Raises
    ------
    ValueError
        If `delta_x` is not a Quantity object.

    """
    if not isinstance(delta_x, units.Quantity):
        raise ValueError(
            "delta_x must be an astropy Quantity object. "
            "value was : {df}".format(df=delta_x)
        )

    n_axis = data_array.shape[axis]
    data_shape = np.ones_like(data_array.shape)
    data_shape[axis] = n_axis
    # win = taper(n_axis).reshape(1, n_axis)
    win = np.broadcast_to(taper(n_axis), data_shape)

    # Fourier Transforms should have a delta_x term multiplied
    # This is the proper normalization of the FT but is not
    # accounted for in an fft.
    if not inverse:
        fourier_array = np.fft.fft(data_array * win, axis=axis)
        fourier_array = np.fft.fftshift(fourier_array, axes=axis)
        fourier_array = fourier_array * delta_x.si
    else:
        fourier_array = np.fft.ifft(data_array, axis=axis)
        fourier_array = np.fft.ifftshift(fourier_array, axes=axis)
        fourier_array = fourier_array / win * delta_x.si

    return fourier_array


def weighted_average(array, uncertainty, weights=None, axis=-1):
    """Compute the weighted average and propagate uncertainty.

    Performs weighted average of input array, and propagates the weighted average into the uncertainty.

    Parameters
    ----------
    array : array_like
        N-dimensional array over which to average an axis.
    uncertainty : array_like
        N-dimensional array of uncertainties associated with each point in input `array`.
    weights : array_like, Optional
        N-dimenional or 1-Dimenaiona array of weights to apply to each data point.
        If weights are one dimensional must be of length N and assumped to be the same for each row M.
        if weights is None, uses inverse variance weighting
    axis : int
        The axis over which the average is taken.

    Returns
    -------
    array : array_like
        MxN-1 dimenionals array averaged array with given weights.
    uncertainty : array_like
        MxN-1 dimensional propagated uncertainty array for given weights.

    Raises
    ------
    ValueError
        If either one of the `array` or `uncertainty` is an astropy Quantity object but the other is not.
        If `array` and `uncertainty` have different shapes.
        If `weights` are 1-Dimensional but length is not the same as the lenght of `array` along the given `axis`.
        If `weights` are N-Dimensional but shaped differs from `array`.

    """
    if isinstance(array, units.Quantity):
        if isinstance(uncertainty, units.Quantity):
            uncertainty = uncertainty << array.unit
        else:
            raise ValueError(
                "Input array is a Quantity Object but "
                "uncertainty is not. Either both arrays must be "
                "Quantity objects or neither must be."
            )
    elif isinstance(uncertainty, units.Quantity):
        raise ValueError(
            "Input array is a not Quantity Objcet but "
            "uncertainty is. Either both arrays must be "
            "Quantity objects or neither must be."
        )
    # check shapes of array and uncertainty
    if not array.shape == uncertainty.shape:
        raise ValueError(
            "Input array and uncertainties must have the same "
            "shape. Array shape: {array}, Uncertainty shape: "
            "{error}".format(array=array.shape, error=uncertainty.shape)
        )
    # if weights is none use uniform? inverse variance?
    if weights is None:
        weights = 1.0 / uncertainty**2
    # check shape of weights
    if np.ndim(weights) == 1 and weights.size != array.shape[axis]:
        raise ValueError(
            "1-Dimenionals weights must have the same shape "
            "as the axis over which the average is taken."
        )
    elif not np.shape(weights) == array.shape:
        raise ValueError(
            "Input array and uncertainties must have the same "
            "shape. Array shape: {array}, Uncertainty shape: "
            "{weights}".format(array=array.shape, weights=weights.shape)
        )
    array_out = np.average(array, weights=weights, axis=axis)
    uncertainty_out = np.sqrt(
        np.sum(uncertainty**2 * np.abs(weights) ** 2, axis=axis)
        / np.abs(np.sum(weights, axis=axis)) ** 2
    )
    return array_out, uncertainty_out


accepted_units = ["mK^2*Mpc^3", units.mK**2 * units.Mpc**3 / littleh**3, "time"]


@units.quantity_input(delays="time", array=accepted_units)
def fold_along_delay(delays, array, uncertainty, weights=None, axis=-1):
    """Fold input array over the delay axis.

    Averages input `array` along the given `axis` by splitting the `delay` array around the value 0 and
    averaging values in `array` correspoding to the same `delay` magnitutde.


    Parameters
    ----------
    delays : Astropy Quantity units equivalent to time.
        A 1-Dimensional numpy array of interferometric delays.
    array : Astropy Quantity units equivalent to time, or power.
        An N-Dimensional Quantity to average across delay magnitudes.
    uncertainty : Astropy Quantity units equivalent to `array`
        An N-Dimensional Quantity of uncertainty values for input `array`
    weights : array_like
        Weights to use while averaging the input `array`. Must have same shape as input array.
        Default: np.ones_like(array)
    axis : int
        The axis over which the input array is to be folded.
        Must have the same shape as the size of input delays.

    Returns
    -------
    array : Astropy Quantity with units equivalent to input `array`
        The N-Dimensional input array folded over the axis specified
        give axis will have size np.shape(array)[axis]/2 if shape is even
        or (np.shape(array)[axis] + 1)/2 if shape is odd
    uncertainty : Astropy Quantity with units equivalent to input `uncertainty`.
        The folded uncertainties corresponding to the input array.

    Raises
    ------
    ValueError
        If shape of the specified axis is not equal to the length of the `delay` array.
        If input `delays` either have no 0 bin or evenly spaced around zero.
        If `uncertainty` is not the same shape as `array`.

    """
    delays = copy.deepcopy(delays)
    array = copy.deepcopy(array)
    uncertainty = copy.deepcopy(uncertainty)
    # This function assumes your array is a block-square array,
    # e.g. all delays are the same.
    if array.shape[axis] != len(delays):
        raise ValueError(
            (
                "Input array must have same length as the "
                "delays along the specified axis."
                "Axis given was {0}, the array has length {1} "
                "but delays are length {2}".format(axis, array.shape[axis], len(delays))
            )
        )
    if len(delays) % 2 != 0 and np.abs(delays).min() != 0:
        raise ValueError(
            (
                "Input delays must have either a delay=0 bin "
                "as the central value or have an even size."
            )
        )

    # check shapes of array and uncertainty
    if not array.shape == uncertainty.shape:
        raise ValueError(
            "Input array and uncertainties must have the same "
            "shape. Array shape: {array}, Uncertainty shape: "
            "{error}".format(array=array.shape, error=uncertainty.shape)
        )

    if weights is None:
        if not array.imag.value.any():
            weights = np.ones_like(array)
        else:
            weights = np.ones_like(array) * (1 + 1j)

    if np.logical_and(np.abs(delays).min() == 0, delays.size % 2):
        split_index = np.argmin(np.abs(delays), axis=axis)
        split_inds = [split_index, split_index + 1]

        neg_vals, zero_bin, pos_vals = np.split(array, split_inds, axis=axis)
        neg_vals = np.flip(neg_vals, axis=axis)

        pos_vals = np.concatenate([zero_bin, pos_vals], axis=axis)
        neg_vals = np.concatenate([zero_bin, neg_vals], axis=axis)

        neg_errors, zero_errors, pos_errors = np.split(
            uncertainty, split_inds, axis=axis
        )
        # the error on the 0 delay mode will go down like 1/sqrt(2) with the
        # inverse vaiance weighting scheme, but we didn't actually gain any information
        # so rescale the error, this _could_ be avoided if we didn't concatenate along
        # the 0th dimension but would make it require two different calculations
        zero_errors *= np.sqrt(2)
        neg_errors = np.flip(neg_errors, axis=axis)
        pos_errors = np.concatenate([zero_errors, pos_errors], axis=axis)
        neg_errors = np.concatenate([zero_errors, neg_errors], axis=axis)

        neg_weights, zero_weights, pos_weights = np.split(
            weights, split_inds, axis=axis
        )
        neg_weights = np.flip(neg_weights, axis=axis)
        pos_weights = np.concatenate([zero_weights, pos_weights], axis=axis)
        neg_weights = np.concatenate([zero_weights, neg_weights], axis=axis)

    else:
        min_val_bool = np.abs(delays) == np.amin(np.abs(delays), axis=axis)
        split_index = np.where(np.logical_and(delays >= 0, min_val_bool))
        split_inds = [np.squeeze(split_index)]

        neg_vals, pos_vals = np.split(array, split_inds, axis=axis)
        neg_vals = np.flip(neg_vals, axis=axis)

        neg_errors, pos_errors = np.split(uncertainty, split_inds, axis=axis)
        neg_errors = np.flip(neg_errors, axis=axis)

        neg_weights, pos_weights = np.split(weights, split_inds, axis=axis)
        neg_weights = np.flip(neg_weights, axis=axis)

    _array = np.stack([pos_vals, neg_vals], axis=0)
    _errors = np.stack([pos_errors, neg_errors], axis=0)
    _weights = np.stack([pos_weights, neg_weights], axis=0)

    if not _array.imag.value.any():
        out_array, out_errors = weighted_average(
            _array.real, _errors.real, weights=_weights, axis=0
        )
    else:
        weight_check = _weights.imag.value.any()
        if not weight_check:
            try:
                _weights.imag = np.ones_like(_weights.real)
            except TypeError:
                _weights = _weights.astype(np.complex64)
                _weights.imag = np.ones_like(_weights.real)
        out_array_real, out_errors_real = weighted_average(
            _array.real, _errors.real, weights=_weights.real, axis=0
        )
        out_array_imag, out_errors_imag = weighted_average(
            _array.imag, _errors.imag, weights=_weights.imag, axis=0
        )

        out_array = out_array_real + 1j * out_array_imag
        out_errors = out_errors_real + 1j * out_errors_imag

    return out_array, out_errors
