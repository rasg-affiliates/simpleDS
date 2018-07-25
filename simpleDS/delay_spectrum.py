"""Calculate Delay Spectrum from pyuvdata object."""
from __future__ import print_function

import os
import sys
import numpy as np
from pyuvdata import UVData
from builtins import range, zip
import astropy.units as units
from astropy.units import Quantity
from astropy import constants as const
from scipy.signal import windows


def jy_to_mk(freqs):
    """Calculate the Jy to mK conversion lambda^2/(2 * K_boltzman)."""
    assert(isinstance(freqs, Quantity))
    jy2t = const.c.to('m/s')**2 / (2 * freqs.to('1/s')**2
                                   * const.k_B)
    return jy2t.to('mK/Jy')


def delay_transform(data_1_array, data_2_array=None,
                    delta_f=1. * units.Hz,
                    window=windows.blackmanharris):
    """Perform the Dealy transform over specified channel rangeself.

    Perform the FFT over frequency using the specified window function
    and cross multiplies all baselines in the data array.

    Arguments:
        data_1_array : (Nbls, Ntimes, Nfreqs) array from utils.get_data_array
                        Can also have shape (Npols, Nbls, Nfreqs, Ntimes)
        data_2_array : same type as data_1_array if take cross-multiplication
                       Defaults to copying data_1_array for auto-correlation
        delta_f: The difference between frequency channels in the data.
                 This is used to properly normalize the Fourier Transform.
                 Must be an astropy Quantity object
        window : Window function used in delay transform.
                 Default is scipy.signal.windows.blackmanharris
    Returns:
        delay_power: (Nbls, Nbls, Ntimes, Nfreqs) array
                     Cross multplies all baselines in data_1_array
                     if pols are present returns
                     (Npols, Nbls, Nbls, Ntimes, Nfreqs)
    """
    if isinstance(data_1_array, Quantity):
        unit = data_1_array.unit
    else:
        unit = 1.

    if not isinstance(delta_f, Quantity):
        raise ValueError('delta_f must be an astropy Quantity object. '
                         'value was : {df}'.format(df=delta_f))

    if data_2_array is None:
        data_2_array = data_1_array.copy()

    if not data_2_array.shape == data_1_array.shape:
        raise ValueError('data_1_array and data_2_array must have same shape, '
                         'but data_1_array has shape {d1_s} and '
                         'data_2_array has shape {d2_s}'
                         .format(d1_s=data_1_array.shape,
                                 d2_s=data_2_array.shape))
    nfreqs = data_1_array.shape[-1]
    win = window(nfreqs).reshape(1, nfreqs)

    # Fourier Transforms should have a delta_f term multiplied
    # This is the proper normalization of the FT but is not
    # accounted for in an fft.
    delay_1_array = np.fft.fft(data_1_array * win, axis=-1)
    delay_1_array = np.fft.fftshift(delay_1_array, axes=-1)
    delay_1_array = delay_1_array * delta_f.to('1/s') * unit

    delay_2_array = np.fft.fft(data_2_array * win, axis=-1)
    delay_2_array = np.fft.fftshift(delay_2_array, axes=-1)
    delay_2_array = delay_2_array * delta_f.to('1/s') * unit

    # Using slicing for cross-multiplication should be quick.
    if len(data_1_array.shape) == 3:
        delay_power = (delay_1_array[None, :, :].conj() *
                       delay_2_array[:, None, :, :])
    else:
        delay_power = (delay_1_array[:, None, :, :].conj() *
                       delay_2_array[:, :, None, :, :])
    return delay_power
