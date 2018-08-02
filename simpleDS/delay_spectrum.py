"""Calculate Delay Spectrum from pyuvdata object."""
from __future__ import print_function

import os
import sys
import numpy as np
from pyuvdata import UVData
import astropy.units as units
from astropy.units import Quantity
from astropy import constants as const
from scipy.signal import windows
from . import utils, cosmo


def jy_to_mk(freqs):
    """Calculate the Jy to mK conversion lambda^2/(2 * K_boltzman)."""
    assert(isinstance(freqs, Quantity))
    jy2t = const.c.to('m/s')**2 / (2 * freqs.to('1/s')**2
                                   * const.k_B)
    return jy2t.to('mK/Jy')


def normalized_fourier_transform(data_array, delta_x=1. * units.Hz, axis=-1,
                                 window=windows.blackmanharris):
    """Perform the Fourier transform over specified axis.

    Perform the FFT over frequency using the specified window function
    and normalizes by delta_x (the discrete of sampling rate along the axis).

    Arguments:
        data_array : (Nbls, Ntimes, Nfreqs) array from utils.get_data_array
                        Can also have shape (Npols, Nbls, Ntimes, Nfreqs)
        delta_x: The difference between frequency channels in the data.
                 This is used to properly normalize the Fourier Transform.
                 Must be an astropy Quantity object
        window : Window function used in delay transform.
                 Default is scipy.signal.windows.blackmanharris
    Returns:
        delay_arry: (Nbls, Ntimes, Nfreqs) array of the Fourier transform along
                    specified axis, and normalized by the provided delta_x
                    if pols are present returns
                    (Npols, Nbls, Ntimes, Nfreqs)
    """
    if isinstance(data_array, Quantity):
        unit = data_array.unit
    else:
        unit = 1.

    if not isinstance(delta_x, Quantity):
        raise ValueError('delta_x must be an astropy Quantity object. '
                         'value was : {df}'.format(df=delta_x))

    n_axis = data_array.shape[axis]
    win = window(n_axis).reshape(1, n_axis)

    # Fourier Transforms should have a delta_x term multiplied
    # This is the proper normalization of the FT but is not
    # accounted for in an fft.
    delay_array = np.fft.fft(data_array * win, axis=axis)
    delay_array = np.fft.fftshift(delay_array, axes=axis)
    delay_array = delay_array * delta_x.to('1/s') * unit

    return delay_array


def combine_nsamples(nsample_1, nsample_2=None):
    """Combine the nsample arrays for use in cross-multiplication.

    Uses numpy slicing to generate array of all sample cross-multiples.
    Used to find the combine samples for a the delay spectrum.
    The geometric mean is taken between nsamples_1 and nsamples_2 because
    nsmaples array is used to compute thermal variance in the delay spectrum.

    Arguments:
        nsample_1 : (Nbls, Ntimes, Nfreqs) array from utils.get_nsamples_array
                    can also have shape (Npols, Nbls, Ntimes, Nfreqs)
        nsample_2 : same type as nsample_1 if take cross-multiplication
                       Defaults to copying nsample_1 for auto-correlation
    Returns:
        samples_out: (Nbls, Nbls, Nfreqs, Ntimes) array of geometric mean of
                     the input sample arrays.
                     Can also have shape (Npols, Nbls, Nbls, Ntimes, Nfreqs)
    """
    if nsample_2 is None:
        nsample_2 = nsample_1.copy()

    if not nsample_1.shape == nsample_2.shape:
        raise ValueError('nsample_1 and nsample_2 must have same shape, '
                         'but nsample_1 has shape {d1_s} and '
                         'nsample_2 has shape {d2_s}'
                         .format(d1_s=nsample_1.shape,
                                 d2_s=nsample_2.shape))

    if len(nsample_1.shape) == 3:
        axis = 0
    else:
        axis = 1

    samples_out = utils.cross_multipy_array(array_1=nsample_1,
                                            array_2=nsample_2,
                                            axis=axis)

    # The nsamples array is used to construct the thermal variance
    # Cross-correlation takes the geometric mean of thermal variance.
    return np.sqrt(samples_out)


def remove_auto_correlations(data_array):
    """Remove the auto-corrlation term from input array.

    Argument:
        data_array : (Nbls, Nbls, Ntimes, Nfreqs)
                     Removes same baseline diagonal along the first 2 diemsions
    Returns:
        data_out : (Nbls * (Nbls-1), Ntimes, Nfreqs) array.
                   if input has pols: (Npols, Nbls * (Nbls -1), Ntimes, Nfreqs)
    """
    if len(data_array.shape) == 4:
        Nbls = data_array.shape[0]
    elif len(data_array.shape) == 5:
        Nbls = data_array.shape[1]
    else:
        raise ValueError('Input data_array must be of type '
                         '(Npols, Nbls, Nbls, Ntimes, Nfreqs) or '
                         '(Nbls, Nbls, Ntimes, Nfreqs) but data_array'
                         'has shape {0}'.format(data_array.shape))
    # make a boolean index array with True off the diagonal and
    # False on the diagonal.
    indices = np.logical_not(np.diag(np.ones(Nbls, dtype=bool)))
    if len(data_array.shape) == 4:
        data_out = data_array[indices]
    else:
        data_out = data_array[:, indices]

    return data_out


def calculate_noise_power(nsamples, freqs, inttime, trcvr, npols):
    """Generate power as given by the radiometry equation.

    noise_power = Tsys/sqrt(delta_frequency * inttime )

    Computes the system temperature using the equation:
        T_sys = 180K * (nu/180MHz)^(-2.55) + T_receiver
    Arguments:
        nsamples: The nsamples array from which to compute thermal variance
        freqs: The observed frequncies
        trcvr: The receiver temperature of the instrument in K
    Returns:
        noise_power: White noise with the same shape as nsamples input.
    """
    if not isinstance(inttime, Quantity):
        raise ValueError('inttime must be an astropy Quantity object. '
                         'value was: {t}'.format(t=inttime))
    if not isinstance(freqs, Quantity):
        raise ValueError('freqs must be an astropy Quantity object. '
                         'value was: {fq}'.format(fq=freqs))
    if not isinstance(trcvr, Quantity):
        raise ValueError('trcvr must be an astropy Quantity object. '
                         ' value was: {temp}'.format(temp=trcvr))

    Tsys = 180. * units.K * np.power(freqs.to('GHz')/(.18 * units.GHz), -2.55)
    Tsys += trcvr.to('K')
    delta_f = np.diff(freqs)[0]
    noise_power = (Tsys.to('K')
                   / np.sqrt(delta_f.to('1/s') * inttime.to('s')
                             * nsamples * npols))
    return noise_power.to('mK')


def generate_noise(noise_power):
    """Generate noise given an input array of noise power.

    Argument:
        noise_power: N-dimensional array of noise power to generate white
                     noise.
    Returns:
        noise: Complex white noise drawn from a Gaussian distribution with
               width given by the value of the input noise_power array.
    """
    # divide by sqrt(2) to conserve total noise amplitude over real and imag
    noise = noise_power * (1 * np.random.normal(size=noise_power.shape)
                           + 1j * np.random.normal(size=noise_power.shape))
    noise /= np.sqrt(2)
    return noise


def calculate_delay_spectrum(uv_even, uv_odd, uvb, trcvr, reds,
                             squeeze=True, window=windows.blackmanharris):
    """Calculate delay cross-correlation between the uv_even and uv_odd.

    Arguments:
        uv_even: One of the pyuvdata objects to cross correlate
        uv_odd: Other pyuvdata object to multiply with uv_even.
        uvb: UVBeam object with relevent beam info.
                Currently assumes 1 beam object can describe all baselines
                Must be a power beam in healpix coordinates and peak normalized
        reds: set of redundant baselines to calculate delay power spectrum.
        nboot: Number of boot strap re-samples to perform (0 to omit step)
        trcvr: Receiver Temperature of antenna to calculate noise power

    Returns:
        delays: Astropy quantity object of delays. Fourier dual to Frequency
                shape (Nfreqs)
        delay_power: (Nbls, Nbls, Ntimes, Nfreqs) of delay powers
                     or (Npols, Nbls, Nbls, Ntimes, Nfreqs)
                     if polarizations are present or squeeze=False
        noise_power: (Nbls, Nbls, Ntimes, Nfreqs) of simulated noise powers
                     or (Npols, Nbls, Nbls, Ntimes, Nfreqs)
                     if polarizations are present or squeeze=False
        thermal_power: (Nbls, Nbls, Ntimes, Nfreqs) of expected thermal power
                     or (Npols, Nbls, Nbls, Ntimes, Nfreqs)
                     if polarizations are present or squeeze=False
    """
    if not np.allclose(uv_even.freq_array, uv_odd.freq_array):
        raise ValueError("Both pyuvdata objects must have the same "
                         "frequencies in order to cross-correlate.")
    freqs = np.squeeze(uv_even.freq_array, axis=0) * units.Hz
    delays = np.fft.fftfreq(len(freqs), d=np.diff(freqs)[0].value)
    delays = np.fft.fftshift(delays) / freqs.unit
    delays = delays.to('s')

    if not np.allclose(uv_even.integration_time, uv_odd.integration_time):
        raise ValueError("Both pyuvdata objects must have the same "
                         "integration time in order to cross-correlate.")

    inttime = utils.get_integration_time(uv_even, reds=reds, squeeze=squeeze)

    # make the inttime a quantity object for units to work
    inttime = inttime * units.s
    # Check if vibiliities are psuedo-Stokes parameters
    # This will decrease the noise estimate
    if np.intersect1d(uv_even.polarization_array, np.arange(1, 5)):
        npols = 2
    else:
        npols = 1
    even_data = utils.get_data_array(uv_even, reds=reds, squeeze=squeeze)
    odd_data = utils.get_data_array(uv_odd, reds=reds, squeeze=squeeze)

    even_samples = utils.get_nsample_array(uv_even, reds=reds, squeeze=squeeze)
    odd_samples = utils.get_nsample_array(uv_odd, reds=reds, squeeze=squeeze)

    even_flags = utils.get_flag_array(uv_even, reds=reds, squeeze=squeeze)
    odd_flags = utils.get_flag_array(uv_odd, reds=reds, squeeze=squeeze)

    # I'm not sure I want to cast them as float right now, but we'll see
    even_flags = np.logical_not(even_flags).astype(float)
    odd_flags = np.logical_not(odd_flags).astype(float)

    # Make all the data Quantity objects to help keep the units right
    even_data = even_data * units.Jy
    odd_data = odd_data * units.Jy

    even_noise = calculate_noise_power(nsamples=even_samples,
                                       freqs=freqs,
                                       inttime=inttime,
                                       trcvr=trcvr, npols=npols)
    odd_noise = calculate_noise_power(nsamples=odd_samples,
                                      freqs=freqs,
                                      inttime=inttime,
                                      trcvr=trcvr, npols=npols)
    # Conver the noise powers to white noise
    even_noise = generate_noise(even_noise)
    odd_noise = generate_noise(odd_noise)

    # save cross-multiplication axis
    if len(np.shape(even_data)) == 3:
        cross_mult_axis = 0
    else:
        cross_mult_axis = 1
    # Generate a waterfall for each cross multiplication of the
    # expected thermal variance.
    thermal_noise_samples = combine_nsamples(even_samples, odd_samples)

    delay_1_array = normalized_fourier_transform(data_array=(even_data
                                                             * even_flags),
                                                 delta_x=np.diff(freqs)[0],
                                                 window=window, axis=-1)
    delay_2_array = normalized_fourier_transform(data_array=(odd_data
                                                             * odd_flags),
                                                 delta_x=np.diff(freqs)[0],
                                                 window=window, axis=-1)

    delay_power = utils.cross_multipy_array(array_1=delay_1_array,
                                            array_2=delay_2_array,
                                            axis=cross_mult_axis)

    noise_1_delay = normalized_fourier_transform(data_array=(even_noise
                                                             * even_flags),
                                                 delta_x=np.diff(freqs)[0],
                                                 window=window)

    noise_2_delay = normalized_fourier_transform(data_array=(odd_noise
                                                             * odd_flags),
                                                 delta_x=np.diff(freqs)[0],
                                                 window=window)

    noise_power = utils.cross_multipy_array(array_1=noise_1_delay,
                                            array_2=noise_2_delay,
                                            axis=cross_mult_axis)

    thermal_power = calculate_noise_power(nsamples=thermal_noise_samples,
                                          freqs=freqs,
                                          inttime=inttime,
                                          trcvr=trcvr, npols=npols)
    thermal_power *= thermal_power

    # Convert from visibility Units (Jy) to comological units (mK^2/(h/Mpc)^3)
    z_mean = np.mean(cosmo.calc_z(freqs))
    X2Y = cosmo.X2Y(z_mean)
    # Calculate the effective bandwith for the given window function
    bandwidth = (freqs[-1] - freqs[0])
    bandwidth *= utils.noise_equivalent_bandwidth(window(len(freqs)))
    unit_conversion = X2Y / bandwidth.to('1/s') / uvb.get_beam_sq_area()

    # the *= operator does not play nicely with multiplying a non-quantity
    # with a quantity
    delay_power = delay_power * unit_conversion * jy_to_mk(freqs)**2
    noise_power = noise_power * unit_conversion
    # The thermal expectation requires the additional delta_f**2 factor
    # for all the units to be correct since we are multiplying them
    # on explicitly

    delta_time = np.diff(np.unique(uv_even.time_array))[0] * units.sday
    lst_bins = uv_even.Ntimes * delta_time.to('s') / inttime.to('s')
    thermal_power = thermal_power * X2Y * np.diff(freqs)[0].to('1/s')
    # This normalization of the thermal power comes from
    # Parsons PSA32 paper appendix B
    thermal_power *= uvb.get_beam_area()**2 / uvb.get_beam_sq_area()
    # Divide by the following factors:
    #   Nbls: baselines should coherently add together
    #   sqrt(2): noise is split between even and odd
    #   sqrt(lst_bins): noise power spectrum averages incoherently over time
    thermal_power /= uv_even.Nbls * np.sqrt(2 * lst_bins)
    # divie the thermal expectation by 2 if visibilities are psuedo Stokes

    return delays, delay_power, noise_power, thermal_power
