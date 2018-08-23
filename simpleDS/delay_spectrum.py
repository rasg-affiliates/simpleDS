# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Matthew Kolopanis
# Licensed under the 3-clause BSD License
"""Calculate Delay Spectrum from pyuvdata object."""
from __future__ import print_function, absolute_import, division

import os
import sys
import numpy as np
from pyuvdata import UVData
import astropy.units as units
from astropy.units import Quantity
from astropy import constants as const
from scipy.signal import windows
from . import utils, cosmo as simple_cosmo


@units.quantity_input(freqs='frequency')
def jy_to_mk(freqs):
    """Calculate the Jy to mK conversion lambda^2/(2 * K_boltzman)."""
    jy2t = const.c.to('m/s')**2 / (2 * freqs.to('1/s')**2
                                   * const.k_B)
    return jy2t.to('mK/Jy')


def normalized_fourier_transform(data_array, delta_x, axis=-1,
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
    delay_array = delay_array * delta_x.si * unit

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

    samples_out = utils.cross_multiply_array(array_1=nsample_1,
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


@units.quantity_input(freqs='frequency', inttime='time', trcvr=units.K)
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
    Tsys = 180. * units.K * np.power(freqs.to('GHz') / (.18 * units.GHz), -2.55)
    Tsys += trcvr.to('K')
    delta_f = np.diff(freqs)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        noise_power = np.ma.masked_invalid(Tsys.to('K')
                                           / np.sqrt(delta_f.to('1/s')
                                                     * inttime.to('s')
                                                     * nsamples * npols))
    return noise_power.filled(0).to('mK')


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


@units.quantity_input(trcvr=units.K)
def calculate_delay_spectrum(uv_even, uv_odd, uvb, trcvr, reds,
                             squeeze=True, window=windows.blackmanharris,
                             cosmo=None):
    """Calculate delay cross-correlation between the uv_even and uv_odd.

    Arguments:
        uv_even: One of the pyuvdata objects to cross correlate
        uv_odd: Other pyuvdata object to multiply with uv_even.
        uvb: UVBeam object with relevent beam info.
                Currently assumes 1 beam object can describe all baselines
                Must be a power beam in healpix coordinates and peak normalized
        reds: set of redundant baselines to calculate delay power spectrum.
        trcvr: Receiver Temperature of antenna to calculate noise power
        window : Window function used in delay transform.
                 Default is scipy.signal.windows.blackmanharris
        cosmo : astropy.cosmology object to specify the assumed cosmology
                for X2Y conversion and cosmological calculations
                Default is WMAP9 in "little h" units (H0=100 Km/s/Mpc)

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
    if not uv_even.vis_units == uv_odd.vis_units:
        raise NotImplementedError(("You are trying to multiply two "
                                   "visibilities with different units. This "
                                   "is currently not a supported feature."))
    if uv_even.vis_units == 'Jy':
        unit = units.Jy
    elif uv_even.vis_units == 'K str':
        unit = units.K * units.sr
    else:
        # if the uv unit is uncalibrated give data a
        # dimensionless_unit
        unit = units.Unit('')
    even_data = even_data * unit
    odd_data = odd_data * unit

    even_noise = calculate_noise_power(nsamples=even_samples,
                                       freqs=freqs,
                                       inttime=inttime,
                                       trcvr=trcvr, npols=1)
    odd_noise = calculate_noise_power(nsamples=odd_samples,
                                      freqs=freqs,
                                      inttime=inttime,
                                      trcvr=trcvr, npols=1)
    # Conver the noise powers to white noise
    even_noise = generate_noise(even_noise) * uvb.get_beam_area(pol=uv_even.polarization_array) / np.sqrt(uvb.get_beam_sq_area(pol=uv_even.polarization_array))
    odd_noise = generate_noise(odd_noise) * uvb.get_beam_area(pol=uv_odd.polarization_array) / np.sqrt(uvb.get_beam_sq_area(pol=uv_odd.polarization_array))

    if unit == units.Jy:
        even_data *= jy_to_mk(freqs) / np.sqrt(uvb.get_beam_sq_area(pol=uv_even.polarization_array))
        odd_data *= jy_to_mk(freqs) / np.sqrt(uvb.get_beam_sq_area(pol=uv_odd.polarization_array))
    elif unit == (units.K * units.sr):
        # multiply by beam_area**2 / beam_square_area to properly normalized
        # the power spectrum
        even_data *= uvb.get_beam_area(pol=uv_even.polarization_array) / (np.sqrt(uvb.get_beam_sq_area(pol=uv_even.polarization_array)) * units.sr)
        odd_data *= uvb.get_beam_area(pol=uv_odd.polarization_array) / (np.sqrt(uvb.get_beam_sq_area(pol=uv_odd.polarization_array)) * units.sr)

        even_data = even_data.to('mK')
        odd_data = odd_data.to('mK')
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

    delay_power = utils.cross_multiply_array(array_1=delay_1_array,
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

    noise_power = utils.cross_multiply_array(array_1=noise_1_delay,
                                             array_2=noise_2_delay,
                                             axis=cross_mult_axis)

    thermal_power = calculate_noise_power(nsamples=thermal_noise_samples,
                                          freqs=freqs,
                                          inttime=inttime,
                                          trcvr=trcvr, npols=npols)
    thermal_power *= thermal_power

    # Convert from visibility Units (Jy) to comological units (mK^2/(h/Mpc)^3)
    z_mean = np.mean(simple_cosmo.calc_z(freqs))
    X2Y = simple_cosmo.X2Y(z_mean, cosmo=cosmo)
    # Calculate the effective bandwith for the given window function
    bandwidth = (freqs[-1] - freqs[0])
    bandwidth *= utils.noise_equivalent_bandwidth(window(len(freqs)))
    unit_conversion = X2Y / bandwidth.to('1/s')

    # the *= operator does not play nicely with multiplying a non-quantity
    # with a quantity
    delay_power = delay_power * unit_conversion
    noise_power = noise_power * unit_conversion
    # The thermal expectation requires the additional delta_f**2 factor
    # for all the units to be correct since we are multiplying them
    # on explicitly

    delta_time = np.diff(np.unique(uv_even.time_array))[0] * units.day
    lst_bins = uv_even.Ntimes * delta_time.to('s') / inttime.to('s')
    thermal_power = thermal_power * X2Y * np.diff(freqs)[0].to('1/s')
    # This normalization of the thermal power comes from
    # Parsons PSA32 paper appendix B
    beam_factor_array = uvb.get_beam_area(pol=uv_even.polarization_array)**2 / uvb.get_beam_sq_area(pol=uv_even.polarization_array)

    if beam_factor_array.shape[0] % 2 == 0:
        mid_index = beam_factor_array.shape[0] // 2
        beam_factor = np.mean(beam_factor_array[mid_index - 1:mid_index + 1])
    else:
        mid_index = (beam_factor_array.shape[0] - 1) // 2
        beam_factor = beam_factor_array[mid_index]

    thermal_power *= beam_factor
    # Divide by the following factors:
    #   Nbls: baselines should coherently add together
    #   sqrt(2): noise is split between even and odd
    #   sqrt(lst_bins): noise power spectrum averages incoherently over time
    thermal_power /= uv_even.Nbls * np.sqrt(2 * lst_bins)
    # divie the thermal expectation by 2 if visibilities are psuedo Stokes

    return delays, delay_power, noise_power, thermal_power


class DelaySpectrum(object):
    """A Delay Spectrum object to hold relevant data."""

    def __init__(self, uv1, uv2, uvb, trcvr, reds, squeeze=True):
        """Initialize the Delay Spectrum Object.

        Arguments
            uv1: One of the pyuvdata objects to cross correlate
            uv2: Other pyuvdata object to multiply with uv_even.
            uvb: UVBeam object with relevent beam info.
                 Currently assumes 1 beam object can describe all baselines
                 Must be power beam in healpix coordinates and peak normalized
            reds: set of redundant baselines to calculate delay power spectrum.
            trcvr: Receiver Temperature of antenna to calculate noise power
                   Must be an astropy Quantity object with units of temperature
        """
        self.freqs = None
        self.delays = None
        self.redshift = None
        self.wavelength = None
        self.Nbls = None
        self.reds = None
        self.k_perpendicular = None
        self.k_parallel = None
        self.power = None
        self.noise = None
        self.thermal_expectation = None
        self.trcvr = None
        self.data_1_array = None
        self.data_2_array = None
        self.nsample_1_array = None
        self.nsample_2_array = None
        self.flag_2_array = None
        self.beam_sq_area = None
        self.beam_area = None

        if not np.allclose(uv_even.freq_array, uv_odd.freq_array):
            raise ValueError("Both pyuvdata objects must have the same "
                             "frequencies in order to cross-correlate.")
        if not np.allclose(uv1.freq_array, uvb.freq_array):
            raise ValueError("The pyuvdata objects and the UVBream object "
                             "must have the same frequencies for "
                             "proper unit conversion.")

        self.freqs = np.squeeze(uv_even.freq_array, axis=0) * units.Hz
        delays = np.fft.fftfreq(len(freqs), d=np.diff(freqs)[0].value)
        delays = np.fft.fftshift(delays) / freqs.unit
        self.delays = delays.to('s')

        self.redshift = cosmo.calc_z(self.freqs).mean()
        self.X2Y = cosmo.X2Y(z_mean)

        self.jy_to_mk = jy_to_mk(self.freqs)

        if not np.allclose(uv1.integration_time, uv2.integration_time):
            raise ValueError("Both pyuvdata objects must have the same "
                             "integration time in order to cross-correlate.")
        self.inttime = utils.get_integration_time(uv_even, reds=reds,
                                                  squeeze=squeeze) * units.s

        delta_t_1 = uv1._calc_single_integration_time()
        delta_t_2 = uv2._calc_single_integration_time()
        if not np.isclose(delta_t_1, delta_t_2):
            raise ValueError("The two UVData objects much have matching "
                             "time sampling rates. "
                             "values were uv1: {0} and uv2: {1}"
                             .format(delta_t_1, delta_t_2))
        self.lst_bins = uv1.Ntimes * delta_time.to('s') / self.inttime.to('s')

        if not isinstance(trcvr, Quantity):
            raise ValueError('trcvr must be an astropy Quantity object. '
                             ' value was: {temp}'.format(temp=trcvr))
        self.trcvr = trcvr
        # Sometimes antenna pairs can be cast into weird types
        # If it is an array of anteanna pairs, convert to baseline numbers
        if isinstance(reds[0], (tuple, np.ndaray, list)):
            reds = list(map(uv1.antnums_to_baseline, reds))
        self.reds = reds
        self.Nbls = len(reds)
        # Check if vibiliities are psuedo-Stokes parameters
        # This will affect the noise estimate
        if np.intersect1d(uv1.polarization_array, np.arange(1, 5)):
            self.npols = 2
        else:
            self.npols = 1

        # Cast the data as Quantity objects for units to work.
        if uv.vis_unit == 'Jy':
            unit = units.Jy
        elif uv.vis_unit == 'K str':
            unit = units.K * units.sr
        else:
            # if the uv unit is uncalibrated give data a
            # dimensionless_unit
            unit = units.Unit('')
        self.data_1_array = utils.get_data_array(uv1, reds=reds,
                                                 squeeze=squeeze) * unit
        self.data_2_array = utils.get_data_array(uv2, reds=reds,
                                                 squeeze=squeeze) * unit

        self.nsample_1_array = utils.get_nsample_array(uv1, reds=reds,
                                                       squeeze=squeeze)
        self.nsample_2_array = utils.get_nsample_array(uv2, reds=reds,
                                                       squeeze=squeeze)

        self.flag_1_array = utils.get_flag_array(uv1, reds=reds,
                                                 squeeze=squeeze)
        self.flag_2_array = utils.get_flag_array(uv2, reds=reds,
                                                 squeeze=squeeze)

        # I'm not sure I want to cast them as float right now, but we'll see
        self.flag_1_array = np.logical_not(self.even_flags).astype(float)
        self.flag_2_array = np.logical_not(self.odd_flags).astype(float)

        if len(np.shape(self.data_1_array)) == 3:
            self.cross_mult_axis = 0
        else:
            self.cross_mult_axis = 1

        self.beam_area = uvb.get_beam_area()
        self.beam_sq_area = uvb.get_beam_sq_area()

        self.window = windows.blackmanharris

    def calculate_delay_spectrum(self, window=None):
        """Perform Delay tranform and cross multiplication of datas.

        Arguments:
            window: The window function to multiply onto the data.
                    Accepts scipy.signal.windows functions or any function
                    whose argument is the len(data) and returns a numpy array.
                    Default: scipy.signal.windows.blackmanharris

        Take the normalized Fourier transform of the data from uv1 and uv2
        objects and cross multiplies.
        Also generates white noie given the frequency range and trcvr and
        calculates the expected noise power.
        """
        if window is None:
            window = self.window
        delta_f = np.diff(self.freqs)[0]

        noise_1_array = calculate_noise_power(nsamples=self.nsample_1_array,
                                              freqs=self.freqs,
                                              inttime=self.inttime,
                                              trcvr=self.trcvr,
                                              npols=self.npols)
        noise_2_array = calculate_noise_power(nsamples=self.nsample_2_array,
                                              freqs=self.freqs,
                                              inttime=self.inttime,
                                              trcvr=self.trcvr,
                                              npols=self.npols)

        NEBW = utils.noise_equivalent_bandwidth(window(len(self.freqs)))
        self.bandwidth = (self.freqs[-1] - self.freqs[0]) * NEBW
        unit_conversion = self.X2Y / self.bandwith.to('1/s') / sef.beam_sq_area

        delay_1_array = normalized_fourier_transform((self.data_1_array
                                                      * self.flag_1_array),
                                                     delta_x=delta_f,
                                                     window=window, axis=-1)
        delay_2_array = normalized_fourier_transform((self.data_2_array
                                                      * self.flag_2_array),
                                                     delta_x=delta_f,
                                                     window=window, axis=-1)

        delay_power = utils.cross_multipy_array(array_1=delay_1_array,
                                                array_2=delay_2_array,
                                                axis=self.cross_mult_axis)

        self.power = delay_power * unit_conversion * self.jy_to_mk**2

        noise_1_delay = normalized_fourier_transform((self.noise_1_array
                                                      * self.flag_1_array),
                                                     delta_x=delta_f,
                                                     window=window, axis=-1)

        noise_2_delay = normalized_fourier_transform((self.noise_2_array
                                                      * self.flag_2_array),
                                                     delta_x=delta_f,
                                                     window=window, axis=-1)

        noise_power = utils.cross_multipy_array(array_1=noise_1_delay,
                                                array_2=noise_2_delay,
                                                axis=self.cross_mult_axis)
        self.noise_power = noise_power * unit_conversion

    def calculate_thermal_sensitivity(self):
        """Calculate the Thermal sensitivity for the power spectrum.

        Uses the 21cmsense_calc formula:
            Tsys**2/(inttime * Nbls * Npols * sqrt(N_lstbins * 2))

        Divide by the following factors:
            Nbls: baselines should coherently add together
            sqrt(2): noise is split between even and odd
            sqrt(lst_bins): noise power spectrum averages incoherently over time
        """
        thermal_noise_samples = combine_nsamples(self.nsample_1_array,
                                                 self.nsample_2_array)
        Tsys = 180. * units.K * np.power(self.freqs / (.18 * units.GHz), -2.55)
        Tsys += trcvr.to('K')
        thermal_power = (Tsys.to('mK')**2
                         / (self.inttime.to('s') * thermal_noise_samples
                            * self.npols * self.Nbls
                            * np.sqrt(2 * self.lst_bins)))

        thermal_power = thermal_power * self.X2Y
        # This normalization of the thermal power comes from
        # Parsons PSA32 paper appendix B
        thermal_power *= self.beam_area**2 / self.beam_sq_area

        self.thermal_power = thermal_power.to('mK ^2 Mpc ^3')
