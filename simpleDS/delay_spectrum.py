# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Matthew Kolopanis
# Licensed under the 3-clause BSD License
"""Calculate Delay Spectrum from pyuvdata object."""
from __future__ import print_function, absolute_import, division

import os
import sys
import copy
import numpy as np
import collections
import warnings
from pyuvdata import UVData, UVBase, utils as uvutils
import astropy.units as units
from astropy.units import Quantity
from astropy import constants as const
from scipy.signal import windows
from . import utils, cosmo as simple_cosmo
from .parameter import UnitParameter


class DelaySpectrum(UVBase):
    """A Delay Spectrum object to hold relevant data."""

    @units.quantity_input(trcvr=units.K)
    def __init__(self, uv=None, uvb=None, trcvr=None, taper=None):
        """Initialize the Delay Spectrum Object.

        If only one UVData Object is specified, data is multiplied by itself.

        Arguments
            uv1: One of the pyuvdata objects to cross correlate.
                 Optional, can be added later
                 If given assumes all baselines in UVData object will be
                 cross multiplied togeter.
            uv2: Other pyuvdata object to multiply with uv1.
                 Optional, can be added later
                 Assunmes baselines are identical to uv1.
            uvb: UVBeam object with relevent beam info.
                 Currently assumes 1 beam object can describe all baselines
                 Must be power beam in healpix coordinates and peak normalized
                 Optional can be added later.
            trcvr: Receiver Temperature of antenna to calculate noise power
                   Must be an astropy Quantity object with units of temperature.
                   Optional can be added later.
            taper: Spectral taper function used during frequency Fourier Transforms
                    Accepts scipy.signal.windows functions or any function
                    whose argument is the len(data) and returns a numpy array.
                    Default: scipy.signal.windows.blackmanharris
                    Optional
        """
        # standard angle tolerance: 10 mas in radians.
        # Should perhaps be decreased to 1 mas in the future
        radian_tol = 10 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)
        self._Ntimes = UnitParameter('Ntimes', description='Number of times',
                                     value_not_quantity=True, expected_type=int)

        self._Nbls = UnitParameter('Nbls', description='Number of baselines',
                                   value_not_quantity=True, expected_type=int)
        # self._Nblts = UnitParameter('Nblts', description='Number of baseline-times '
        #                            '(i.e. number of spectra). Not necessarily '
        #                            'equal to Nbls * Ntimes', expected_type=int)
        desc = ('Number of frequency channels per spectral window')
        self._Nfreqs = UnitParameter('Nfreqs', description=desc,
                                     value_not_quantity=True, expected_type=int)

        self._Npols = UnitParameter('Npols', description='Number of polarizations',
                                    value_not_quantity=True, expected_type=int)

        desc = ('Number of delay channels. Must be equal to (Nfreqs) with '
                'FFT usage. However may differ if a more intricate '
                'Fourier Transform is used.')
        self._Ndelays = UnitParameter('Ndelays', description=desc,
                                      value_not_quantity=True, expected_type=int)

        desc = ('Number of UVData objects which have been read. '
                'Only a maximum of 2 UVData objects can be read into a single '
                'DelaySpectrum Object. Only 1 UVData must be ready to enable '
                'delay transformation and power spectrum estimation.')
        self._Nuv = UnitParameter('Nuv', description=desc, expected_type=int,
                                  acceptable_vals=[0, 1, 2], value=0,
                                  value_not_quantity=True)

        dec = ('The number of spectral windows over which the Delay Transform '
               'is performed. All spectral windows must be the same size.')
        self._Nspws = UnitParameter('Nspws', description=desc,
                                    expected_type=int, value_not_quantity=True)
        # Fourier domain information
        desc = ('String indicating which domain the data is in. '
                'Allowed values are "delay", "frequency"')
        self._data_type = UnitParameter('data_type', form='str', expected_type=str,
                                        value_not_quantity=True,
                                        description=desc, value='frequency',
                                        acceptable_vals=['delay', 'frequency'])

        desc = ('Array of the visibility data, shape: (Nspws, Nuv, Npols, Nbls, Ntimes, '
                'Nfreqs), type = complex float, in units of self.vis_units')
        self._data_array = UnitParameter('data_array', description=desc,
                                         form=('Nspws', 'Nuv', 'Npols',
                                               'Nbls', 'Ntimes', 'Nfreqs'),
                                         expected_type=(np.complex, np.complex128))

        desc = ('Array of the simulation noise visibility data, '
                'shape: (Nspws, Nuv, Npols, Nbls, Ntimes, Nfreqs), '
                'type = complex float, in units of self.vis_units. '
                'Noise simulation generated assuming the sky is 180K@180MHz '
                'relation with the input receiver temperature, nsample_array, '
                'and integration times.')
        self._noise_array = UnitParameter('noise_array', description=desc,
                                          form=('Nspws', 'Nuv', 'Npols',
                                                'Nbls', 'Ntimes', 'Nfreqs'),
                                          expected_type=(np.complex, np.complex128))

        desc = 'Visibility units, options are: "uncalib", "Jy" or "K str"'
        self._vis_units = UnitParameter('vis_units', description=desc,
                                        value_not_quantity=True,
                                        form='str', expected_type=str,
                                        acceptable_vals=["uncalib", "Jy", "K str"])

        desc = ('Number of data points averaged into each data elementself. '
                'Uses the same convention as a UVData object:'
                'NOT required to be an integer, type = float, same shape as data_array.'
                'The product of the integration_time and the nsample_array '
                'value for a visibility reflects the total amount of time '
                'that went into the visibility.')
        self._nsample_array = UnitParameter('nsample_array', description=desc,
                                            value_not_quantity=True,
                                            form=('Nspws', 'Nuv', 'Npols',
                                                  'Nbls', 'Ntimes', 'Nfreqs'),
                                            expected_type=(np.float))

        desc = ('Boolean flag, True is flagged, shape: same as data_array.')
        self._flag_array = UnitParameter('flag_array', description=desc,
                                         value_not_quantity=True,
                                         form=('Nspws', 'Nuv', 'Npols',
                                               'Nbls', 'Ntimes', 'Nfreqs'),
                                         expected_type=np.bool)

        desc = ('Array of lsts, center of integration, shape (Ntimes), '
                'UVData objects must be LST aligned before adding data '
                'to the DelaySpectrum object.'
                'units radians')
        self._lst_array = UnitParameter('lst_array', description=desc,
                                        form=('Ntimes',),
                                        expected_type=np.float,
                                        tols=radian_tol)

        desc = ('Array of first antenna indices, shape (Nbls), '
                'type = int, 0 indexed')
        self._ant_1_array = UnitParameter('ant_1_array', description=desc,
                                          value_not_quantity=True,
                                          expected_type=int, form=('Nbls',))

        desc = ('Array of second antenna indices, shape (Nbls), '
                'type = int, 0 indexed')
        self._ant_2_array = UnitParameter('ant_2_array', description=desc,
                                          value_not_quantity=True,
                                          expected_type=int, form=('Nbls',))

        desc = ('Array of baseline indices, shape (Nbls), '
                'type = int; baseline = 2048 * (ant1+1) + (ant2+1) + 2^16')
        self._baseline_array = UnitParameter('baseline_array',
                                             description=desc,
                                             value_not_quantity=True,
                                             expected_type=int, form=('Nbls',))

        desc = 'Array of frequencies, shape (Nspws, Nfreqs), units Hz'
        self._freq_array = UnitParameter('freq_array', description=desc,
                                         form=('Nspws', 'Nfreqs'),
                                         expected_type=np.float,
                                         tols=1e-3)

        dest = 'Array of delay, shape (Ndelays), units ns'
        self._delay_array = UnitParameter('delay_array', description=desc,
                                          form=('Ndelays',),
                                          expected_type=np.float,
                                          tols=self._freq_array.tols)

        desc = ('Array of polarization integers, shape (Npols). '
                'Uses same convention as pyuvdata: '
                'pseudo-stokes 1:4 (pI, pQ, pU, pV);  '
                'circular -1:-4 (RR, LL, RL, LR); linear -5:-8 (XX, YY, XY, YX).')
        self._polarization_array = UnitParameter('polarization_array',
                                                 description=desc,
                                                 value_not_quantity=True,
                                                 expected_type=int,
                                                 acceptable_vals=list(np.arange(-8, 0)) + list(np.arange(1, 5)),
                                                 form=('Npols',))
        desc = ('Nominal (u,v,w) vector of baselines in units of meters')
        self._uvw = UnitParameter('uvw', description=desc,
                                  expected_type=np.float,
                                  form=(3))

        desc = ('System receiver temperature used in noise simulation. '
                'Stored as array of length (Nfreqs), but may be passed as a '
                'single scalar. Must be a Quantity with units compatible to K.')
        self._trcvr = UnitParameter('trcvr', description=desc,
                                    expected_type=np.float, form=('Nspws', 'Nfreqs'))

        desc = ('Mean redshift of given frequencies. Calculated with assumed '
                'cosmology.')
        self._redshift = UnitParameter('redshift', description=desc,
                                       expected_type=np.float, form=('Nspws',))

        desc = ('Cosmological wavenumber of spatial modes probed perpendicular '
                ' to the line of sight.')
        self._k_perpendicular = UnitParameter('k_perpendicular',
                                              description=desc,
                                              expected_type=np.float, form=('Nspws',))

        desc = ('Cosmological wavenumber of spatial modes probed along the line of sight. '
                'This value is awlays calculated, however it is not a realistic '
                'probe of k_parallel over large bandwidths. This code '
                'assumes k_tau >> k_perpendicular and as a results '
                'k_tau  is interpreted as k_parallel.')
        self._k_parallel = UnitParameter('k_parallel', description=desc,
                                         expected_type=np.float, form=('Nspws', 'Ndelays'))

        desc = ('The cross-multiplied power spectrum estimates. '
                'Units are converted to cosmological frame (mK^2/(hMpc^-1)^3).')
        self._power_array = UnitParameter('power_array', description=desc,
                                          expected_type=np.complex, required=False,
                                          form=('Nspws', 'Npols', 'Nbls', 'Nbls',
                                                'Ntimes', 'Ndelays'))
        desc = ('The predicted thermal sensitivity for input data assuming all '
                'data will be averaged together.'
                'Units are converted to cosmological frame (mK^2/(hMpc^-1)^3).')
        self._power_array = UnitParameter('power_array', description=desc,
                                          expected_type=np.complex, required=False,
                                          form=('Nspws', 'Npols', 'Nbls', 'Nbls',
                                                'Ntimes', 'Ndelays'))

        desc = ('The cross-multiplied simulated noise power spectrum estimates. '
                'Units are converted to cosmological frame (mK^2/(hMpc^-1)^3).')
        self._noise_power = UnitParameter('noise_power', description=desc,
                                          expected_type=np.complex, required=False,
                                          form=('Nspws', 'Npols', 'Nbls', 'Nbls',
                                                'Ntimes', 'Ndelays'))

        desc = ('The cosmological unit conversion factor applied to the data. '
                'This factor does not include andy Jansky to Kelvin-steradian '
                'factors. Only used for conversion from (k*sr)^2 to mk^2*[h/Mpc]^-3')
        self._unit_conversion = UnitParameter('unit_conversion',
                                              description=desc, required=False,
                                              expected_type=np.float,
                                              form=('Nspws', 'Npols', 'Ndelays'))

        desc = ('The integral of the power beam area. Shape = (Nspws, Npols, Nfreqs)')
        self._beam_area = UnitParameter('beam_area', description=desc,
                                        form=('Nspws', 'Npols', 'Nfreqs'),
                                        expected_type=np.float)
        desc = ('The integral of the squared power beam squared area. '
                'Shape = (Nspws, Npols, Nfreqs)')
        self._beam_sq_area = UnitParameter('beam_sq_area', description=desc,
                                           form=('Nspws', 'Npols', 'Nfreqs'),
                                           expected_type=np.float)
        desc = ('Length of the integration in seconds, has shape '
                '(1, Npols, Nbls, Ntimes, 1). units s, assumes inegration time '
                ' is the same for all spectral windows and all frequncies in a '
                'spectral window. '
                'Assumes the same convention as pyuvdata, where this is the '
                'target amount of time a measurement is integrated over. '
                'Spectral window dimension allows for frequency dependent '
                'filtering to be properly tracked for noise simulations.')
        self._integration_time = UnitParameter('integration_time',
                                               description=desc,
                                               form=(1, 'Npols',
                                                     'Nbls', 'Ntimes', 1),
                                               expected_type=np.float)

        desc = ('Spectral taper function used during Fourier Transform. Functions like scipy.signal.windows.blackmanharris')
        self._taper = UnitParameter('taper', description=desc,
                                    form=(), expected_type=collections.Callable,
                                    value=windows.blackmanharris,
                                    value_not_quantity=True)

        super(DelaySpectrum, self).__init__()

        if uv is not None:
            if not isinstance(uv, (list, np.ndarray, tuple)):
                uv = [uv]
            for _uv in uv:
                self.add_uvdata(_uv)

        if uvb is not None:
            if self.Nuv == 0:
                raise ValueError("Please Load data before attaching a UVBeam.")
            self.add_uvbeam(uvb)

        if trcvr is not None:
            self.add_trcvr(trcvr=trcvr)

        if taper is not None:
            self.set_taper(taper=taper)

    def set_taper(self, taper=None):
        """Set spectral taper function used during Fourier Transform.

        Raises:
            ValueError: Input spectral taper must be a function or callable whose arguments are the length of the band over which Fourier Transform is taken.

        Arguments:
            taper: type callable;
                   must be a function or callable whose arguments are a scalar N. This is interpreted as the length of the spectral window to be returned. Examples scipy.signal.windows.blackmanharris
        """
        if not callable(taper):
            raise ValueError("Input spectral taper must be a function or "
                             "callable whose arguments are "
                             "the length of the band "
                             "over which Fourier Transform is taken.")
        else:
            self.taper = taper

    def set_delay(self):
        """Set the data type to delay."""
        consistent_units = [units.Jy * units.Hz, units.K * units.sr * units.Hz,
                            units.dimensionless_unscaled * units.Hz]
        if not any([self.data_array.unit.is_equivalent(u) for u in consistent_units]):
            raise units.UnitConversionError("Data is not in units consistent "
                                            "with the delay domain. "
                                            "Cannot set data_type to delay.")
        else:
            self.data_type = 'delay'

    def set_frequency(self):
        """Set the data type to frequency."""
        consistent_units = [units.Jy, units.K * units.sr,
                            units.dimensionless_unscaled]
        if not any([self.data_array.unit.is_equivalent(u) for u in consistent_units]):
            raise units.UnitConversionError("Data is not in units consistent "
                                            "with the frequency domain. "
                                            "Cannot set data_type to frequency.")
        else:
            self.data_type = 'frequency'

    def check(self, check_extra=True, run_check_acceptability=True):
        """Add some extra checks on top of checks on UVBase class.

        Check that required parameters exist. Check that parameters have
        appropriate shapes and optionally that the values are acceptable.

        Args:
            check_extra: If true, check all parameters, otherwise only check
                required parameters.
            run_check_acceptability: Option to check if values in parameters
                are acceptable. Default is True.
        """
        if self.data_type == 'delay':
            self.set_delay()
        else:
            self.set_frequency()

        if self.Nbls != len(np.unique(self.baseline_array)):
            raise ValueError('Nbls must be equal to the number of unique '
                             'baselines in the data_array')

        if self.Ntimes != len(np.unique(self.lst_array)):
            raise ValueError('Ntimes must be equal to the number of unique '
                             'times in the lst_array')

        if check_extra:
            p_check = [p for p in self.required()] + [p for p in self.extra()]
        else:
            p_check = [p for p in self.required()]

        for p in p_check:
            param = getattr(self, p)
            # Check required parameter exists
            if param.value is None:
                if param.required is True:
                    raise ValueError('Required UnitParameter ' + p
                                     + ' has not been set.')
            else:
                # Check parameter shape
                eshape = param.expected_shape(self)
                # default value of eshape is ()
                if eshape == 'str' or (eshape == () and param.expected_type == 'str'):
                    # Check that it's a string
                    if not isinstance(param.value, str):
                        raise ValueError('UnitParameter ' + p + ' expected to be '
                                         'string, but is not')
                else:
                    # Check the shape of the parameter value. Note that np.shape
                    # returns an empty tuple for single numbers. eshape should do the same.
                    if not np.shape(param.value) == eshape:
                        raise ValueError('UnitParameter {param} is not expected shape. '
                                         'Parameter shape is {pshape}, expected shape is '
                                         '{eshape}.'.format(param=p, pshape=np.shape(param.value),
                                                            eshape=eshape))
                    if eshape == ():
                        # Single element
                        if not isinstance(param.value, param.expected_type):
                            raise ValueError('UnitParameter ' + p + ' is not the appropriate'
                                             ' type. Is: ' + str(type(param.value))
                                             + '. Should be: ' + str(param.expected_type))
                    else:
                        # UnitParameters cannot have list entries
                        # Array
                        if isinstance(param.value, list):
                            raise ValueError('UnitParameter ' + p + ' is a list. '
                                             'UnitParameters are incompatible with lists')
                        if isinstance(param.value, units.Quantity):
                            if not isinstance(param.value.value.item(0), param.expected_type):
                                raise ValueError('UnitParameter ' + p + ' is not the appropriate'
                                                 ' type. Is: ' + str(param.value.dtype)
                                                 + '. Should be: ' + str(param.expected_type))
                        else:
                            if not isinstance(param.value.item(0), param.expected_type):
                                raise ValueError('UnitParameter ' + p + ' is not the appropriate'
                                                 ' type. Is: ' + str(param.value.dtype)
                                                 + '. Should be: ' + str(param.expected_type))

                if run_check_acceptability:
                    accept, message = param.check_acceptability()
                    if not accept:
                        raise ValueError('UnitParameter ' + p + ' has unacceptable values. '
                                         + message)
        return True

    def add_uvdata(self, uv, spectral_windows=None):
        """Add the relevant uvdata object data to DelaySpectrum object.

        Unloads the data, flags, and nsamples arrays from the input UVData
        object (or subclass) into local storage.

        Raises :
            ValueError:
                        Input data object must be an instance or subclass of UVData
                        A DelaySpectrum object can only perform a Fourier Transform along a single baseline vector. Downselect the input UVData object to only have one redundant baseline type.

        Arguments:
            uv  : A UVData object or subclass of UVData to add to the existing
                  datasets
        """
        if not isinstance(uv, UVData):
            raise ValueError('Input data object must be an instance or '
                             'subclass of UVData.')

        red_groups, uvw_centers, lengths, conjugates = uv.get_baseline_redundancies()
        if len(red_groups) > 1:
            raise ValueError('A DelaySpectrum object can only perform a Fourier '
                             'Transform along a single baseline vector. '
                             'Downselect the input UVData object to only have '
                             'one redundant baseline type.')

        this = DelaySpectrum()
        this.Ntimes = uv.Ntimes
        this.Nbls = uv.Nbls
        this.Nfreqs = uv.Nfreqs
        this.vis_units = uv.vis_units
        this.Npols = uv.Npols
        this.Nspws = 1
        this.Nuv = 1
        this.lst_array = np.unique(uv.lst_array)
        this.polarization_array = uv.polarization_array
        if this.vis_units == 'Jy':
            data_unit = units.Jy
        elif this.vis_units == 'K str':
            data_unit = units.K * units.sr
        else:
            data_unit = units.dimensionless_unscaled

        this.freq_array = uv.freq_array * units.Hz
        this._freq_array.tols = (this._freq_array.tols[0], this._freq_array.tols[1] * units.Hz)

        this.baseline_array = uv.get_baseline_nums()
        this.ant_1_array, this.ant_2_array = np.transpose(uv.get_antpairs(), [1, 0])
        temp_data = np.zeros(shape=this._data_array.expected_shape(this),
                             dtype=np.complex128)
        temp_data[:, :, :, :, :] = utils.get_data_array(uv, reds=this.baseline_array, squeeze=False)
        this.data_array = copy.deepcopy(temp_data) * data_unit
        this._data_array.tols = (this._data_array.tols[0], this._data_array.tols[1] * data_unit)

        temp_data = np.zeros(shape=this._nsample_array.expected_shape(this),
                             dtype=np.float)
        temp_data[:, :, :, :, :] = utils.get_nsample_array(uv, reds=this.baseline_array, squeeze=False)
        this.nsample_array = copy.deepcopy(temp_data)

        temp_data = np.ones(shape=this._flag_array.expected_shape(this),
                            dtype=np.bool)
        temp_data[:, :, :, :, :] = utils.get_flag_array(uv, reds=this.baseline_array, squeeze=False)
        this.flag_array = copy.deepcopy(temp_data)

        temp_data = np.zeros(shape=this._integration_time.expected_shape(this),
                             dtype=np.float)
        temp_data[:, :, :, :, :] = utils.get_integration_time(uv, reds=this.baseline_array, squeeze=False)

        this.integration_time = copy.deepcopy(temp_data) * units.s
        # initialize the beam_area and beam_sq_area to help with selections later
        this.beam_area = np.ones(this._beam_area.expected_shape(this)) * np.inf * units.sr
        this.beam_sq_area = np.ones(this._beam_sq_area.expected_shape(this)) * np.inf * units.sr**2

        this.trcvr = np.ones(shape=this._trcvr.expected_shape(this)) * np.inf * units.K

        this.uvw = uvw_centers[0] * units.m
        this._uvw.tols = (this._uvw.tols[0], this._uvw.tols[1] * units.m)

        this.generate_noise()

        if this.data_array.unit == units.K * units.sr:
            # if the visibilities are in K steradian then the noise should be too.
            # reshape to have form of Nspws, Nuv, Npols, Nbls, Ntimes, Nfreqs
            this.noise_array = this.noise_array * utils.jy_to_mk(this.freq_array).reshape(this.Nspws, 1, 1, 1, 1, this.Nfreqs)
        elif this.data_array.unit == units.dimensionless_unscaled:
            warnings.warn("Data is uncalibrated. Unable to covert noise array "
                          "to unicalibrated units.", UserWarning)

        if self.freq_array is not None:
            this.select_spectral_windows(freqs=self.freq_array)
        else:
            this.select_spectral_windows(spectral_windows=spectral_windows)
        this._delay_array.tols = (this._delay_array.tols[0],
                                  this._delay_array.tols[1] * this.delay_array.unit)

        # Make sure both objects are self consistent before trying to join them
        # self.check(check_extra=True, run_check_acceptability=True)
        this.check(check_extra=False, run_check_acceptability=True)

        # check for compatibility of all parameters before adding UVData data
        # to object. Parameters are compatible if all parameters are equal
        # excluding the data_array, flag_array, and nsample_array
        # want to run checks and error before any data is written
        if self.data_array is not None and not self.data_array.unit.is_equivalent(this.data_array.unit):
            errmsg = ("Input data object is in units "
                      "incompatible with saved DelaySpectrum units."
                      "Saved units are: {dspec}, "
                      "input units are: {uvin}."
                      .format(dspec=self.data_array.unit,
                              uvin=this.data_array.unit))
            raise units.UnitConversionError(errmsg)

        if self.Nuv == 2:
            raise ValueError('This DelaySpectrum Object has already '
                             'been loaded with two datasets. Create '
                             'a new object to cross-multipy different '
                             'data.')

        for p in self:
            my_parm = getattr(self, p)
            other_parm = getattr(this, p)
            if p not in ['_data_array', '_flag_array', '_nsample_array',
                         '_noise_array', '_Nuv', '_beam_area', '_beam_sq_area',
                         '_trcvr', '_taper', '_lst_array']:
                if my_parm.value is not None and my_parm != other_parm:
                    raise ValueError("Input data differs from previously "
                                     "loaded data. Parameter {name} is not "
                                     "the same.".format(name=p))
            elif p in ['_lst_array']:
                if my_parm.value is not None and my_parm != other_parm:
                    time_diff = np.abs(my_parm.value - other_parm.value) * 12. * units.h / np.pi
                    warnings.warn("Input LST arrays differ on average "
                                  "by {time:}. Keeping LST array stored from "
                                  "the first data set read."
                                  .format(time=time_diff.mean().to('min')),
                                  UserWarning)
            elif p in ['_beam_area, _beam_sq_area', '_trcvr']:
                if my_parm.value is not None and my_parm != other_parm:
                    if np.isfinite(my_parm.value.value).all() and np.isfinite(other_parm.value.value).all():
                        raise ValueError("Input data differs from previously "
                                         "loaded data. Parameter {name} is not "
                                         "the same".format(name=p))

        # Increment by one the number of read uvdata objects
        self._Nuv.value += 1
        for p in self:
            my_parm = getattr(self, p)
            if p not in ['_data_array', '_flag_array', '_nsample_array',
                         '_noise_array', '_Nuv', '_beam_area', '_beam_sq_area',
                         '_trcvr', '_taper']:
                if my_parm.value is None:
                    parm = getattr(this, p)
                    setattr(self, p, parm)
            elif p in ['_beam_area', '_beam_sq_area', '_trcvr']:
                if my_parm.value is None:
                    parm = getattr(this, p)
                    setattr(self, p, parm)
                elif np.isinf(my_parm.value.value).all():
                    parm = getattr(this, p)
                    setattr(self, p, parm)
            elif p in ['_data_array', '_flag_array',
                       '_nsample_array', '_noise_array']:
                parm = getattr(this, p)
                if my_parm.value is not None:
                    tmp_data = np.zeros(my_parm.expected_shape(self),
                                        dtype=my_parm.expected_type)
                    tmp_data[:, :self.Nuv - 1] = my_parm.value[:]

                    if isinstance(my_parm.value, units.Quantity):
                        tmp_data = tmp_data * my_parm.value.unit

                    my_parm.value = tmp_data
                    my_parm.value[:, self.Nuv - 1] = parm.value[0]
                    setattr(self, p, my_parm)
                else:
                    setattr(self, p, parm)
        self.check(check_extra=False, run_check_acceptability=True)

    def select_spectral_windows(self, spectral_windows=None, freqs=None, inplace=True):
        """Select the spectral windows from loaded data.

        Raises:
            ValueError:
                    Spectral window tuples must only have two elements (stard_index, end_index)
                    Spectra windows must all have the same size
        Arguments:
            spectral_windows: tuple of tuples, or tuple of indices, or list of lists, or list of indices; Default selection is (0, Nfreqs)
                              spectral windows ranges like (start_index, end_index) where the indices are the frequency channel numbers.
            inplace: Bool; Default True
                     choose whether spectral window selection is done inplcae on the object, or a new object is returned.
        """
        if inplace:
            this = self
        else:
            this = copy.deepcopy(self)

        if freqs is not None:
            freq_arr_use = this.freq_array[:, :]
            spectral_windows = []
            for _fq in freqs:
                sp = []
                for _f in _fq:
                    for _fq_array in freq_arr_use:
                        if _f in _fq_array:
                            sp.extend(np.where(_f == _fq_array)[0])
                        else:
                            raise ValueError("Frequency {f} not found in "
                                             "frequency array".format(_f.to('Mhz')))
                spectral_windows.append(sp)
            spectral_windows = [[sp[0], sp[-1]] for sp in spectral_windows]
        if spectral_windows is None:
            spectral_windows = [(0, this.Nfreqs - 1)]

        spectral_windows = uvutils._get_iterable(spectral_windows)
        if not isinstance(spectral_windows[0], (list, tuple, np.ndarray)):
            spectral_windows = [spectral_windows]

        if not all(len(x) == 2 for x in spectral_windows):
            raise ValueError("Spectral window tuples must only have two "
                             "elements (stard_index, end_index)")

        spectral_windows = np.asarray(spectral_windows)
        Nspws = np.shape(spectral_windows)[0]
        # check all windows are the same length
        # iteratively select and cut data
        # or possibly slice the array and reshape?
        num_freqs = spectral_windows[:, 1] + 1 - spectral_windows[:, 0]
        if not all(num_freqs == num_freqs[0]):
            raise ValueError("Spectral windows must all have the same size.")
        freq_chans = spectral_windows[:, 0, None] + np.arange(num_freqs[0])
        this.Nspws = Nspws
        this.Nfreqs = np.int(num_freqs[0])
        this.freq_array = np.take(this.freq_array, freq_chans)
        this.trcvr = np.take(this.trcvr, freq_chans)

        freq_chans = freq_chans.flatten()
        # For beam_area and beam_sq_area transpose to (Npols, Nspws, Nfreqs)
        # Take flattened freqs, then reshape tp correct shapes and transpose back
        beam_sq_area = np.transpose(this.beam_sq_area, [1, 0, 2])
        beam_sq_area = np.take(beam_sq_area, freq_chans)
        beam_sq_area = beam_sq_area.reshape(this.Npols, this.Nspws, this.Nfreqs)
        this.beam_sq_area = np.transpose(beam_sq_area, [1, 0, 2])

        beam_area = np.transpose(this.beam_area, [1, 0, 2])
        beam_area = np.take(beam_area, freq_chans)
        beam_area = beam_area.reshape(this.Npols, this.Nspws, this.Nfreqs)
        this.beam_area = np.transpose(beam_area, [1, 0, 2])
        # to make take easier, reorder to Nuv, Npols, Nbls, Ntimes, Nspws, Nfreqs
        this.data_array = this._take_spectral_windows_from_data_like_array(this.data_array, freq_chans)
        this.noise_array = this._take_spectral_windows_from_data_like_array(this.noise_array, freq_chans)
        this.nsample_array = this._take_spectral_windows_from_data_like_array(this.nsample_array, freq_chans)
        this.flag_array = this._take_spectral_windows_from_data_like_array(this.flag_array, freq_chans)
        this.integration_time = this.integration_time.reshape(this._integration_time.expected_shape(this))
        # this.integration_time = np.tile(this.integration_time, (Nspws, 1, 1), subok=True)
        # this.check(check_extra=True, run_check_acceptability=True)
        # This seems obvious for an FFT but in the case that something more
        # sophisticated is added later this hook will exist.
        this.Ndelays = np.int(this.Nfreqs)
        delays = np.fft.fftfreq(this.Ndelays,
                                d=np.diff(this.freq_array[0])[0].value)
        delays = np.fft.fftshift(delays) / this.freq_array.unit
        this.delay_array = delays.to('ns')

        this.update_cosmology()

        if not inplace:
            return this
        else:
            return

    def _take_spectral_windows_from_data_like_array(self, data, inds):
        """Reshape and take spectral windows along the frequency axis.

        This take is only for arrays of shape like self.data_array
        Arguments:
                data: array of shape self.data_array to be selected
                inds: flattened indices of frequency array to select
        """
        data = np.transpose(data, [1, 2, 3, 4, 0, 5])
        data = np.take(data, inds, axis=5)
        # reshape to get Nspws, Nfreqs correct along 2 last dimensions
        data = data.reshape(self.Nuv, self.Npols, self.Nbls,
                            self.Ntimes, self.Nspws, self.Nfreqs)
        #  reorder to Nsps, Nuv, Npos, Nbls, Ntimes, Nfreqs
        data = np.transpose(data, [4, 0, 1, 2, 3, 5])
        return data

    def generate_noise(self):
        """Simulate noise based on meta-data of observation."""
        noise_power = self.calculate_noise_power()
        self.noise_array = utils.generate_noise(noise_power)

    def update_cosmology(self, cosmo=None):
        """Update cosmological information with the assumed cosmology.

        Arguments:
            cosmo: input assumed cosmology. Must be an astropy cosmology object.
        """
        # find the mean redshift for each spectral window
        self.redshift = simple_cosmo.calc_z(self.freq_array).mean(axis=1)
        self.k_parallel = simple_cosmo.eta2kparr(self.delay_array.reshape(1, self.Ndelays),
                                                 self.redshift.reshape(self.Nspws, 1), cosmo=cosmo)

        uvw_wave = np.linalg.norm(self.uvw.value) * self.uvw.unit
        mean_freq = np.mean(self.freq_array.value, axis=1) * self.freq_array.unit
        uvw_wave = uvw_wave / (const.c / mean_freq.to('1/s')).to('m')
        self.k_perpendicular = simple_cosmo.u2kperp(uvw_wave, self.redshift,
                                                    cosmo=cosmo)

    def add_uvbeam(self, uvb, no_read_trcvr=False):
        """Add the beam_area and beam_square_area integrals into memory.

        Also adds receiver temperature information if set in UVBeam object.
        Arguments:
            uvb: UVBeam object with relevent beam info.
                 Currently assumes 1 beam object can describe all baselines
                 Must be a power beam in healpix coordinates and peak normalized
            no_read_trcvr: (Bool, default False)
                           Flag to Not read trcvr from UVBeam object even if set in UVBeam.
                           This is useful if a trcvr wants to be manually set but
                           a beam read from a file which also contains receiver temperature information.
        """
        for spw, freqs in enumerate(self.freq_array):
            _beam = uvb.select(frequencies=freqs.to('Hz').value, inplace=False)
            for pol_cnt, pol in enumerate(self.polarization_array):
                self.beam_area[spw, pol_cnt, :] = _beam.get_beam_area(pol=pol) * units.sr
                self.beam_sq_area[spw, pol_cnt, :] = _beam.get_beam_sq_area(pol=pol) * units.sr**2

                if _beam.receiver_temperature_array is not None and not no_read_trcvr:
                    self.trcvr[spw, :] = _beam.receiver_temperature_array[0] * units.K

    @units.quantity_input(trcvr=units.K)
    def add_trcvr(self, trcvr):
        """Add the receiver temperature used to generate noise simulation.

        Arguments:
                trcvr: (astropy Quantity, units: Kelvin)
                       (Nspws, Nfreqs) array of temperatures
                       if a single temperature, it is assumed to be constant at all frequencies.
        """
        if trcvr.size == 1:
            self.trcvr = np.ones(shape=self._trcvr.expected_shape(self)) * trcvr
        elif trcvr.shape[0] != self.Nspws or trcvr.shape[1] != self.Nfreqs:
            raise ValueError("If input receiver temperature is not a scalar "
                             "Quantity, must shape (Nspws, Nfreqs). "
                             "Expected shape was {s1}, but input shape "
                             "was {s2}".format(s1=(self.Nspws, self.Nfreqs),
                                               s2=trcvr.shape))
        else:
            self.trcvr = trcvr

    def delay_transform(self):
        """Perform a delay transform on the stored data array.

        If data is set to frequency domain, fourier transforms to delay space.
        If data is set to delay domain, inverse fourier transform to frequency space.
        """
        if self.data_array.unit == units.dimensionless_unscaled:
            warnings.warn("Fourier Transforming uncalibrated data. Units will "
                          "not have physical meaning. "
                          "Data will be arbitrarily scaled.", UserWarning)
        if self.data_type == 'frequency':
            self.normalized_fourier_transform()
            self.set_delay()
        elif self.data_type == 'delay':
            self.normalized_fourier_transform(inverse=True)
            self.set_frequency()
        else:
            raise ValueError('Unknown data type: {dt}. Unable to perform '
                             'delay transformation.'.format(dt=self.data_type))

    def normalized_fourier_transform(self, inverse=False):
        """Perform a normalized Fourier Transform along frequency dimension.

        Local wrapper for function normalized_fourier_transform.
        Uses astropy quantities to properly normalize an FFT accounting for the Volume factor and units.

        Arguments:
                inverse: (bool; default False)
                         perform the inverse Fourier Transform with np.fft.ifft
        """
        if inverse is True:
            delta_x = np.diff(self.delay_array)[0]
        else:
            delta_x = np.diff(self.freq_array[0])[0]
        float_flags = np.logical_not(self.flag_array).astype(float)
        self.data_array = utils.normalized_fourier_transform((self.data_array
                                                              * float_flags),
                                                             delta_x=delta_x,
                                                             axis=-1,
                                                             taper=self.taper,
                                                             inverse=inverse)
        self.noise_array = utils.normalized_fourier_transform((self.noise_array
                                                               * float_flags),
                                                              delta_x=delta_x,
                                                              axis=-1,
                                                              taper=self.taper,
                                                              inverse=inverse)

    def calculate_noise_power(self):
        """Use the radiometry equation to generate the expected noise power."""
        Tsys = 180. * units.K * np.power(self.freq_array.to('GHz') / (.18 * units.GHz), -2.55)
        Tsys += self.trcvr.to('K')
        Tsys = Tsys.reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs)
        delta_f = np.diff(self.freq_array[0])[0]
        # if any of the polarizations are psuedo-stokes then there Should
        # be a factor of 1/sqrt(2) factor divided as per Cheng 2018
        npols_noise = np.array([2 if p in np.arange(1, 5) else 1
                                for p in self.polarization_array])
        npols_noise = npols_noise.reshape(1, 1, self.Npols, 1, 1, 1)
        # when there are 0's of infs in the trcvr or integration_time arrays
        # we would notmally get an error/warning form numpy but we can supress
        # those for now.
        with np.errstate(divide='ignore', invalid='ignore'):
            noise_power = Tsys.to('K') / np.sqrt(delta_f.to('1/s')
                                                 * self.integration_time.to('s')
                                                 * self.nsample_array
                                                 * npols_noise)
        # Want to put noise into Jy units
        # the normalization of noise is defined to have this 1/beam_integral
        # factor in temperature units, so we need to multiply it get them into
        # Janskys
        noise_power = np.ma.masked_invalid(noise_power.to('mK')
                                           * self.beam_area.reshape(self.Nspws, 1, self.Npols, 1, 1, self.Nfreqs)
                                           / utils.jy_to_mk(self.freq_array).reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs))
        noise_power = noise_power.filled(0)
        return noise_power.to('Jy')

    def calculate_delay_spectrum(self, run_check=True,
                                 run_check_acceptability=True):
        """Perform Delay tranform and cross multiplication of datas.

        Take the normalized Fourier transform of the data in objects and cross multiplies baselines.
        Also generates white noise given the frequency range and trcvr and calculates the expected noise power.
        """
        if self.Nuv == 0:
            raise ValueError("No data has be loaded. Add UVData objects before "
                             "calling calculate_delay_spectrum.")

        NEBW = utils.noise_equivalent_bandwidth(self.taper(self.Nfreqs))
        self.bandwidth = (self.freq_array[0][-1] - self.freq_array[0][0]) * NEBW
        self.unit_conversion = (simple_cosmo.X2Y(self.redshift).reshape(self.Nspws, 1, 1)
                                / self.bandwidth.to('1/s') / self.beam_sq_area)

        if self.data_type == 'delay':
            self.delay_transform()

        if self.data_array.unit.is_equivalent(units.Jy):
            jy_to_ksr = utils.jy_to_mk(self.freq_array)
            # Need to reshape the conversion factor to be broadcastable with
            # the stored data
            self.data_array = self.data_array.copy() * jy_to_ksr.reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs)
            self.noise_array = self.noise_array.copy() * jy_to_ksr.reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs)

        # else:
        #     tmp_data_array = self.data_array.copy()
        #     tmp_noise_aray = self.noise_array.copy()

        self.delay_transform()

        if self.Nuv == 1:
            delay_power = utils.cross_multiply_array(array_1=self.data_array[:, 0],
                                                     axis=2)
            noise_power = utils.cross_multiply_array(array_1=self.noise_array[:, 0],
                                                     axis=2)
        else:
            delay_power = utils.cross_multiply_array(array_1=self.data_array[:, 0],
                                                     array_2=self.data_array[:, 1],
                                                     axis=2)
            noise_power = utils.cross_multiply_array(array_1=self.noise_array[:, 0],
                                                     array_2=self.noise_array[:, 1],
                                                     axis=2)

        self.power_array = delay_power * self.unit_conversion.reshape(self.Nspws, self.Npols, 1, 1, 1, self.Ndelays)
        self.noise_power = noise_power * self.unit_conversion.reshape(self.Nspws, self.Npols, 1, 1, 1, self.Ndelays)
        self.calculate_thermal_sensitivity()

    def calculate_thermal_sensitivity(self):
        """Calculate the Thermal sensitivity for the power spectrum.

        Uses the 21cmsense_calc formula:
            Tsys**2/(inttime * Nbls * Npols * sqrt(N_lstbins * 2))

        Divide by the following factors:
            Nbls: baselines should coherently add together
            sqrt(2): noise is split between even and odd
            sqrt(lst_bins): noise power spectrum averages incoherently over time
        """
        if self.Nuv == 1:
            thermal_noise_samples = utils.combine_nsamples(self.nsample_array[:, 0],
                                                           self.nsample_array[:, 0],
                                                           axis=2)
        else:
            thermal_noise_samples = utils.combine_nsamples(self.nsample_array[:, 0],
                                                           self.nsample_array[:, 1],
                                                           axis=2)
        # lst_array is stored in radians, multiply by 12*3600/np.pi to convert
        # to seconds s
        if self.lst_array.size > 1:
            delta_t = np.diff(self.lst_array)[0] * 12. / np.pi * 3600 * units.s
        else:
            delta_t = self.integration_time.to('s')
        lst_bins = (np.size(self.lst_array) * delta_t / self.integration_time.to('s'))
        npols_noise = np.array([2 if p in np.arange(1, 5) else 1
                                for p in self.polarization_array])
        npols_noise = npols_noise.reshape(1, 1, self.Npols, 1, 1, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            Tsys = 180. * units.K * np.power(self.freq_array / (.18 * units.GHz), -2.55)
            Tsys += self.trcvr.to('K')
            Tsys = Tsys.reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs)
            thermal_power = (Tsys.to('mK')**2
                             / (self.integration_time.to('s') * thermal_noise_samples
                                * npols_noise * self.Nbls
                                * np.sqrt(2 * lst_bins)))

            thermal_power = thermal_power * simple_cosmo.X2Y(self.redshift).reshape(self.Nspws, 1, 1, 1, 1, 1)
            # This normalization of the thermal power comes from
            # Parsons PSA32 paper appendix B
            thermal_power *= (self.beam_area**2 / self.beam_sq_area).reshape(self.Nspws, self.Npols, 1, 1, 1, self.Nfreqs)
        thermal_power = np.ma.masked_invalid(thermal_power)
        self.thermal_power = thermal_power.filled(0).to('mK^2 Mpc^3')
