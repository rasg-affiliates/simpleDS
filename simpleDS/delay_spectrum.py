# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Calculate Delay Spectrum from pyuvdata object."""
import os
import copy
import h5py
import warnings
import numpy as np
from itertools import chain, product

import astropy.units as units
from astropy import constants as const
from astropy.cosmology.core import Cosmology as reference_cosmology_object
from astropy.cosmology.units import littleh, with_H0
from pyuvdata.uvbase import UVBase
from pyuvdata import UVData, utils as uvutils

from scipy.signal import windows
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from . import utils, cosmo as simple_cosmo
from .parameter import UnitParameter

from collections.abc import Callable


class DelaySpectrum(UVBase):
    """A Delay Spectrum object to hold relevant data.

    If only one UVData Object is specified, data is multiplied by itself.

    Parameters
    ----------
    uv1 : pyuvdata object, list of no more than 2 pyuvdata objects, optional
         Objects to cross correlate. Assumes all baselines in UVData object will be
         cross multiplied togeter, required to have only one reduntant group as
         computed with pyuvdata redundancy functions.
         input during initialization is optional, but must be set later with add_uvdata()
    uvb : UVBeam object, optional
        Containts relevent beam info. Currently assumes 1 beam object can describe all baselines
        Must be power beam in healpix coordinates and peak normalized.
        input during initialization is optional, but must be set later with add_uvbeam()
    trcvr :  Astropy Quantity, units: Kelvin, optional
        Receiver Temperature of antenna to calculate noise power
        input during initialization is optional, but must be set later with add_trcvr()
    taper : function or Callable, optional
        Spectral taper function used during frequency Fourier Transforms
        Accepts scipy.signal.windows functions or any function
        whose argument is the len(data) and returns a numpy array.
        Default: scipy.signal.windows.blackmanharris but can be overwritten later set_taper()

    Attributes
    ----------
    UnitParameter Objects
        For a full list of all attributes, types, and expected forms please
        consult the documentation at (http://simpleds.readthedocs.io/en/latest/dspec_parameters.html).

    """

    @units.quantity_input(trcvr=units.K)
    def __init__(self, uv=None, uvb=None, trcvr=None, taper=None):
        """Initialize the object."""
        # standard angle tolerance: 10 mas in radians.
        # Should perhaps be decreased to 1 mas in the future
        radian_tol = 10 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0) * units.rad
        self._Ntimes = UnitParameter(
            "Ntimes",
            description="Number of times",
            value_not_quantity=True,
            expected_type=int,
        )

        self._Nbls = UnitParameter(
            "Nbls",
            description="Number of baselines",
            value_not_quantity=True,
            expected_type=int,
        )

        desc = (
            "Number of antennas with data present (i.e. number of unique"
            "entries in ant_1_array and ant_2_array). May be smaller"
            "than the number of antennas in the array"
        )
        self._Nants_data = UnitParameter(
            "Nants_data", description=desc, expected_type=int, value_not_quantity=True
        )

        desc = (
            "Number of antennas in the array. May be larger "
            "than the number of antennas with data"
        )
        self._Nants_telescope = UnitParameter(
            "Nants_telescope",
            description=desc,
            expected_type=int,
            value_not_quantity=True,
        )

        desc = "Number of frequency channels per spectral window"
        self._Nfreqs = UnitParameter(
            "Nfreqs", description=desc, value_not_quantity=True, expected_type=int
        )

        self._Npols = UnitParameter(
            "Npols",
            description="Number of polarizations",
            value_not_quantity=True,
            expected_type=int,
        )

        desc = (
            "Number of delay channels. Must be equal to (Nfreqs) with "
            "FFT usage. However may differ if a more intricate "
            "Fourier Transform is used."
        )
        self._Ndelays = UnitParameter(
            "Ndelays", description=desc, value_not_quantity=True, expected_type=int
        )

        desc = (
            "Number of UVData objects which have been read. "
            "Only a maximum of 2 UVData objects can be read into a single "
            "DelaySpectrum Object. Only 1 UVData must be ready to enable "
            "delay transformation and power spectrum estimation."
        )
        self._Nuv = UnitParameter(
            "Nuv",
            description=desc,
            expected_type=int,
            acceptable_vals=[0, 1, 2],
            value=0,
            value_not_quantity=True,
        )

        desc = (
            "The number of spectral windows over which the Delay Transform "
            "is performed. All spectral windows must be the same size."
        )
        self._Nspws = UnitParameter(
            "Nspws", description=desc, expected_type=int, value_not_quantity=True
        )
        # Fourier domain information
        desc = (
            "String indicating which domain the data is in. "
            'Allowed values are "delay", "frequency"'
        )
        self._data_type = UnitParameter(
            "data_type",
            form="str",
            expected_type=str,
            value_not_quantity=True,
            description=desc,
            value="frequency",
            acceptable_vals=["delay", "frequency"],
        )

        desc = (
            "Array of the visibility data, shape: (Nspws, Nuv, Npols, Nbls, Ntimes, "
            "Nfreqs), type = complex float, in units of self.vis_units"
        )
        self._data_array = UnitParameter(
            "data_array",
            description=desc,
            form=("Nspws", "Nuv", "Npols", "Nbls", "Ntimes", "Nfreqs"),
            expected_type=np.complex64,
            expected_units=(
                units.Jy,
                units.Jy * units.Hz,
                units.K * units.sr * units.Hz,
                units.K * units.sr,
                units.dimensionless_unscaled,
                units.dimensionless_unscaled * units.Hz,
            ),
        )

        desc = (
            "Array of the simulation noise visibility data, "
            "shape: (Nspws, Nuv, Npols, Nbls, Ntimes, Nfreqs), "
            "type = complex float, in units of self.vis_units. "
            "Noise simulation generated assuming the sky is 180K@180MHz "
            "relation with the input receiver temperature, nsample_array, "
            "and integration times."
        )
        self._noise_array = UnitParameter(
            "noise_array",
            description=desc,
            form=("Nspws", "Nuv", "Npols", "Nbls", "Ntimes", "Nfreqs"),
            expected_type=np.complex64,
            expected_units=(
                units.Jy,
                units.Jy * units.Hz,
                units.K * units.sr * units.Hz,
                units.K * units.sr,
                units.dimensionless_unscaled,
                units.dimensionless_unscaled * units.Hz,
            ),
        )

        desc = 'Visibility units, options are: "uncalib", "Jy" or "K str"'
        self._vis_units = UnitParameter(
            "vis_units",
            description=desc,
            value_not_quantity=True,
            form="str",
            expected_type=str,
            acceptable_vals=["uncalib", "Jy", "K str"],
        )

        desc = (
            "Number of data points averaged into each data elementself. "
            "Uses the same convention as a UVData object:"
            "NOT required to be an integer, type = float, same shape as data_array."
            "The product of the integration_time and the nsample_array "
            "value for a visibility reflects the total amount of time "
            "that went into the visibility."
        )
        self._nsample_array = UnitParameter(
            "nsample_array",
            description=desc,
            value_not_quantity=True,
            form=("Nspws", "Nuv", "Npols", "Nbls", "Ntimes", "Nfreqs"),
            expected_type=np.float64,
        )

        desc = "Boolean flag, True is flagged, shape: same as data_array."
        self._flag_array = UnitParameter(
            "flag_array",
            description=desc,
            value_not_quantity=True,
            form=("Nspws", "Nuv", "Npols", "Nbls", "Ntimes", "Nfreqs"),
            expected_type=bool,
        )

        desc = (
            "Array of lsts, center of integration, shape (Ntimes), "
            "UVData objects must be LST aligned before adding data "
            "to the DelaySpectrum object."
            "units radians"
        )
        self._lst_array = UnitParameter(
            "lst_array",
            description=desc,
            form=("Ntimes",),
            expected_type=np.float64,
            tols=radian_tol,
            expected_units=units.rad,
        )

        desc = "Array of first antenna indices, shape (Nbls), " "type = int, 0 indexed"
        self._ant_1_array = UnitParameter(
            "ant_1_array",
            description=desc,
            value_not_quantity=True,
            expected_type=int,
            form=("Nbls",),
        )

        desc = "Array of second antenna indices, shape (Nbls), " "type = int, 0 indexed"
        self._ant_2_array = UnitParameter(
            "ant_2_array",
            description=desc,
            value_not_quantity=True,
            expected_type=int,
            form=("Nbls",),
        )

        desc = (
            "Array of baseline indices, shape (Nbls), "
            "type = int; baseline = 2048 * (ant1+1) + (ant2+1) + 2^16"
        )
        self._baseline_array = UnitParameter(
            "baseline_array",
            description=desc,
            value_not_quantity=True,
            expected_type=int,
            form=("Nbls",),
        )

        desc = "Array of frequencies, shape (Nspws, Nfreqs), units Hz"
        self._freq_array = UnitParameter(
            "freq_array",
            description=desc,
            form=("Nspws", "Nfreqs"),
            expected_type=np.float64,
            tols=1e-3 * units.Hz,
            expected_units=units.Hz,
        )

        desc = "Array of delay, shape (Ndelays), units ns"
        self._delay_array = UnitParameter(
            "delay_array",
            description=desc,
            form=("Ndelays",),
            expected_type=np.float64,
            tols=1e-3 * units.ns,
            expected_units=units.ns,
        )

        desc = (
            "Array of polarization integers, shape (Npols). "
            "Uses same convention as pyuvdata: "
            "pseudo-stokes 1:4 (pI, pQ, pU, pV);  "
            "circular -1:-4 (RR, LL, RL, LR); linear -5:-8 (XX, YY, XY, YX)."
        )
        self._polarization_array = UnitParameter(
            "polarization_array",
            description=desc,
            value_not_quantity=True,
            expected_type=int,
            acceptable_vals=list(np.arange(-8, 0)) + list(np.arange(1, 5)),
            form=("Npols",),
        )
        desc = "Nominal (u,v,w) vector of baselines in units of meters"
        self._uvw = UnitParameter(
            "uvw",
            description=desc,
            expected_type=np.float64,
            form=(3),
            expected_units=units.m,
        )

        desc = (
            "System receiver temperature used in noise simulation. "
            "Stored as array of length (Nfreqs), but may be passed as a "
            "single scalar. Must be a Quantity with units compatible to K."
        )
        self._trcvr = UnitParameter(
            "trcvr",
            description=desc,
            expected_type=np.float64,
            form=("Nspws", "Nfreqs"),
            expected_units=units.K,
        )

        desc = (
            "Mean redshift of given frequencies. Calculated with assumed " "cosmology."
        )
        self._redshift = UnitParameter(
            "redshift",
            description=desc,
            expected_type=np.float64,
            form=("Nspws",),
            expected_units=units.dimensionless_unscaled,
        )

        _kval_units = (1.0 / units.Mpc, littleh / units.Mpc)

        desc = (
            "Cosmological wavenumber of spatial modes probed perpendicular "
            " to the line of sight. In python 2 this unit is always in 1/Mpc. "
            "For python 3 users, it is possible to convert the littleh/Mpc "
            "using the littleh_units boolean flag in update_cosmology."
        )
        self._k_perpendicular = UnitParameter(
            "k_perpendicular",
            description=desc,
            expected_type=np.float64,
            form=("Nspws",),
            expected_units=_kval_units,
        )

        desc = (
            "Cosmological wavenumber of spatial modes probed along the line of sight. "
            "This value is awlays calculated, however it is not a realistic "
            "probe of k_parallel over large bandwidths. This code "
            "assumes k_tau >> k_perpendicular and as a results "
            "k_tau  is interpreted as k_parallel. "
            "In python 2 this unit is always in 1/Mpc. "
            "For python 3 users, it is possible to convert the littleh/Mpc "
            "using the littleh_units boolean flag in update_cosmology."
        )
        self._k_parallel = UnitParameter(
            "k_parallel",
            description=desc,
            expected_type=np.float64,
            form=("Nspws", "Ndelays"),
            expected_units=_kval_units,
        )
        _power_units = (
            (units.mK**2 * units.Mpc**3),
            (units.mK**2 * (units.Mpc / littleh) ** 3),
            (units.Jy * units.Hz) ** 2,
            (units.Hz) ** 2,
        )

        desc = (
            "The cross-multiplied power spectrum estimates. "
            "Units are converted to cosmological frame (mK^2/(hMpc^-1)^3). "
            "Will have shape ('Nspws', 'Npols', 'Nbls', 'Nbls', 'Ntimes', 'Ndelays').\n"
            "For uncalibrated data the cosmological power is not well defined "
            "the power array instead represents the power in the delay domain "
            "and will have units (Hz^2). "
            "In python 2 this unit is always in mK^2 Mpc^3. "
            "For python 3 users, it is possible to convert the mK^2 / (littleh/Mpc)^3 "
            "using the littleh_units boolean flag in update_cosmology."
        )
        self._power_array = UnitParameter(
            "power_array",
            description=desc,
            expected_type=np.complex64,
            required=False,
            form=("Nspws", "Npols", "Nbls", "Nbls", "Ntimes", "Ndelays"),
            expected_units=_power_units,
        )
        _noise_power_units = (
            (units.mK**2 * units.Mpc**3),
            (units.mK**2 * (units.Mpc / littleh) ** 3),
            (units.Jy * units.Hz) ** 2,
        )
        desc = (
            "The cross-multiplied simulated noise power spectrum estimates. "
            "Units are converted to cosmological frame (mK^2/(hMpc^-1)^3)."
            "Will have shape ('Nspws', 'Npols', 'Nbls', 'Nbls', 'Ntimes', 'Ndelays').\n"
            "For uncalibrated data the noise simulation is not well defined "
            "but is still calculated and will have units (Jy Hz)^2. "
            "In python 2 this unit is always in mK^2 Mpc^3. "
            "For python 3 users, it is possible to convert the mK^2 / (littleh/Mpc)^3 "
            "using the littleh_units boolean flag in update_cosmology."
        )
        self._noise_power = UnitParameter(
            "noise_power",
            description=desc,
            expected_type=np.complex64,
            required=False,
            form=("Nspws", "Npols", "Nbls", "Nbls", "Ntimes", "Ndelays"),
            expected_units=_noise_power_units,
        )

        _thermal_power_units = (
            (units.mK**2 * units.Mpc**3),
            (units.mK**2 * (units.Mpc / littleh) ** 3),
            (units.K * units.sr * units.Hz**3) ** 2,
        )
        desc = (
            "The predicted thermal variance of the input data averaged over "
            "all input baselines."
            "Will have shape ('Nspws', 'Npols', 'Nbls', 'Nbls', 'Ntimes').\n"
            "Units are converted to cosmological frame (mK^2/(hMpc^-1)^3). "
            "In python 2 this unit is always in mK^2 Mpc^3. "
            "For python 3 users, it is possible to convert the mK^2 / (littleh/Mpc)^3 "
            "using the littleh_units boolean flag in update_cosmology."
        )
        self._thermal_power = UnitParameter(
            "thermal_power",
            description=desc,
            expected_type=np.float64,
            required=False,
            form=("Nspws", "Npols", "Nbls", "Nbls", "Ntimes"),
            expected_units=_thermal_power_units,
        )

        _conversion_units = (
            (units.mK**2 * units.Mpc**3 / (units.Jy * units.Hz) ** 2),
            (units.mK**2 * units.Mpc**3 / (units.K * units.sr * units.Hz) ** 2),
            (units.mK**2 * (units.Mpc / littleh) ** 3 / (units.Jy * units.Hz) ** 2),
            (
                units.mK**2
                * (units.Mpc / littleh) ** 3
                / (units.K * units.sr * units.Hz) ** 2
            ),
            (units.dimensionless_unscaled),
        )
        desc = (
            "The cosmological unit conversion factor applied to the data. "
            'Has the form ("Nspws", "Npols"). Accounts for all beam polarizations.'
            "Depending on units of input visibilities it may take units of "
            "mK^2/(h/Mpc)^3 / (K * sr * Hz)^2 or mK^2/[h/Mpc]^3 / (Jy * Hz)^2"
        )
        self._unit_conversion = UnitParameter(
            "unit_conversion",
            description=desc,
            required=False,
            expected_type=np.float64,
            form=("Nspws", "Npols"),
            expected_units=_conversion_units,
        )
        _tconversion_units = (
            (
                units.mK**2
                * units.Mpc**3
                / (units.K * units.sr * units.Hz**3) ** 2
            ),
            (
                units.mK**2
                * (units.Mpc / littleh) ** 3
                / (units.K * units.sr * units.Hz) ** 2
            ),
        )
        desc = (
            "The cosmological unit conversion factor applied to the thermal noise estimate. "
            'Has the form ("Nspws", "Npols"). Accounts for all beam polarizations.'
            "Always has units mK^2 Mpc^3 /( K^2 sr^2 Hz^2)"
        )
        self._thermal_conversion = UnitParameter(
            "thermal_conversion",
            description=desc,
            required=False,
            expected_type=np.float64,
            form=("Nspws", "Npols"),
            expected_units=_tconversion_units,
        )
        desc = "The integral of the power beam area. Shape = (Nspws, Npols, Nfreqs)"
        self._beam_area = UnitParameter(
            "beam_area",
            description=desc,
            form=("Nspws", "Npols", "Nfreqs"),
            expected_type=np.float64,
            expected_units=units.sr,
        )
        desc = (
            "The integral of the squared power beam squared area. "
            "Shape = (Nspws, Npols, Nfreqs)"
        )
        self._beam_sq_area = UnitParameter(
            "beam_sq_area",
            description=desc,
            form=("Nspws", "Npols", "Nfreqs"),
            expected_type=np.float64,
            expected_units=units.sr,
        )
        desc = (
            "Length of the integration in seconds, has shape "
            "(Npols, Nbls, Ntime). units s, assumes inegration time "
            " is the same for all spectral windows and all frequncies in a "
            "spectral window. "
            "Assumes the same convention as pyuvdata, where this is the "
            "target amount of time a measurement is integrated over. "
            "Spectral window dimension allows for frequency dependent "
            "filtering to be properly tracked for noise simulations."
        )
        self._integration_time = UnitParameter(
            "integration_time",
            description=desc,
            form=("Nbls", "Ntimes"),
            expected_type=np.float64,
            expected_units=units.s,
        )

        desc = "Spectral taper function used during Fourier Transform. Functions like scipy.signal.windows.blackmanharris"
        self._taper = UnitParameter(
            "taper",
            description=desc,
            form=(),
            expected_type=Callable,
            value=windows.blackmanharris,
            value_not_quantity=True,
        )

        desc = "Astropy cosmology object cabale of performing necessary cosmological calculations. Defaults to WMAP 9-Year."
        self._cosmology = UnitParameter(
            "cosmology",
            description=desc,
            form=(),
            expected_type=reference_cosmology_object,
            value=simple_cosmo.default_cosmology.get(),
            value_not_quantity=True,
        )

        desc = (
            "Orientation of the physical dipole corresponding to what is"
            "labelled as the x polarization. Options are 'east'"
            "(indicating east/west orientation) and 'north' (indicating"
            "north/south orientation)"
        )
        self._x_orientation = UnitParameter(
            "x_orientation",
            description=desc,
            required=False,
            expected_type=str,
            acceptable_vals=["east", "north"],
        )

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

    @property
    def _visdata_params(self):
        """List of strings giving the visdata-like parameters."""
        return ["data_array", "nsample_array", "flag_array", "noise_array"]

    @property
    def visdata_like_parameters(self):
        """Iterate through defined parameters which are visdata-like (not metadata-like or power-like)."""
        for key in self._visdata_params:
            if hasattr(self, key):
                yield getattr(self, key)

    @property
    def _power_params(self):
        """List of strings giving the power-like parameters."""
        return ["power_array", "noise_power"]

    @property
    def power_like_parameters(self):
        """Iterate through defined parameters which are power-like (not metadata-like or power-like)."""
        for key in self._power_params:
            if hasattr(self, key):
                yield getattr(self, key)

    @property
    def _thermal_params(self):
        """List of thermal_power  like params."""
        return ["thermal_power"]

    @property
    def thermal_like_parameters(self):
        """Iterate through the parameters which are themal-like."""
        for key in self._thermal_params:
            if hasattr(self, key):
                yield getattr(self, key)

    @property
    def metadata_only(self):
        """
        Property that determines whether this is a metadata only object.

        An object is metadata only if data_array, nsample_array and flag_array
        are all None.
        """
        metadata_only = all(
            d is None
            for d in chain(
                self.visdata_like_parameters,
                self.power_like_parameters,
                self.thermal_like_parameters,
            )
        )

        for param_name in self._visdata_params:
            getattr(self, "_" + param_name).required = not metadata_only

        if not metadata_only and any(
            d is not None
            for d in chain(self.power_like_parameters, self.thermal_like_parameters)
        ):
            for param_name in chain(self._power_params, self._thermal_params):
                getattr(self, "_" + param_name).required = not metadata_only
        elif not metadata_only:
            for param_name in chain(self._power_params, self._thermal_params):
                getattr(self, "_" + param_name).required = metadata_only

        return metadata_only

    def set_taper(self, taper=None):
        """Set spectral taper function used during Fourier Transform.

        Parameters
        ----------
        taper : function or Callable, Optional
            Spectral taper function used during frequency Fourier Transforms
            Accepts scipy.signal.windows functions or any function
            whose argument is the len(data) and returns a numpy array.
            Default: scipy.signal.windows.blackmanharris

        Raises
        ------
        ValueError
            If taper input is not callable.


        """
        if taper is None:
            taper = windows.blackmanharris

        if not callable(taper):
            raise ValueError(
                "Input spectral taper must be a function or "
                "callable whose arguments are "
                "the length of the band "
                "over which Fourier Transform is taken."
            )
        else:
            self.taper = taper

    def set_delay(self):
        """Set the data type to delay.

        Raises
        ------
        UnitConversionError
            if data is inconsistent with delay units.

        """
        consistent_units = [
            units.Jy * units.Hz,
            units.K * units.sr * units.Hz,
            units.dimensionless_unscaled * units.Hz,
        ]
        if not any([self.data_array.unit.is_equivalent(u) for u in consistent_units]):
            raise units.UnitConversionError(
                "Data is not in units consistent "
                "with the delay domain. "
                "Cannot set data_type to delay."
            )
        else:
            self.data_type = "delay"

    def set_frequency(self):
        """Set the data type to frequency.

        Raises
        ------
        UnitConversionError
            if data is inconsistent with frequency units.

        """
        consistent_units = [units.Jy, units.K * units.sr, units.dimensionless_unscaled]
        if not any([self.data_array.unit.is_equivalent(u) for u in consistent_units]):
            raise units.UnitConversionError(
                "Data is not in units consistent "
                "with the frequency domain. "
                "Cannot set data_type to frequency."
            )
        else:
            self.data_type = "frequency"

    def check(self, check_extra=True, run_check_acceptability=True):
        """Add some extra checks on top of checks on UVBase class.

        Check that required parameters exist. Check that parameters have
        appropriate shapes and optionally that the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            If true, check all parameters, otherwise only check required parameters.
        run_check_acceptability : bool
            Option to check if values in parameters are acceptable.

        Raises
        ------
        ValueError
            If any parameter values are inconsistent with expected types, shapes, or ranges of values.

        """
        if not self.metadata_only:
            if self.data_type == "delay":
                self.set_delay()
            else:
                self.set_frequency()

        if self.Nbls != len(np.unique(self.baseline_array)):
            raise ValueError(
                "Nbls must be equal to the number of unique "
                "baselines in the data_array"
            )

        if self.Ntimes != len(np.unique(self.lst_array)):
            raise ValueError(
                "Ntimes must be equal to the number of unique " "times in the lst_array"
            )

        if check_extra:
            p_check = [p for p in self.required()] + [p for p in self.extra()]
        else:
            p_check = [p for p in self.required()]

        for p in p_check:
            param = getattr(self, p)
            # Check required parameter exists
            if param.value is None:
                if param.required is True:
                    raise ValueError(
                        "Required UnitParameter " + p + " has not been set."
                    )
            else:
                # Check parameter shape
                eshape = param.expected_shape(self)
                # default value of eshape is ()
                if eshape == "str" or (eshape == () and param.expected_type == "str"):
                    # Check that it's a string
                    if not isinstance(param.value, str):
                        raise ValueError(
                            "UnitParameter " + p + " expected to be "
                            "string, but is not"
                        )
                else:
                    # Check the shape of the parameter value. Note that np.shape
                    # returns an empty tuple for single numbers. eshape should do the same.
                    if not np.shape(param.value) == eshape:
                        raise ValueError(
                            f"UnitParameter {p} is not expected shape. "
                            f"Parameter shape is {np.shape(param.value)}, expected shape is "
                            f"{eshape}."
                        )
                    if eshape == ():
                        # Single element
                        if not isinstance(param.value, param.expected_type):
                            raise ValueError(
                                "UnitParameter " + p + " is not the appropriate"
                                " type. Is: "
                                + str(type(param.value))
                                + ". Should be: "
                                + str(param.expected_type)
                            )
                    else:
                        # UnitParameters cannot have list entries
                        # Array
                        if isinstance(param.value, list):
                            raise ValueError(
                                "UnitParameter " + p + " is a list. "
                                "UnitParameters are incompatible with lists"
                            )
                        if isinstance(param.value, units.Quantity):
                            if not param.value.unit.is_equivalent(param.expected_units):
                                raise units.UnitConversionError(
                                    "UnitParameter " + p + " "
                                    f"has units {param.value.unit} "
                                    "which are not equivalent to "
                                    f"expected units of {param.expected_units}."
                                )
                            if not isinstance(
                                param.value.value.item(0), param.expected_type
                            ):
                                raise ValueError(
                                    "UnitParameter " + p + " is not the appropriate"
                                    " type. Is: "
                                    + str(param.value.dtype)
                                    + ". Should be: "
                                    + str(param.expected_type)
                                )
                        else:
                            if not isinstance(param.value.item(0), param.expected_type):
                                raise ValueError(
                                    "UnitParameter " + p + " is not the appropriate"
                                    " type. Is: "
                                    + str(param.value.dtype)
                                    + ". Should be: "
                                    + str(param.expected_type)
                                )

                if run_check_acceptability:
                    accept, message = param.check_acceptability()
                    if not accept:
                        raise ValueError(
                            "UnitParameter "
                            + p
                            + " has unacceptable values. "
                            + message
                        )
        return True

    def add_uvdata(self, uv, spectral_windows=None, tol=1.0):
        """Add the relevant uvdata object data to DelaySpectrum object.

        Unloads the data, flags, and nsamples arrays from the input UVData
        object (or subclass) into local storage.

        Parameters
        ----------
        uv : UVdata object
            A UVData object or subclass of UVData to add to the existing datasets

        spectral_windows : tuple of tuples, or tuple of indices, or list of lists, or list of indices; Default selection is (0, Nfreqs)
            spectral windows ranges like (start_index, end_index) where the indices are the frequency channel numbers.

        tol : float
            Tolerance in meters of the redundancy allowed for pyuvdata.get_redundancies calculation

        Raises
        ------
        ValueError
            Input data object must be an instance or subclass of UVData.
            If input UVData object has more than 1 unique baseline type within tolerance.
            Input UVData object has parameters inconsistent with any existing data

        """
        if not isinstance(uv, UVData):
            raise ValueError(
                "Input data object must be an instance or " "subclass of UVData."
            )

        red_groups, uvw_centers, lengths, conjugates = uv.get_redundancies(
            tol=tol, include_conjugates=True
        )
        if len(red_groups) > 1:
            raise ValueError(
                "A DelaySpectrum object can only perform a Fourier "
                "Transform along a single baseline vector. "
                "Downselect the input UVData object to only have "
                "one redundant baseline type."
            )

        this = DelaySpectrum()
        this.Ntimes = uv.Ntimes
        this.Nbls = uv.Nbls
        this.Nfreqs = uv.Nfreqs
        this.Nants_data = uv.Nants_data
        this.Nants_telescope = uv.Nants_telescope
        this.x_orientation = uv.x_orientation
        this.vis_units = uv.vis_units
        this.Npols = uv.Npols
        this.Nspws = 1
        this.Nuv = 1
        this.lst_array = (
            uv.lst_array[uv.baseline_array == uv.baseline_array[0]] << units.rad
        )
        lst_inds = np.argsort(np.unwrap(this.lst_array))
        this.polarization_array = uv.polarization_array
        if this.vis_units == "Jy":
            data_unit = units.Jy
        elif this.vis_units == "K str":
            data_unit = units.K * units.sr
        else:
            data_unit = units.dimensionless_unscaled

        this.freq_array = uv.freq_array << units.Hz
        this._freq_array.tols = (
            this._freq_array.tols[0],
            this._freq_array.tols[1] * units.Hz,
        )

        this.baseline_array = uv.get_baseline_nums()
        this.ant_1_array, this.ant_2_array = np.transpose(uv.get_antpairs(), [1, 0])
        temp_data = np.zeros(
            shape=this._data_array.expected_shape(this), dtype=np.complex128
        )
        temp_data[:, :, :, :, :] = utils.get_data_array(
            uv, reds=this.baseline_array, squeeze=False
        )
        this.data_array = copy.deepcopy(temp_data) << data_unit
        this._data_array.tols = (
            this._data_array.tols[0],
            this._data_array.tols[1] * data_unit,
        )

        temp_data = np.zeros(
            shape=this._nsample_array.expected_shape(this), dtype=np.float64
        )
        temp_data[:, :, :, :, :] = utils.get_nsample_array(
            uv, reds=this.baseline_array, squeeze=False
        )
        this.nsample_array = copy.deepcopy(temp_data)

        temp_data = np.ones(shape=this._flag_array.expected_shape(this), dtype=bool)
        temp_data[:, :, :, :, :] = utils.get_flag_array(
            uv, reds=this.baseline_array, squeeze=False
        )
        this.flag_array = copy.deepcopy(temp_data)

        temp_data = np.zeros(
            shape=this._integration_time.expected_shape(this), dtype=np.float64
        )
        temp_data[:, :] = utils.get_integration_time(
            uv, reds=this.baseline_array, squeeze=False
        )

        this.integration_time = copy.deepcopy(temp_data) << units.s

        # reorder to have increasing lsts for every array like Ntimes
        this.lst_array = np.unwrap(this.lst_array)[lst_inds]
        this.data_array = this.data_array[..., lst_inds, :]
        this.flag_array = this.flag_array[..., lst_inds, :]
        this.nsample_array = this.nsample_array[..., lst_inds, :]
        this.integration_time = this.integration_time[:, lst_inds]
        # initialize the beam_area and beam_sq_area to help with selections later
        this.beam_area = (
            np.ones(this._beam_area.expected_shape(this)) * np.inf << units.sr
        )
        this.beam_sq_area = (
            np.ones(this._beam_sq_area.expected_shape(this)) * np.inf << units.sr
        )

        this.trcvr = np.ones(shape=this._trcvr.expected_shape(this)) * np.inf << units.K

        this.uvw = uvw_centers[0] << units.m
        this._uvw.tols = (this._uvw.tols[0], this._uvw.tols[1] * units.m)

        this.generate_noise()

        if this.data_array.unit == units.K * units.sr:
            # if the visibilities are in K steradian then the noise should be too.
            # reshape to have form of Nspws, Nuv, Npols, Nbls, Ntimes, Nfreqs
            this.noise_array = this.noise_array * utils.jy_to_mk(
                this.freq_array
            ).reshape(this.Nspws, 1, 1, 1, 1, this.Nfreqs)
        elif this.data_array.unit == units.dimensionless_unscaled:
            warnings.warn(
                "Data is uncalibrated. Unable to covert noise array "
                "to unicalibrated units.",
                UserWarning,
            )

        if self.freq_array is not None:
            this.select_spectral_windows(freqs=self.freq_array)
        else:
            this.select_spectral_windows(spectral_windows=spectral_windows)
        this._delay_array.tols = (
            this._delay_array.tols[0],
            this._delay_array.tols[1] * this.delay_array.unit,
        )

        # Make sure both objects are self consistent before trying to join them
        # self.check(check_extra=True, run_check_acceptability=True)
        this.check(check_extra=False, run_check_acceptability=True)

        # check for compatibility of all parameters before adding UVData data
        # to object. Parameters are compatible if all parameters are equal
        # excluding the data_array, flag_array, and nsample_array
        # want to run checks and error before any data is written
        if self.data_array is not None and not self.data_array.unit.is_equivalent(
            this.data_array.unit
        ):
            errmsg = (
                "Input data object is in units "
                "incompatible with saved DelaySpectrum units."
                f"Saved units are: {self.data_array.unit}, "
                f"input units are: {this.data_array.unit}."
            )
            raise units.UnitConversionError(errmsg)

        if self.Nuv == 2:
            raise ValueError(
                "This DelaySpectrum Object has already "
                "been loaded with two datasets. Create "
                "a new object to cross-multipy different "
                "data."
            )

        for p in self:
            my_parm = getattr(self, p)
            other_parm = getattr(this, p)
            if p not in [
                "_data_array",
                "_flag_array",
                "_nsample_array",
                "_noise_array",
                "_Nuv",
                "_beam_area",
                "_beam_sq_area",
                "_trcvr",
                "_taper",
                "_lst_array",
            ]:
                if my_parm.value is not None and my_parm != other_parm:
                    raise ValueError(
                        "Input data differs from previously "
                        f"loaded data. Parameter {p} is not "
                        "the same."
                    )
            elif p in ["_lst_array"]:
                if my_parm.value is not None and my_parm != other_parm:
                    time_diff = (
                        np.abs(my_parm.value - other_parm.value)
                        * 12.0
                        * units.h
                        / (np.pi * units.rad)
                    )
                    warnings.warn(
                        "Input LST arrays differ on average "
                        f"by {time_diff.mean().to('min')}. Keeping LST array stored from "
                        "the first data set read.",
                        UserWarning,
                    )
            # these next lines don't seem reasonable
            # the this object will never have a beam area, beam sq area or trcvr
            # that isfinite().all() = Tru
            # elif p in ["_beam_area, _beam_sq_area", "_trcvr"]:
            #     if my_parm.value is not None and my_parm != other_parm:
            #         if (
            #             np.isfinite(my_parm.value.value).all()
            #             and np.isfinite(other_parm.value.value).all()
            #         ):
            #             raise ValueError(
            #                 "Input data differs from previously "
            #                 "loaded data. Parameter {name} is not "
            #                 "the same".format(name=p)
            #             )
            else:
                pass

        # Increment by one the number of read uvdata objects
        self._Nuv.value += 1
        for p in self:
            my_parm = getattr(self, p)
            if p not in [
                "_data_array",
                "_flag_array",
                "_nsample_array",
                "_noise_array",
                "_Nuv",
                "_beam_area",
                "_beam_sq_area",
                "_trcvr",
                "_taper",
            ]:
                if my_parm.value is None:
                    parm = getattr(this, p)
                    setattr(self, p, parm)
            elif p in ["_beam_area", "_beam_sq_area", "_trcvr"]:
                if my_parm.value is None:
                    parm = getattr(this, p)
                    setattr(self, p, parm)
                elif np.isinf(my_parm.value.value).all():
                    parm = getattr(this, p)
                    setattr(self, p, parm)
            elif p in ["_data_array", "_flag_array", "_nsample_array", "_noise_array"]:
                parm = getattr(this, p)
                if my_parm.value is not None:
                    tmp_data = np.zeros(
                        my_parm.expected_shape(self), dtype=my_parm.value.dtype
                    )
                    tmp_data[:, : self.Nuv - 1] = my_parm.value[:]

                    if isinstance(my_parm.value, units.Quantity):
                        tmp_data = tmp_data << my_parm.value.unit

                    my_parm.value = tmp_data
                    my_parm.value[:, self.Nuv - 1] = parm.value[0]
                    setattr(self, p, my_parm)
                else:
                    setattr(self, p, parm)
        self.check(check_extra=False, run_check_acceptability=True)

    def remove_cosmology(self):
        """Remove cosmological conversion from any power spectrum estimates."""
        if self.cosmology is None:
            raise ValueError(f"Cannot remove cosmology of type {type(None)}")

        self.k_parallel = None
        self.k_perpendicular = None
        # If power spectrum estimation has already occurred, need to re-normalize
        # in the new cosmological framework.
        if self.power_array is not None:
            if self.power_array.unit.is_equivalent(
                units.mK**2 * units.Mpc**3 / littleh**3
            ):
                self.power_array = self.power_array.to(
                    units.mK**2 * units.Mpc**3, with_H0(self.cosmology.H0)
                )
                self.noise_power = self.noise_power.to(
                    units.mK**2 * units.Mpc**3, with_H0(self.cosmology.H0)
                )
            # only divide by the conversion when power array is in cosmological units
            # e.g. not if this is the first time, or if calculate_delay_spectrum was just called.
            if (
                self.unit_conversion is not None
                and not self.power_array.unit.is_equivalent(
                    (
                        units.Jy**2 * units.Hz**2,
                        units.K**2 * units.sr**2 * units.Hz**2,
                    )
                )
            ):
                with np.errstate(divide="ignore", invalid="ignore"):
                    self.power_array = self.power_array / self.unit_conversion.reshape(
                        self.Nspws, self.Npols, 1, 1, 1, 1
                    )
                    self.noise_power = self.noise_power / self.unit_conversion.reshape(
                        self.Nspws, self.Npols, 1, 1, 1, 1
                    )

        if self.thermal_power is not None:
            if self.thermal_power.unit.is_equivalent(
                units.mK**2 * units.Mpc**3 / littleh**3
            ):
                self.thermal_power = self.thermal_power.to(
                    units.mK**2 * units.Mpc**3, with_H0(self.cosmology.H0)
                )
            # only divide by the conversion when power array is in cosmological units
            # e.g. not if this is the first time, or if calculate_delay_spectrum was just called.
            if (
                self.thermal_conversion is not None
                and not self.thermal_power.unit.is_equivalent(
                    (
                        units.Jy**2 * units.Hz**2,
                        units.K**2 * units.sr**2 * units.Hz**6,
                    )
                )
            ):
                with np.errstate(divide="ignore", invalid="ignore"):
                    self.thermal_power = (
                        self.thermal_power
                        / self.thermal_conversion.reshape(
                            self.Nspws, self.Npols, 1, 1, 1
                        )
                    )

        self.unit_conversion = None
        self.thermal_conversion = None
        return

    def update_cosmology(self, cosmology=None, littleh_units=False):
        """Update cosmological information with the assumed cosmology.

        Parameters
        ----------
        cosmology : Astropy Cosmology Object, optional
            input assumed cosmology. Must be an astropy cosmology object. Defaults to Planck15
        littleh_units: Bool, default:False
                       automatically convert to to mK^2 / (littleh / Mpc)^3.

        Raises
        ------
        ValueError
            If input cosomolgy is not an astropy cosmology object

        """
        # remove any previously applied cosmologies
        self.remove_cosmology()

        if cosmology is not None:
            if not isinstance(cosmology, reference_cosmology_object):
                raise ValueError(
                    "Input cosmology must a sub-class of astropy.cosmology.core.Cosmology"
                )
            self.cosmology = cosmology
        else:
            self.cosmology = simple_cosmo.default_cosmology.get()
        # find the mean redshift for each spectral window
        self.redshift = simple_cosmo.calc_z(self.freq_array).mean(axis=1)

        self.k_parallel = simple_cosmo.eta2kparr(
            self.delay_array.reshape(1, self.Ndelays),
            self.redshift.reshape(self.Nspws, 1),
            cosmo=self.cosmology,
        )

        uvw_wave = np.linalg.norm(self.uvw.value) << self.uvw.unit
        mean_freq = np.mean(self.freq_array.value, axis=1) << self.freq_array.unit
        uvw_wave = uvw_wave / (const.c / mean_freq.to("1/s")).to("m")
        self.k_perpendicular = simple_cosmo.u2kperp(
            uvw_wave, self.redshift, cosmo=self.cosmology
        )

        # If power spectrum estimation has already occurred, need to re-normalize
        # in the new cosmological framework.
        if self.power_array is not None:

            if self.power_array.unit.is_equivalent((units.Jy * units.Hz) ** 2):
                # This additoinal units.sr term in the c/2*K_b expression may seem
                # weird, however, the temperature to Jy conversion factor is defined
                # such that there is a beam integral, or a sr factor, included.
                # this helps the units work out.

                # This is the full unit conversion integral.
                # See liu et al 2014ab or write the visibility equation and convert to cosmological units without pulling anything outside the integral.
                integration_array = (
                    self.freq_array.reshape(self.Nspws, 1, self.Nfreqs) ** 4
                    / simple_cosmo.X2Y(
                        simple_cosmo.calc_z(self.freq_array), cosmo=self.cosmology
                    ).reshape(self.Nspws, 1, self.Nfreqs)
                    * self.taper(self.Nfreqs).reshape(1, 1, self.Nfreqs) ** 2
                    * self.beam_sq_area.reshape(self.Nspws, self.Npols, self.Nfreqs)
                )
                self.unit_conversion = (
                    const.c**2 * units.sr / (2 * const.k_B)
                ) ** 2 / integrate.trapz(
                    integration_array.value,
                    x=self.freq_array.value.reshape(self.Nspws, 1, self.Nfreqs),
                    axis=-1,
                ).reshape(
                    self.Nspws, self.Npols
                )
                self.unit_conversion = self.unit_conversion / (
                    integration_array.unit * self.freq_array.unit
                )
                self.unit_conversion = self.unit_conversion << units.Unit(
                    "mK^2 Mpc^3 /( Jy^2 Hz^2)"
                )
            elif self.power_array.unit.is_equivalent(
                (units.K * units.sr * units.Hz) ** 2
            ):
                integration_array = (
                    1.0
                    / simple_cosmo.X2Y(
                        simple_cosmo.calc_z(self.freq_array), cosmo=self.cosmology
                    ).reshape(self.Nspws, 1, self.Nfreqs)
                    * self.taper(self.Nfreqs).reshape(1, 1, self.Nfreqs) ** 2
                    * self.beam_sq_area.reshape(self.Nspws, self.Npols, self.Nfreqs)
                )
                self.unit_conversion = 1.0 / integrate.trapz(
                    integration_array.value,
                    x=self.freq_array.value.reshape(self.Nspws, 1, self.Nfreqs),
                    axis=-1,
                ).reshape(self.Nspws, self.Npols)
                self.unit_conversion = self.unit_conversion / (
                    integration_array.unit * self.freq_array.unit
                )
                self.unit_conversion = self.unit_conversion << units.Unit(
                    "mK^2 Mpc^3 /( K^2 sr^2 Hz^2)"
                )
            else:
                self.unit_conversion = (
                    np.ones((self.Npols, self.Nspws)) << units.dimensionless_unscaled
                )

            self.power_array = self.power_array * self.unit_conversion.reshape(
                self.Nspws, self.Npols, 1, 1, 1, 1
            )
            self.noise_power = self.noise_power * self.unit_conversion.reshape(
                self.Nspws, self.Npols, 1, 1, 1, 1
            )
            if not self.data_array.unit.is_equivalent(
                units.dimensionless_unscaled * units.Hz
            ):
                self.power_array = self.power_array << units.Unit("mK^2 * Mpc^3")
                self.noise_power = self.noise_power << units.Unit("mK^2 * Mpc^3")

        if self.thermal_power is not None:

            integration_array = (
                self.freq_array.reshape(self.Nspws, 1, self.Nfreqs) ** 4
                / simple_cosmo.X2Y(
                    simple_cosmo.calc_z(self.freq_array), cosmo=self.cosmology
                ).reshape(self.Nspws, 1, self.Nfreqs)
                * self.taper(self.Nfreqs).reshape(1, 1, self.Nfreqs) ** 2
                * self.beam_sq_area.reshape(self.Nspws, self.Npols, self.Nfreqs)
            )
            thermal_conversion = 1.0 / integrate.trapz(
                integration_array.value,
                x=self.freq_array.value.reshape(self.Nspws, 1, self.Nfreqs),
                axis=-1,
            ).reshape(self.Nspws, self.Npols)
            thermal_conversion = thermal_conversion << 1.0 / (
                integration_array.unit * self.freq_array.unit
            )
            self.thermal_conversion = thermal_conversion << units.Unit(
                "mK^2 Mpc^3 /( K^2 sr^2 Hz^6)"
            )
            self.thermal_power = self.thermal_power * self.thermal_conversion.reshape(
                self.Nspws, self.Npols, 1, 1, 1
            )

            self.thermal_power = self.thermal_power << units.Unit("mK^2 Mpc^3")

        if littleh_units:
            self.k_perpendicular = self.k_perpendicular.to(
                littleh / units.Mpc, with_H0(self.cosmology.H0)
            )
            self.k_parallel = self.k_parallel.to(
                littleh / units.Mpc, with_H0(self.cosmology.H0)
            )
            if self.power_array is not None:
                self.power_array = self.power_array.to(
                    units.mK**2 * units.Mpc**3 / littleh**3,
                    with_H0(self.cosmology.H0),
                )
                self.noise_power = self.noise_power.to(
                    units.mK**2 * units.Mpc**3 / littleh**3,
                    with_H0(self.cosmology.H0),
                )
            if self.thermal_power is not None:
                self.thermal_power = self.thermal_power.to(
                    units.mK**2 * units.Mpc**3 / littleh**3,
                    with_H0(self.cosmology.H0),
                )

    @units.quantity_input(
        frequencies=units.Hz,
        delays=units.s,
        lsts=units.rad,
        lst_range=units.rad,
    )
    def _select_preprocess(
        self,
        antenna_nums=None,
        bls=None,
        spws=None,
        frequencies=None,
        freq_chans=None,
        delays=None,
        delay_chans=None,
        lsts=None,
        lst_range=None,
        polarizations=None,
        uv_index=None,
        exact_spw_match=False,
    ):
        """Gather all the indices necessary to make selections on metadata and data.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object.
        bls : array_like of tuple or int, optional
            The baselines to keep in the object.
            Can either be a tuple of antenna numbers (e.g. (0, 1)) the baseline number or a combination of both.
        spectral_windows : array_like of int
            The spectral windows to keep in the object.
        frequencies : Astropy Quantity object, optional
            The frequencies to keep in the object.
            Values must match frequencies which already exist in the object.
            Only available when in 'frequency' mode
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
            When multiple spectral windows are present, this will apply to all.
            Only available when in 'frequency' mode
        delays : Astropy Quantity object, optional
            The delays to kee in the object.
            Values must match delays which already exist in the object.
            Only available when in 'delay' mode.
        delay_chans : array_like of int, optional
            The delays channel numbers to keep in the object.
            When multiple spectral windows are present, this will apply to all.
            Only available when in 'delay' mode.
        lsts : Astropy Quantity object, optional
            The time to keep in the object.
            Values must match times which already exist in the object.
        lst_range : list of Astropy Quantity, optional
            A list of length 2 with start and end times to select.
            All time values which fall in this range will be selected.
        polarizations : array_like of int, optional
            The polarizations to keep in the object.
        exact_spw_match : bool, optional
            When True, will only select spectral windows whose entire set of
            frequencies matches the input frequencies


        Returns
        -------
        spw_inds : list of ints
            list of indices for spectral windows to keep. Can be None (keep all)
        freq_inds : list of ints
            list of indices into each frequency axis to keep. Can be None (keep all)
        delay_inds : list of ints
            list of indices into the delay axis to keep. Can be None (keep all)
        bl_inds : list of ints
            list of indices into the baseline array to keep. Can be None (keep all)
        time_inds : list of ints
            list of indices into the lst_array to keep. Can be None (keep all)
        pol_inds : list of ints
            list of indices into the polarization_array to keep. Can be None (keep all)

        """
        spw_inds = set()
        freq_inds = [set() for spw in range(self.Nspws)]
        delay_inds = set()
        bl_inds = set()
        lst_inds = set()
        pol_inds = set()
        uv_inds = set()

        if antenna_nums is not None:
            inds1, inds2 = set(), set()
            for ant in uvutils._get_iterable(antenna_nums):
                if not (ant in self.ant_1_array or ant in self.ant_2_array):
                    raise ValueError(
                        f"Antenna {ant:d} is not present in either ant_1_array "
                        f"or ant_2_array."
                    )
                wh1 = np.nonzero(self.ant_1_array == ant)[0]
                inds1.update(wh1)

                wh2 = np.nonzero(self.ant_2_array == ant)[0]
                inds2.update(wh2)
            # only keep baseline indices if both the antennas were present
            # in the antenna_numbers
            bl_inds.update(inds1.intersection(inds2))

        if bls is not None:
            tmp_bl_inds = set()
            if isinstance(bls, tuple) and (len(bls) == 2 or len(bls) == 3):
                bls = [bls]
            if len(bls) == 0 or not all(isinstance(item, tuple) for item in bls):
                if any(isinstance(bl, (int, np.int64, np.int_, np.intc)) for bl in bls):
                    warnings.warn(
                        "Input baseline array is a mix of integers and tuples of "
                        "integers. Assuming all integers not in a tuple as baseline "
                        "numbers and converting to antenna pairs."
                    )
                    bls = [
                        uvutils.baseline_to_antnums(bl, self.Nants_telescope)
                        if isinstance(bl, (int, np.int_, np.intc))
                        else bl
                        for bl in bls
                    ]
                else:
                    raise ValueError(
                        "bls must be a list of tuples of antenna numbers (optionally with polarization)."
                    )
            if not all(
                [isinstance(item[0], (int, np.integer)) for item in bls]
                + [isinstance(item[1], (int, np.integer)) for item in bls]
            ):
                raise ValueError(
                    "bls must be a list of tuples of antenna numbers (optionally with polarization)."
                )
            if all([len(item) == 3 for item in bls]):
                if polarizations is not None:
                    raise ValueError(
                        "Cannot provide length-3 tuples and also specify polarizations."
                    )
                if not all([isinstance(item[2], str) for item in bls]):
                    raise ValueError(
                        "The third element in each bl must be a polarization string"
                    )

            bl_pols = set()
            for bl in bls:
                wh1 = np.where(
                    np.logical_and(self.ant_1_array == bl[0], self.ant_2_array == bl[1])
                )[0]
                wh2 = np.where(
                    np.logical_and(self.ant_1_array == bl[1], self.ant_2_array == bl[0])
                )[0]

                if len(wh1) > 0:
                    tmp_bl_inds.update(wh1)
                    if len(bl) == 3:
                        bl_pols.add(bl[2])
                elif len(wh2) > 0:
                    tmp_bl_inds.update(wh2)
                    if len(bl) == 3:
                        bl_pols.add(bl[2][::-1])
                else:
                    raise ValueError(f"Baseline {bl} has no data associated with it.")

            if len(bl_pols) > 0:
                polarizations = bl_pols

            # bool of a set is True if it is not-empty
            if bool(bl_inds):
                # if non-empty only keep the intersection
                bl_inds.intersection_update(tmp_bl_inds)
            else:
                bl_inds.update(tmp_bl_inds)

        if bl_inds == set(np.arange(self.Nbls)):
            bl_inds = set()

        if spws is not None:
            spws = uvutils._get_iterable(spws)
            if any(not isinstance(spw, (int, np.int_, np.int_)) for spw in spws):
                raise ValueError(
                    "Input spws must be an array_like of integers corresponding "
                    "to spectral windows."
                )
            if any(spw >= self.Nspws for spw in spws):
                raise ValueError(
                    "Input spectral window values must be less than the number "
                    "of spectral windows currently on the object."
                )
            spw_inds.update(spws)

        if freq_chans is not None:
            freq_chans = uvutils._get_iterable(freq_chans)
            if np.array(freq_chans).ndim > 1:
                freq_chans = np.array(freq_chans).flatten()

            if frequencies is None:
                frequencies = units.Quantity(
                    sorted(set(self.freq_array[:, freq_chans].flatten()))
                )
            else:
                frequencies = units.Quantity(
                    sorted(
                        set(frequencies.flatten()).intersection(
                            self.freq_array[:, freq_chans].flatten()
                        )
                    )
                )

        if frequencies is not None:
            frequencies = frequencies.flatten()
            for spw, fq_array in enumerate(self.freq_array):
                for f in frequencies:
                    if f not in self.freq_array.flatten():
                        raise ValueError(
                            f"Frequency {f} not present in the frequency array."
                        )
                    freq_inds[spw].update(np.nonzero(fq_array == f)[0])
            # if we are also making selections on spectral windows, take intersection
            # otherwise the frequencies will define new spectral windows
            if bool(spw_inds):
                spw_inds.intersection_update(
                    set([ind for ind, t in enumerate(freq_inds) if len(t) > 0])
                )
            else:
                spw_inds.update([ind for ind, t in enumerate(freq_inds) if len(t) > 0])

            lens = np.unique([len(freq_inds[spw]) for spw in spw_inds])
            lens = lens[lens != 0]

            # reduce the frequencies inidces to only the remaining number of sepctral windows
            freq_inds = [freq_inds[spw] for spw in spw_inds]

            if lens.size > 1:
                if not exact_spw_match:
                    raise ValueError(
                        "Frequencies provided for selection will result in a non-rectangular "
                        "frequency array. Please ensure that all remaining spectral windows will "
                        "have the same number of frequencies. Settings exact_spw_match to True "
                        "will automatically drop spectral windows of uneven size."
                    )
                else:
                    spw_inds.intersection_update(
                        [
                            list(spw_inds)[i]
                            for i, fq in enumerate(freq_inds)
                            if len(fq) == self.Nfreqs
                        ]
                    )
                    freq_inds = [fq for fq in freq_inds if len(fq) == self.Nfreqs]

            if (
                self.power_array is not None
                and lens.size == 1
                and lens.item() != self.Nfreqs
            ):
                warnings.warn(
                    "This object has already been converted to a power spectrum "
                    "and a frequency selection is being performed. This will result "
                    "in an inconsistent data_array and power_array. Moreover all "
                    "parameters with shape Ndelay will retain their old shape "
                    "if a delay selection is not also performed."
                )

            # if the frequency and spectral window selections make no cut
            # reset them to null
            if all(set(fq) == set(np.arange(self.Nfreqs)) for fq in freq_inds):
                freq_inds = [set() for spw in range(self.Nspws)]

            if spw_inds == set(np.arange(self.Nspws)):
                spw_inds = set()

        if delay_chans is not None:
            delay_chans = uvutils._get_iterable(delay_chans)
            if np.array(delay_chans).ndim > 1:
                delay_chans = np.array(delay_chans).flatten()
            if delays is None:
                delays = units.Quantity(sorted(set(self.delay_array[delay_chans])))
            else:
                delays = units.Quantity(
                    sorted(
                        set(delays.flatten()).intersection(
                            self.delay_array[delay_chans]
                        )
                    )
                )
                if not bool(set(delays)):
                    raise ValueError(
                        "The intersection of the input delays and delay_chans "
                        "is empty. This will result in an object with no data."
                    )

        if delays is not None:
            delays = delays.flatten()
            for d in delays:
                if d not in self.delay_array:
                    raise ValueError(
                        f"The input delay {d} is not present in the delay_array."
                    )
                delay_inds.update(np.nonzero(self.delay_array == d)[0])

            if delay_inds == set(np.arange(self.Ndelays)):
                delay_inds = set()

        if lst_range is not None:
            lst_range = lst_range.flatten()
            if lst_range.size != 2:
                raise ValueError(
                    "Parameter lst_range must be an Astropy "
                    "Quantity object with size 2 (e.g. [start_lst, end_lst] * units.rad)"
                )

            tmp_inds = np.nonzero(
                np.logical_and(
                    lst_range[0] <= self.lst_array, self.lst_array <= lst_range[1]
                )
            )[0]
            if lsts is None:
                lsts = units.Quantity(sorted(set(self.lst_array[tmp_inds])))
            else:
                lsts = units.Quantity(
                    sorted(set(self.lst_array[tmp_inds]).intersection(lsts.flatten()))
                )

        if lsts is not None:
            lsts = lsts.flatten()
            for _lst in lsts:
                if _lst not in self.lst_array:
                    raise ValueError(
                        f"The input lst {_lst} is not present in the lst_array."
                    )
                lst_inds.update(np.nonzero(self.lst_array == _lst)[0])

            if lst_inds == set(np.arange(self.Ntimes)):
                lst_inds = set()

        if polarizations is not None:
            polarizations = uvutils._get_iterable(polarizations)

            if np.array(polarizations).ndim > 1:
                polarizations = np.array(polarizations).flatten()

            for p in polarizations:
                if isinstance(p, str):
                    p_num = uvutils.polstr2num(p, x_orientation=self.x_orientation)
                else:
                    p_num = p

                if p_num not in self.polarization_array:
                    raise ValueError(
                        f"Polarization {p_num} not present in polarization_array."
                    )

                pol_inds.update(np.nonzero(self.polarization_array == p_num)[0])

            if pol_inds == set(np.arange(self.Npols)):
                pol_inds = set()

        if uv_index is not None:
            uv_index = uvutils._get_iterable(uv_index)

            if np.array(uv_index).ndim > 1:
                uv_index = np.array(uv_index).flatten()

            uv_inds.update(uv_index)

        if len(uv_inds) == self.Nuv:
            uv_inds = set()
        elif len(uv_inds) > self.Nuv:
            raise ValueError(
                "The number of UVData objects in this DelaySpectrum object "
                f"is {self.Nuv} but the number of selections attempted is "
                f"{len(uv_inds)}. Restrict the uv_index to be less than "
                "or equal to the number of UVData objects."
            )

        # any empty sets will be replaced with None
        spw_inds = sorted(spw_inds) or None
        freq_inds = [sorted(fq) or None for fq in freq_inds]
        if all(fq is None for fq in freq_inds):
            freq_inds = None
        delay_inds = sorted(delay_inds) or None
        bl_inds = sorted(bl_inds) or None
        lst_inds = sorted(lst_inds) or None
        pol_inds = sorted(pol_inds) or None
        uv_inds = sorted(uv_inds) or None

        return spw_inds, freq_inds, delay_inds, bl_inds, lst_inds, pol_inds, uv_inds

    def _select_metadata(
        self,
        spw_inds=None,
        freq_inds=None,
        delay_inds=None,
        bl_inds=None,
        lst_inds=None,
        pol_inds=None,
        uv_inds=None,
    ):
        """Perform select on everything but data- and power-sized arrays.

        Parameters
        ----------
        spw_inds : list of ints
            list of indices for spectral windows to keep. Can be None (keep all)
        freq_inds : list of ints
            list of indices into each frequency axis to keep. Can be None (keep all)
        delay_inds : list of ints
            list of indices into the delay axis to keep. Can be None (keep all)
        bl_inds : list of ints
            list of indices into the baseline array to keep. Can be None (keep all)
        time_inds : list of ints
            list of indices into the lst_array to keep. Can be None (keep all)
        pol_inds : list of ints
            list of indices into the polarization_array to keep. Can be None (keep all)

        """
        if spw_inds is not None:
            self.Nspws = len(spw_inds)
            self.freq_array = np.take(self.freq_array, spw_inds, axis=0)
            self.beam_area = np.take(self.beam_area, spw_inds, axis=0)
            self.beam_sq_area = np.take(self.beam_sq_area, spw_inds, axis=0)
            self.trcvr = np.take(self.trcvr, spw_inds, axis=0)

            if self.k_parallel is not None:
                self.k_parallel = np.take(self.k_parallel, spw_inds, axis=0)

            if self.k_perpendicular is not None:
                self.k_perpendicular = np.take(self.k_perpendicular, spw_inds, axis=0)

            if self.redshift is not None:
                self.redshift = np.take(self.redshift, spw_inds, axis=0)

            if self.unit_conversion is not None:
                self.unit_conversion = np.take(self.unit_conversion, spw_inds, axis=0)

            if self.thermal_conversion is not None:
                self.thermal_conversion = np.take(
                    self.thermal_conversion, spw_inds, axis=0
                )

        if freq_inds is not None:
            self.Nfreqs = len(freq_inds[0])
            tmp_freqs = np.zeros((self.Nspws, self.Nfreqs)) * self.freq_array.unit
            tmp_trcvr = np.zeros((self.Nspws, self.Nfreqs)) * self.trcvr.unit
            tmp_beam = (
                np.zeros((self.Nspws, self.Npols, self.Nfreqs)) * self.beam_area.unit
            )
            tmp_sq_beam = (
                np.zeros((self.Nspws, self.Npols, self.Nfreqs)) * self.beam_sq_area.unit
            )
            for spw, inds in enumerate(freq_inds):
                tmp_freqs[spw] = np.take(self.freq_array[spw], inds, axis=0)
                tmp_trcvr[spw] = np.take(self.trcvr[spw], inds, axis=0)
                tmp_beam[spw] = np.take(self.beam_area[spw], inds, axis=1)
                tmp_sq_beam[spw] = np.take(self.beam_sq_area[spw], inds, axis=1)

            self.freq_array = tmp_freqs
            self.trcvr = tmp_trcvr
            self.beam_area = tmp_beam
            self.beam_sq_area = tmp_sq_beam

        if delay_inds is not None:
            self.Ndelays = len(delay_inds)
            self.k_parallel = np.take(self.k_parallel, delay_inds)
            self.delay_array = np.take(self.delay_array, delay_inds)

        if bl_inds is not None:
            self.Nbls = len(bl_inds)
            self.baseline_array = np.take(self.baseline_array, bl_inds, axis=0)
            self.ant_1_array = np.take(self.ant_1_array, bl_inds, axis=0)
            self.ant_2_array = np.take(self.ant_2_array, bl_inds, axis=0)
            self.integration_time = np.take(self.integration_time, bl_inds, axis=0)
            self.Nants_data = len(set(self.ant_1_array).union(self.ant_2_array))

        if lst_inds is not None:
            self.Ntimes = len(lst_inds)
            self.lst_array = np.take(self.lst_array, lst_inds, axis=0)
            self.integration_time = np.take(self.integration_time, lst_inds, axis=1)

        if pol_inds is not None:
            self.Npols = len(pol_inds)
            self.polarization_array = np.take(self.polarization_array, pol_inds)
            self.beam_area = np.take(self.beam_area, pol_inds, axis=1)
            self.beam_sq_area = np.take(self.beam_sq_area, pol_inds, axis=1)

            if self.unit_conversion is not None:
                self.unit_conversion = np.take(self.unit_conversion, pol_inds, axis=1)

            if self.thermal_conversion is not None:
                self.thermal_conversion = np.take(
                    self.thermal_conversion, pol_inds, axis=1
                )
        if uv_inds is not None:
            self.Nuv = len(uv_inds)

    def select(
        self,
        antenna_nums=None,
        bls=None,
        spws=None,
        frequencies=None,
        freq_chans=None,
        delays=None,
        delay_chans=None,
        lsts=None,
        lst_range=None,
        polarizations=None,
        uv_index=None,
        inplace=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Downselect data to keep on the object along various axes.

        Axes that can be selected along include antenna names or numbers,
        antenna pairs, frequencies, times and polarizations. Specific
        baseline-time indices can also be selected, but this is not commonly
        used.
        The history attribute on the object will be updated to identify the
        operations performed.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object.
        bls : array_like of tuple or int, optional
            The baselines to keep in the object.
            Can either be a tuple of antenna numbers (e.g. (0, 1)) the baseline number or a combination of both.
        spws : array_like of int
            The spectral windows to keep in the object.
        frequencies : Astropy Quantity object, optional
            The frequencies to keep in the object.
            Values must match frequencies which already exist in the object.
            Only available when in 'frequency' mode
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
            When multiple spectral windows are present, this will apply to all.
            Only available when in 'frequency' mode
        delays : Astropy Quantity object, optional
            The delays to kee in the object.
            Values must match delays which already exist in the object.
            Only available when in 'delay' mode.
        delay_chans : array_like of int, optional
            The delays channel numbers to keep in the object.
            When multiple spectral windows are present, this will apply to all.
            Only available when in 'delay' mode.
        lsts : Astropy Quantity object, optional
            The time to keep in the object.
            Values must match times which already exist in the object.
        lst_range : list of Astropy Quantity, optional
            A list of length 2 with start and end times to select.
            All time values which fall in this range will be selected.
        polarizations : array_like of int, optional
            The polarizations to keep in the object.
        uv_index : array_like of int, optional
            The index of the UVData objects to keep on the object.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).
        inplace : bool
            Option to perform the select directly on self or return a new DelaySpectrum
            object with just the selected data (the default is True, meaning the
            select will be done on self).

        """
        if inplace:
            ds_object = self
        else:
            ds_object = copy.deepcopy(self)

        (
            spw_inds,
            freq_inds,
            delay_inds,
            bl_inds,
            lst_inds,
            pol_inds,
            uv_inds,
        ) = ds_object._select_preprocess(
            antenna_nums,
            bls,
            spws,
            frequencies,
            freq_chans,
            delays,
            delay_chans,
            lsts,
            lst_range,
            polarizations,
            uv_index,
        )
        # do select operations on everything except data_array, flag_array and nsample_array
        # noise_array, power_array, noise_power, and thermal_power
        ds_object._select_metadata(
            spw_inds, freq_inds, delay_inds, bl_inds, lst_inds, pol_inds, uv_inds
        )

        if not self.metadata_only:

            for inds, axis in zip(
                [spw_inds, uv_inds, pol_inds, bl_inds, lst_inds], [0, 1, 2, 3, 4]
            ):
                if inds is not None:
                    for param_name, param in zip(
                        ds_object._visdata_params, ds_object.visdata_like_parameters
                    ):
                        setattr(ds_object, param_name, np.take(param, inds, axis=axis))

            # handle the frequencies for each remaining spectral window
            if freq_inds is not None:
                for param_name, param in zip(
                    ds_object._visdata_params, ds_object.visdata_like_parameters
                ):
                    tmp = np.zeros(
                        getattr(ds_object, "_" + param_name).expected_shape(ds_object),
                        dtype=getattr(ds_object, param_name).dtype,
                    )

                    if isinstance(param, units.Quantity):
                        tmp <<= param.unit

                    for spw, inds in enumerate(freq_inds):
                        tmp[spw] = np.take(param[spw], inds, axis=4)
                    setattr(ds_object, param_name, tmp)

            for inds, axis in zip(
                [spw_inds, pol_inds, bl_inds, bl_inds, lst_inds, delay_inds],
                [0, 1, 2, 3, 4, 5],
            ):
                if inds is not None:
                    for param_name, param in zip(
                        ds_object._power_params, ds_object.power_like_parameters
                    ):
                        if param is not None:
                            setattr(
                                ds_object, param_name, np.take(param, inds, axis=axis)
                            )

            for inds, axis in zip(
                [spw_inds, pol_inds, bl_inds, bl_inds, lst_inds], [0, 1, 2, 3, 4]
            ):
                if inds is not None:
                    for param_name, param in zip(
                        ds_object._thermal_params, ds_object.thermal_like_parameters
                    ):
                        if param is not None:
                            setattr(
                                ds_object,
                                param_name,
                                np.take(param, inds, axis=axis),
                            )

        use_littleh = self.k_perpendicular.unit == units.Unit(littleh / units.Mpc)
        ds_object.update_cosmology(littleh_units=use_littleh)

        # check if object is uv_object-consistent
        if run_check:
            ds_object.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return ds_object

    def _read_header(self, header):
        """Read DelaySpectrum object Header data from an hdf5 save file.

        Parameters
        ----------
        header : hdf5 group object
            The group object from a save file representing the header data.

        """
        self.Ntimes = int(header["Ntimes"][()])
        self.Nbls = int(header["Nbls"][()])
        self.Nfreqs = int(header["Nfreqs"][()])
        self.Npols = int(header["Npols"][()])

        if "Nants_data" in header:
            self.Nants_data = int(header["Nants_data"][()])
        else:
            self.Nants_data = len(
                set(header["ant_1_array"][()]).union(header["ant_2_array"][()])
            )

        if "Nants_telescope" in header:
            self.Nants_telescope = int(header["Nants_telescope"][()])
        else:
            warnings.warn(
                "Nants_telescope is not present in the header of this save file. "
                "This Delay Spectrum object was created by an old version of simpleDS. "
                "Assuming Nants_telescope is the equivalend to Nants_data. "
                "Please update this parameter with its true value."
            )
            self.Nants_telescope = self.Nants_data

        if "x_orientation" in header:
            self.x_orientation = header["x_orientation"][()].decode("utf-8")
        else:
            warnings.warn(
                "The parameter x_orientation is not present in the header of this "
                "save file. This Delay Spectrum object was created by an old version "
                "of simpleDS. x_orientation will be set to None. This can make some "
                "polarization conversion difficult."
            )
            self.x_orientation = None

        if "Ndelays" in header:
            self.Ndelays = int(header["Ndelays"][()])

        self.Nuv = int(header["Nuv"][()])
        self.Nspws = int(header["Nspws"][()])
        self.data_type = header["data_type"][()].decode("utf-8")
        self.vis_units = header["vis_units"][()].decode("utf-8")

        self.lst_array = header["lst_array"][:] * units.Unit(
            header["lst_array"].attrs["unit"]
        )
        self.ant_1_array = header["ant_1_array"][()]
        self.ant_2_array = header["ant_2_array"][()]
        self.baseline_array = header["baseline_array"][()]
        self.freq_array = header["freq_array"][:, :] * units.Unit(
            header["freq_array"].attrs["unit"]
        )

        if "delay_array" in header:
            self.delay_array = header["delay_array"][:] * units.Unit(
                header["delay_array"].attrs["unit"]
            )

        self.polarization_array = header["polarization_array"][()]

        self.uvw = header["uvw"][:] * units.Unit(header["uvw"].attrs["unit"])
        self.integration_time = header["integration_time"][:, :] * units.Unit(
            header["integration_time"].attrs["unit"]
        )
        if "trcvr" in header:
            self.trcvr = header["trcvr"][:, :] * units.Unit(
                header["trcvr"].attrs["unit"]
            )
        else:
            self.trcvr = np.full_like(self.freq_array.value, np.Inf) * units.K

        if "redshift" in header:
            self.readshift = header["redshift"][()]

        if "beam_area" in header:
            self.beam_area = header["beam_area"][:, :] * units.Unit(
                header["beam_area"].attrs["unit"]
            )
        if "beam_sq_area" in header:
            self.beam_sq_area = header["beam_sq_area"][:, :] * units.Unit(
                header["beam_sq_area"].attrs["unit"]
            )

        if header["taper"][()] != np.string_(self.taper.__name__):
            warnings.warn(
                "Saved taper function has a different name than "
                "the default (blackmanharris).\n"
                "Functions are not serializable and cannot be saved by hdf5. "
                "Custom taper functions must be reassigned to the object "
                "after reading. Here is some more information on your custom taper:\n"
                f"Name: {header['taper'][()]}\nClass: {header['taper'].attrs['class']}\n"
                f"Module: {header['taper'].attrs['module']}"
            )
            self.taper = None
        else:
            self.taper = windows.blackmanharris

    def _get_data(
        self,
        dgrp,
        antenna_nums,
        bls,
        spws,
        frequencies,
        freq_chans,
        delays,
        delay_chans,
        lsts,
        lst_range,
        polarizations,
        uv_index,
        cosmology,
        littleh_units,
    ):
        """Read the data like arrays from disk.

        This is an internal function used to read data from a save file.
        This is separated from the full read to allow for patial I/O and metadata only reads.

        Parameters
        ----------
        drgp : h5py dataset
            The dataset object on disk to read

        Returns
        -------
            None

        """
        (
            spw_inds,
            freq_inds,
            delay_inds,
            bl_inds,
            lst_inds,
            pol_inds,
            uv_inds,
        ) = self._select_preprocess(
            antenna_nums,
            bls,
            spws,
            frequencies,
            freq_chans,
            delays,
            delay_chans,
            lsts,
            lst_range,
            polarizations,
            uv_index,
        )

        if spw_inds is not None:
            spw_frac = len(spw_inds) / float(self.Nspws)
        else:
            spw_frac = 1

        if freq_inds is not None:
            freq_frac = len(freq_inds[0]) / float(self.Nfreqs)
        else:
            freq_frac = 1

        if delay_inds is not None:
            delay_frac = len(delay_inds) / float(self.Ndelays)
        else:
            delay_frac = 1

        if bl_inds is not None:
            bl_frac = len(bl_inds) / float(self.Nbls)
        else:
            bl_frac = 1

        if lst_inds is not None:
            lst_frac = len(lst_inds) / float(self.Ntimes)
        else:
            lst_frac = 1

        if pol_inds is not None:
            pol_frac = len(pol_inds) / float(self.Npols)
        else:
            pol_frac = 1

        if uv_inds is not None:
            uv_frac = len(uv_inds) / float(self.Nuv)
        else:
            uv_frac = 1

        select_fracs = [
            spw_frac,
            freq_frac,
            delay_frac,
            bl_frac,
            lst_frac,
            pol_frac,
            uv_frac,
        ]
        min_frac = np.min(select_fracs)

        if min_frac == 1:
            self.data_array = dgrp["visdata"][:, :, :, :, :, :] * units.Unit(
                dgrp["visdata"].attrs["unit"]
            )
            self.noise_array = dgrp["visnoise"][:, :, :, :, :, :] * units.Unit(
                dgrp["visnoise"].attrs["unit"]
            )

            self.flag_array = dgrp["flags"][:, :, :, :, :, :]
            self.nsample_array = dgrp["nsamples"][:, :, :, :, :, :]

            if "data_power" in dgrp:
                self.power_array = dgrp["data_power"][:, :, :, :, :, :] * units.Unit(
                    dgrp["data_power"].attrs["unit"]
                )

            if "noise_power" in dgrp:
                self.noise_power = dgrp["noise_power"][:, :, :, :, :, :] * units.Unit(
                    dgrp["noise_power"].attrs["unit"]
                )

            if "thermal_power" in dgrp:
                self.thermal_power = dgrp["thermal_power"][:, :, :, :, :] * units.Unit(
                    dgrp["thermal_power"].attrs["unit"]
                )
        else:
            self.update_cosmology(cosmology=cosmology, littleh_units=littleh_units)
            self._select_metadata(
                spw_inds, freq_inds, delay_inds, bl_inds, lst_inds, pol_inds, uv_inds
            )

            # open references to datasets
            visdata_dset = dgrp["visdata"]
            visnoise_dset = dgrp["visnoise"]
            flags_dset = dgrp["flags"]
            nsamples_dset = dgrp["nsamples"]

            # iterate over smallest frac first then grab remaining
            visdata_ind_set = [
                spw_inds,
                uv_inds,
                pol_inds,
                bl_inds,
                lst_inds,
                freq_inds,
            ]
            visdata_axes = [0, 1, 2, 3, 4, 5]
            visdata_inds = np.argsort(
                [spw_frac, uv_frac, pol_frac, bl_frac, lst_frac, freq_frac]
            )

            for count, axis_ind in enumerate(visdata_inds):
                if (
                    visdata_axes[axis_ind] == 5
                    and visdata_ind_set[axis_ind] is not None
                ):
                    for fq_inds in visdata_ind_set[axis_ind]:
                        slice = tuple(
                            np.s_[:] if _cnt != visdata_axes[axis_ind] else fq_inds
                            for _cnt in range(len(visdata_dset.shape))
                        )
                        if count == 0:
                            visdata = visdata_dset[slice] * units.Unit(
                                visdata_dset.attrs["unit"]
                            )
                            visnoise = visnoise_dset[slice] * units.Unit(
                                visnoise_dset.attrs["unit"]
                            )
                            flags = flags_dset[slice]
                            nsamples = nsamples_dset[slice]
                        else:
                            visdata = visdata[slice]
                            visnoise = visnoise[slice]
                            flags = flags[slice]
                            nsamples = nsamples[slice]
                else:
                    slice = tuple(
                        np.s_[:]
                        if _cnt != visdata_axes[axis_ind]
                        else (visdata_ind_set[axis_ind] or np.s_[:])
                        for _cnt in range(len(visdata_dset.shape))
                    )
                    if count == 0:
                        visdata = visdata_dset[slice] * units.Unit(
                            visdata_dset.attrs["unit"]
                        )
                        visnoise = visnoise_dset[slice] * units.Unit(
                            visnoise_dset.attrs["unit"]
                        )
                        flags = flags_dset[slice]
                        nsamples = nsamples_dset[slice]
                    else:
                        visdata = visdata[slice]
                        visnoise = visnoise[slice]
                        flags = flags[slice]
                        nsamples = nsamples[slice]

            self.data_array = visdata
            self.noise_array = visnoise
            self.flag_array = flags
            self.nsample_array = nsamples

            # iterate over smallest frac first then grab remaining
            power_ind_set = [spw_inds, pol_inds, bl_inds, bl_inds, lst_inds, delay_inds]
            power_axes = [0, 1, 2, 3, 4, 5]
            power_inds = np.argsort(
                [spw_frac, pol_frac, bl_frac, bl_frac, lst_frac, delay_frac]
            )

            for count, axis_ind in enumerate(power_inds):
                if "data_power" in dgrp:
                    power_dset = dgrp["data_power"]
                    _power = True
                else:
                    _power = False

                if "noise_power" in dgrp:
                    noise_power_dset = dgrp["noise_power"]
                    _noise = True
                else:
                    _noise = False

                if _power:
                    slice = tuple(
                        np.s_[:]
                        if _cnt != power_axes[axis_ind]
                        else (power_ind_set[axis_ind] or np.s_[:])
                        for _cnt in range(len(power_dset.shape))
                    )
                    if count == 0:
                        power_data = power_dset[slice] * units.Unit(
                            power_dset.attrs["unit"]
                        )
                    else:
                        power_data = power_data[slice]

                if _noise:
                    slice = tuple(
                        np.s_[:]
                        if _cnt != power_axes[axis_ind]
                        else (power_ind_set[axis_ind] or np.s_[:])
                        for _cnt in range(len(noise_power_dset.shape))
                    )
                    if count == 0:
                        power_noise = noise_power_dset[slice] * units.Unit(
                            noise_power_dset.attrs["unit"]
                        )
                    else:
                        power_noise = power_noise[slice]

            if _power:
                self.power_array = power_data
            if _noise:
                self.noise_power = power_noise

            if "thermal_power" in dgrp:
                thermal_dset = dgrp["thermal_power"]

                # iterate over smallest frac first then grab remaining
                power_ind_set = [spw_inds, pol_inds, bl_inds, bl_inds, lst_inds]
                power_axes = [0, 1, 2, 3, 4]
                power_inds = np.argsort(
                    [spw_frac, pol_frac, bl_frac, bl_frac, lst_frac]
                )

                for count, axis_ind in enumerate(power_inds):
                    slice = tuple(
                        np.s_[:]
                        if _cnt != power_axes[axis_ind]
                        else (power_ind_set[axis_ind] or np.s_[:])
                        for _cnt in range(len(thermal_dset.shape))
                    )
                    if count == 0:
                        thermal_data = thermal_dset[slice] * units.Unit(
                            thermal_dset.attrs["unit"]
                        )
                    else:
                        thermal_data = thermal_data[slice]

                    self.thermal_power = thermal_data

    def read(
        self,
        filename,
        cosmology=None,
        littleh_units=False,
        antenna_nums=None,
        bls=None,
        spws=None,
        frequencies=None,
        freq_chans=None,
        delays=None,
        delay_chans=None,
        lsts=None,
        lst_range=None,
        polarizations=None,
        uv_index=None,
        read_data=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Read a saved DelaySpectrum object from hdf5 file.

        Parameters
        ----------
        filename : str
            Name of file to read
        cosmology : Astropy Cosmology Object, optional
            Input assumed cosmology. Must be an astropy cosmology object.
            Cosmology objects cannot be serialized in hdf5 objects.
            Input cosmology object is used to perform cosmological normalization when data is read.
            Defaults to Planck15
        littleh_units: bool, default: False
           automatically convert to to mK^2 / (littleh / Mpc)^3 if power type arrays
           are present in the data
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object.
        bls : array_like of tuple or int, optional
            The baselines to keep in the object.
            Can either be a tuple of antenna numbers (e.g. (0, 1)) the baseline number or a combination of both.
        spectral_windows : array_like of int
            The spectral windows to keep in the object.
        frequencies : Astropy Quantity object, optional
            The frequencies to keep in the object.
            Values must match frequencies which already exist in the object.
            Only available when in 'frequency' mode
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
            When multiple spectral windows are present, this will apply to all.
            Only available when in 'frequency' mode
        delays : Astropy Quantity object, optional
            The delays to keep in the object.
            Values must match delays which already exist in the object.
            Only available when in 'delay' mode.
        delay_chans : array_like of int, optional
            The delays channel numbers to keep in the object.
            When multiple spectral windows are present, this will apply to all.
            Only available when in 'delay' mode.
        lsts : Astropy Quantity object, optional
            The time to keep in the object.
            Values must match times which already exist in the object.
        lst_range : list of Astropy Quantity, optional
            A list of length 2 with start and end times to select.
            All time values which fall in this range will be selected.
        polarizations : array_like of int, optional
            The polarizations to keep in the object.
        uv_index : array_like of int, optional
            The index of the UVData objects to keep on the object.
        read_data : bool, default: True
            Whether to read the data or return a metadata only object.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run). Ignored if read_data is False.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
            Ignored if read_data is False.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done). Ignored if read_data is False.


        """
        if not os.path.exists(filename):
            raise IOError(filename + " not found")

        with h5py.File(filename, "r") as f:
            # extract header information
            header = f["/Header"]
            self._read_header(header)

            if read_data:
                # Now read in the data
                dgrp = f["/Data"]
                self._get_data(
                    dgrp,
                    antenna_nums,
                    bls,
                    spws,
                    frequencies,
                    freq_chans,
                    delays,
                    delay_chans,
                    lsts,
                    lst_range,
                    polarizations,
                    uv_index,
                    cosmology,
                    littleh_units,
                )

        self.update_cosmology(cosmology=cosmology, littleh_units=littleh_units)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        return

    def _write_header(self, header):
        """Write all metadata to the given hdf5 header.

        Parameters
        ----------
        header : hdf5 Group object

        """
        header["Ntimes"] = self.Ntimes
        header["Nbls"] = self.Nbls
        header["Nfreqs"] = self.Nfreqs
        header["Npols"] = self.Npols
        header["Nants_data"] = self.Nants_data
        header["Nants_telescope"] = self.Nants_telescope
        if self.x_orientation is not None:
            header["x_orientation"] = self.x_orientation.encode()

        if self.Ndelays is not None:
            header["Ndelays"] = self.Ndelays

        header["Nuv"] = self.Nuv
        header["Nspws"] = self.Nspws
        header["data_type"] = self.data_type.encode()
        header["vis_units"] = self.vis_units.encode()

        header["lst_array"] = self.lst_array.value
        header["lst_array"].attrs["unit"] = self.lst_array.unit.to_string()
        header["ant_1_array"] = self.ant_1_array
        header["ant_2_array"] = self.ant_2_array
        header["baseline_array"] = self.baseline_array
        header["freq_array"] = self.freq_array.value
        header["freq_array"].attrs["unit"] = self.freq_array.unit.to_string()

        if self.delay_array is not None:
            header["delay_array"] = self.delay_array.value
            header["delay_array"].attrs["unit"] = self.delay_array.unit.to_string()

        header["polarization_array"] = self.polarization_array

        header["uvw"] = self.uvw.value
        header["uvw"].attrs["unit"] = self.uvw.unit.to_string()

        header["integration_time"] = self.integration_time.value
        header["integration_time"].attrs[
            "unit"
        ] = self.integration_time.unit.to_string()

        if self.trcvr is not None:
            if np.all(np.isfinite(self.trcvr)):
                header["trcvr"] = self.trcvr.value
                header["trcvr"].attrs["unit"] = self.trcvr.unit.to_string()

        if self.redshift is not None:
            header["redshift"] = self.redshift

        if self.beam_area is not None:
            header["beam_area"] = self.beam_area.value
            header["beam_area"].attrs["unit"] = self.beam_area.unit.to_string()
        if self.beam_sq_area is not None:
            header["beam_sq_area"] = self.beam_sq_area.value
            header["beam_sq_area"].attrs["unit"] = self.beam_sq_area.unit.to_string()

        # Objects are not serializable by hdf5.
        # For now save the name and module and issue a print/warning/error
        # on read if it is not the default?
        if np.string_(self.taper.__name__) != np.string_(
            windows.blackmanharris.__name__
        ):
            warnings.warn(
                "The given taper function has a different name than "
                "the default (blackmanharris). "
                "Functions are not serializable and cannot be saved by hdf5 but "
                "information will be stored to help readers re-initialize the taper function."
            )
        header["taper"] = np.string_(self.taper.__name__)
        header["taper"].attrs["module"] = np.string_(self.taper.__module__)
        header["taper"].attrs["class"] = np.string_(self.taper.__class__)

    def write(
        self,
        filename,
        overwrite=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        data_compression=None,
        flags_compression="lzf",
        nsample_compression="lzf",
    ):
        """Write the DelaySpectrum object out to an hdf5 file.

        Parameters
        ----------
        filename : str
            Name of file to write.
        overwrite : bool
            If True will overwrite the file if it already exsits.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file. Default is True.
        check_extra : bool
            Option to check optional parameters as well as required ones.
            Default is True.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters
            before writing the file. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the data_array. Default is None
            (no filter/compression).
        flags_compression : str
            HDF5 filter to apply when writing the flags_array.
            Default is the LZF filter.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array.
            Default is the LZF filter.

        """
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if os.path.exists(filename):
            if overwrite:
                print("File exists; overwriting file")
            else:
                raise IOError(
                    "File exists. If overwriting is desired set the clobber keyword to True."
                )

        restore_cosmo = False
        if self.power_array is not None and self.power_array.unit.is_equivalent(
            self._power_array.expected_units
        ):
            warnings.warn(
                "Cannot write DelaySpectrum objects to file when power is in "
                "cosmological units. Removing cosmological conversion factors.",
                UserWarning,
            )
            self.remove_cosmology()
            restore_cosmo = True

        with h5py.File(filename, "w") as h5file:
            header = h5file.create_group("Header")
            self._write_header(header)

            dgrp = h5file.create_group("Data")

            for _name, _data in zip(
                ["visdata", "visnoise"], [self.data_array, self.noise_array]
            ):
                visdata = dgrp.create_dataset(
                    _name,
                    chunks=True,
                    data=_data.value,
                    compression=data_compression,
                )
                visdata.attrs["unit"] = _data.unit.to_string()

            dgrp.create_dataset(
                "flags",
                chunks=True,
                data=self.flag_array,
                compression=flags_compression,
            )
            dgrp.create_dataset(
                "nsamples",
                chunks=True,
                data=self.nsample_array.astype(np.float32),
                compression=nsample_compression,
            )

            if self.power_array is not None:
                for _name, _data in zip(
                    ["data_power", "noise_power"], [self.power_array, self.noise_power]
                ):
                    power_data = dgrp.create_dataset(
                        _name,
                        chunks=True,
                        data=_data.value,
                        compression=data_compression,
                    )
                    power_data.attrs["unit"] = _data.unit.to_string()

            if self.thermal_power is not None:
                tpower = dgrp.create_dataset(
                    "thermal_power",
                    chunks=True,
                    data=self.thermal_power.value.astype(np.float32),
                    compression=nsample_compression,
                    dtype=np.float32,
                )
                tpower.attrs["unit"] = self.thermal_power.unit.to_string()

        # restore old cosmology if there was one
        if restore_cosmo:
            self.update_cosmology()

        return

    def initialize_save_file(
        self,
        filename,
        overwrite=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        data_compression=None,
        flags_compression="lzf",
        nsample_compression="lzf",
    ):
        """Initialize the full hdf5 file onto disk to allow for partial writing.

        Parameters
        ----------
        filename : str
            Name of file to write.
        overwrite : bool
            If True will overwrite the file if it already exsits.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file. Default is True.
        check_extra : bool
            Option to check optional parameters as well as required ones.
            Default is True.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters
            before writing the file. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the data_array. Default is None
            (no filter/compression).
        flags_compression : str
            HDF5 filter to apply when writing the flags_array.
            Default is the LZF filter.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array.
            Default is the LZF filter.

        Notes
        -----
        When partially writing out data, this function should be called first to
        initialize the file on disk. The data is then actually written by
        calling the write_partial method, with the same filename as the one
        specified in this function.

        """
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if os.path.exists(filename):
            if overwrite:
                print("File exists; overwriting file")
            else:
                raise IOError(
                    "File exists. If overwriting is desired set the clobber keyword to True."
                )

        with h5py.File(filename, "w") as h5file:
            header = h5file.create_group("Header")
            self._write_header(header)

            dgrp = h5file.create_group("Data")
            data_shape = self._data_array.expected_shape(self)
            dtype = self.data_array.dtype
            _unit = self.data_array.unit.to_string()

            for _name in ["visdata", "visnoise"]:
                visdata = dgrp.create_dataset(
                    _name,
                    data_shape,
                    chunks=True,
                    compression=data_compression,
                    dtype=dtype,
                )
                visdata.attrs["unit"] = _unit

            dgrp.create_dataset(
                "flags",
                self._flag_array.expected_shape(self),
                chunks=True,
                compression=flags_compression,
                dtype=self.flag_array.dtype,
            )

            dgrp.create_dataset(
                "nsamples",
                self._nsample_array.expected_shape(self),
                chunks=True,
                compression=nsample_compression,
                dtype=self.nsample_array.dtype,
            )

            power_shape = self._power_array.expected_shape(self)
            power_unit = self._power_array.expected_units[-2].to_string()
            dtype = (
                self.power_array.dtype if self.power_array is not None else np.complex64
            )
            for _name in ["data_power", "noise_power"]:
                power_data = dgrp.create_dataset(
                    _name,
                    power_shape,
                    chunks=True,
                    compression=data_compression,
                    dtype=dtype,
                )
                power_data.attrs["unit"] = power_unit

            power_unit = self._thermal_power.expected_units[-1].to_string()
            tpower = dgrp.create_dataset(
                "thermal_power",
                self._thermal_power.expected_shape(self),
                chunks=True,
                compression=nsample_compression,
                dtype=np.float32,
            )
            tpower.attrs["unit"] = power_unit

        return

    def write_partial(self, filename):
        """Write section of data into an alrady initialized hdf5 file.

        Parameters
        ----------
        filename : str
            Name of already initializd file to insert delay spetrum data.

        """
        pass
        # check that the file already exists
        if not os.path.exists(filename):
            raise AssertionError(
                f"{filename} does not exists; please first initialize it with initialize_uvh5_file"
            )

        other = DelaySpectrum()
        other.read(filename, read_data=False)
        (
            spw_inds,
            freq_inds,
            delay_inds,
            bl_inds,
            lst_inds,
            pol_inds,
            uv_inds,
        ) = other._select_preprocess(
            antenna_nums=list(set(self.ant_1_array).union(self.ant_2_array)),
            bls=[
                (i, j)
                for i, j in zip(
                    *uvutils.baseline_to_antnums(
                        self.baseline_array, self.Nants_telescope
                    )
                )
            ],
            frequencies=self.freq_array.flatten(),
            delays=self.delay_array.flatten(),
            lsts=self.lst_array,
            polarizations=self.polarization_array,
            exact_spw_match=True,
            uv_index=list(range(self.Nuv)),
        )

        if spw_inds is not None:
            if np.unique(np.diff(spw_inds)).size <= 1:
                spw_reg_spaced = True
                spw_start = spw_inds[0]
                spw_end = spw_inds[-1] + 1
                if len(spw_inds) == 1:
                    d_spw = 1
                else:
                    d_spw = spw_inds[1] - spw_inds[0]
                spw_inds = np.s_[spw_start:spw_end:d_spw]
            else:
                spw_reg_spaced = False
        else:
            spw_reg_spaced = True
            spw_inds = np.s_[:]

        if freq_inds is not None:
            # freq inds is a (Nspw, Nfeqs) list of indices, check all
            if all(np.unique(np.diff(fq)).size <= 1 for fq in freq_inds):
                freq_reg_spaced = True
                for cnt, fq_ind in enumerate(freq_inds):
                    fq_start = fq_ind[0]
                    fq_end = fq_ind[-1] + 1
                    if len(fq_ind) == 1:
                        d_fq = 1
                    else:
                        d_fq = fq_ind[1] - fq_ind[0]

                    freq_inds[cnt] = np.s_[fq_start:fq_end:d_fq]
            else:
                freq_reg_spaced = False
        else:
            freq_reg_spaced = True
            freq_inds = [np.s_[:]] * self.Nspws

        if delay_inds is not None:
            if np.unique(np.diff(delay_inds)).size <= 1:
                delay_reg_spaced = True
                delay_start = delay_inds[0]
                delay_end = delay_inds[-1] + 1
                if len(delay_inds) == 1:
                    d_delay = 1
                else:
                    d_delay = delay_inds[1] - delay_inds[0]
                delay_inds = np.s_[delay_start:delay_end:d_delay]
            else:
                delay_reg_spaced = False
        else:
            delay_reg_spaced = True
            delay_inds = np.s_[:]

        if bl_inds is not None:
            if np.unique(np.diff(bl_inds)).size <= 1:
                bl_reg_spaced = True
                bl_start = bl_inds[0]
                bl_end = bl_inds[-1] + 1
                if len(bl_inds) == 1:
                    d_bl = 1
                else:
                    d_bl = bl_inds[1] - bl_inds[0]
                bl_inds = np.s_[bl_start:bl_end:d_bl]
            else:
                bl_reg_spaced = False
        else:
            bl_reg_spaced = True
            bl_inds = np.s_[:]

        if lst_inds is not None:
            if np.unique(np.diff(lst_inds)).size <= 1:
                lst_reg_spaced = True
                lst_start = lst_inds[0]
                lst_end = lst_inds[-1] + 1
                if len(lst_inds) == 1:
                    d_lst = 1
                else:
                    d_lst = lst_inds[1] - lst_inds[0]
                lst_inds = np.s_[lst_start:lst_end:d_lst]
            else:
                lst_reg_spaced = False
        else:
            lst_reg_spaced = True
            lst_inds = np.s_[:]

        if pol_inds is not None:
            if np.unique(np.diff(pol_inds)).size <= 1:
                pol_reg_spaced = True
                pol_start = pol_inds[0]
                pol_end = pol_inds[-1] + 1
                if len(pol_inds) == 1:
                    d_pol = 1
                else:
                    d_pol = pol_inds[1] - pol_inds[0]
                pol_inds = np.s_[pol_start:pol_end:d_pol]
            else:
                pol_reg_spaced = False
        else:
            pol_reg_spaced = True
            pol_inds = np.s_[:]

        uv_inds_reg_spaced = True
        if uv_inds is not None:
            # this is going to take some thinking
            # and the use case is very niche
            raise NotImplementedError(
                "UV selection on partial write is not supported at this time."
            )
        else:
            uv_inds = np.s_[:]

        restore_cosmo = False
        if self.power_array is not None and self.power_array.unit.is_equivalent(
            self._power_array.expected_units
        ):
            warnings.warn(
                "Cannot write DelaySpectrum objects to file when power is in "
                "cosmological units. Removing cosmological conversion factors.",
                UserWarning,
            )
            self.remove_cosmology()
            restore_cosmo = True

        with h5py.File(filename, "r+") as f:
            dgrp = f["/Data"]
            visdata_dset = dgrp["visdata"]
            visnoise_dset = dgrp["visnoise"]
            flags_dset = dgrp["flags"]
            nsamples_dset = dgrp["nsamples"]

            data_reg_spaced = [
                spw_reg_spaced,
                uv_inds_reg_spaced,
                pol_reg_spaced,
                bl_reg_spaced,
                lst_reg_spaced,
                freq_reg_spaced,
            ]

            if np.count_nonzero(data_reg_spaced) >= 5:
                # fancy indexing is available
                for fq in freq_inds:
                    data_inds = tuple(
                        [spw_inds, uv_inds, pol_inds, bl_inds, lst_inds, fq]
                    )
                    visdata_dset[data_inds] = self.data_array.to_value(
                        visdata_dset.attrs["unit"]
                    )

                    visnoise_dset[data_inds] = self.noise_array.to_value(
                        visnoise_dset.attrs["unit"]
                    )

                    flags_dset[data_inds] = self.flag_array
                    nsamples_dset[data_inds] = self.nsample_array
            else:
                reg_spaced = np.nonzero(data_reg_spaced)[0]
                non_reg_inds = np.nonzero(np.logical_not(data_reg_spaced))[0]
                for fq in freq_inds:
                    indices = [spw_inds, uv_inds, pol_inds, bl_inds, lst_inds, fq]
                    non_reg = [np.ravel(indices[i]) for i in non_reg_inds]
                    for mesh_ind in product(*non_reg):
                        _inds = tuple(
                            indices[_cnt]
                            if _cnt in reg_spaced
                            else mesh_ind[np.nonzero(_cnt == non_reg_inds)[0].item()]
                            for _cnt in range(len(visdata_dset.shape))
                        )
                        data_inds = tuple(
                            indices[_cnt]
                            if _cnt in reg_spaced
                            else np.nonzero(
                                non_reg[np.nonzero(_cnt == non_reg_inds)[0].item()]
                                == mesh_ind[np.nonzero(_cnt == non_reg_inds)[0].item()]
                            )[0].item()
                            for _cnt in range(len(visdata_dset.shape))
                        )
                        visdata_dset[_inds] = self.data_array[data_inds].to_value(
                            visdata_dset.attrs["unit"]
                        )
                        visnoise_dset[_inds] = self.noise_array[data_inds].to_value(
                            visnoise_dset.attrs["unit"]
                        )
                        flags_dset[_inds] = self.flag_array[data_inds]
                        nsamples_dset[_inds] = self.nsample_array[data_inds]

            if self.power_array is not None:
                power_reg_spaced = [
                    spw_reg_spaced,
                    pol_reg_spaced,
                    bl_reg_spaced,
                    bl_reg_spaced,
                    lst_reg_spaced,
                    delay_reg_spaced,
                ]
                data_power_dset = dgrp["data_power"]
                noise_power_dset = dgrp["noise_power"]
                if np.count_nonzero(power_reg_spaced) >= 5:
                    inds = tuple(
                        [spw_inds, pol_inds, bl_inds, bl_inds, lst_inds, delay_inds]
                    )
                    data_power_dset[inds] = self.power_array.to_value(
                        data_power_dset.attrs["unit"]
                    )
                    noise_power_dset[inds] = self.noise_power.to_value(
                        noise_power_dset.attrs["unit"]
                    )
                else:
                    reg_spaced = np.nonzero(power_reg_spaced)[0]
                    non_reg_inds = np.nonzero(np.logical_not(power_reg_spaced))[0]
                    indices = [
                        spw_inds,
                        pol_inds,
                        bl_inds,
                        bl_inds,
                        lst_inds,
                        delay_inds,
                    ]
                    non_reg = [np.ravel(indices[i]) for i in non_reg_inds]
                    for mesh_ind in product(*non_reg):
                        _inds = tuple(
                            indices[_cnt]
                            if _cnt in reg_spaced
                            else mesh_ind[np.nonzero(_cnt == non_reg_inds)[0].item()]
                            for _cnt in range(len(visdata_dset.shape))
                        )
                        data_inds = tuple(
                            indices[_cnt]
                            if _cnt in reg_spaced
                            else np.nonzero(
                                non_reg[np.nonzero(_cnt == non_reg_inds)[0].item()]
                                == mesh_ind[np.nonzero(_cnt == non_reg_inds)[0].item()]
                            )[0].item()
                            for _cnt in range(len(visdata_dset.shape))
                        )
                        data_power_dset[_inds] = self.power_array[data_inds].to_value(
                            data_power_dset.attrs["unit"]
                        )
                        noise_power_dset[_inds] = self.noise_power[data_inds].to_value(
                            noise_power_dset.attrs["unit"]
                        )

            if self.thermal_power is not None:
                thermal_reg_spaced = [
                    spw_reg_spaced,
                    pol_reg_spaced,
                    bl_reg_spaced,
                    bl_reg_spaced,
                    lst_reg_spaced,
                ]
                thermal_dset = dgrp["thermal_power"]
                if np.count_nonzero(thermal_reg_spaced) >= 4:
                    inds = tuple([spw_inds, pol_inds, bl_inds, bl_inds, lst_inds])
                    thermal_dset[inds] = self.thermal_power.to_value(
                        thermal_dset.attrs["unit"]
                    )
                else:
                    reg_spaced = np.nonzero(thermal_reg_spaced)[0]
                    non_reg_inds = np.nonzero(np.logical_not(thermal_reg_spaced))[0]
                    indices = [spw_inds, pol_inds, bl_inds, bl_inds, lst_inds]
                    non_reg = [np.ravel(indices[i]) for i in non_reg_inds]
                    for mesh_ind in product(*non_reg):
                        _inds = tuple(
                            indices[_cnt]
                            if _cnt in reg_spaced
                            else mesh_ind[np.nonzero(_cnt == non_reg_inds)[0].item()]
                            for _cnt in range(len(thermal_dset.shape))
                        )
                        data_inds = tuple(
                            indices[_cnt]
                            if _cnt in reg_spaced
                            else np.nonzero(
                                non_reg[np.nonzero(_cnt == non_reg_inds)[0].item()]
                                == mesh_ind[np.nonzero(_cnt == non_reg_inds)[0].item()]
                            )[0].item()
                            for _cnt in range(len(thermal_dset.shape))
                        )

                        thermal_dset[_inds] = self.thermal_power[data_inds].to_value(
                            thermal_dset.attrs["unit"]
                        )

        # restore old cosmology if there was one
        if restore_cosmo:
            self.update_cosmology()

        return

    def select_spectral_windows(self, spectral_windows=None, freqs=None, inplace=True):
        """Select the spectral windows from loaded data.

        Parameters
        ----------
        spectral_windows: tuple of tuples, or tuple of indices, or list of lists, or list of indices; Default selection is (0, Nfreqs)
            Spectral windows ranges like (start_index, end_index) where the indices are the frequency channel numbers.
        freqs : Quantity array of the shape (Nspws, Nfreqs) units equivalent to frequency, optional
            The frequency values to select from the held data set. Input frequencies must match exactly to some subset of frequencies stored in self.freq_array.

        inplace : Bool; Default True
                 choose whether spectral window selection is done inplace on the object, or a new object is returned.

        Returns
        -------
        DelaySpectrum Object: Default None
            If inplace is False then returns new object with given spectral windows

        Raises
        ------
        ValueError
            Spectral window tuples must only have two elements (stard_index, end_index)
            Spectra windows must all have the same size
            If given frequencies not present in frequency array.

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
                            raise ValueError(
                                f"Frequency {_f.to('Mhz'):f} not found in "
                                "frequency array"
                            )
                spectral_windows.append(sp)
            spectral_windows = [[sp[0], sp[-1]] for sp in spectral_windows]
        if spectral_windows is None:
            spectral_windows = [(0, this.Nfreqs - 1)]

        spectral_windows = uvutils._get_iterable(spectral_windows)
        if not isinstance(spectral_windows[0], (list, tuple, np.ndarray)):
            spectral_windows = [spectral_windows]

        if not all(len(x) == 2 for x in spectral_windows):
            raise ValueError(
                "Spectral window tuples must only have two "
                "elements (stard_index, end_index)"
            )

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
        this.Nfreqs = int(num_freqs[0])
        this.freq_array = np.take(this.freq_array, freq_chans)
        this.trcvr = np.take(this.trcvr, freq_chans)

        freq_chans = freq_chans.flatten()
        # For beam_area and beam_sq_area transpose to (Npols, Nspws, Nfreqs)
        # Take flattened freqs, then reshape to correct shapes and transpose back
        beam_sq_area = np.transpose(this.beam_sq_area, [1, 0, 2])
        beam_sq_area = beam_sq_area.reshape(this.Npols, -1)
        beam_sq_area = np.take(beam_sq_area, freq_chans, axis=1)
        beam_sq_area = beam_sq_area.reshape(this.Npols, this.Nspws, this.Nfreqs)
        this.beam_sq_area = np.transpose(beam_sq_area, [1, 0, 2])

        beam_area = np.transpose(this.beam_area, [1, 0, 2])
        beam_area = beam_area.reshape(this.Npols, -1)
        beam_area = np.take(beam_area, freq_chans, axis=1)
        beam_area = beam_area.reshape(this.Npols, this.Nspws, this.Nfreqs)
        this.beam_area = np.transpose(beam_area, [1, 0, 2])

        # to make take easier, reorder to Nuv, Npols, Nbls, Ntimes, Nspws, Nfreqs
        this.data_array = this._take_spectral_windows_from_data_like_array(
            this.data_array, freq_chans
        )
        this.noise_array = this._take_spectral_windows_from_data_like_array(
            this.noise_array, freq_chans
        )
        this.nsample_array = this._take_spectral_windows_from_data_like_array(
            this.nsample_array, freq_chans
        )
        this.flag_array = this._take_spectral_windows_from_data_like_array(
            this.flag_array, freq_chans
        )

        # This seems obvious for an FFT but in the case that something more
        # sophisticated is added later this hook will exist.
        this.Ndelays = int(this.Nfreqs)
        delays = np.fft.fftfreq(this.Ndelays, d=np.diff(this.freq_array[0])[0].value)
        delays = np.fft.fftshift(delays) << 1.0 / this.freq_array.unit
        this.delay_array = delays << units.ns

        this.update_cosmology()

        this.check(check_extra=True, run_check_acceptability=True)

        if not inplace:
            return this
        else:
            return

    def _take_spectral_windows_from_data_like_array(self, data, inds):
        """Reshape and take spectral windows along the frequency axis.

        This take is only for arrays of shape like self.data_array
        Parameters
        ----------
        data : self.data_array like
            array of shape self.data_array to be selected
        inds : array_like of ind
            flattened indices of frequency array to select
        """
        data = np.transpose(data, [1, 2, 3, 4, 0, 5])
        # easily flatten the last two axes for spectral window selection.
        shape = tuple([_s for _s in data.shape[:4]])
        shape = shape + (-1,)
        data = data.reshape(shape)
        data = np.take(data, inds, axis=4)
        # reshape to get Nspws, Nfreqs correct along 2 last dimensions
        data = data.reshape(
            self.Nuv, self.Npols, self.Nbls, self.Ntimes, self.Nspws, self.Nfreqs
        )
        #  reorder to Nsps, Nuv, Npos, Nbls, Ntimes, Nfreqs
        data = np.transpose(data, [4, 0, 1, 2, 3, 5])
        return data

    def generate_noise(self):
        """Simulate noise based on meta-data of observation."""
        noise_power = self.calculate_noise_power()
        self.noise_array = utils.generate_noise(noise_power)

    def add_uvbeam(
        self,
        uvb,
        no_read_trcvr=False,
        use_exact=False,
        bounds_error=False,
        fill_value="extrapolate",
        kind="cubic",
    ):
        """Add the beam_area and beam_square_area integrals into memory.

        Also adds receiver temperature information if set in UVBeam object.
        By default will interpolate the beam_area and beam_sq_area value
        to the frequencies defined in the DelaySpectrum object.

        Exact values from the UVBeam object can be used if frequencies match
        exactly with the DelaySpectrum object by setting the 'use_exact' flag.

        Parameters
        ----------
        uvb : UVBeam object
            Reads `beam_area` and `beam_sq_area` from input UVBeam object.
            Currently assumes 1 beam object can describe all baselines
            Must be a power beam in healpix coordinates and peak normalized
        no_read_trcvr : (Bool, default False)
           Flag to Not read trcvr from UVBeam object even if set in UVBeam.
           This is useful if a trcvr wants to be manually set but
           a beam read from a file which also contains receiver temperature information.
        use_exact : Bool
            Use frequencies exactly out of the UVBeam object. If frequencies are
            not found use the UVBeam's interp to interpolate along the frequency
            dimension.
        bounds_error : Bool
            scipy.interpolate.interp1d bounds_error flag. When set to True,
            interpolate will error if new frequencies lay outisde of the bounds
            of the old frequencies
        fill_value : float or str
            scipy.interpolate.interp1d fill_value flag. If set to a float, will
            use that float to fill values outside of the bounds when bounds_error
            is False. If set to 'extrapolate' will extrapolate values outside
            of the original bounds.
        kind : str
            scipy.interpolate.interp1d kind flag. defines the type of interpolation
            (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’)

        """
        if use_exact:
            for spw, freqs in enumerate(self.freq_array):
                if any(f not in uvb.freq_array.squeeze() for f in freqs.to_value("Hz")):
                    uvb.freq_interp_kind = kind
                    _beam = uvb.interp(freq_array=freqs.to("Hz").value, new_object=True)
                else:
                    _beam = uvb.select(frequencies=freqs.to("Hz").value, inplace=False)

                for pol_cnt, pol in enumerate(self.polarization_array):
                    self.beam_area[spw, pol_cnt, :] = units.Quantity(
                        _beam.get_beam_area(pol=pol), unit=units.sr
                    )
                    self.beam_sq_area[spw, pol_cnt, :] = units.Quantity(
                        _beam.get_beam_sq_area(pol=pol), unit=units.sr
                    )

                    if (
                        _beam.receiver_temperature_array is not None
                        and not no_read_trcvr
                    ):
                        self.trcvr[spw, :] = units.Quantity(
                            _beam.receiver_temperature_array[0], unit=units.K
                        )
        else:
            for pol_cnt, pol in enumerate(self.polarization_array):
                beam_sq_interp = interp1d(
                    np.asarray(uvb.freq_array.squeeze()),
                    uvb.get_beam_sq_area(pol=pol),
                    bounds_error=bounds_error,
                    fill_value=fill_value,
                    kind=kind,
                )
                beam_area_interp = interp1d(
                    np.asarray(uvb.freq_array.squeeze()),
                    uvb.get_beam_area(pol=pol),
                    bounds_error=bounds_error,
                    fill_value=fill_value,
                    kind=kind,
                )
                if uvb.receiver_temperature_array is not None and not no_read_trcvr:
                    trcvr_interp = interp1d(
                        np.asarray(uvb.freq_array.squeeze()),
                        uvb.receiver_temperature_array[0],
                        bounds_error=bounds_error,
                        fill_value=fill_value,
                        kind=kind,
                    )

                for spw, freqs in enumerate(self.freq_array):
                    self.beam_area[spw, pol_cnt, :] = units.Quantity(
                        beam_area_interp(freqs.to_value("Hz")),
                        unit=units.sr,
                        dtype=np.float64,
                    )

                    self.beam_sq_area[spw, pol_cnt, :] = units.Quantity(
                        beam_sq_interp(freqs.to_value("Hz")),
                        unit=units.sr,
                        dtype=np.float64,
                    )

                    if uvb.receiver_temperature_array is not None and not no_read_trcvr:
                        self.trcvr[spw, :] = units.Quantity(
                            trcvr_interp(freqs.to_value("Hz")),
                            unit=units.K,
                            dtype=np.float64,
                        )

    @units.quantity_input(trcvr=units.K)
    def add_trcvr(self, trcvr):
        """Add the receiver temperature used to generate noise simulation.

        Parameters
        ----------
        trcvr: astropy Quantity, units: Kelvin
               (Nspws, Nfreqs) array of temperatures
               if a single temperature, it is assumed to be constant at all frequencies.

        Raises
        ------
        ValueError
            If input trcvr is not a single scalar but also not correctly shaped.

        """
        if trcvr.size == 1:
            self.trcvr = np.ones(shape=self._trcvr.expected_shape(self)) * trcvr
        elif trcvr.shape[0] != self.Nspws or trcvr.shape[1] != self.Nfreqs:
            raise ValueError(
                "If input receiver temperature is not a scalar "
                "Quantity, must shape (Nspws, Nfreqs). "
                f"Expected shape was {(self.Nspws, self.Nfreqs)}, but input shape "
                f"was {trcvr.shape}"
            )
        else:
            self.trcvr = trcvr

    def normalized_fourier_transform(self, inverse=False):
        """Perform a normalized Fourier Transform along frequency dimension.

        Local wrapper for function normalized_fourier_transform.
        Uses astropy quantities to properly normalize an FFT accounting for the Volume factor and units.

        Parameters
        ----------
        inverse: bool, default False
                 perform the inverse Fourier Transform with np.fft.ifft

        """
        if inverse is True:
            delta_x = np.diff(self.delay_array)[0]
        else:
            delta_x = np.diff(self.freq_array[0])[0]
        float_flags = np.logical_not(self.flag_array).astype(float)
        self.data_array = utils.normalized_fourier_transform(
            (self.data_array * float_flags),
            delta_x=delta_x,
            axis=-1,
            taper=self.taper,
            inverse=inverse,
        )
        self.noise_array = utils.normalized_fourier_transform(
            (self.noise_array * float_flags),
            delta_x=delta_x,
            axis=-1,
            taper=self.taper,
            inverse=inverse,
        )

    def delay_transform(self):
        """Perform a delay transform on the stored data array.

        If data is set to frequency domain, fourier transforms to delay space.
        If data is set to delay domain, inverse fourier transform to frequency space.
        """
        if self.data_array.unit == units.dimensionless_unscaled:
            warnings.warn(
                "Fourier Transforming uncalibrated data. Units will "
                "not have physical meaning. "
                "Data will be arbitrarily scaled.",
                UserWarning,
            )
        if self.data_type == "frequency":
            self.normalized_fourier_transform()
            self.set_delay()
        elif self.data_type == "delay":
            self.normalized_fourier_transform(inverse=True)
            self.set_frequency()
        else:
            raise ValueError(
                f"Unknown data type: {self.data_type}. Unable to perform "
                "delay transformation."
            )

    def calculate_noise_power(self):
        """Use the radiometry equation to generate the expected noise power."""
        Tsys = (
            180.0
            * units.K
            * np.power(self.freq_array.to("GHz") / (0.18 * units.GHz), -2.55)
        )
        Tsys += self.trcvr.to("K")
        Tsys = Tsys.reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs)
        delta_f = np.diff(self.freq_array[0])[0]
        # if any of the polarizations are psuedo-stokes then there Should
        # be a factor of 1/sqrt(2) factor divided as per Cheng 2018
        npols_noise = np.array(
            [2 if p in np.arange(1, 5) else 1 for p in self.polarization_array]
        )
        npols_noise = npols_noise.reshape(1, 1, self.Npols, 1, 1, 1)
        # when there are 0's of infs in the trcvr or integration_time arrays
        # we would notmally get an error/warning form numpy but we can supress
        # those for now.
        with np.errstate(divide="ignore", invalid="ignore"):
            noise_power = Tsys.to("K") / np.sqrt(
                delta_f.to("1/s")
                * self.integration_time.to("s").reshape(
                    1, 1, 1, self.Nbls, self.Ntimes, 1
                )
                * self.nsample_array
            )
        # Want to put noise into Jy units
        # the normalization of noise is defined to have this 1/beam_integral
        # factor in temperature units, so we need to multiply it get them into
        # Janskys
        noise_power = np.ma.masked_invalid(
            noise_power.to("mK")
            * self.beam_area.reshape(self.Nspws, 1, self.Npols, 1, 1, self.Nfreqs)
            / utils.jy_to_mk(self.freq_array).reshape(
                self.Nspws, 1, 1, 1, 1, self.Nfreqs
            )
        )
        noise_power = noise_power.filled(0)
        return noise_power.to("Jy")

    def calculate_delay_spectrum(
        self,
        run_check=True,
        run_check_acceptability=True,
        cosmology=None,
        littleh_units=False,
    ):
        """Perform Delay tranform and cross multiplication of datas.

        Take the normalized Fourier transform of the data in objects and cross multiplies baselines.
        Also generates white noise given the frequency range and trcvr and calculates the expected noise power.

        Parameters
        ----------
        cosmology: Astropy.Cosmology subclass
            Default: None (uses cosmology object saved in self.cosmology)
            Input assumed cosmology
            Setting this value will overwrite the cosmology set on the DelaySpectrum Object.
        littleh_units: Bool, default:False
                       automatically convert to to mK^2 / (littleh / Mpc)^3. Only applies in python 3.

        Raises
        ------
        ValueError
            If called before loading a uvdata object

        """
        if self.Nuv == 0:
            raise ValueError(
                "No data has be loaded. Add UVData objects before "
                "calling calculate_delay_spectrum."
            )
        if self.data_type == "frequency":
            self.delay_transform()

        if self.Nuv == 1:
            self.power_array = utils.cross_multiply_array(
                array_1=self.data_array[:, 0], axis=2
            )
            self.noise_power = utils.cross_multiply_array(
                array_1=self.noise_array[:, 0], axis=2
            )
        else:
            self.power_array = utils.cross_multiply_array(
                array_1=self.data_array[:, 0], array_2=self.data_array[:, 1], axis=2
            )
            self.noise_power = utils.cross_multiply_array(
                array_1=self.noise_array[:, 0], array_2=self.noise_array[:, 1], axis=2
            )
        self.calculate_thermal_sensitivity()
        self.update_cosmology(cosmology=cosmology, littleh_units=littleh_units)

    def calculate_thermal_sensitivity(self):
        """Calculate the Thermal sensitivity for the power spectrum.

        Uses the 21cmsense_calc formula:
            Tsys**2/(inttime * Nbls * Npols * sqrt(N_lstbins * 2))

        Divide by the following factors:
            Nbls:
                baselines should coherently add together.
            Npols:
                number of linear polarizations combined (2 if psuedo-Stokes).
            sqrt(2):
                noise is split between real and imaginary.
            sqrt(lst_bins):
                noise power spectrum averages incoherently over time.


        """
        if self.Nuv == 1:
            thermal_noise_samples = utils.combine_nsamples(
                self.nsample_array[:, 0], self.nsample_array[:, 0], axis=2
            )
        else:
            thermal_noise_samples = utils.combine_nsamples(
                self.nsample_array[:, 0], self.nsample_array[:, 1], axis=2
            )
        npols_noise = np.array(
            [2 if p in np.arange(1, 5) else 1 for p in self.polarization_array]
        )
        npols_noise = npols_noise.reshape(1, self.Npols, 1, 1, 1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Tsys = (
                180.0
                * np.power(self.freq_array.to("GHz") / (0.18 * units.GHz), -2.55).value
            ) << units.K
            Tsys += self.trcvr.to("K")
            Tsys = Tsys.reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs)
            thermal_power = (
                Tsys.to("mK")
                * self.beam_area.reshape(self.Nspws, self.Npols, 1, 1, 1, self.Nfreqs)
                / np.sqrt(
                    self.integration_time.to("s").reshape(
                        1, 1, 1, self.Nbls, self.Ntimes, 1
                    )
                    * thermal_noise_samples.reshape(
                        self.Nspws,
                        self.Npols,
                        self.Nbls,
                        self.Nbls,
                        self.Ntimes,
                        self.Nfreqs,
                    )
                    * npols_noise
                    # sqrt 2 factor because taking complex variance
                    # but only interested in real part of the power spectrum
                    * np.sqrt(2)
                )
            )
            # integrate the noise temperature over the bands being Fourier Transformed
            thermal_power = integrate.trapz(
                np.ma.masked_invalid(thermal_power.value) ** 2
                * self.taper(self.Nfreqs).reshape(1, 1, 1, 1, 1, self.Nfreqs) ** 2
                * self.freq_array.reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs) ** 4,
                x=self.freq_array.value.reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs),
                axis=-1,
            ) << (
                # four power from nu^4 one power from the integration
                self.freq_array.unit**5
                * thermal_power.unit**2
            )
        self.thermal_power = thermal_power << units.Unit("K^2 sr^2 Hz^6")
