# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Calculate Delay Spectrum from pyuvdata object."""
import os
import copy
import h5py
import numpy as np
import warnings
from pyuvdata import UVData, utils as uvutils
from pyuvdata.uvbase import UVBase
import astropy.units as units
from astropy import constants as const
from astropy.cosmology.core import Cosmology as reference_cosmology_object

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
        # self._Nblts = UnitParameter('Nblts', description='Number of baseline-times '
        #                            '(i.e. number of spectra). Not necessarily '
        #                            'equal to Nbls * Ntimes', expected_type=int)
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
            expected_type=(np.complex, np.complex128),
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
            expected_type=(np.complex, np.complex128),
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
            expected_type=(np.float),
        )

        desc = "Boolean flag, True is flagged, shape: same as data_array."
        self._flag_array = UnitParameter(
            "flag_array",
            description=desc,
            value_not_quantity=True,
            form=("Nspws", "Nuv", "Npols", "Nbls", "Ntimes", "Nfreqs"),
            expected_type=np.bool,
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
            expected_type=np.float,
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
            expected_type=np.float,
            tols=1e-3 * units.Hz,
            expected_units=units.Hz,
        )

        desc = "Array of delay, shape (Ndelays), units ns"
        self._delay_array = UnitParameter(
            "delay_array",
            description=desc,
            form=("Ndelays",),
            expected_type=np.float,
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
            expected_type=np.float,
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
            expected_type=np.float,
            form=("Nspws", "Nfreqs"),
            expected_units=units.K,
        )

        desc = (
            "Mean redshift of given frequencies. Calculated with assumed " "cosmology."
        )
        self._redshift = UnitParameter(
            "redshift",
            description=desc,
            expected_type=np.float,
            form=("Nspws",),
            expected_units=units.dimensionless_unscaled,
        )

        _kval_units = (1.0 / units.Mpc, units.littleh / units.Mpc)

        desc = (
            "Cosmological wavenumber of spatial modes probed perpendicular "
            " to the line of sight. In python 2 this unit is always in 1/Mpc. "
            "For python 3 users, it is possible to convert the littleh/Mpc "
            "using the littleh_units boolean flag in update_cosmology."
        )
        self._k_perpendicular = UnitParameter(
            "k_perpendicular",
            description=desc,
            expected_type=np.float,
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
            expected_type=np.float,
            form=("Nspws", "Ndelays"),
            expected_units=_kval_units,
        )
        _power_units = (
            (units.mK ** 2 * units.Mpc ** 3),
            (units.mK ** 2 * (units.Mpc / units.littleh) ** 3),
            (units.Hz ** 2),
        )

        desc = (
            "The cross-multiplied power spectrum estimates. "
            "Units are converted to cosmological frame (mK^2/(hMpc^-1)^3)."
            "For uncalibrated data the cosmological power is not well defined "
            "the power array instead represents the power in the delay domain "
            "adn will have units (Hz^2). "
            "In python 2 this unit is always in mK^2 Mpc^3. "
            "For python 3 users, it is possible to convert the mK^2 / (littleh/Mpc)^3 "
            "using the littleh_units boolean flag in update_cosmology."
        )
        self._power_array = UnitParameter(
            "power_array",
            description=desc,
            expected_type=np.complex,
            required=False,
            form=("Nspws", "Npols", "Nbls", "Nbls", "Ntimes", "Ndelays"),
            expected_units=_power_units,
        )
        _noise_power_units = (
            (units.mK ** 2 * units.Mpc ** 3),
            (units.mK ** 2 * (units.Mpc / units.littleh) ** 3),
            (units.Jy * units.Hz) ** 2,
        )
        desc = (
            "The cross-multiplied simulated noise power spectrum estimates. "
            "Units are converted to cosmological frame (mK^2/(hMpc^-1)^3)."
            "For uncalibrated data the noise simulation is not well defined "
            "but is still calculated and will have units (Jy Hz)^2. "
            "In python 2 this unit is always in mK^2 Mpc^3. "
            "For python 3 users, it is possible to convert the mK^2 / (littleh/Mpc)^3 "
            "using the littleh_units boolean flag in update_cosmology."
        )
        self._noise_power = UnitParameter(
            "noise_power",
            description=desc,
            expected_type=np.complex,
            required=False,
            form=("Nspws", "Npols", "Nbls", "Nbls", "Ntimes", "Ndelays"),
            expected_units=_noise_power_units,
        )

        _thermal_power_units = (
            (units.mK ** 2 * units.Mpc ** 3),
            (units.mK ** 2 * (units.Mpc / units.littleh) ** 3),
        )
        desc = (
            "The predicted thermal variance of the input data averaged over "
            "all input baselines."
            "Units are converted to cosmological frame (mK^2/(hMpc^-1)^3). "
            "In python 2 this unit is always in mK^2 Mpc^3. "
            "For python 3 users, it is possible to convert the mK^2 / (littleh/Mpc)^3 "
            "using the littleh_units boolean flag in update_cosmology."
        )
        self._thermal_power = UnitParameter(
            "thermal_power",
            description=desc,
            expected_type=np.float,
            required=False,
            form=("Nspws", "Npols", "Nbls", "Nbls", "Ntimes"),
            expected_units=_thermal_power_units,
        )

        _conversion_units = (
            (units.mK ** 2 * units.Mpc ** 3 / (units.Jy * units.Hz) ** 2),
            (units.mK ** 2 * units.Mpc ** 3 / (units.K * units.sr * units.Hz) ** 2),
            (
                units.mK ** 2
                * (units.Mpc / units.littleh) ** 3
                / (units.Jy * units.Hz) ** 2
            ),
            (
                units.mK ** 2
                * (units.Mpc / units.littleh) ** 3
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
            expected_type=np.float,
            form=("Nspws", "Npols"),
            expected_units=_conversion_units,
        )
        _tconversion_units = (
            (units.mK ** 2 * units.Mpc ** 3 / (units.K * units.sr * units.Hz) ** 2),
            (
                units.mK ** 2
                * (units.Mpc / units.littleh) ** 3
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
            expected_type=np.float,
            form=("Nspws", "Npols"),
            expected_units=_tconversion_units,
        )
        desc = "The integral of the power beam area. Shape = (Nspws, Npols, Nfreqs)"
        self._beam_area = UnitParameter(
            "beam_area",
            description=desc,
            form=("Nspws", "Npols", "Nfreqs"),
            expected_type=np.float,
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
            expected_type=np.float,
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
            expected_type=np.float,
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
                            "UnitParameter {param} is not expected shape. "
                            "Parameter shape is {pshape}, expected shape is "
                            "{eshape}.".format(
                                param=p, pshape=np.shape(param.value), eshape=eshape
                            )
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
                                    "has units {0} "
                                    "which are not equivalent to "
                                    "expected units of {1}.".format(
                                        param.value.unit, param.expected_units
                                    )
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
        this.vis_units = uv.vis_units
        this.Npols = uv.Npols
        this.Nspws = 1
        this.Nuv = 1
        this.lst_array = np.unique(uv.lst_array) * units.rad
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
            shape=this._nsample_array.expected_shape(this), dtype=np.float
        )
        temp_data[:, :, :, :, :] = utils.get_nsample_array(
            uv, reds=this.baseline_array, squeeze=False
        )
        this.nsample_array = copy.deepcopy(temp_data)

        temp_data = np.ones(shape=this._flag_array.expected_shape(this), dtype=np.bool)
        temp_data[:, :, :, :, :] = utils.get_flag_array(
            uv, reds=this.baseline_array, squeeze=False
        )
        this.flag_array = copy.deepcopy(temp_data)

        temp_data = np.zeros(
            shape=this._integration_time.expected_shape(this), dtype=np.float
        )
        temp_data[:, :] = utils.get_integration_time(
            uv, reds=this.baseline_array, squeeze=False
        )

        this.integration_time = copy.deepcopy(temp_data) << units.s
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
                "Saved units are: {dspec}, "
                "input units are: {uvin}.".format(
                    dspec=self.data_array.unit, uvin=this.data_array.unit
                )
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
                        "loaded data. Parameter {name} is not "
                        "the same.".format(name=p)
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
                        "by {time:}. Keeping LST array stored from "
                        "the first data set read.".format(
                            time=time_diff.mean().to("min")
                        ),
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
                        my_parm.expected_shape(self), dtype=my_parm.expected_type
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
            raise ValueError("Cannot remove cosmology of type {}".format(type(None)))

        self.k_parallel = None
        self.k_perpendicular = None
        # If power spectrum estimation has already occurred, need to re-normalize
        # in the new cosmological framework.
        if self.power_array is not None:
            if self.power_array.unit.is_equivalent(
                units.mK ** 2 * units.Mpc ** 3 / units.littleh ** 3
            ):
                self.power_array = self.power_array.to(
                    units.mK ** 2 * units.Mpc ** 3, units.with_H0(self.cosmology.H0)
                )
                self.noise_power = self.noise_power.to(
                    units.mK ** 2 * units.Mpc ** 3, units.with_H0(self.cosmology.H0)
                )
            # only divide by the conversion when power array is in cosmological units
            # e.g. not if this is the first time, or if calculate_delay_spectrum was just called.
            if (
                self.unit_conversion is not None
                and not self.power_array.unit.is_equivalent(
                    (
                        units.Jy ** 2 * units.Hz ** 2,
                        units.K ** 2 * units.sr ** 2 * units.Hz ** 2,
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
                units.mK ** 2 * units.Mpc ** 3 / units.littleh ** 3
            ):
                self.thermal_power = self.thermal_power.to(
                    units.mK ** 2 * units.Mpc ** 3, units.with_H0(self.cosmology.H0)
                )
            # only divide by the conversion when power array is in cosmological units
            # e.g. not if this is the first time, or if calculate_delay_spectrum was just called.
            if (
                self.thermal_conversion is not None
                and not self.thermal_power.unit.is_equivalent(
                    (
                        units.Jy ** 2 * units.Hz ** 2,
                        units.K ** 2 * units.sr ** 2 * units.Hz ** 2,
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
                    const.c ** 2 * units.sr / (2 * const.k_B)
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
                1.0
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
                "mK^2 Mpc^3 /( K^2 sr^2 Hz^2)"
            )
            self.thermal_power = self.thermal_power * self.thermal_conversion.reshape(
                self.Nspws, self.Npols, 1, 1, 1
            )
            self.thermal_power = self.thermal_power << units.Unit("mK^2 Mpc^3")

        if littleh_units:
            self.k_perpendicular = self.k_perpendicular.to(
                "littleh/Mpc", units.with_H0(self.cosmology.H0)
            )
            self.k_parallel = self.k_parallel.to(
                "littleh/Mpc", units.with_H0(self.cosmology.H0)
            )
            if self.power_array is not None:
                self.power_array = self.power_array.to(
                    "mK^2 Mpc^3/littleh^3", units.with_H0(self.cosmology.H0)
                )
                self.noise_power = self.noise_power.to(
                    "mK^2 Mpc^3/littleh^3", units.with_H0(self.cosmology.H0)
                )
            if self.thermal_power is not None:
                self.thermal_power = self.thermal_power.to(
                    "mK^2 Mpc^3/littleh^3", units.with_H0(self.cosmology.H0)
                )

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
        if "Ndelays" in header:
            self.Ndelays = int(header["Ndelays"][()])

        self.Nuv = int(header["Nuv"][()])
        self.Nspws = int(header["Nspws"][()])
        self.data_type = header["data_type"][()]
        self.vis_units = header["vis_units"][()]

        self.lst_array = header["lst_array"][:] * units.Unit(
            header["lst_array"].attrs["unit"]
        )
        self.ant_1_array = header["ant_1_array"][:]
        self.ant_2_array = header["ant_2_array"][:]
        self.baseline_array = header["baseline_array"][:]
        self.freq_array = header["freq_array"][:, :] * units.Unit(
            header["freq_array"].attrs["unit"]
        )

        if "delay_array" in header:
            self.delay_array = header["delay_array"][:] * units.Unit(
                header["delay_array"].attrs["unit"]
            )

        self.polarization_array = header["polarization_array"][:]

        self.uvw = header["uvw"][:] * units.Unit(header["uvw"].attrs["unit"])
        self.integration_time = header["integration_time"][:, :] * units.Unit(
            header["integration_time"].attrs["unit"]
        )
        if "trcvr" in header:
            print(header["trcvr"][:, :])
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
                "Name: {name}\nClass: {classname}\nModule: {module}".format(
                    name=header["taper"][()],
                    classname=header["taper"].attrs["class"],
                    module=header["taper"].attrs["module"],
                )
            )
            self.taper = None
        else:
            self.taper = windows.blackmanharris

    def _get_data(self, dgrp):
        """Read the data like arrays from disk.

        Parameters
        ----------
        drgp : h5py dataset
            The dataset object on disk to read
        data_array_dtype : numpy dtype object
            the custom dtype to read data as.

        Returns
        -------
            None

        """
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

    def read(
        self,
        filename,
        cosmology=None,
        littleh_units=False,
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
        littleh_units: Bool, default: False
           automatically convert to to mK^2 / (littleh / Mpc)^3 if power type arrays
           are present in the data
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

            # Now read in the data
            dgrp = f["/Data"]
            self._get_data(dgrp)

        self.update_cosmology(cosmology=cosmology, littleh_units=littleh_units)

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
        if self.Ndelays is not None:
            header["Ndelays"] = self.Ndelays

        header["Nuv"] = self.Nuv
        header["Nspws"] = self.Nspws
        header["data_type"] = self.data_type
        header["vis_units"] = self.vis_units

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
                raise IOError("File exists; skipping")

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
                    _name, chunks=True, data=_data.value, compression=data_compression,
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
                                "Frequency {f} not found in "
                                "frequency array".format(_f.to("Mhz"))
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
        this.Nfreqs = np.int(num_freqs[0])
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
        this.Ndelays = np.int(this.Nfreqs)
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
                    self.beam_area[spw, pol_cnt, :] = (
                        _beam.get_beam_area(pol=pol) << units.sr
                    )
                    self.beam_sq_area[spw, pol_cnt, :] = (
                        _beam.get_beam_sq_area(pol=pol) << units.sr
                    )

                    if (
                        _beam.receiver_temperature_array is not None
                        and not no_read_trcvr
                    ):
                        self.trcvr[spw, :] = (
                            _beam.receiver_temperature_array[0] << units.K
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
                    self.beam_area[spw, pol_cnt, :] = (
                        beam_area_interp(freqs.to_value("Hz")) << units.sr
                    )

                    self.beam_sq_area[spw, pol_cnt, :] = (
                        beam_sq_interp(freqs.to_value("Hz")) << units.sr
                    )

                    if uvb.receiver_temperature_array is not None and not no_read_trcvr:
                        self.trcvr[spw, :] = (
                            trcvr_interp(freqs.to_value("Hz")) << units.K
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
                "Expected shape was {s1}, but input shape "
                "was {s2}".format(s1=(self.Nspws, self.Nfreqs), s2=trcvr.shape)
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
            x = self.delay_array
        else:
            x = self.freq_array

        self.data_array = utils.normalized_fourier_transform(
            x,
            self.data_array,
            flag_array=self.flag_array,
            axis=-1,
            taper=self.taper,
            inverse=inverse,
        )
        self.noise_array = utils.normalized_fourier_transform(
            x,
            self.noise_array,
            flag_array=self.flag_array,
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
                "Unknown data type: {dt}. Unable to perform "
                "delay transformation.".format(dt=self.data_type)
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
        # lst_array is stored in radians, multiply by 12*3600/np.pi to convert
        # to seconds s
        if self.lst_array.size > 1:
            delta_t = np.diff(self.lst_array)[0] * 12.0 * units.h / (np.pi * units.rad)
            delta_t = delta_t.to("s")
        else:
            delta_t = self.integration_time.item(0).to("s")
        lst_bins = (
            np.size(self.lst_array) * delta_t / self.integration_time.item(0).to("s")
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
                    * self.Nbls
                    * np.sqrt(2 * lst_bins)
                )
            )
            # integrate the noise temperature over the bands being Fourier Transformed
            thermal_power = integrate.trapz(
                np.ma.masked_invalid(thermal_power.value) ** 2
                * self.taper(self.Nfreqs).reshape(1, 1, 1, 1, 1, self.Nfreqs) ** 2,
                x=self.freq_array.value.reshape(self.Nspws, 1, 1, 1, 1, self.Nfreqs),
                axis=-1,
            ) << (self.freq_array.unit * thermal_power.unit ** 2)
        self.thermal_power = thermal_power << units.Unit("K^2 sr^2 Hz^2")
