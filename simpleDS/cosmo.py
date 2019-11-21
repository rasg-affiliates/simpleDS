# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Relevent Cosmology units and transforms for Power Spectrum estimation.

All cosmological calculations and converions follow from Liu et al 2014a
Phys. Rev. D 90, 023018 or 	arXiv:1404.2596
"""

import numpy as np
from astropy import constants as const
from astropy import units
from astropy.cosmology import Planck15, default_cosmology

# the emission frequency of 21m photons in the Hydrogen's rest frame
f21 = 1420405751.7667 * units.Hz

# Using WMAP 9-year cosmology as the default
# move in the the little-h unit frame by setting H0=100
default_cosmology.set(Planck15)


@units.quantity_input(freq="frequency")
def calc_z(freq):
    """Calculate the redshift from a given frequency or frequncies.

    Parameters
    ----------
    freq : Astropy Quantity Object units equivalent to frequency
        The frequency to calculate the redshift of 21cm emission

    Returns
    -------
    redshift : float
        The redshift consistent with 21cm observations of the input frequency.

    """
    return (f21 / freq).si.value - 1


def calc_freq(redshift):
    """Calculate the frequency or frequencies of a given 21cm redshift.

    Parameters
    ----------
    redshift : float
        The redshift of the expected 21cm emission

    Returns
    -------
    freq : Astropy Quantity Object units equivalent to frequency
        Frequency of the emission in the rest frame of emission

    """
    return f21 / (1 + redshift)


def u2kperp(u, z, cosmo=None):
    """Convert baseline length u to k_perpendicular.

    Parameters
    ----------
    u : float
        The baseline separation of two interferometric antennas in units of wavelength
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    kperp : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale perpendicular to the line of sight probed by the baseline length u.

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return 2 * np.pi * u / cosmo.comoving_transverse_distance(z)


@units.quantity_input(kperp="wavenumber")
def kperp2u(kperp, z, cosmo=None):
    """Convert comsological k_perpendicular to baseline length u.

    Parameters
    ----------
    kperp : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale perpendicular to the line of sight.
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    u : float
        The baseline separation of two interferometric antennas in units of
        wavelength which probes the spatial scale given by kperp

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return kperp * cosmo.comoving_transverse_distance(z) / (2 * np.pi)


@units.quantity_input(eta="time")
def eta2kparr(eta, z, cosmo=None):
    """Conver delay eta to k_parallel (comoving 1./Mpc along line of sight).

    Parameters
    ----------
    eta : Astropy Quantity object with units equivalent to time.
        The inteferometric delay observed in units compatible with time.
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    kparr : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight probed by the input delay (eta).

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (
        eta * (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z)) / (const.c * (1 + z) ** 2)
    ).to("1/Mpc")


@units.quantity_input(kparr="wavenumber")
def kparr2eta(kparr, z, cosmo=None):
    """Convert k_parallel (comoving 1/Mpc along line of sight) to delay eta.

    Parameters
    ----------
    kparr : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    eta : Astropy Quantity units equivalent to time
        The inteferometric delay which probes the spatial scale given by kparr.

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (
        kparr * const.c * (1 + z) ** 2 / (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))
    ).to("s")


def X2Y(z, cosmo=None):
    """Convert units from interferometric delay power to comsological power.

    Converts power spectrum from interferometric units of eta, u to
    cosmological k_par, k_perp.

    Parameters
    ----------
    z : float
        redshift for cosmological conversion
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    X2Y : Astropy Quantity units of Mpc^3*s/sr
        The power spectrum unit conversion factor

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    X2Y = const.c * (1 + z) ** 2 * cosmo.comoving_transverse_distance(z) ** 2
    X2Y /= cosmo.H0 * f21 * cosmo.efunc(z) * units.sr
    return X2Y.to("Mpc^3*s/sr")
