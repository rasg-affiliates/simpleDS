"""Relevent Cosmology units and transforms for Power Spectrum estimation.

All cosmological calculations and converions follow from Liu et al 2014a
Phys. Rev. D 90, 023018 or 	arXiv:1404.2596
"""

import os
import sys
import numpy as np
from astropy import constants as const
from astropy import units
from astropy.units import Quantity
from astropy.cosmology import WMAP9, default_cosmology

# the emission frequency of 21m photons
f21 = 1420405751.7667 * units.Hz

# Using WMAP 9-year cosmology as the default
# move in the the little-h unit frame by setting H0=100
little_h_cosmo = WMAP9.clone(name='h-units', H0=100)
default_cosmology.set(little_h_cosmo)


@units.quantity_input(freq='frequency')
def calc_z(freq):
    """Calculate the redshift from a given frequency or frequncies."""
    return (f21 / freq).si.value - 1


def calc_freq(redshift):
    """Calculate the frequency or frequencies of a given 21cm redshift."""
    return f21 / (1 + redshift)


def u2kperp(u, z, cosmo=None):
    """Convert baseline length u to k_perpendicular."""
    if cosmo is None:
        cosmo = default_cosmology.get()
    return 2 * np.pi * u / cosmo.comoving_distance(z)


@units.quantity_input(kperp='wavenumber')
def kperp2u(kperp, z, cosmo=None):
    """Convert comsological k_perpendicular to baseline length u."""
    if cosmo is None:
        cosmo = default_cosmology.get()
    return kperp * cosmo.comoving_distance(z) / (2 * np.pi)


@units.quantity_input(eta='time')
def eta2kparr(eta, z, cosmo=None):
    """Conver delay eta to k_parallel (comoving 1./Mpc along line of sight."""
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (eta * (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))
            / (const.c * (1 + z)**2)).to('1/Mpc')


@units.quantity_input(kparr='wavenumber')
def kparr2eta(kparr, z, cosmo=None):
    """Convert k_parallel (comoving 1/Mpc along line of sight) to delay eta."""
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (kparr * const.c * (1 + z)**2
            / (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))).to('s')


def X2Y(z, cosmo=None):
    """Convert units from interfometric delay power to comsological power.

    Converts power spectrum from interfometric units of eta, u to
    cosmological k_par, k_perp.

    Arguments:
        z: redshift for cosmological conversion
    Returns:
        X2Y: The power spectrum unit conversion factor
    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    X2Y = const.c * (1 + z)**2 * cosmo.comoving_distance(z)**2
    X2Y /= cosmo.H0 * f21 * cosmo.efunc(z)
    return X2Y.to('Mpc^3*s')