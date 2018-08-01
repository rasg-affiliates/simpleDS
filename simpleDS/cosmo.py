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
from astropy.cosmology import WMAP9

# the emission frequency of 21m photons
f21 = 1420405751.7667 * units.Hz

# Using WMAP 9-year cosmology as the default
# move in the the little-h unit frame by setting H0=100
little_h_cosmo = WMAP9.clone(name='WMAP9 h-units', H0=100)


def calc_z(freq):
    """Calculate the redshift from a given frequency or frequncies."""
    if not isinstance(freq, Quantity):
        raise ValueError("Input freq must be an astropy Quantity. "
                         "value was: {}".format(freq))
    return (f21 / freq).si.value - 1


def u2kperp(u, z, cosmo=little_h_cosmo):
    """Convert baseline length u to k_perpendicular."""
    return 2 * np.pi * u / cosmo.comoving_distance(z)


def kperp2u(kperp, z, cosmo=little_h_cosmo):
    """Convert comsological k_perpendicular to baseline length u."""
    if not isinstance(kperp, Quantity):
        raise ValueError('input kperp must be an astropy Quantity object. '
                         'value was: {0}'.format(kperp))
    return kperp * cosmo.comoving_distance(z) / (2 * np.pi)


def eta2kparr(eta, z, cosmo=little_h_cosmo):
    """Conver delay eta to k_parallel (comoving 1./Mpc along line of sight)."""
    if not isinstance(eta, Quantity):
        raise ValueError('input eta must be an astropy Quantity object. '
                         'value was: {0}'.format(eta))
    return (eta * (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))
            / (const.c * (1 + z)**2)).to('1/Mpc')


def kparr2eta(kparr, z, cosmo=little_h_cosmo):
    """Convert k_parallel (comoving 1/Mpc along line of sight) to delay eta."""
    if not isinstance(kparr, Quantity):
        raise ValueError('input kparr must be an astropy Quantity object. '
                         'value was: {0}'.format(kparr))
    return (kparr * const.c * (1 + z)**2
            / (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))).to('s')


def X2Y(z, cosmo=little_h_cosmo):
    """Convert units from interfometric delay power to comsological power.

    Converts power spectrum from interfometric units of eta, u to
    cosmological k_par, k_perp.

    Arguments:
        z: redshift for cosmological conversion
    Returns:
        X2Y: The power spectrum unit conversion factor
    """
    X2Y = const.c * (1 + z)**2 * cosmo.comoving_distance(z)**2
    X2Y /= cosmo.H0 * f21 * cosmo.efunc(z)
    return X2Y.to('Mpc^3*s')
