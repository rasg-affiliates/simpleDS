# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Matthew Kolopanis
# Licensed under the 3-clause BSD License
"""Tests for the UnitParameter class."""
from __future__ import print_function, absolute_import, division

import numpy as np

import nose.tools as nt
from simpleDS import parameter as unp
from astropy import units
from pyuvdata import parameter as uvp


def test_error_from_mixed_list():
    """Test that error is raised if input value is list of mixed Quantities."""
    nt.assert_raises(ValueError, unp.UnitParameter, name='unp1',
                     value=[1 * units.m, 1 * units.Hz])


def test_incompatible_tolerance_units():
    """Test error is raised if tolerances have incompatible units."""
    nt.assert_raises(units.UnitConversionError, unp.UnitParameter, name='unp1',
                     value=3 * units.m, tols=1 * units.ns)


def test_incompatible_tolerance_units_from_tuple():
    """Test error is raised if tolerance is a tuple and has incomplatible units."""
    nt.assert_raises(units.UnitConversionError, unp.UnitParameter, name='unp1',
                     value=3 * units.m, tols=(1e-9, 1 * units.ns))


def test_tols_cast_up_to_tuple():
    """Test if a single scalar is given as tols it is upgraded to a tuple."""
    unp1 = unp.UnitParameter(name='unp1', value=3 * units.m, tols=1e-3)
    nt.assert_true(isinstance(unp1.tols, tuple))


def test_error_raised_if_not_quantity_and_no_flag_set():
    """Test an Error is raised if input value is not a quantity and the flag is not set."""
    nt.assert_raises(ValueError, unp.UnitParameter, name='unp1', value=3)


def test_tols_cast_up_to_correct_units():
    """Test if a single scalar is given as tols it is upgraded to a tuple with correct Units."""
    unp1 = unp.UnitParameter(name='unp1', value=3 * units.m, tols=1e-3)
    nt.assert_true(isinstance(unp1.tols, tuple))
    nt.assert_equal(units.m, unp1.tols[1].unit)


def test_error_from_quantity_tolerance():
    """Test error is raised if tolerance is a Quantity Object."""
    nt.assert_raises(ValueError, unp.UnitParameter, name='unp1',
                     value=3 * units.m, tols=(1e-9, 1) * units.m)


def test_unitparameters_not_equal_to_uvparameter():
    """Test that UnitParameters are not equal to UVParameters."""
    unp1 = unp.UnitParameter(name='unp1', value=3 * units.m)
    uvp1 = uvp.UVParameter(name='uvp1', value=3)
    nt.assert_not_equal(unp1, uvp1)


def test_value_class_inequality():
    """Test false is returned if values have different classes."""
    unp1 = unp.UnitParameter(name='unp1', value=[3 * units.m])
    unp2 = unp.UnitParameter(name='unp2', value=np.array(3) * units.m)
    nt.assert_not_equal(unp1, unp2)
    nt.assert_not_equal(unp2, unp1)
    unp3 = unp.UnitParameter(name='unp3', value='test string', value_not_quantity=True)
    nt.assert_not_equal(unp1, unp3)


def test_value_unit_inequality():
    """Test parameters are not equal if units are different."""
    unp1 = unp.UnitParameter(name='unp1', value=np.array(3) * units.m)
    unp2 = unp.UnitParameter(name='unp2', value=np.array(3) * units.Hz)
    nt.assert_not_equal(unp1, unp2)


def test_not_equal_shapes():
    """Test parameters with different Quantity shapes are not equal."""
    unp1 = unp.UnitParameter(name='unp1', value=np.array([1, 2]) * units.m)
    unp2 = unp.UnitParameter(name='unp2', value=np.array([1, 2, 3]) * units.m)
    nt.assert_not_equal(unp1, unp2)


def test_value_inequality():
    """Test inequality holds for different arrays."""
    unp1 = unp.UnitParameter(name='unp1', value=np.array([1, 2, 3]) * units.m,
                             tols=(1e-05, 1e-08 * units.m))
    unp2 = unp.UnitParameter(name='unp2', value=np.array([1, 2, 4]) * units.m,
                             tols=(1e-05, 1e-08 * units.m))
    nt.assert_not_equal(unp1, unp2)


def test_value_equality():
    """Test equality holds for different arrays."""
    unp1 = unp.UnitParameter(name='unp1', value=np.array([1, 2, 3]) * units.m,
                             tols=(1e-05, 1e-08 * units.m))
    unp2 = unp.UnitParameter(name='unp2', value=np.array([1, 2, 3]) * units.m,
                             tols=(1e-05, 1e-08 * units.m))
    nt.assert_equal(unp1, unp2)
