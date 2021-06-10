# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Tests for the UnitParameter class."""
from __future__ import print_function, absolute_import, division

import numpy as np

import pytest
from simpleDS import parameter as unp
from astropy import units
from astropy.cosmology import WMAP9
from pyuvdata import parameter as uvp
import pyuvdata.tests as uvtest


class NonParameter(object):
    """A Dummy object for comparison."""

    def __init__(self):
        """Do Nothing."""
        pass


def test_error_from_mixed_list():
    """Test that error is raised if input value is list of mixed Quantities."""
    pytest.raises(
        ValueError,
        unp.UnitParameter,
        name="unp1",
        value=[1 * units.m, 1 * units.Hz],
        expected_units=units.Hz,
    )


def test_incompatible_tolerance_units():
    """Test error is raised if tolerances have incompatible units."""
    pytest.raises(
        units.UnitConversionError,
        unp.UnitParameter,
        name="unp1",
        value=3 * units.m,
        tols=1 * units.ns,
        expected_units=units.m,
    )


def test_incompatible_tolerance_units_from_tuple():
    """Test error is raised if tolerance is a tuple and has incomplatible units."""
    pytest.raises(
        units.UnitConversionError,
        unp.UnitParameter,
        name="unp1",
        value=3 * units.m,
        tols=(1e-9, 1 * units.ns),
        expected_units=units.m,
    )


def test_tols_cast_up_to_tuple():
    """Test if a single scalar is given as tols it is upgraded to a tuple."""
    unp1 = unp.UnitParameter(
        name="unp1", value=3 * units.m, tols=1e-3, expected_units=units.m
    )
    assert isinstance(unp1.tols, tuple)


def test_error_raised_if_not_quantity_and_no_flag_set():
    """Test an Error is raised if input value is not a quantity and the flag is not set."""
    pytest.raises(
        ValueError, unp.UnitParameter, name="unp1", value=3, expected_units=units.m
    )


def test_tols_cast_up_to_correct_units():
    """Test if a single scalar is given as tols it is upgraded to a tuple with correct Units."""
    unp1 = unp.UnitParameter(
        name="unp1", value=3 * units.m, tols=1e-3, expected_units=units.m
    )
    assert isinstance(unp1.tols, tuple)
    assert units.m == unp1.tols[1].unit


def test_error_from_quantity_tolerance():
    """Test error is raised if tolerance is a Quantity Object."""
    pytest.raises(
        ValueError,
        unp.UnitParameter,
        name="unp1",
        value=3 * units.m,
        tols=(1e-9, 1) * units.m,
        expected_units=units.m,
    )


def test_error_if_no_units_with_quantity():
    """Check and error is raised if instantiating a UnitParameter without expected units."""
    pytest.raises(
        ValueError,
        unp.UnitParameter,
        name="unp1",
        value=3 * units.m,
        tols=(1e-9, 1 * units.m),
    )


def test_incompatible_expected_units():
    """Test error is raised if value and expected_units are not equivalent."""
    pytest.raises(
        ValueError,
        unp.UnitParameter,
        name="unp1",
        value=3 * units.m,
        tols=(1e-9, 1 * units.m),
        expected_units=units.Jy,
    )


def test_unitparameters_equal_to_uvparameter_with_same_value():
    """Test that UnitParameters are equal to UVParameters if their values match."""
    unp1 = unp.UnitParameter(
        name="unp1",
        value=units.Quantity(3, unit=units.m, dtype=int),
        expected_units=units.m,
    )
    uvp1 = uvp.UVParameter(name="uvp1", value=np.int64(3))
    warn_message = [
        "A UnitParameter with quantity value is being cast to "
        "UVParameter. All quantity information will be lost. "
        "If this is a comparison that fails, you may need "
        "to alter the unit of the value to match expected "
        "UVParameter units."
    ]
    with uvtest.check_warnings(UserWarning, match=warn_message):
        assert unp1 == uvp1

    with uvtest.check_warnings(UserWarning, match=warn_message):
        assert uvp1 == unp1


def test_unitparameters_not_equal_to_uvp_with_different_type():
    """Test that UnitParameters are not equal to UVParameters if their types differ."""
    unp1 = unp.UnitParameter(name="unp1", value=3 * units.m, expected_units=units.m)
    uvp1 = uvp.UVParameter(name="uvp1", value=3)
    warn_message = [
        "A UnitParameter with quantity value is being cast to "
        "UVParameter. All quantity information will be lost. "
        "If this is a comparison that fails, you may need "
        "to alter the unit of the value to match expected "
        "UVParameter units."
    ]
    with uvtest.check_warnings(UserWarning, match=warn_message):
        assert unp1 != uvp1

    with uvtest.check_warnings(UserWarning, match=warn_message):
        assert uvp1 != unp1


def test_non_required_unitparameters_equal_to_uvparameter():
    """Test that UnitParameter is equal to UVParameters, non-required version."""
    unp1 = unp.UnitParameter(
        name="unp1",
        value=3 * units.m,
        required=False,
        spoof_val=10,
        expected_units=units.m,
    )
    uvp1 = uvp.UVParameter(
        name="uvp1", value=3.0, required=False, spoof_val=10, expected_type=np.float32
    )
    warn_message = [
        "A UnitParameter with quantity value is being cast to "
        "UVParameter. All quantity information will be lost. "
        "If this is a comparison that fails, you may need "
        "to alter the unit of the value to match expected "
        "UVParameter units."
    ]
    with uvtest.check_warnings(UserWarning, warn_message):
        assert unp1 == uvp1

    with uvtest.check_warnings(UserWarning, warn_message):
        assert uvp1 == unp1


def test_non_required_non_value_unitparameter_is_equal_to_uvparameter():
    """Test a UnitParameter with value_not_quantity is equal to a UVParameter, non-requred version."""
    unp1 = unp.UnitParameter(
        name="unp1", value=3, required=False, spoof_val=10, value_not_quantity=True
    )
    uvp1 = uvp.UVParameter(name="uvp1", value=3, required=False, spoof_val=10)
    assert unp1 == uvp1
    assert uvp1 == unp1


def test_non_value_unitparameter_is_equal_to_uvparameter():
    """Test a UnitParameter with value_not_quantity is equal to a UVParameter."""
    unp1 = unp.UnitParameter(
        name="unp1", value=3.0, expected_type=np.float32, value_not_quantity=True
    )
    uvp1 = uvp.UVParameter(name="uvp1", value=3.0, expected_type=np.float32)
    assert unp1 == uvp1
    assert uvp1 == unp1


def test_value_class_inequality():
    """Test false is returned if values have different classes."""
    unp1 = unp.UnitParameter(name="unp1", value=[3 * units.m], expected_units=units.m)
    unp2 = unp.UnitParameter(
        name="unp2", value=np.array(3) * units.m, expected_units=units.m
    )
    assert unp1 != unp2
    assert unp2 != unp1
    unp3 = unp.UnitParameter(name="unp3", value="test string", value_not_quantity=True)
    assert unp1 != unp3


def test_value_unit_inequality():
    """Test parameters are not equal if units are different."""
    unp1 = unp.UnitParameter(
        name="unp1", value=np.array(3) * units.m, expected_units=units.m
    )
    unp2 = unp.UnitParameter(
        name="unp2", value=np.array(3) * units.Hz, expected_units=units.Hz
    )
    assert unp1 != unp2


def test_value_unit_compatibility():
    """Test parameters with compatible units can be compared as equal."""
    unp1 = unp.UnitParameter(
        name="unp1", value=np.array(3) * units.ns, expected_units=units.s
    )
    unp2 = unp.UnitParameter(
        name="unp2", value=np.array(3) * 1.0 / units.GHz, expected_units=units.s
    )
    assert unp1 == unp2


def test_not_equal_shapes():
    """Test parameters with different Quantity shapes are not equal."""
    unp1 = unp.UnitParameter(
        name="unp1", value=np.array([1, 2]) * units.m, expected_units=units.m
    )
    unp2 = unp.UnitParameter(
        name="unp2", value=np.array([1, 2, 3]) * units.m, expected_units=units.m
    )
    assert unp1 != unp2


def test_value_inequality():
    """Test inequality holds for different arrays."""
    unp1 = unp.UnitParameter(
        name="unp1",
        value=np.array([1, 2, 3]) * units.m,
        tols=(1e-05, 1e-08 * units.m),
        expected_units=units.m,
    )
    unp2 = unp.UnitParameter(
        name="unp2",
        value=np.array([1, 2, 4]) * units.m,
        tols=(1e-05, 1e-08 * units.m),
        expected_units=units.m,
    )
    assert unp1 != unp2


def test_value_equality():
    """Test equality holds for different arrays."""
    unp1 = unp.UnitParameter(
        name="unp1",
        value=np.array([1, 2, 3]) * units.m,
        tols=(1e-05, 1e-08 * units.m),
        expected_units=units.m,
    )
    unp2 = unp.UnitParameter(
        name="unp2",
        value=np.array([1, 2, 3]) * units.m,
        tols=(1e-05, 1e-08 * units.m),
        expected_units=units.m,
    )
    assert unp1 == unp2


def test_calls_to_super_class_are_equal():
    """Test that calls to the super class equalit function work."""
    unp1 = unp.UnitParameter(name="unp1", value=3, value_not_quantity=True)
    unp2 = unp.UnitParameter(name="unp2", value=3, value_not_quantity=True)
    assert unp1 == unp2


def test_not_equal_to_different_object():
    """Test object is not equal to general other object."""
    unp1 = unp.UnitParameter(name="unp1", value=3, value_not_quantity=True)
    test_obj = NonParameter()
    assert unp1 != test_obj


def test_cosmologies_not_equal_value():
    """Test two cosmologies are not equal if a value is different."""
    wmap = WMAP9.clone()
    test_cosmo = WMAP9.clone(Neff=10)

    unp1 = unp.UnitParameter(name="wmap", value=wmap, value_not_quantity=True)
    unp2 = unp.UnitParameter(
        name="test_cosmo", value=test_cosmo, value_not_quantity=True
    )
    assert unp1 != unp2


def test_cosmologies_not_equal_quantity():
    """Test two cosmologies are not equal if a quantity is different."""
    wmap = WMAP9.clone()
    test_cosmo = WMAP9.clone(Tcmb0=10 * units.K)

    unp1 = unp.UnitParameter(name="wmap", value=wmap, value_not_quantity=True)
    unp2 = unp.UnitParameter(
        name="test_cosmo", value=test_cosmo, value_not_quantity=True
    )
    assert unp1 != unp2
