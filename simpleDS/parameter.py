# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Define Unit Parameter objects: Subclasses of the UVParameter object.

These objects extend the functionality of pyuvdata UVParameter objects to also
include compatibility with Astropy Units and Quantity objects.
"""
import numpy as np
import warnings
import copy
from pyuvdata import parameter as uvp, utils as uvutils
import astropy.units as units
from astropy.cosmology.core import Cosmology as reference_cosmology_object


class UnitParameter(uvp.UVParameter):
    r"""SubClass of UVParameters with astropy quantity compatibility.

    Adds checks for Astropy Quantity objects and equality between Quantites.


    For a complete list of input parameters

    Attributes
    ----------
    name : str
        A string giving the name of the attribute. Used as the associated
        property name in classes based on UVBase.

    required : bool
        A boolean indicating whether this is required metadata for
        the class with this UVParameter as an attribute. Default is True.

    value : str, astropy Quantity object, or scalar
        The value of the data or metadata.

    spoof_val : any but same kind as `value`
        A fake value that can be assigned to a non-required
        UVParameter if the metadata is required for a particular file-type.
        This is not an attribute of required UVParameters.

    form : str or tuple
        Either 'str' or a tuple giving information about the expected
        shape of the value. Elements of the tuple may be the name of other
        UVParameters that indicate data shapes. \n
        Examples:\n
            'str': a string value\n
            ('Nblts', 3): the value should be an array of shape: Nblts (another UVParameter name), 3

    description : str
        A string description of the data or metadata in the object.

    expected_type : type
        The type that the data or metadata should be.
        Default is np.int or str if form is 'str'

    expected_units : astropy.unit object
        The expected units of the input astropy Quantity.

    acceptable_vals : List, Optional.
        List giving allowed values for elements of value.

    acceptable_range : tuple, Optional.
        Tuple giving a range of allowed magnitudes for elements of value.

    tols: tuple of scalar and astropy Quantity or single astropy Quantity
        Tolerances for testing the equality of UVParameters. Either a
        single absolute value or a tuple of relative and absolute values to
        be used by np.isclose()

    value_not_quantity : Boolean, default False
        Boolean flag used to specify that input value is not an astropy Quantity object,
        but a UnitParameter is desired over a UVParameter.

    strict_type_check : bool
        When True, the input expected_type is used exactly, otherwise a more generic
        type is found to allow changes in precions
        or to/from numpy dtypes to not break checks.


    Raises
    ------
    ValueError
        If input value is a list of objects with different unit equivalencies.
        If no expected unit is provided.
        If tolerance is a Quantity, but be a single value.
        If value is not a Quantity and value_not_quantity is False
    UnitConversionError
        If the value and expected units have different equivalencies.
        If the value and absolute tolerance have different equivalencies.

    """

    def __init__(
        self,
        name,
        required=True,
        value=None,
        spoof_val=None,
        form=(),
        description="",
        expected_type=int,
        acceptable_vals=None,
        acceptable_range=None,
        expected_units=None,
        tols=(1e-05, 1e-08),
        value_not_quantity=False,
        strict_type_check=False,
    ):
        """Initialize the UVParameter."""
        self.value_not_quantity = value_not_quantity
        self.expected_units = expected_units
        if isinstance(value, list) and isinstance(value[0], units.Quantity):
            try:
                value = units.Quantity(value)
            except units.UnitConversionError:
                raise ValueError(
                    "Unable to create UnitParameter objects "
                    "from lists whose elements have "
                    "non-comaptible units."
                )
        if isinstance(value, units.Quantity):
            if self.expected_units is None:
                raise ValueError(
                    "Input Quantity must also be accompained "
                    "by the expected unit or equivalent unit. "
                    "Please set parameter expected_units to "
                    "an instance of an astropy Units object."
                )
            if not value.unit.is_equivalent(self.expected_units):
                raise units.UnitConversionError(
                    "Input value has units {0} "
                    "which are not equivalent to "
                    "expected units of {1}".format(value.unit, self.expected_units)
                )
            if isinstance(tols, units.Quantity):
                if tols.size > 1:
                    raise ValueError(
                        "Tolerance values that are Quantity "
                        "objects must be a single value to "
                        "represent the absolute tolerance."
                    )
                else:
                    tols = tuple((0, tols))
            if len(uvutils._get_iterable(tols)) == 1:
                # single value tolerances are assumed to be absolute
                tols = tuple((0, tols))
            if not isinstance(tols[1], units.Quantity):
                print(
                    "Given absolute tolerance did not all have units. "
                    "Applying units from parameter value."
                )
                tols = tuple((tols[0], tols[1] * value.unit))
            if not tols[1].unit.is_equivalent(value.unit):
                raise units.UnitConversionError(
                    "Given absolute tolerance "
                    "did not all have units "
                    "compatible with given "
                    "parameter value."
                )
            tol_unit = tols[1].unit
            tols = tuple((tols[0], tols[1].value))
            super(UnitParameter, self).__init__(
                name=name,
                required=required,
                value=value,
                spoof_val=spoof_val,
                form=form,
                description=description,
                expected_type=expected_type,
                acceptable_vals=acceptable_vals,
                acceptable_range=acceptable_range,
                tols=tols,
                strict_type_check=strict_type_check,
            )

            self.tols = (tols[0], tols[1] * tol_unit)
        else:
            if value_not_quantity or value is None:
                super(UnitParameter, self).__init__(
                    name=name,
                    required=required,
                    value=value,
                    spoof_val=spoof_val,
                    form=form,
                    description=description,
                    expected_type=expected_type,
                    acceptable_vals=acceptable_vals,
                    acceptable_range=acceptable_range,
                    tols=tols,
                    strict_type_check=strict_type_check,
                )
            else:
                raise ValueError(
                    "Input value array is not an astropy Quantity"
                    " object and the user did not specify "
                    "value_not_quantity flag."
                )

    def __eq__(self, other):
        """Equal if classes match and values are identical."""
        if isinstance(other, self.__class__):
            # if both are UnitParameter objects then do new comparison
            if not isinstance(self.value, other.value.__class__):
                print(
                    "{name} parameter value classes are different. Left is "
                    "{lclass}, right is {rclass}".format(
                        name=self.name,
                        lclass=self.value.__class__,
                        rclass=other.value.__class__,
                    )
                )
                return False
            if isinstance(self.value, units.Quantity):
                # check shapes are the same
                if not isinstance(self.tols[1], units.Quantity):
                    self.tols = (self.tols[0], self.tols[1] * self.value.unit)
                if self.value.shape != other.value.shape:
                    print(
                        "{name} parameter value is array, shapes are "
                        "different".format(name=self.name)
                    )
                    return False
                elif not self.value.unit.is_equivalent(other.value.unit):
                    print(
                        "{name} parameter is Quantity, but have "
                        "non-compatible units ".format(name=self.name)
                    )
                    return False
                else:
                    # astropy.units has a units.allclose but only for python 3
                    # already know the units are compatible so
                    # Convert other to self's units and compare values
                    other.value = other.value.to(self.value.unit)

                    if not np.allclose(
                        self.value.value,
                        other.value.value,
                        rtol=self.tols[0],
                        atol=self.tols[1].value,
                    ):
                        print(
                            "{name} parameter value is array, values are not "
                            "close".format(name=self.name)
                        )
                        return False
                    else:
                        return True

            elif isinstance(self.value, reference_cosmology_object):
                cosmo_dict = copy.deepcopy(self.value.__dict__)
                # remove string entries from the dict
                cosmo_dict.pop("name", None)
                cosmo_dict.pop("__doc__", None)
                for p in cosmo_dict:
                    parm = getattr(self.value, p)
                    other_parm = getattr(other.value, p)
                    # This line is not necessarily going to be hit by the equality checker
                    # Changing a value in an astropy cosmology object also updates other values
                    # As necessary so it may find non-quantities that are not equal first
                    # But it is a good checkt to have
                    if isinstance(parm, units.Quantity):
                        if not np.allclose(parm.value, other_parm.to(parm.unit).value):
                            print(
                                "Assumed Cosmologies are not equal. "
                                "{name} parameter values are not close".format(name=p)
                            )
                            return False
                    elif isinstance(parm, (np.ndarray, list, tuple)):
                        try:
                            if not np.allclose(
                                np.asarray(parm), np.asarray(other_parm)
                            ):
                                print(
                                    "Assumed Cosmologies are not equal. "
                                    "{name} parameter values are not close".format(
                                        name=p
                                    )
                                )
                                return False
                        except TypeError:
                            try:

                                if not all(
                                    np.isclose(p_val, other_val)
                                    for p_val, other_val in zip(parm, other_parm)
                                ):
                                    print(
                                        "Assumed Cosmologies are not equal. "
                                        "{name} parameter values are not close".format(
                                            name=p
                                        )
                                    )
                                    return False

                            except units.UnitConversionError:
                                return False

                        except units.UnitConversionError:
                            return False
                    elif isinstance(parm, dict):
                        try:
                            # Try a naive comparison first
                            # this will fail if keys are the same
                            # but cases differ.
                            # so only look for exact equality
                            # then default to the long test below.
                            if parm == other_parm:
                                return True
                        except ValueError:
                            pass
                            # this dict probably contains arrays
                            # we will need to check each item individually

                        # check to see if they are equal other than
                        # upper/lower case keys
                        self_lower = {k.lower(): v for k, v in parm.items()}
                        other_lower = {k.lower(): v for k, v in other_parm.items()}
                        if set(self_lower.keys()) != set(other_lower.keys()):
                            print(
                                "Assumed Cosmologies are not equal. "
                                "{name} parameter values are not close".format(name=p)
                            )
                            return False
                        else:
                            # need to check if values are close,
                            # not just equal
                            values_close = True
                            for key in self_lower.keys():
                                try:
                                    if not np.allclose(
                                        self_lower[key], other_lower[key]
                                    ):
                                        values_close = False
                                except (TypeError):
                                    # this isn't a type that can be
                                    # handled by np.isclose,
                                    # test for equality
                                    if self_lower[key] != other_lower[key]:
                                        values_close = False
                                except units.UnitConversionError:
                                    return False
                            if values_close is False:
                                print(
                                    "Assumed Cosmologies are not equal. "
                                    "{name} parameter values are not close".format(
                                        name=p
                                    )
                                )
                                return False
                            else:
                                return True
                    elif parm != other_parm:
                        return False

                return True

            else:
                return super(UnitParameter, self).__eq__(other)

        elif issubclass(self.__class__, other.__class__):
            # If trying to compare a UnitParameter to a UVParameter
            # value of the quantity must match the UVParameter
            return self.to_uvp().__eq__(other)
        else:
            print(
                "{name} parameter value classes are different and one "
                "is not a subclass of the other. Left is "
                "{lclass}, right is {rclass}".format(
                    name=self.name, lclass=self.__class__, rclass=other.__class__
                )
            )
            return False

    def __ne__(self, other):
        """Not Equal."""
        return not self.__eq__(other)

    def to_uvp(self):
        """Cast self as a UVParameter."""
        if self.value_not_quantity:
            if self.required:
                return uvp.UVParameter(
                    name=self.name,
                    required=self.required,
                    value=self.value,
                    form=self.form,
                    description=self.description,
                    expected_type=self.expected_type,
                    acceptable_vals=self.acceptable_vals,
                    acceptable_range=self.acceptable_range,
                    tols=(self.tols[0], self.tols[1]),
                    strict_type_check=True,
                )
            else:
                return uvp.UVParameter(
                    name=self.name,
                    required=self.required,
                    value=self.value,
                    spoof_val=self.spoof_val,
                    form=self.form,
                    description=self.description,
                    expected_type=self.expected_type,
                    acceptable_vals=self.acceptable_vals,
                    acceptable_range=self.acceptable_range,
                    tols=(self.tols[0], self.tols[1]),
                    strict_type_check=True,
                )
        else:
            # what sould happen here? Warn the user we are comparing a qunatity
            # back to a UVP? might lose units or something. Should it be cast to si?
            # That could mess up things that are intentionally not stored in si.
            warnings.warn(
                "A UnitParameter with quantity value is being cast to "
                "UVParameter. All quantity information will be lost. "
                "If this is a comparison that fails, you may need "
                "to alter the unit of the value to match expected "
                "UVParameter units.",
                UserWarning,
            )
            if self.required:
                return uvp.UVParameter(
                    name=self.name,
                    required=self.required,
                    value=self.value.value,
                    form=self.form,
                    description=self.description,
                    expected_type=self.expected_type,
                    acceptable_vals=self.acceptable_vals,
                    acceptable_range=self.acceptable_range,
                    tols=(self.tols[0], self.tols[1].value),
                    strict_type_check=True,
                )

            else:
                return uvp.UVParameter(
                    name=self.name,
                    required=self.required,
                    value=self.value.value,
                    spoof_val=self.spoof_val,
                    form=self.form,
                    description=self.description,
                    expected_type=self.expected_type,
                    acceptable_vals=self.acceptable_vals,
                    acceptable_range=self.acceptable_range,
                    tols=(self.tols[0], self.tols[1].value),
                    strict_type_check=True,
                )
