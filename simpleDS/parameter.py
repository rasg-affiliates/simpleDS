# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Matthew Kolopanis
# Licensed under the 3-clause BSD License
"""Define Unit Parameter objects: Subclasses of the UVParameter object.

These objects extend the functionality of pyuvdata UVParameter objects to also
include compatibility with Astropy Units and Quantity objects.
"""
from __future__ import print_function, absolute_import, division

import numpy as np
import warnings
from pyuvdata import parameter as uvp, utils as uvutils
import astropy.units as units


class UnitParameter(uvp.UVParameter):
    """SubClass of UVParameters with astropy quantity compatibility.

    Adds checks for Astropy Quantity objects and equality between Quantites.
    """

    def __init__(self, name, required=True, value=None, spoof_val=None,
                 form=(), description='', expected_type=int,
                 acceptable_vals=None, acceptable_range=None,
                 tols=(1e-05, 1e-08), value_not_quantity=False):
        """Initialize the UVParameter.

        Extra keywords:
            value_not_quantity: (Boolean, default False)
                                Boolean flag used to specify that input value
                                is not an astropy Quantity object, but a
                                UnitParameter is desired over a UVParameter.
        """
        if isinstance(value, list) and isinstance(value[0], units.Quantity):
            try:
                value = units.Quantity(value)
            except units.UnitConversionError:
                raise ValueError("Unable to create UnitParameter objects "
                                 "from lists whose elements have "
                                 "non-comaptible units.")
        if isinstance(value, units.Quantity):
            if isinstance(tols, units.Quantity):
                if tols.size > 1:
                    raise ValueError("Tolerance values that are Quantity "
                                     "objects must be a single value to "
                                     "represent the absolute tolerance.")
                else:
                    tols = tuple((0, tols))
            if len(uvutils._get_iterable(tols)) == 1:
                # single value tolerances are assumed to be absolute
                tols = tuple((0, tols))
            if not isinstance(tols[1], units.Quantity):
                print("Given absolute tolerance did not all have units. "
                      "Applying units from parameter value.")
                tols = tuple((tols[0], tols[1] * value.unit))
            if not tols[1].unit.is_equivalent(value.unit):
                raise units.UnitConversionError("Given absolute tolerance "
                                                "did not all have units "
                                                "compatible with given "
                                                "parameter value.")
            tol_unit = tols[1].unit
            tols = tuple((tols[0], tols[1].value))
            super(UnitParameter, self).__init__(name=name, required=required,
                                                value=value,
                                                spoof_val=spoof_val, form=form,
                                                description=description,
                                                expected_type=expected_type,
                                                acceptable_vals=acceptable_vals,
                                                acceptable_range=acceptable_range,
                                                tols=tols)

            self.tols = (tols[0], tols[1] * tol_unit)
        else:
            if value_not_quantity:
                super(UnitParameter, self).__init__(name=name, required=required,
                                                    value=value,
                                                    spoof_val=spoof_val, form=form,
                                                    description=description,
                                                    expected_type=expected_type,
                                                    acceptable_vals=acceptable_vals,
                                                    acceptable_range=acceptable_range,
                                                    tols=tols)
            else:
                raise ValueError("Input value array is not an astropy Quantity"
                                 " object and the user did not specify "
                                 "value_not_quantity flag.")

    def __eq__(self, other):
        """Equal if classes match and values are identical."""
        if isinstance(other, self.__class__):
            # if both are UnitParameter objects then do new comparison
            if not isinstance(self.value, other.value.__class__):
                print('{name} parameter value classes are different. Left is '
                      '{lclass}, right is {rclass}'.format(name=self.name,
                                                           lclass=self.value.__class__,
                                                           rclass=other.value.__class__))
                return False
            if isinstance(self.value, units.Quantity):
                # check shapes are the same
                if self.value.shape != other.value.shape:
                    print('{name} parameter value is array, shapes are '
                          'different'.format(name=self.name))
                    return False
                elif self.value.unit != other.value.unit:
                    print('{name} parameter is Quantity, but have different '
                          'units '.format(name=self.name))
                    return False
                elif not units.allclose(self.value, other.value,
                                        rtol=self.tols[0], atol=self.tols[1]):
                    print('{name} parameter value is array, values are not '
                          'close'.format(name=self.name))
                    return False
                else:
                    return True
            elif isinstance(self.value, list):
                if self.value.shape != other.value.shape:
                    print('{name} parameter value is a list, shapes are '
                          'different'.format(name=self.name))
                    return False
                elif not (any(isinstance(_val, units.Quantity) for _val in self.value) and any(isinstance(_val, units.Quantity) for _val in other.value)):
                    print("{name} is a list, but one has Quantities "
                          "and the other does not".format(name=self.name))
                    return False
                else:
                    if not (all(isinstance(_val, units.Quantity) for _val in self.value)
                            and all(isinstance(_val, units.Quantity) for _val in other.value)):
                        raise NotImplementedError("Comparison of lists whose "
                                                  "elements are not all Quantity "
                                                  "objects is not currently "
                                                  "supported.")
                    else:
                        for (_self_val, _other_val) in zip(self.value, other.value):
                            if _self_val.unit != _other_val.unit:
                                print('{name} parameter is a list of Quantities, '
                                      'but have different units'
                                      .format(name=self.name))
                                return False
                            elif not units.allclose(_self_val, _other_val,
                                                    rtol=self.tols[0],
                                                    atol=self.tols[1]):
                                print("{name} parameter value are lists, values "
                                      "are not close".format(name=self.name))
                                return False
                            else:
                                return True
            else:
                return super(UnitParameter, self).__eq__(other)

    def __neq__(self, other):
        """Not Equal."""
        return not self.__eq__(other)
