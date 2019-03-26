# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Matthew Kolopanis
# Licensed under the 3-clause BSD License
"""Setup testing environment, define useful testing functions."""
from __future__ import absolute_import, division, print_function

import os
import warnings
import sys
import numpy as np
import six
from pyuvdata.tests import skip

from simpleDS.data import DATA_PATH


def skipIf_py2(test_func):
    """Define a decorator to skip tests that require python 3 for astropy units."""
    reason = 'Astropy.units.littleh is a python3 only unit. Skipping in python 2'
    if six.PY2:
        return skip(reason)(test_func)
    return test_func
