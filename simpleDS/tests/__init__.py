# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Setup testing environment, define useful testing functions."""
from __future__ import absolute_import, division, print_function

import six
import pytest

from simpleDS.data import DATA_PATH  # noqa


# defines a decorator to skip tests that require PY3.
reason = "Astropy.units.littleh is a python3 only unit. Skipping in python 2"
skipIf_py2 = pytest.mark.skipif(not six.PY3, reason=reason)
