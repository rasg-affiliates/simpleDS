# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Init file for simpleDS."""


from __future__ import print_function, absolute_import

from . import version  # noqa
from .utils import *  # noqa
from .delay_spectrum import *  # noqa

__version__ = version.version
