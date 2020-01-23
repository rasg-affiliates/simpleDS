# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Init file for simpleDS."""

from pkg_resources import get_distribution, DistributionNotFound

# Set the version automatically from the package details.
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: nocover
    # package is not installed
    pass

from .utils import *  # noqa
from .delay_spectrum import *  # noqa
