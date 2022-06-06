# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Setup testing environment, define useful testing functions."""
from __future__ import absolute_import, division, print_function

from pathlib import Path
from setuptools_scm import get_version
from pkg_resources import get_distribution, DistributionNotFound

from .branch_scheme import branch_scheme

from simpleDS.data import DATA_PATH  # noqa


try:  # pragma: nocover
    # get accurate version for developer installs
    version_str = get_version(Path(__file__).parent.parent, local_scheme=branch_scheme)

    __version__ = version_str

except (LookupError, ImportError):
    try:
        # Set the version automatically from the package details.
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:  # pragma: nocover
        # package is not installed
        pass
