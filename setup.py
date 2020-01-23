# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License
"""Setup modules simpleDS."""
from __future__ import absolute_import, division, print_function

from setuptools import setup
import glob
import io
from pygitversion import branch_scheme

with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_args = {
    "name": "simpleDS",
    "author": "rasg-affiliates",
    "license": "BSD",
    "description": ("A Simple Delay Spectrum calculator for radio interferometers. "),
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "package_dir": {"simpleDS": "simpleDS"},
    "packages": ["simpleDS", "simpleDS.tests"],
    "scripts": glob.glob("scripts/*"),
    "use_scm_version": {"local_scheme": branch_scheme},
    "include_package_data": True,
    "install_requires": ["numpy>1.17", "astropy>=4.0", "scipy", "pyuvdata>=1.5"],
    "extras_require": {
        "healpix": ["astropy-healpix"],
        "all": ["astropy-healpix", "pytest", "pytest-cov"],
    },
    "tests_require": ["pytest"],
}

if __name__ == "__main__":
    setup(**setup_args)
