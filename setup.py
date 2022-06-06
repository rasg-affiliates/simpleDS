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
    "name": "simple_delay_spectrum",
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
    "install_requires": ["numpy>=1.18", "astropy>=5.0.4", "scipy", "pyuvdata>=1.4.2"],
    "extras_require": {
        "healpix": ["astropy-healpix"],
        "all": ["astropy-healpix", "pytest", "pytest-cov"],
    },
    "tests_require": ["pytest"],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
}

if __name__ == "__main__":
    setup(**setup_args)
