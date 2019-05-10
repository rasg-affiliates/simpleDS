# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Setup modules simpleDS."""
from __future__ import absolute_import, division, print_function

from setuptools import setup
import glob
import os
from os import listdir
import sys
import json
import io

from simpleDS import version  # noqa (pycodestyle complains about import below code)

data = [version.git_origin, version.git_hash,
        version.git_description, version.git_branch]
with open(os.path.join('simpleDS', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup_args = {
    'name': 'simpleDS',
    'author': 'Radio Astronomy Software Group',
    'license': 'BSD',
    'description': ('A Simple Delay Spectrum calculator for radio '
                    'interferometers. '),
    'long_description': readme,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'simpleDS': 'simpleDS'},
    'packages': ['simpleDS', 'simpleDS.tests'],
    'scripts': glob.glob('scripts/*'),
    'version': version.version,
    'include_package_data': True,
    'setup_requires': ["pytest-runner", 'numpy>=1.15', 'six>=1.10'],
    'install_requires': ['numpy>1.15',
                         'astropy>2.0',
                         'scipy',
                         'pyuvdata>=1.3.8',
                         'six>=1.10'],
    'tests_require': ['pytest'],
}

if __name__ == '__main__':
    setup(**setup_args)
