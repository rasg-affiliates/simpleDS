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
    'author': 'Matthew Kolopanis',
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
    'install_requires': ['numpy>1.10', 'astropy>1.2', 'nose', 'pyuvdata',
                         'future'],
    'test_suite': 'nose'
}

if __name__ == '__main__':
    setup(**setup_args)
