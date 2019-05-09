# -*- coding: utf-8 -*-
"""Format the readme.md file into the sphinx index.rst file."""
from __future__ import absolute_import, division, print_function

import codecs
import os
import inspect
import re
import pypandoc
from astropy.time import Time


def write_index_rst(readme_file=None, write_file=None):
    """Create index on the fly."""
    t = Time.now()
    t.out_subfmt = 'date_hms'
    out = ('.. simpleDS documentation master file, created by\n'
           '   make_index.py on {date}\n\n').format(date=t.iso)

    print(readme_file)
    if readme_file is None:
        main_path = os.path.dirname(os.path.dirname(os.path.abspath(inspect.stack()[0][1])))
        readme_file = os.path.join(main_path, 'README.md')

    readme_md = pypandoc.convert_file(readme_file, 'md')
    readme_text = pypandoc.convert_file(readme_file, 'rst')

    title_badge_text = (
        'simpleDS\n========\n\n'
        '.. image:: https://travis-ci.com/RadioAstronomySoftwareGroup/simpleDS.svg?branch=master\n'
        '    :target: https://travis-ci.com/RadioAstronomySoftwareGroup/simpleDS\n\n'
        '.. image:: https://circleci.com/gh/RadioAstronomySoftwareGroup/simpleDS.svg?style=svg\n'
        '    :target: https://circleci.com/gh/RadioAstronomySoftwareGroup/simpleDS\n\n'
        '.. image:: https://codecov.io/gh/RadioAstronomySoftwareGroup/simpleDS/branch/master/graph/badge.svg\n'
        '  :target: https://codecov.io/gh/RadioAstronomySoftwareGroup/simpleDS\n\n')

    readme_text = pypandoc.convert_file(readme_file, 'rst')

    begin_desc = 'SimpleDS is currently in a working *BETA* state'
    start_desc = str.find(readme_text, begin_desc)

    readme_text = title_badge_text + readme_text[start_desc:]

    end_text = '# Documentation'
    regex = re.compile(end_text.replace(' ', r'\s+'))
    loc = re.search(regex, readme_text).start()
    tutorial_notebook_file = os.path.join(os.path.abspath('../examples'), 'simpleds_tutorial.ipynb')

    out += readme_text[0:loc]
    out += ('\n\nFurther Documentation\n====================================\n'
            '.. toctree::\n'
            '   :maxdepth: 2\n\n'
            # '   index\n'
            '   dspec_parameters\n'
            '   DelaySpectrum\n'
            '   cosmo\n'
            '   utils\n'
            '   examples/simpleDS_tutorial\n'
            )

    out.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\xa0", " ")

    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, 'index.rst')
    F = codecs.open(write_file, 'w', 'utf-8')
    F.write(out)
    print("wrote " + write_file)
