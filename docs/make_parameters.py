# -*- coding: utf-8 -*-
"""Format the simpleDS object parameters into a sphinx rst file."""
from __future__ import absolute_import, division, print_function

import os
import inspect
from simpleDS import DelaySpectrum
from astropy.time import Time


def write_dataparams_rst(write_file=None):
    """Write a parameter rst on the fly."""
    dspec = DelaySpectrum()
    out = "SimpleDS Parameters\n==========================\n"
    out += (
        "These are the standard attributes of DelaySpectrum objects.\n\nUnder the hood "
        "they are actually properties based on UnitParameter object which themselves are "
        "based on a UVParameter.\n\n"
    )
    out += "Required\n----------------\n"
    out += (
        "These parameters are required to have a sensible DelaySpectrum object and \n"
        "are required for most kinds of power spectrum estimation."
    )
    out += "\n\n"
    for thing in dspec.required():
        obj = getattr(dspec, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"

    out += "Optional\n----------------\n"
    out += (
        "These parameters are not required to prepare a DelaySpectrum object "
        "for power spectrum estimation. However, some become required once "
        "the data has been Fourier Transformed into delay space."
    )
    out += "\n\n"
    for thing in dspec.extra():
        obj = getattr(dspec, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"
    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += "last updated: {date}".format(date=t.iso)
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "dspec_parameters.rst")
    F = open(write_file, "w")
    F.write(out)
    print("wrote " + write_file)
