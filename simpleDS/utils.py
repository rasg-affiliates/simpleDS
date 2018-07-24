"""Read and Write support for PAPER miriad files with pyuvdata."""

import os
import sys
import numpy as np
from pyuvdata import UVData, utils as uvutils
from builtins import range, map
from astropy import constants as const

def import_calfile(filename=None):
    """Import the calfile from the PAPER array."""
    if filename is None:
        raise ValueError("Must Supply calfile.")
    if filename.split('.')[-1] == 'py':
        # If the input name has a .py attached it needs to be removed
        # before it can be imported
        filename = '.'.join(filename.split('.')[:-1])
    if not os.path.exists(filename + '.py'):
        raise IOError(filename + '.py not found')
    cal_dir_name = os.path.dirname(filename)
    cal_base_name = os.path.basename(filename)
    sys.path.append(cal_dir_name)
    exec("import {0} as calfile".format(cal_base_name))
    return calfile


def read_paper_miriad(filename, calfile=None, antpos_file=None, **kwargs):
    """Read PAPER miriad files and return pyuvdata object.

    One of a calfile or an antpos_file is required to generate uvws.
    kwargs passed to numpy.genfromtxt if antpos file is provided.
    """
    if not isinstance(filename, (list, np.ndarray)):
        filename = [filename]
    uv = UVData()
    uv.read_miriad(filename)

    if all(x is None for x in [calfile, antpos_file]):
        raise ValueError("Either an antpos_file file or a calfile "
                         "is required to generate uvw array.")
    elif calfile:
        calfile = import_calfile(calfile)
        # load an AtennaArray object from calfile with a dummy Frequency
        aa = calfile.get_aa(np.array([.1]))
        antpos = np.array([aa.get_baseline(0, i, src='z')
                           for i in range(len(aa.ants))])
        antpos *= const.c.to('m/ns').value  # Convert light-nanoseconds to meters
        antpos -= antpos.mean(0)
    else:
        if not os.path.exists(antpos_file):
            raise IOError("{0} not found.".format(antpos_file))
        antpos = np.genfromtxt(antpos_file, **kwargs)
    antpos_ecef = uvutils.ECEF_from_ENU(antpos, *uv.telescope_location_lat_lon_alt)
    antpos_itrf = antpos_ecef - uv.telescope_location
    good_ants = list(map(int, uv.antenna_names))
    antpos_itrf = np.take(antpos_itrf, good_ants, axis=0)
    setattr(uv, 'antenna_positions', antpos_itrf)
    uv.set_uvws_from_antenna_positions()

    return uv
