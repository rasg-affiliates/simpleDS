#! /usr/bin/env python
"""A command line script to conver PAPER miriad files to newer file formats."""

import os
import sys
import numpy as np
import argparse
import six
from six.moves import range
from pyuvdata import UVData, utils as uvutils
from pyuvdata import __version__ as uvversion


def read_paper_miriad(filename, antpos_file=None, **kwargs):
    """Read PAPER miriad files and return pyuvdata object.

    Uses a UVData object to read PAPER miriad files, caught known issues and
    insert missing required attributes into the UVData object before returning to the user.

    Parameters
    ----------
    filename : st
        pyuvdata compatible PAPER miriad file
    antpols_file : str
        A file of antenna postions.
        This is assumed to be a csv or other file readable by numpy.genfromtxt
        Required to generate uvws.
    kwargs : any
        kwargs are passed to the following functions:
            - numpy.genfromtxt
            - UVData.read
        Please consult the documentation for these functions for lists of keywords.

    Returns
    -------
    uv : UVData object
        Correctly formatted pyuvdata object from input PAPER data

    Raises
    ------
    ValueError
        If no `antpos_file` provided.
    IOError
        If `antpos_file` does not exist.

    """
    uv = UVData()
    # find keywords that pass to uv.read
    uvdata_argspec = six.get_function_code(uv.read).co_varnames
    kwargs_uvdata = {key: kwargs[key] for key in kwargs if key in uvdata_argspec}
    uv.read_miriad(filename, **kwargs_uvdata, run_check=False)

    if antpos_file is None:
        raise ValueError("An antpos_file file " "is required to generate uvw array.")

    if not os.path.exists(antpos_file):
        raise IOError("{0} not found.".format(antpos_file))

    # find keywords that pass to np.genfromtxt
    genfromtxt_argpsec = six.get_function_code(np.genfromtxt).co_varnames
    kwargs_genfromtxt = {
        key: kwargs[key] for key in kwargs if key in genfromtxt_argpsec
    }

    antpos = np.genfromtxt(antpos_file, **kwargs_genfromtxt)

    antpos_ecef = uvutils.ECEF_from_ENU(antpos, *uv.telescope_location_lat_lon_alt)
    antpos_itrf = antpos_ecef - uv.telescope_location
    setattr(uv, "Nants_telescope", antpos_itrf.shape[0])
    ant_names = [str(x) for x in range(uv.Nants_telescope)]
    setattr(uv, "antenna_names", ant_names)
    ant_nums = [x for x in range(uv.Nants_telescope)]
    setattr(uv, "antenna_numbers", ant_nums)
    setattr(uv, "antenna_positions", antpos_itrf)

    uv.set_uvws_from_antenna_positions()

    # PAPER uses Eastward pointing baselines, look for and baselines
    # facing west, conjugate them and flip the antenna order in the UV
    # object.
    conj_bls = np.argwhere(uv.uvw_array.squeeze()[:, 0] < 0)

    uv.data_array[conj_bls] = np.conj(uv.data_array[conj_bls])
    uv.ant_1_array[conj_bls], uv.ant_2_array[conj_bls] = (
        uv.ant_2_array[conj_bls],
        uv.ant_1_array[conj_bls],
    )
    uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array, uv.ant_2_array)
    # Rebuild the uvw array
    uv.set_uvws_from_antenna_positions()

    if "FRF_NEBW" in uv.extra_keywords:
        uv.integration_time = np.ones_like(uv.integration_time)
        uv.integration_time *= uv.extra_keywords["FRF_NEBW"]

    return uv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "A command line reader to conver PAPER Miriad files into a newer "
            "former uvdata complatible file. This requires pyuvdata version < 1.5"
        )
    )
    parser.add_argument("filename", type=str, help="PAPER Correlator Miriad file.")
    parser.add_argument(
        "antpos_file",
        type=str,
        help="The antenna positions for corresponding PAPER Miriad file.",
    )
    parser.add_argument(
        "--outfmt",
        "-o",
        dest="format",
        type=str,
        choices=["miriad", "uvfits", "uvh5"],
        default="miriad",
    )

    args = parser.parse_args()

    if float(".".join(uvversion.split(".")[:-1])) >= 1.5:
        print("This script is only compatible with pyuvdata version < 1.5.")
        sys.exit(0)

    uv = read_paper_miriad(
        filename=args.filename,
        antpos_file=args.antpos_file,
        skip_header=3,
        usecols=[1, 2, 3],
    )
    if args.format == "miriad":
        uv.write_miriad(args.filename, clobber=True)
    elif args.format == "uvfits":
        out_name = ".".join(args.filename.split(".")[:-1]) + ".uvfits"
        uv.write_uvfits(out_name, clobber=True)
    elif args.format == "uvh5":
        out_name = ".".join(args.filename.split(".")[:-1]) + ".uvh5"
        uv.write_uvh5(out_name, clobber=True)
