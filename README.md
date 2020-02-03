# Simple Delay Spectrum (SimpleDS)

![](https://github.com/rasg-affiliates/simpleDS/workflows/Run%20Tests/badge.svg?branch=master)
[![Build Status](https://travis-ci.com/rasg-affiliates/simpleDS.svg?branch=master)](https://travis-ci.com/rasg-affiliates/simpleDS)
[![CircleCI](https://circleci.com/gh/rasg-affiliates/simpleDS.svg?style=svg)](https://circleci.com/gh/rasg-affiliates/simpleDS)
[![codecov](https://codecov.io/gh/rasg-affiliates/simpleDS/branch/master/graph/badge.svg)](https://codecov.io/gh/rasg-affiliates/simpleDS)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)



SimpleDS is currently in a working *BETA* state.  All code will execute and tests pass, however there are still a number of bugs to fix and enhancements to make.


`simpleDS` is a python package used to calculate the Delay Power Spectrum
of interferometric radio data. It performs the Fourier Transform of
visibility data along the Frequency dimension
and cross-multiplies redundant baseline information (if available).
`simpleDS` attempts to calculate the Delay Spectrum in the simplest manner
using only Fast Fourier Transforms (FFTs) and calculating the beam_squared_area
directly from a UVBeam compatible beam map.
This calculator requires [`pyuvdata`](https://github.com/RadioAstronomySoftwareGroup/pyuvdata.git) for data handling and beam area calculation.

# Motivation
The main goals are:

1. Provide a simple, user-friendly interface for computing a delay Fourier transform on interferometric radio data.
2. Perform the delay power spectrum estimation on radio interferometric data using only Fast Fourier Transforms (FFTs) to provide a mathematically simple framework for analysis.
3. Perform explicit unit conversions on data with as few approximations as possible.

# Package Details
simpleDS has one major user class:

* DelaySpectrum: supports Fourier transformation of UVData compatible radio visibility data. Also can perform power spectrum estimation and redundant baseline cross multiplication. Creates noise realization of the input data product to track analysis steps and verify normalization. Attempts to produce a theoretical thermal noise limit for the input data power spectrum.

and one minor cosmological conversion module:

* cosmo: Uses astropy.cosmology to compute relevant cosmological factors for 21cm radio data to convert between interferometric (f, u,v,w) to (k<sub>&parallel;</sub>, k<sub>&perp;</sub>) units.

# Installation

## Dependencies
First install dependencies.

* numpy >= 1.18
* scipy
* astropy >= 4.0
* h5py (for uvh5 compatibility with pyuvdata, optional)
* astropy-healpix used for UVBeam interpolations through pyuvdata (optional, only used with `use_exact` keyword for `add_uvbeam`)
* pyuvdata >=1.4.2 (conda install -c conda-forge pyuvdata, `pip install pyuvdata`, or use the [development version](https://github.com/RadioAstronomySoftwareGroup/pyuvdata.git))

For anaconda users, we suggest using conda to install astropy, numpy and scipy.

## Installing simpleDS
Clone the repo using
`git clone https://github.com/rasg-affiliates/simpleDS.git`

Navigate into the directory and run `pip install .`
To also install the optional `astropy-healpix` use `pip install .[healpix]`.
For all packages necessary for the testing suite use `pip install .[all]`

## Running Tests
We use `pytest` to execute the tests for this pacakge.
From the source `simpleDS` directory run: `python -m pytest` or `pytest`.


# Versioning
We use a `generation.major.minor` version number format.

- The `generation` number for very significant improvements, major rewrites, and API breaking changes.
- The `major` number to indicate substantial package changes.
- The `minor` number to release smaller incremental updates which usually do not include breaking API changes.

We do our best to provide a significant period of deprecation warnings for all breaking changes to the API. We track all changes in our [changelog](https://github.com/rasg-affiliates/simpleDS/blob/master/CHANGELOG.md).

# Documentation
A tutorial with example usage is hosted on [ReadTheDocs](https://simpleds.readthedocs.io).
