# Simple Delay Spectrum (SimpleDS)

[![Build Status](https://travis-ci.org/mkolopanis/simpleDS.svg?branch=master)](https://travis-ci.org/mkolopanis/simpleDS)
[![Coverage Status](https://coveralls.io/repos/github/mkolopanis/simpleDS/badge.svg?branch=master)](https://coveralls.io/github/mkolopanis/simpleDS?branch=master)


`simpleDS` is a python package used to calculate the Delay Power Spectrum
of interferometric radio data. It performs the Fourier Transform of
visibility data along the Frequency dimension
and cross-multiplies redundant baseline information (if available).
`simpleDS` attempts to calculate the Delay Spectrum in the simplest manner
using only Fast Fourier Transforms (FFTs) and calculating the beam_squared_area
directly from a UVBeam compatible beam map.
This calculator requires `pyuvdata` for data handling and beam area calculation.

# Motivation
The main goals are:

1. Provide a simple, user-friendly interface for computing a delay Fourier transform on interferometric radio data.
2. Perform the delay power spectrum estimation on radio interferometric data using only Fast Fourier Transforms (FFTs) to provide a mathematically simple framework for analysis.
3. Perform explicit unit conversions on data with as few approximations as possible.

# Package Details
simpleDS has one major user class:

* DelaySpectrum: supports Fourier transformation of UVData compatible radio visibility data. Also can perform power spectrum estimation and redundant baseline cross multiplication. Creates noise realization of the input data product to track analysis steps and verify normalization. Attempts to produce a theoretical thermal noise limit for the input data power spectrum.

and one minor cosmological conversion module:

* cosmo: Uses astropy.cosmology to compute relevant cosmological factors for 21cm radio data to convert between interferometric (f, u,v,w) to (k\_parallel, k\_perpendicular) units.

# Installation

## Dependencies
First install dependencies.

* numpy >= 1.10
* scipy
* astropy >= 2.0
* h5py (for uvh5 compatibility with pyuvdata)
* six (for compatibility between python 2 and 3)
* pyuvdata (conda install -c conda-forge pyuvdata, `pip install pyuvdata`, or use the development version  https://github.com/HERA-RadioAstronomySoftwareGroup/pyuvdata.git)

For anaconda users, we suggest using conda to install astropy, numpy and scipy.

## Installing simpleDS
Clone the repo using
`git clone https://github.com/mkolopanis/simpleDS.git`

Navigate into the directory and run `python setup.py install` or `pip install .`

## Running Tests
Requires installation of `nose` package.
From the source `simpleDS` directory run: `nosetests simpleDS`.
