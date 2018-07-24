# Simple Delay Spectrum (SimpleDS)

[![Build Status](https://travis-ci.org/mkolopanis/simpleDS.svg?branch=master)](https://travis-ci.org/mkolopanis/simpleDS)
[![Coverage Status](https://coveralls.io/repos/github/mkolopanis/simpleDS/badge.svg?branch=master)](https://coveralls.io/github/mkolopanis/simpleDS?branch=master)

`simpleDS` is a python package used to calculate the Delay Power Spectrum
of interferometric radio data. It performs the Fourier Transform of
visibility data along the Frequency dimension
and cross-multiplies redundant baseline information (if available).
`simpleDS` attempts to caclulate the Delay Spectrum in the simplest manner
using only Fast Fourier Transforms (FFTs) and calculating the beam_squared_area
directly from a UVBeam compatible beam map.
This calculator requires `pyuvdata` for data handling and beam area calculation.

## Installation

### Dependencies
First install dependencies.

* numpy >= 1.10
* astropy >= 2.0
* future (for python3 forward compatibility)
* pyuvdata (`pip install pyuvdata` or use https://github.com/HERA-Team/pyuvdata.git)

For anaconda users, we suggest using conda to install astropy, numpy and scipy and conda-forge
for aipy (```conda install -c conda-forge aipy```).

### Installing hera_qm
Clone the repo using
`git clone https://github.com/mkolopanis/simpleDS.git`

Navigate into the directory and run `python setup.py install` or `pip install .`

### Running Tests
Requires installation of `nose` package.
From the source `simpleDs` directory run: `nosetests simpleDS`.
