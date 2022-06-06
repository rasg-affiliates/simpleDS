# Changelog
All notable changes to this project will be documented in this file.

## [2.0.1] 06 June 2022

### Added
- File initialization for partial I/O

### Changed
- Thermal noise estimation to be more consistent with theoretical expectations.
- requires astorpy>=5.0.4 for unit compatibility.
- dropped python 3.7 support

### Fixed
- Compatibility with numpy 1.20
- Unnecessary `Nbls` factor in thermal estimation.

## [2.0.0] 23 January 2020

### Added
- basic I/O capabilities

### Changed
- specifies minimum dependency for pyuvdata >= 1.4.2
- Removed `version.py`. Instead use `pygitversion` to do versioning. This also improves the install-ability of the code. Does not rely on having dependencies previously installed.
- Removed active python 2.7 support. Python 2.7 support frozen in version 1.0.2

## [1.0.2] 21 November 2019
### Added
- ReadTheDocs support for simpleDS Documentation
- Versioning definition.

### Changed
- Default cosmology is now Planck15
- specifies minimum dependency for pyuvdata >= 1.3.8
- Tutorial notebook moved to ReadTheDocs and added more commentary.
- file compatibility to use uvfits not uvh5 since uvh5 is optional dependency.
- test package changed from nose to pytest
- Updated to DeprecationWarning for pyuvdata warnings
- updated warning text from pyuvdata
- Transition from nose to pytest

### Fixed
- various typos.
- Bug where `update_cosmology` would not convert littleh units back before trying to re-normalize power spectrum.
- Can call `calculate_delay_spectrum` twice now.
- Bug where frequency array was not cast to GHz in the `thermal_power` calculation.
- bug where `taper` was not squared in `thermal_power` calculation.

## [1.0.1] 19 April 2019
### Added
- update_cosmology now recomputes cosmological conversions and re-normalizes power spectrum
- thermal_conversion parameter to track the conversion of the thermal_power array to cosmological units
- cosmology object support in UnitParameters
- littleh units compatibility in python 3
- littleh conversion in update_cosmology for python 3
### Changed
- Moved some mathematical operations out of return statements to save memory
- Various typos
- Propagated `littleh_units` keyword up to `calculate_delay_spectrum` function.
- Docstring Formatting
- Moved all cosmological calculations to update_cosmology function
### Fixed
- definition of acceptable units for data_array and noise_array. Previously had units.K not units.K * units.sr * units.Hz
- definition of units for power_array and noise_power when data is uncalibrated
- definition of units for unit_conversion and thermal_conversion when data is uncalibrated

## [1.0.0] - 13 March 2019
### Added
- DelaySpectrum Object for handing data, checking units, and performing Fourier Transform
- Parameter Class to handle units and verify shape, type, etc of data in the DelaySpectrum Object, subclass of UVParameter.
- IPython notebook to provide some examples oh how to create and interact with DelaySpectrum Object.
- simple LST aligning function to align even and odd data sets. Must be called before loading UVData objects into DelaySpectrum object.
- Test data of and even and odd observation for example notebook.

### Changed
- Complete API overhaul
- Power spectrum normalization computes integrals over bandpass instead of assuming constant factors across a sub-band.
- travis integration with osx and linux. Also python 2.7, 3.6, and 3.7.
- Moved some functions from delay_spectrum to utils package
- Thermal noise expectation calculation and normalization (work in progress)
- DelaySpectrum attempts to extract the receiver_temperature_array from a UVBeam object if applicable.
- `integration_time` array now only nbls, ntimes in shape. This hopefully should not cause a problem with pyuvdata since all input arrays must only have redundant baselines.
- Added tolerance to simple lst_align function to allow slight differences between uvdata objects
### Fixed
- Various PEP8 typos
- Spectral Window Selection bug
- Unit normalization in cosmo.X2Y now includes 1/sr term
- Broadcasting of computations along polarization axis

## [v0.1.0] -28 January 2019
- Prototype release version
### Added
- travis integration
- Method to read PAPER miriad files
- cosmological conversion module
- normalized Fourier transform with inclusion of units
- Noise realization to follow data during delay transformation
- Python 2 and 3 compatibility

## Changed
- No longer dependent on aipy. Requires antenna position files to read PAPER miriad files
- better separation of keyword arguments in read_paper_miriad
- unit handling in cross-multiplication to make unit multiplication as safe as possible.

### Fixed
- Beam square area normalization for noise as well.
- Properly Conjugates paper baselines when adding to UVData object.
- various typos and PEP8
- PendingDeprecationWarning handling when reading PAPER miriad files
