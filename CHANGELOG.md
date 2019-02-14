# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
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

### Fixed
- Various PEP8 typos
- Spectral Window Selection bug
- Unit normalization in cosmo.X2Y now includes 1/sr term
- Broadcasting of beam_area and beam_sq_area along the polarization axis

## [v0.1.0] - 1/28/2019
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
