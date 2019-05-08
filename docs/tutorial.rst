.. _tutorial:

Tutorial
========

.. testsetup::
   from __future__ import absolute_import, division, print_function

-------------
DelaySpectrum
-------------

DelaySpectrum: Initialization
-----------------------------
Loading data into a DelaySpectrum object can be done during object creation, or later manually.
A pyuvdata compatible file must be read by pyuvdata and imported into the DelaySpectrum.
DelaySpectrum objects assume the UVData object consists of only one redudant baseline type.


A UVBeam file is also necessary which holds relevant information about the `beam_area` and `beam_sq_area` for power spectrum normalizaiton.

If you are interested in using the built in noise simulation a receiver temperature (`trcvr`) must also be specified in the form of an astropy.Quantity object.

b) At creation
**************
::

  >>> import os
  >>> from pyuvdata import UVData, UVBeam
  >>> from pyuvdata.data import DATA_PATH as UVDATA_PATH
  >>> from simpleDS import DelaySpectrum
  >>> from simpleDS.data import DATA_PATH
  >>> from astropy import units

  # It would normally be necessary to also down-select to only one set of
  # redundant baselines but this PAPER data file is already in that format.
  >>> data_file = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
  >>> beam_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
  >>> UV = UVData()
  >>> UV.read_uvfits(data_file)
  >>> UVB = UVBeam()
  >>> UVB.read_beamfits(beam_file)

  >>> dspec = DelaySpectrum(uv=UV, uvb=UVB, trcvr=144*units.K)


a) After Creation
*****************
::

  >>> import os
  >>> from pyuvdata import UVData, UVBeam
  >>> from pyuvdata.data import DATA_PATH as UVDATA_PATH
  >>> from simpleDS import DelaySpectrum
  >>> from simpleDS.data import DATA_PATH
  >>> from astropy import units

  # It would normally be necessary to also down-select to only one set of
  # redundant baselines but this PAPER data file is already in that format.
  >>> data_file = os.path.join(UVDATA_PATH, 'test_redundant_array.uvfits')
  >>> beam_file = os.path.join(DATA_PATH, 'test_redundant_array.beamfits')
  >>> UV = UVData()
  >>> UV.read_uvfits(data_file)
  >>> UVB = UVBeam()
  >>> UVB.read_beamfits(beam_file)

  >>> dspec = DelaySpectrum()
  >>> dspec.add_uvdata(uv=UV)
  >>> dspec.add_uvbeam(uvb=UVB)
  >>> dspec.add_trcvr(trcvr=144*units.K)
