SimpleDS Parameters
==========================
These are the standard attributes of DelaySpectrum objects.

Under the hood they are actually properties based on UnitParameter object which themselves are based on a UVParameter.

Required
----------------
These parameters are required to have a sensible DelaySpectrum object and
are required for most kinds of power spectrum estimation.

**Nbls**
     Number of baselines

**Ndelays**
     Number of delay channels. Must be equal to (Nfreqs) with FFT usage. However may differ if a more intricate Fourier Transform is used.

**Nfreqs**
     Number of frequency channels per spectral window

**Npols**
     Number of polarizations

**Nspws**
     Number of UVData objects which have been read. Only a maximum of 2 UVData objects can be read into a single DelaySpectrum Object. Only 1 UVData must be ready to enable delay transformation and power spectrum estimation.

**Ntimes**
     Number of times

**Nuv**
     Number of UVData objects which have been read. Only a maximum of 2 UVData objects can be read into a single DelaySpectrum Object. Only 1 UVData must be ready to enable delay transformation and power spectrum estimation.

**ant_1_array**
     Array of first antenna indices, shape (Nbls), type = int, 0 indexed

**ant_2_array**
     Array of second antenna indices, shape (Nbls), type = int, 0 indexed

**baseline_array**
     Array of baseline indices, shape (Nbls), type = int; baseline = 2048 * (ant1+1) + (ant2+1) + 2^16

**beam_area**
     The integral of the power beam area. Shape = (Nspws, Npols, Nfreqs)

**beam_sq_area**
     The integral of the squared power beam squared area. Shape = (Nspws, Npols, Nfreqs)

**cosmology**
     Astropy cosmology object cabale of performing necessary cosmological calculations. Defaults to WMAP 9-Year.

**data_array**
     Array of the visibility data, shape: (Nspws, Nuv, Npols, Nbls, Ntimes, Nfreqs), type = complex float, in units of self.vis_units

**data_type**
     String indicating which domain the data is in. Allowed values are "delay", "frequency"

**delay_array**
     Array of frequencies, shape (Nspws, Nfreqs), units Hz

**flag_array**
     Boolean flag, True is flagged, shape: same as data_array.

**freq_array**
     Array of frequencies, shape (Nspws, Nfreqs), units Hz

**integration_time**
     Length of the integration in seconds, has shape (Npols, Nbls, Ntime). units s, assumes inegration time  is the same for all spectral windows and all frequncies in a spectral window. Assumes the same convention as pyuvdata, where this is the target amount of time a measurement is integrated over. Spectral window dimension allows for frequency dependent filtering to be properly tracked for noise simulations.

**k_parallel**
     Cosmological wavenumber of spatial modes probed along the line of sight. This value is awlays calculated, however it is not a realistic probe of k_parallel over large bandwidths. This code assumes k_tau >> k_perpendicular and as a results k_tau  is interpreted as k_parallel. In python 2 this unit is always in 1/Mpc. For python 3 users, it is possible to convert the littleh/Mpc using the littleh_units boolean flag in update_cosmology.

**k_perpendicular**
     Cosmological wavenumber of spatial modes probed perpendicular  to the line of sight. In python 2 this unit is always in 1/Mpc. For python 3 users, it is possible to convert the littleh/Mpc using the littleh_units boolean flag in update_cosmology.

**lst_array**
     Array of lsts, center of integration, shape (Ntimes), UVData objects must be LST aligned before adding data to the DelaySpectrum object.units radians

**noise_array**
     Array of the simulation noise visibility data, shape: (Nspws, Nuv, Npols, Nbls, Ntimes, Nfreqs), type = complex float, in units of self.vis_units. Noise simulation generated assuming the sky is 180K@180MHz relation with the input receiver temperature, nsample_array, and integration times.

**nsample_array**
     Number of data points averaged into each data elementself. Uses the same convention as a UVData object:NOT required to be an integer, type = float, same shape as data_array.The product of the integration_time and the nsample_array value for a visibility reflects the total amount of time that went into the visibility.

**polarization_array**
     Array of polarization integers, shape (Npols). Uses same convention as pyuvdata: pseudo-stokes 1:4 (pI, pQ, pU, pV);  circular -1:-4 (RR, LL, RL, LR); linear -5:-8 (XX, YY, XY, YX).

**redshift**
     Mean redshift of given frequencies. Calculated with assumed cosmology.

**taper**
     Spectral taper function used during Fourier Transform. Functions like scipy.signal.windows.blackmanharris

**trcvr**
     System receiver temperature used in noise simulation. Stored as array of length (Nfreqs), but may be passed as a single scalar. Must be a Quantity with units compatible to K.

**uvw**
     Nominal (u,v,w) vector of baselines in units of meters

**vis_units**
     Visibility units, options are: "uncalib", "Jy" or "K str"

Optional
----------------
These parameters are not required to prepare a DelaySpectrum object for power spectrum estimation. However, some become required once the data has been Fourier Transformed into delay space.

**noise_power**
     The cross-multiplied simulated noise power spectrum estimates. Units are converted to cosmological frame (mK^2/(hMpc^-1)^3).For uncalibrated data the noise simulation is not well defined but is still calculated and will have units (Jy Hz)^2. In python 2 this unit is always in mK^2 Mpc^3. For python 3 users, it is possible to convert the mK^2 / (littleh/Mpc)^3 using the littleh_units boolean flag in update_cosmology.

**power_array**
     The cross-multiplied power spectrum estimates. Units are converted to cosmological frame (mK^2/(hMpc^-1)^3).For uncalibrated data the cosmological power is not well defined the power array instead represents the power in the delay domain adn will have units (Hz^2). In python 2 this unit is always in mK^2 Mpc^3. For python 3 users, it is possible to convert the mK^2 / (littleh/Mpc)^3 using the littleh_units boolean flag in update_cosmology.

**thermal_conversion**
     The cosmological unit conversion factor applied to the thermal noise estimate. Has the form ("Nspws", "Npols"). Accounts for all beam polarizations.Always has units mK^2 Mpc^3 /( K^2 sr^2 Hz^2)

**thermal_power**
     The predicted thermal variance of the input data averaged over all input baselines.Units are converted to cosmological frame (mK^2/(hMpc^-1)^3). In python 2 this unit is always in mK^2 Mpc^3. For python 3 users, it is possible to convert the mK^2 / (littleh/Mpc)^3 using the littleh_units boolean flag in update_cosmology.

**unit_conversion**
     The cosmological unit conversion factor applied to the data. Has the form ("Nspws", "Npols"). Accounts for all beam polarizations.Depending on units of input visibilities it may take units of mK^2/(h/Mpc)^3 / (K * sr * Hz)^2 or mK^2/[h/Mpc]^3 / (Jy * Hz)^2

last updated: 2019-05-09
