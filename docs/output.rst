.. _output:

Simulation output structure
###########################

TORAX file output is written to a ``state_history.nc`` netCDF file. The ``output_dir``
string is set in the config dict **runtime_params** key, with default
``'/tmp/torax_results_<YYYYMMDD_HHMMSS>/'``.

Currently we do not support backwards compatibility for old netCDF files. We
have some limited support for being able to use old TORAX netCDF files with
plotting (see :ref:`plotting`) but don't make any guarantees on this!

The output is an `[xarray] <https://docs.xarray.dev>`_ DataSet. The Dataset
is a flat structure containing ``xarray.DataArray`` type data variables,
corresponding to a subset of the TORAX internal ``Geometry``, ``core_profiles``,
``core_sources``, and ``core_transport`` objects. Note that while sharing similar
names and contents, these internal objects do not exactly correspond to IMAS
objects.

Also, note that, as of June 2024, TORAX does not have a specific COCOS it
adheres to, yet. Our team is working on aligning to a specific standard COCOS
on both the input and output. (CHEASE, which is COCOS 2 is still the supported
way to provide geometric inputs to TORAX, and we will continue to support CHEASE
as an input method, regardless of which COCOS we choose.)

Dimensions
==========

The Dataset variables can have the following dimensions:

* (time)
* (space)
* (time, space)

There are two named variants of spatial dimension:

* **rho_cell**: corresponding to the ``torax.fvm`` cell grid (see :ref:`fvm`).
* **rho_face**: corresponding to the ``torax.fvm`` face grid (see :ref:`fvm`).

In all subsequent lists, the dimensions associated with each variable or coordinate
will be surrounded by parentheses, e.g. (time, rho_cell).

Coordinates
===========

The Dataset contains the following Coordinates.

* ``time`` (time)
    Times corresponding to each simulation timestep, in units of [s].

* ``rho_cell`` (rho_cell)
    Toroidal flux coordinate (see :ref:`glossary`) on the fvm cell grid, in units of [m].
    The array size is set in the input config by ``geometry['nrho']``.

* ``rho_cell_norm`` (rho_cell)
    Normalized toroidal flux coordinate (see :ref:`glossary`) on the fvm cell grid.
    The array size is set in the input config by ``geometry['nrho']``.

* ``rho_face`` (rho_face)
    Toroidal flux coordinate (see :ref:`glossary`) on the fvm face grid, in units of [m].
    The array size is ``geometry['nrho']+1``.

* ``rho_face_norm`` (rho_face)
    Normalized toroidal flux coordinate (see :ref:`glossary`) on the fvm face grid.
    The array size is ``geometry['nrho']+1``.

In xarray, Coordinates are also embedded within the data variables. For example,
a data variable like ``j_bootstrap``, with dimensions (time, rho_cell), will be associated
with both ``r_cell`` and ``r_cell_norm``.

Data variables
==============

All TORAX output data variables are listed in thematic order. Please note that
this list is expected to grow as more simulation output post-processing is developed.

Geometry
--------

``vpr`` (rho_cell)
  Volume derivative :math:`\frac{dV}{d \rho}`, on the cell grid, in [:math:`m^2`].

``vpr_face`` (rho_face)
  Volume derivative :math:`\frac{dV}{d \rho}`, on the face grid, in [:math:`m^2`].

``spr`` (rho_cell)
  Surface derivative :math:`\frac{dS}{d \rho}`, on the cell grid, in [:math:`m`].

``spr_face`` (rho_face)
  Surface derivative :math:`\frac{dS}{d \rho}`, on the face grid, in [:math:`m`].

core_profiles
-------------

``temp_el`` (time, rho_cell)
  Electron temperature in :math:`[keV]`.

``temp_ion`` (time, rho_cell)
  Ion temperature in :math:`[keV]`.

``ne`` (time, rho_cell)
  Electron density in units of ``nref``.

``ni`` (time, rho_cell)
  Main ion density in units of ``nref``.

``nref`` (time)
  Reference density in :math:`[m^{-3}]`.

``psi`` (time, rho_cell)
  Poloidal flux :math:`(\psi)` in :math:`[Wb]`.

``psidot`` (time, rho_cell)
  Loop voltage :math:`V_{loop}=\frac{\partial\psi}{\partial t}`.

``q_face`` (time, rho_face)
  q-profile on face grid.

``s_face`` (time, rho_face)
  Magnetic shear on face grid.

``sigma`` (time, rho_cell)
  Plasma conductivity on cell grid.

``j_bootstrap`` (time, rho_cell)
  Bootstrap current density on cell grid, in :math:`[A/m^2]`

``j_bootstrap_face`` (time, rho_face)
  Bootstrap current density on face grid, in :math:`[A/m^2]`

``generic_current_source`` (time, rho_cell)
  External non-inductive current density on cell grid, as defined by the generic ``generic_current_source`` source, in :math:`[A/m^2]`.

``generic_current_source_face`` (time, rho_face)
  External non-inductive current density on face grid as defined by the generic ``generic_current_source`` source, in :math:`[A/m^2]`.

``johm`` (time, rho_cell)
  Ohmic current density on cell grid in :math:`[A/m^2]`.

``johm_face`` (time, rho_face)
  Ohmic current density on face grid in :math:`[A/m^2]`.

``jtot`` (time, rho_cell)
  Total current density on cell grid in :math:`[A/m^2]`.

``jtot_face`` (time, rho_face)
  Total current density on face grid in :math:`[A/m^2]`.

core_sources
------------

Any source which is not included in the input config, will `not` have a corresponding
output in ``state_history.nc``. This needs to be taken into account in analysis scripts and plotting tools.

``Qext_e`` (time, rho_cell)
  External electron heat source density, as defined by the generic ``generic_ion_el_heat_source``, in :math:`[W/m^3]`.

``Qext_i`` (time, rho_cell)
  External ion heat source density, as defined by the generic ``generic_ion_el_heat_source``, in :math:`[W/m^3]`.

``Qfus_e`` (time, rho_cell)
  Fusion electron heat source density in :math:`[W/m^3]`.

``Qfus_i`` (time, rho_cell)
  Fusion ion heat source density in :math:`[W/m^3]`.

``Qohm`` (time, rho_cell)
  Ohmic electron heat source density in :math:`[W/m^3]`.

``Qei`` (time, rho_cell)
  Ion-electron heat exchange density in :math:`[W/m^3]`.
  Positive values means heat source for ions, and heat sink for electrons.

``s_puff`` (time, rho_cell)
  Gas puff particle source density  in :math:`[s^{-1} m^{-3}]`.

``s_generic`` (time, rho_cell)
  Generic particle source density  in :math:`[s^{-1} m^{-3}]`.

``s_pellet`` (time, rho_cell)
  Pellet particle source density  in :math:`[s^{-1} m^{-3}]`.


core_transport
--------------

``chi_face_el`` (time, rho_face)
  Electron heat conductivity on face grid in :math:`m^2/s`

``chi_face_ion`` (time, rho_face)
  Ion heat conductivity on face grid in :math:`m^2/s`

``d_face_el`` (time, rho_face)
  Electron particle diffusivity on face grid in :math:`m^2/s`

``v_face_el`` (time, rho_face)
  Electron particle convection on face grid in :math:`m/s`

Examples
========

To demonstrate xarray and numpy manipulations of output data, the following code carries out
volume integration of ``Qfus_e`` and ``Qfus_i`` at the time closest to t=1. The result equals
the input config ``sources['generic_ion_el_heat_source']['Ptot']`` at the time closest to t=1.

``ds`` is the xarray.DataSet. The netCDF file is assumed to be in the working directory. ``vpr``
is assumed to not be time varying.

.. code-block:: python

  import numpy as np
  import xarray as xr

  ds = xr.open_dataset('state_history.nc')
  Ptot = np.trapz((ds.Qext_i+ds.Qext_e).sel(time=1.0, method='nearest') * ds.vpr, ds.r_cell)