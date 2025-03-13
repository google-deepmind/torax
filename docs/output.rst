.. _output:

Simulation output structure
###########################

TORAX file output is written to a ``state_history.nc`` netCDF file. The ``output_dir``
string is set in the config dict **runtime_params** key, with default
``'/tmp/torax_results_<YYYYMMDD_HHMMSS>/'``.

Currently we do not support backwards compatibility for old netCDF files. The
prior flat structure of TORAX outputs in a single dataset is at v0.2.0.

The output is an `[xarray] <https://docs.xarray.dev>`_ DataTree.

The Dataset is a hierarchical structure containing ``xarray.DataSet``s
corresponding to a subset of the TORAX internal ``Geometry``, ``core_profiles``,
``core_sources``, ``core_transport`` and ``post_processed_outputs`` each of
which contains ``xarray.DataArray`` type data variables. Note that while sharing
similar names and contents, these internal objects do not exactly correspond to
IMAS objects.

Also, note that, as of June 2024, TORAX does not have a specific COCOS it
adheres to, yet. Our team is working on aligning to a specific standard COCOS
on both the input and output. (CHEASE, which is COCOS 2 is still the supported
way to provide geometric inputs to TORAX, and we will continue to support CHEASE
as an input method, regardless of which COCOS we choose.)

Dimensions
==========

The DataTree/Dataset variables can have the following dimensions:

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

All the ``Dataset``s in the output contains the following Coordinates.

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
with both ``rho_cell`` and ``rho_cell_norm``.

Top level dataset
=================
The top level dataset contains an indicator of whether the simulation completed
successfully.

``sim_error`` ()
  Indicator if the simulation completed successfully, 0 if successful, 1 if not.


Child datasets
==============
The following datasets are child nodes, the title of each section is the name of
the child ``DataTree``.

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
  Plasma conductivity on cell grid, in :math:`[S/m]`.

``j_bootstrap`` (time, rho_cell)
  Bootstrap current density on cell grid, in :math:`[A/m^2]`

``j_bootstrap_face`` (time, rho_face)
  Bootstrap current density on face grid, in :math:`[A/m^2]`

``core_profiles_generic_current_source`` (time, rho_cell)
  External non-inductive current density on cell grid, as defined by the generic ``generic_current_source`` source, in :math:`[A/m^2]`.

``johm`` (time, rho_cell)
  Ohmic current density on cell grid in :math:`[A/m^2]`.

``jtot`` (time, rho_cell)
  Total current density on cell grid in :math:`[A/m^2]`.

``jtot_face`` (time, rho_face)
  Total current density on face grid in :math:`[A/m^2]`.

``Ip_profile_face`` (time, rho_face)
  Current profile on face grid, in :math:`[A]`.

``I_bootstrap`` (time)
  Total bootstrap current, in :math:`[A]`.

core_sources
------------

Any source which is not included in the input config, will `not` have a corresponding
output in ``state_history.nc``. This needs to be taken into account in analysis scripts and plotting tools.
In future we aim to populate core_sources in a more structured way.

``generic_ion_el_heat_source_el`` (time, rho_cell)
  External electron heat source density, as defined by the generic ``generic_ion_el_heat_source``, in :math:`[W/m^3]`.

``generic_ion_el_heat_source_ion`` (time, rho_cell)
  External ion heat source density, as defined by the generic ``generic_ion_el_heat_source``, in :math:`[W/m^3]`.

``generic_current_source`` (time, rho_cell)
  Generic externl current source density in :math:`[A/m^2]`.

``fusion_heat_source_el`` (time, rho_cell)
  Fusion electron heat source density in :math:`[W/m^3]`.

``fusion_heat_source_ion`` (time, rho_cell)
  Fusion ion heat source density in :math:`[W/m^3]`.

``ohmic_heat_source`` (time, rho_cell)
  Ohmic electron heat source density in :math:`[W/m^3]`.

``qei_source`` (time, rho_cell)
  Ion-electron heat exchange density in :math:`[W/m^3]`.
  Positive values means heat source for ions, and heat sink for electrons.

``gas_puff_source`` (time, rho_cell)
  Gas puff particle source density  in :math:`[s^{-1} m^{-3}]`.

``generic_particle_source`` (time, rho_cell)
  Generic particle source density  in :math:`[s^{-1} m^{-3}]`.

``pellet_source`` (time, rho_cell)
  Pellet particle source density  in :math:`[s^{-1} m^{-3}]`.

``electron_cyclotron_source_el`` (time, rho_cell) [:math:`W/m^3`]:
  Electron cyclotron heating power density.

``electron_cyclotron_source_j`` (time, rho_cell) [:math:`A/m^2`]:
  Electron cyclotron current.


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

post_processed_outputs
----------------------

These outputs are calculated by the post_processing module, for both
analysis and inspection.

``pressure_thermal_ion_face`` (time, rho_face) [Pa]:
  Ion thermal pressure on the face grid.

``pressure_thermal_el_face`` (time, rho_face) [Pa]:
  Electron thermal pressure on the face grid.

``pressure_thermal_tot_face`` (time, rho_face) [Pa]:
  Total thermal pressure on the face grid.

``te_volume_avg`` (time) [keV]:
  Volume average electron temperature.

``ti_volume_avg`` (time) [keV]:
  Volume average ion temperature.

``ne_volume_avg`` (time) [nref m^-3]:
  Volume average electron density.

``ni_volume_avg`` (time) [nref m^-3]:
  Volume average ion density.

``fgw_ne_volume_avg`` (time) [dimensionless]:
  Greenwald fraction from volume-averaged electron density.

``pprime_face`` (time, rho_face) [Pa/Wb]:
  Derivative of total pressure with respect to poloidal flux on the face grid.

``W_thermal_ion`` (time) [J]:
  Ion thermal stored energy.

``W_thermal_el`` (time) [J]:
  Electron thermal stored energy.

``W_thermal_tot`` (time) [J]:
  Total thermal stored energy.

``Wpol`` (time) [J]
  Total magnetic energy

``q95`` (time) [dimensionless]
  Safety-factor at 95% of the normalized poloidal flux coordinate.

``li3`` (time) [dimensionless]:
  Normalized plasma internal inductance, ITER convention

``tauE`` (time) [s]:
  Thermal confinement time defined as ``W_thermal_tot`` / ``P_heating``, where
  ``P_heating`` is the total heating power into the plasma, including external
  contributions and fusion heating. Radiative losses are not subtracted from
  heating power.

``H98`` (time) [dimensionless]:
  H-mode confinement quality factor with respect to the ITER98y2 scaling law,
  defined as ``tauE`` / ``tau98_scaling``, where ``tau98_scaling`` is the
  confinement time defined by the ITER98y2 scaling law, derived from the ITER
  H-mode confinement database. As for ``tauE``, radiative losses are not
  subtracted from the ``P_loss`` term used to calculate the empirical scaling
  law confinement time.

``H97L`` (time) [dimensionless]:
  L-mode confinement quality factor with respect to the ITER97L scaling law
  derived from the ITER L-mode confinement database. Defined similarly to ``H98``
  above, but using the ITER97L scaling law for the confinement time.

``H20`` (time) [dimensionless]:
  H-mode confinement quality factor with respect to the ITER20 scaling law
  derived from the updated (2020) ITER confinement database. Defined similarly
  to ``H98`` above, but using the updated ITER20 scaling law law for the
  confinement time.

``FFprime_face`` (time, rho_face) [m^2 T^2 / Wb]:
  :math:`FF'` on the face grid, where F is the toroidal flux function, and
  F' is its derivative with respect to the poloidal flux.

``psi_norm_face`` (time, rho_face) [dimensionless]:
  Normalized poloidal flux on the face grid.

``P_sol_ion`` (time) [W]:
  Total ion heating power exiting the plasma with all sources:
  auxiliary heating + ion-electron exchange + fusion.

``P_sol_el`` (time) [W]:
  P_sol_el: Total electron heating power exiting the plasma with all sources
  and sinks: auxiliary heating + ion-electron exchange + Ohmic + fusion +
  radiation sinks.

``P_sol_tot`` (time) [W]:
  Total heating power exiting the plasma with all sources and sinks.

``P_external_ion`` (time) [W]:
  Total external ion heating power: auxiliary heating + Ohmic.

``P_external_el`` (time) [W]:
  Total external electron heating power: auxiliary heating + Ohmic.

``P_external_tot`` (time) [W]:
  Total external heating power: auxiliary heating + Ohmic.

``P_ei_exchange_ion`` (time) [W]:
  Electron-ion heat exchange power to ions.

``P_ei_exchange_el`` (time) [W]:
  Electron-ion heat exchange power to electrons.

``P_generic_ion`` (time) [W]:
  Total `generic_ion_el_heat_source` power to ions.

``P_generic_el`` (time) [W]:
  Total `generic_ion_el_heat_source` power to electrons.

``P_generic_tot`` (time) [W]:
  Total `generic_ion_el_heat_source` power.

``P_alpha_ion`` (time) [W]:
  Total fusion power to ions.

``P_alpha_el`` (time) [W]:
  Total fusion power to electrons.

``P_alpha_tot`` (time) [W]:
  Total fusion power to plasma.

``P_ohmic`` (time) [W]:
  Ohmic heating power to electrons.

``P_brems`` (time) [W]:
  Bremsstrahlung electron heat sink.

``P_ecrh`` (time) [W]:
  Total electron cyclotron source power.

``I_ecrh`` (time) [A]:
  Total electron cyclotron source current.

``I_generic`` (time) [A]:
  Total generic source current.

``Q_fusion`` (time):
  Fusion power gain.

``P_icrh_el`` (time) [W]:
  Ion cyclotron resonance heating to electrons.

``P_icrh_ion`` (time) [W]:
  Ion cyclotron resonance heating to ions.

``P_icrh_tot`` (time) [W]:
  Total ion cyclotron resonance heating power.

``P_LH_hi_dens`` (time) [W]: H-mode transition power for high density branch,
  according to Eq 3 from Martin 2008.

``P_LH_min`` (time) [W]: Minimum H-mode transition power at the minimum density
  ``ne_min_P_LH``, according to Eq 4 from Ryter 2014.

``P_LH`` (time) [W]: H-mode transition power taken as the maximum of
  ``P_LH_min`` and ``P_LH_hi_dens``. ``P_LH_min`` and ``P_LH_hi_dens`` are kept
  in output for increased introspectability.

``ne_min_P_LH`` (time) [nref]:  Density corresponding to the minimum P_LH,
  according to Eq 3 from Ryter 2014.

``E_cumulative_fusion`` (time) [J]:
  Total cumulative fusion energy.

``E_cumulative_external`` (time) [J]:
  Total external injected energy (Ohmic + auxiliary heating).

geometry
--------

The geometry dataset contains the following data variables.

Geometry
--------

``Phi`` (time, rho_cell) [Wb]
  Toroidal magnetic flux at each radial grid point.

``Phi_face`` (time, rho_face) [Wb]
  Toroidal magnetic flux at each radial face.

``Rmaj`` (time) [m]
  Major radius.

``Rmin`` (time) [m]
  Minor radius.

``B0`` (time) [T]
  Magnetic field strength at the magnetic axis.

``volume`` (time, rho_cell) [:math:`m^3`]
  Plasma volume enclosed by each flux surface.

``volume_face`` (time, rho_face) [:math:`m^3`]
  Plasma volume enclosed by each flux surface at the faces.

``area`` (time, rho_cell) [:math:`m^2`]
  Poloidal cross-sectional area of each flux surface.

``area_face`` (time, rho_face) [:math:`m^2`]
  Poloidal cross-sectional area of each flux surface at the faces.

``vpr`` (time, rho_cell) [:math:`m^3`]
  Derivative of plasma volume enclosed by each flux surface with respect to the normalized toroidal flux coordinate rho_norm.

``vpr_face`` (time, rho_face) [:math:`m^3`]
  Derivative of plasma volume enclosed by each flux surface at the faces, with respect to the normalized toroidal flux coordinate rho_face_norm.

``spr`` (time, rho_cell) [:math:`m^2`]
  Derivative of plasma surface area enclosed by each flux surface, with respect to the normalized toroidal flux coordinate rho_norm.

``spr_face`` (time, rho_face) [:math:`m^2`]
  Derivative of plasma surface area enclosed by each flux surface at the faces, with respect to the normalized toroidal flux coordinate rho_face_norm.

``delta_face`` (time, rho_face) [dimensionless]
  Average triangularity of each flux surface at the faces.

``elongation``(time, rho_cell) [dimensionless]
  Elongation of each flux surface.

``elongation_face`` (time, rho_face) [dimensionless]
  Elongation of each flux surface at the faces.

``g0`` (time, rho_cell) [:math:`m^2`]
  Flux surface averaged :math:`\nabla V`, the radial derivative of the plasma volume.

``g0_face`` (time, rho_face) [:math:`m^2`]
  Flux surface averaged :math:`\nabla V` on the faces.

``g1`` (time, rho_cell) [:math:`m^4`]
  Flux surface averaged :math:`(\nabla V)^2`.

``g1_face`` (time, rho_face) [:math:`m^4`]
  Flux surface averaged :math:`(\nabla V)^2` at the faces.

``g2`` (time, rho_cell) [:math:`m^2`]
  Flux surface averaged :math:`\frac{(\nabla V)^2}{R^2}`, where R is the major radius along the flux surface being averaged.

``g2_face`` (time, rho_face) [:math:`m^2`]
  Flux surface averaged :math:`\frac{(\nabla V)^2}{R^2}` at the faces.

``g3`` (time, rho_cell) [:math:`m^{-2}`]
  Flux surface averaged :math:`\frac{1}{R^2}`.

``g3_face`` (time, rho_face) [:math:`m^{-2}`]
  Flux surface averaged :math:`\frac{1}{R^2}` at the faces.

``g2g3_over_rhon`` (time, rho_cell) [dimensionless]
  Ratio of g2g3 to the normalized toroidal flux coordinate rho_norm.

``g2g3_over_rhon_face`` (time, rho_face) [dimensionless]
  Ratio of g2g3 to the normalized toroidal flux coordinate rho_norm on the face grid.

``F`` (time, rho_cell) [:math:`T m`]
  Flux function :math:`F=B_{tor}R` , constant on any given flux surface.

``F_face`` (time, rho_face) [:math:`T m`]
  Flux function :math:`F=B_{tor}R`  on the face grid.

``Rin`` (time, rho_cell) [m]
  Inner radius of each flux surface.

``Rin_face`` (time, rho_face) [m]
  Inner radius of each flux surface at the faces.

``Rout``(time, rho_cell) [m]
  Outer radius of each flux surface.

``Rout_face`` (time, rho_face) [m]
  Outer radius of each flux surface at the faces.

``Phibdot`` (time) [Wb/s]
  Time derivative of the toroidal magnetic flux.

``_z_magnetic_axis`` (time) [m]
  Vertical position of the magnetic axis.

Examples
========

To demonstrate xarray and numpy manipulations of output data, the following code carries out
volume integration of ``fusion_heat_source_el`` and ``fusion_heat_source_ion`` at the time closest to t=1. The result equals
the input config ``sources['fusion_heat_source']['Ptot']`` at the time closest to t=1.

``dt`` is the xarray.DataTree. The netCDF file is assumed to be in the working directory. ``vpr``
is assumed to not be time varying.

.. code-block:: python

  import numpy as np
  from torax import output

  data_tree = output.safe_load_state_file('state_history.nc').sel(time=1.0, method='nearest')
  fusion_heat_source_el = data_tree.children['core_sources'].dataset['fusion_heat_source_el']
  fusion_heat_source_ion = data_tree.children['core_sources'].dataset['fusion_heat_source_ion']

  Ptot = np.trapz((fusion_heat_source_el + fusion_heat_source_ion) * data_tree.vpr, data_tree.rho_cell_norm)
