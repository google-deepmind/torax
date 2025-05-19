.. _output:

Simulation output structure
###########################

TORAX file output is written to a ``state_history.nc`` netCDF file. If running with
the ``run_simulation_main.py`` or ``run_torax`` script, the ``output_dir``
is set via flag ``--output_dir``, with default
``/tmp/torax_results_<YYYYMMDD_HHMMSS>/``.

We currently support the below output structure for Torax V1.

The output is an `[xarray DataTree] <https://docs.xarray.dev/en/latest/generated/xarray.DataTree.html>`_.

The DataTree is a hierarchical structure containing three ``xarray.DataSet``s:

* ``numerics`` containing simulation numeric quantities.
* ``scalars`` containing scalar quantities.
* ``profiles`` containing 1D profile quantities.

Where sensible we have aimed to match names used by the IMAS standard.

Also, note that, TORAX does not have a specific COCOS it
adheres to, yet. Our team is working on aligning to a specific standard COCOS
on both the input and output. (CHEASE, which is COCOS 2 is still the supported
way to provide geometric inputs to TORAX, and we will continue to support CHEASE
as an input method, regardless of which COCOS we choose.)

Dimensions
==========

The DataTree/Dataset variables can have the following dimensions:

* (time)
* (time, space)

There are three named variants of spatial dimension:

* **rho_cell_norm**: corresponding to the ``torax.fvm`` cell grid.
* **rho_face_norm**: corresponding to the ``torax.fvm`` face grid.
* **rho_norm**: corresponding to the ``torax.fvm`` cell grid plus boundary values.

See :ref:`fvm` for more details on the grids.

In all subsequent lists, the dimensions associated with each variable or coordinate
will be surrounded by parentheses, e.g. (time, rho_norm).

Coordinates
===========

All the ``Dataset`` objects in the output contains the following Coordinates. In order
for users of TORAX outputs to not have to worry about TORAX internals such as
interpolation routines or the grid on which a value is computed we provide
outputs on three different grids depending on the available data for an output.
Some TORAX outputs are only computed on the face grid (such as transport
coefficients), some only on the cell (such as source profiles) and some are
computed on both (like the core profiles evolved by the PDE).
In cases where both are computed we merge the computed quantities into a single
output on the cell grid as well as the left and right face values. These
different grids exist due to the finite-volume method.

* ``time`` (time)
    Times corresponding to each simulation timestep, in units of [:math:`s`].

* ``rho_norm`` (rho_cell + boundary values)
   Normalized toroidal flux coordinate (see :ref:`glossary`) on the fvm cell grid.
   The array size is set in the input config by ``geometry['nrho']+2``.

* ``rho_cell_norm`` (rho_cell)
    Normalized toroidal flux coordinate (see :ref:`glossary`) on the fvm cell grid.
    The array size is set in the input config by ``geometry['nrho']``.

* ``rho_face_norm`` (rho_face)
    Normalized toroidal flux coordinate (see :ref:`glossary`) on the fvm face grid.
    The array size is ``geometry['nrho']+1``.

The coordinate for a given data variable is then attached to the given variable.

Top level dataset
=================
The top level dataset contains the config used to run the simulation. This can
be used to rerun a simulation with the same input. Otherwise the top level
only has references to the child datasets.

``attrs`` ()
  The ``attrs`` field of a Dataset is used to store metadata about the Dataset
  as a dictionary. We use this field to store the input config as a json string
  under the ``config`` key.

To retrieve the input config, see :ref:`output_examples` below.

Child datatrees
===============
The following datatrees are child nodes, the title of each section is the name of
the child ``DataTree``.

numerics
--------
The ``numerics`` dataset contains the following data variables.

``sim_error`` ()
  Indicator if the simulation completed successfully, 0 if successful, 1 if not.

``sawtooth_crash`` (time)
  Boolean array with a length equal to the number of simulation timesteps,
  indicating whether the state at that timestep corresponds to a
  post-sawtooth-crash state.

``outer_solver_iterations`` (time)
  Number of outer solver iterations. This will either be 1 or in the case of
  any adaptive steps being taken, 1+`num_adaptive_steps`

``inner_solver_iterations`` (time)
  Number of inner solver iterations.

profiles
--------

This dataset contains radial profiles of various plasma parameters at different times. The radial coordinate is the normalized toroidal flux coordinate.
Note that the output structure is dependent on the input config for the ``geometry``, ``transport`` and ``sources`` fields.

For ``sources`` certain profiles are only output if the source is active.

For ``geometry`` certain profiles are only output if ``circular`` geometry is not used.

For ``transport`` certain profiles are only output if the ``bohm-gyrobohm`` model is used.

``T_e`` (time, rho_norm)
  Electron temperature profile [:math:`keV`].

``T_i`` (time, rho_norm)
  Ion temperature profile [:math:`keV`].

``psi`` (time, rho_norm)
  Poloidal flux profile :math:`\psi` [:math:`Wb`].

``v_loop`` (time, rho_norm)
  Loop voltage profile :math:`V_{loop}=\frac{\partial\psi}{\partial t}` [:math:`V`].

``n_e`` (time, rho_norm)
  Electron density profile [:math:`m^{-3}`].

``n_i`` (time, rho_norm)
  Main ion density profile [:math:`m^{-3}`].

``n_impurity`` (time, rho_norm)
  Impurity density profile [:math:`m^{-3}`].

``Z_impurity`` (time, rho_norm)
  Effective charge profile of the impurity species [dimensionless].

``j_total`` (time, rho_norm)
  Total current density profile [:math:`A/m^2`].

``Ip_profile`` (time, rho_face_norm)
  Total current profile on the face grid [:math:`A`].

``q`` (time, rho_face_norm)
  Safety factor profile on the face grid [dimensionless].

``magnetic_shear`` (time, rho_face_norm)
  Magnetic shear profile on the face grid [dimensionless].

``chi_turb_i`` (time, rho_face_norm)
  Turbulent ion heat conductivity profile [:math:`m^2/s`].

``chi_turb_e`` (time, rho_face_norm)
  Turbulent electron heat conductivity profile [:math:`m^2/s`].

``D_turb_e`` (time, rho_face_norm)
  Turbulent electron particle diffusivity profile on the face grid [:math:`m^2/s`].

``V_turb_e`` (time, rho_face_norm)
  Turbulent electron particle convection profile on the face grid [:math:`m/s`].

``chi_bohm_e`` (time, rho_face_norm) [:math:`m^2/s`]
  Bohm electron heat conductivity profile on the face grid. Only output if active.

``chi_gyrobohm_e`` (time, rho_face_norm) [:math:`m^2/s`]
  Gyro-Bohm electron heat conductivity profile on the face grid. Only output if active.

``chi_bohm_i`` (time, rho_face_norm) [:math:`m^2/s`]
  Bohm ion heat conductivity profile on the face grid. Only output if active.

``chi_gyrobohm_i`` (time, rho_face_norm) [:math:`m^2/s`]
  Gyro-Bohm ion heat conductivity profile on the face grid. Only output if active.

``ei_exchange`` (time, rho_cell_norm)
  Ion-electron heat exchange density profile on the cell grid [:math:`W/m^3`]. Positive values mean heat source for ions, and heat sink for electrons.

``j_bootstrap`` (time, rho_norm)
  Bootstrap current density profile [:math:`A/m^2`].

``sigma_parallel`` (time, rho_cell_norm)
  Plasma conductivity parallel to the magnetic field profile on the cell grid [:math:`S/m`].

``p_cyclotron_radiation_e`` (time, rho_cell_norm) [:math:`W/m^3`]
  Cyclotron radiation heat sink density profile on the cell grid. Only output if ``cyclotron_radiation`` source is active.

``p_ecrh_e`` (time, rho_cell_norm)
  Electron cyclotron heating power density profile on the cell grid [:math:`W/m^3`]. Only output if ``ecrh`` source is active.

``j_ecrh`` (time, rho_cell_norm)
  Electron cyclotron heating current density profile on the cell grid [:math:`A/m^2`]. Only output if ``ecrh`` source is active.

``p_icrh_i`` (time, rho_cell_norm)
  Ion cyclotron heating power density ion heating profile on the cell grid [:math:`W/m^3`]. Only output if ``icrh`` source is active.

``p_icrh_e`` (time, rho_cell_norm)
  Ion cyclotron heating power density electron heating profile on the cell grid [:math:`W/m^3`]. Only output if ``icrh`` source is active.

``p_alpha_i`` (time, rho_cell_norm)
  Fusion alpha heating power density profile to ions on the cell grid [:math:`W/m^3`]. Only output if ``fusion`` source is active.

``p_impurity_radiation_e`` (time, rho_cell_norm)
  Impurity radiation heat sink density profile on the cell grid [:math:`W/m^3`]. Only output if ``impurity_radiation`` source is active.

``p_ohmic_e`` (time, rho_cell_norm)
  Ohmic heat sink density profile on the cell grid [:math:`W/m^3`]. Only output if ``ohmic`` source is active.

``p_generic_heat_i`` (time, rho_cell_norm)
  Generic external ion heat source density profile on the cell grid [:math:`W/m^3`]. Only output if ``generic_heat`` source is active.

``p_alpha_e`` (time, rho_cell_norm)
  Fusion alpha heating power density profile to electrons on the cell grid [:math:`W/m^3`]. Only output if ``fusion`` source is active.

``p_generic_heat_e`` (time, rho_cell_norm)
  Generic external electron heat source density profile on the cell grid [:math:`W/m^3`]. Only output if ``generic_heat`` source is active.

``j_generic_current`` (time, rho_cell_norm)
  Generic external non-inductive current density profile on the cell grid [:math:`A/m^2`]. Only output if ``generic_current`` source is active.

``s_gas_puff`` (time, rho_cell_norm)
  Gas puff particle source density profile on the cell grid [:math:`s^{-1} m^{-3}`]. Only output if ``gas_puff`` source is active.

``s_generic_particle`` (time, rho_cell_norm)
  Generic particle source density profile on the cell grid [:math:`s^{-1} m^{-3}`]. Only output if ``generic_particle`` source is active.

``s_pellet`` (time, rho_cell_norm)
  Pellet particle source density profile on the cell grid [:math:`s^{-1} m^{-3}`]. Only output if ``pellet`` source is active.

``pressure_thermal_i`` (time, rho_face_norm)
  Ion thermal pressure profile [:math:`Pa`].

``pressure_thermal_e`` (time, rho_face_norm)
  Electron thermal pressure profile [:math:`Pa`].

``pressure_thermal_total`` (time, rho_face_norm)
  Total thermal pressure profile [:math:`Pa`].

``pprime`` (time, rho_face_norm)
  Derivative of total pressure with respect to poloidal flux [:math:`Pa/Wb`].

``FFprime`` (time, rho_face_norm)
  :math:`FF'` profile on the face grid [:math:`m^2 T^2 / Wb`].

``psi_norm`` (time, rho_face_norm)
  Normalized poloidal flux profile [dimensionless].

``j_external`` (time, rho_cell_norm)
  Total external current density profile (including generic and ECRH current) [:math:`A/m^2`].

``j_ohmic`` (time, rho_cell_norm)
  Ohmic current density profile [:math:`A/m^2`].

``Phi`` (time, rho_norm)
  Toroidal magnetic flux at each radial grid point [:math:`Wb`].

``volume`` (time, rho_norm)
  Plasma volume enclosed by each flux surface [:math:`m^3`].

``area`` (time, rho_norm)
  Poloidal cross-sectional area of each flux surface [:math:`m^2`].

``vpr`` (time, rho_norm)
  Derivative of plasma volume enclosed by each flux surface with respect to the normalized toroidal flux coordinate rho_norm [:math:`m^3`].

``spr`` (time, rho_norm)
  Derivative of plasma surface area enclosed by each flux surface, with respect to the normalized toroidal flux coordinate rho_norm [:math:`m^2`].

``elongation`` (time, rho_norm)
  Elongation of each flux surface [dimensionless].

``g0`` (time, rho_norm)
  Flux surface averaged :math:`\nabla V`, the radial derivative of the plasma volume [:math:`m^2`].

``g1`` (time, rho_norm)
  Flux surface averaged :math:`(\nabla V)^2` [:math:`m^4`].

``g2`` (time, rho_norm)
  Flux surface averaged :math:`\frac{(\nabla V)^2}{R^2}`, where R is the major radius along the flux surface being averaged [:math:`m^2`].

``g3`` (time, rho_norm)
  Flux surface averaged :math:`\frac{1}{R^2}` [:math:`m^{-2}`].

``g2g3_over_rhon`` (time, rho_norm)
  Ratio of g2g3 to the normalized toroidal flux coordinate rho_norm [dimensionless].

``F`` (time, rho_norm)
  Flux function :math:`F=B_{tor}R` , constant on any given flux surface [:math:`T m`].

``R_in`` (time, rho_norm)
  Inner (minimum) radius of each flux surface [:math:`m`].

``R_out`` (time, rho_norm)
  Outer (maximum) radius of each flux surface [:math:`m`].

``psi_from_geo`` (time, rho_cell_norm)
  Poloidal flux calculated from geometry (NOT psi calculated self-consistently by the TORAX PDE) on the cell grid [:math:`Wb`].

``psi_from_Ip`` (time, rho_norm)
  Poloidal flux calculated from the current profile in the geometry file (NOT psi calculated self-consistently by the TORAX PDE) [:math:`Wb`].

``g0_over_vpr_face`` (time, rho_face_norm)
  Ratio of g0 to vpr on the face grid [dimensionless].

``g1_over_vpr`` (time, rho_cell_norm)
  Ratio of g1 to vpr on the cell grid [dimensionless].

``g1_over_vpr2`` (time, rho_cell_norm)
  Ratio of g1 to vpr squared on the cell grid [dimensionless].

``g1_over_vpr2_face`` (time, rho_face_norm)
  Ratio of g1 to vpr squared on the face grid [dimensionless].

``g1_over_vpr_face`` (time, rho_face_norm)
  Ratio of g1 to vpr on the face grid [dimensionless].

``r_mid`` (time, rho_cell_norm)
  Mid-plane radius of each flux surface on the cell grid [:math:`m`].

``r_mid_face`` (time, rho_face_norm)
  Mid-plane radius of each flux surface on the face grid [:math:`m`].


scalars
-------

This dataset contains time-dependent scalar quantities describing global plasma properties and characteristics.

``Ip`` (time)
  Plasma current [:math:`A`].

``n_ref`` (time)
  Reference density used for normalization [:math:`m^{-3}`].

``vloop_lcfs`` (time)
  Loop voltage at the last closed flux surface (LCFS) [:math:`Wb/s` or :math:`V`]. This is a scalar value derived from the `v_loop` profile.

``W_thermal_i`` (time)
  Total ion thermal stored energy [:math:`J`].

``W_thermal_e`` (time)
  Total electron thermal stored energy [:math:`J`].

``W_thermal_total`` (time)
  Total thermal stored energy [:math:`J`].

``tau_E`` (time)
  Thermal confinement time [:math:`s`].

``H89P`` (time)
  H-mode confinement quality factor with respect to the ITER89-P scaling law [dimensionless].

``H98`` (time)
  H-mode confinement quality factor with respect to the ITER98y2 scaling law [dimensionless].

``H97L`` (time)
  L-mode confinement quality factor with respect to the ITER97L scaling law [dimensionless].

``H20`` (time)
  H-mode confinement quality factor with respect to the ITER20 scaling law [dimensionless].

``P_SOL_i`` (time)
  Total ion heating power exiting the plasma across the LCFS [:math:`W`].

``P_SOL_e`` (time)
  Total electron heating power exiting the plasma across the LCFS [:math:`W`].

``P_SOL_total`` (time)
  Total heating power exiting the plasma across the LCFS [:math:`W`].

``P_aux_i`` (time)
  Total auxiliary ion heating power [:math:`W`].

``P_aux_e`` (time)
  Total auxiliary electron heating power [:math:`W`].

``P_aux_total`` (time)
  Total auxiliary heating power [:math:`W`] (sum of ion and electron auxiliary heating).

``P_external_injected`` (time)
  Total externally injected power into the plasma [:math:`W`]. This is likely equivalent to `P_external_tot`.

``P_ei_exchange_i`` (time)
  Total electron-ion heat exchange power to ions [:math:`W`].

``P_ei_exchange_e`` (time)
  Total electron-ion heat exchange power to electrons [:math:`W`].

``P_aux_generic_i`` (time)
  Total generic auxiliary heating power to ions [:math:`W`].

``P_aux_generic_e`` (time)
  Total generic auxiliary heating power to electrons [:math:`W`].

``P_aux_generic_total`` (time)
  Total generic auxiliary heating power [:math:`W`].

``P_alpha_i`` (time)
  Total fusion alpha heating power to ions [:math:`W`].

``P_alpha_e`` (time)
  Total fusion alpha heating power to electrons [:math:`W`].

``P_alpha_total`` (time)
  Total fusion alpha heating power [:math:`W`].

``P_ohmic_e`` (time)
  Total Ohmic heating power to electrons [:math:`W`].

``P_bremsstrahlung_e`` (time)
  Total Bremsstrahlung electron heat sink power [:math:`W`].

``P_cyclotron_e`` (time)
  Total cyclotron radiation heat sink power [:math:`W`].

``P_ecrh_e`` (time)
  Total electron cyclotron source power to electrons [:math:`W`].

``P_radiation_e`` (time)
  Total radiative heat sink power (including Bremsstrahlung, Cyclotron, and other radiation) to electrons [:math:`W`].

``I_ecrh`` (time)
  Total electron cyclotron source current [:math:`A`].

``I_aux_generic`` (time)
  Total generic auxiliary current [:math:`A`].

``Q_fusion`` (time)
  Fusion power gain [dimensionless].

``P_icrh_e`` (time)
  Total ion cyclotron resonance heating power to electrons [:math:`W`].

``P_icrh_i`` (time)
  Total ion cyclotron resonance heating power to ions [:math:`W`].

``P_icrh_total`` (time)
  Total ion cyclotron resonance heating power [:math:`W`].

``P_LH_high_density`` (time)
  H-mode transition power for the high density branch, according to the Martin 2008 scaling law [:math:`W`].

``P_LH_min`` (time)
  H-mode transition power at the density corresponding to the minimum transition power, from Ryter 2014. [:math:`W`].

``P_LH`` (time)
  Calculated H-mode transition power, taken as the maximum of ``P_LH_min`` and ``P_LH_high_density``. This does not include an accurate calculation for the low density branch. [:math:`W`].

``n_e_min_P_LH`` (time)
  Electron density at which the minimum H-mode transition power occurs [:math:`m^{-3}`].

``E_fusion`` (time)
  Total cumulative fusion energy produced [:math:`J`].

``E_aux`` (time)
  Total cumulative auxiliary injected energy (Ohmic + auxiliary heating) [:math:`J`].

``T_e_volume_avg`` (time)
  Volume-averaged electron temperature [:math:`keV`].

``T_i_volume_avg`` (time)
  Volume-averaged ion temperature [:math:`keV`].

``n_e_volume_avg`` (time)
  Volume-averaged electron density [dimensionless].

``n_i_volume_avg`` (time)
  Volume-averaged main ion density [dimensionless].

``n_e_line_avg`` (time)
  Line-averaged electron density [dimensionless].

``n_i_line_avg`` (time)
  Line-averaged main ion density [dimensionless].

``fgw_n_e_volume_avg`` (time)
  Greenwald fraction from volume-averaged electron density [dimensionless].

``fgw_n_e_line_avg`` (time)
  Greenwald fraction from line-averaged electron density [dimensionless].

``q95`` (time)
  Safety factor at 95% of the normalized poloidal flux coordinate [dimensionless].

``W_pol`` (time)
  Total poloidal magnetic energy [:math:`J`].

``li3`` (time)
  Normalized plasma internal inductance (ITER convention) [dimensionless].

``dW_thermal_dt`` (time)
  Time derivative of the total thermal stored energy [:math:`W`].

``rho_q_min`` (time)
  Normalized toroidal flux coordinate at which the minimum safety factor occurs [dimensionless].

``q_min`` (time)
  Minimum safety factor [dimensionless].

``rho_q_3_2_first`` (time)
  Normalized toroidal flux coordinate of the first surface where q = 3/2 [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_3_2_second`` (time)
  Normalized toroidal flux coordinate of the second surface where q = 3/2 [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_2_1_first`` (time)
  Normalized toroidal flux coordinate of the first surface where q = 2 [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_2_1_second`` (time)
  Normalized toroidal flux coordinate of the second surface where q = 2 [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_3_1_first`` (time)
  Normalized toroidal flux coordinate of the first surface where q = 3 [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_3_1_second`` (time)
  Normalized toroidal flux coordinate of the second surface where q = 3 [dimensionless]. Values of -inf indicate no such surface exists.

``I_bootstrap`` (time)
  Total bootstrap current [:math:`A`].

``R_major`` (time)
  Major radius [:math:`m`].

``a_minor`` (time)
  Minor radius [:math:`m`].

``B_0`` (time)
  Magnetic field strength at the magnetic axis [:math:`T`].

``Phi_b_dot`` (time)
  Time derivative of the total toroidal magnetic flux [:math:`Wb/s`].

``Phi_b`` (time)
  Total toroidal magnetic flux [:math:`Wb`].

``drho`` (time)
  Radial grid spacing in the unnormalized rho coordinate [:math:`m`].

``drho_norm`` ()
  Radial grid spacing in the normalized rho coordinate [dimensionless]. This is a fixed scalar value.

``rho_b`` (time)
  Value of the unnormalized rho coordinate at the boundary [:math:`m`].

.. _output_examples:

Working with output data
========================

To demonstrate xarray and numpy manipulations of output data, the following code carries out
volume integration of ``alpha_e`` and ``alpha_i`` at the time closest to t=1. The result equals
the input config ``sources['fusion']['P_total']`` at the time closest to t=1.

The netCDF file is assumed to be in the working directory.

.. code-block:: python

  import numpy as np
  from torax import output

  data_tree = output.load_state_file('state_history.nc').sel(time=1.0, method='nearest')
  alpha_electron = data_tree.profiles.alpha_e
  alpha_ion = data_tree.profiles.alpha_i
  vpr = data_tree.profiles.vpr.sel(rho_norm=data_tree.rho_cell_norm)

  P_total = np.trapz((alpha_el + alpha_ion) * vpr, data_tree.rho_cell_norm)


It is possible to retrieve the input config from the output for debugging
purposes or to rerun the simulation.

.. code-block:: python

  import json
  import torax
  from torax import output

  data_tree = output.load_state_file('state_history.nc')
  config_dict = json.loads(data_tree.attrs['config'])
  # Check which transport model was used.
  print(config_dict['transport']['transport_model'])
  # We can also use ToraxConfig to run the simulation again.
  torax_config = torax.ToraxConfig.from_dict(config_dict)
  new_output = torax.run_simulation(torax_config)

