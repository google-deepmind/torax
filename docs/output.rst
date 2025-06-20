.. _output:

Simulation output structure
###########################

TORAX file output is written to a ``state_history.nc`` netCDF file. If running
with the ``run_simulation_main.py`` or ``run_torax`` script, the ``output_dir``
is set via flag ``--output_dir``, with default
``/tmp/torax_results_<YYYYMMDD_HHMMSS>/``.

We currently support the below output structure for Torax V1.

The output is an `[xarray DataTree] <https://docs.xarray.dev/en/latest/generated/xarray.DataTree.html>`_.

The DataTree is a hierarchical structure containing three ``xarray.DataSet``
objects:

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
* **rho_norm**: corresponding to the ``torax.fvm`` cell grid plus boundary
  values.

See :ref:`fvm` for more details on the grids.

In all subsequent lists, the dimensions associated with each variable or
coordinate will be surrounded by parentheses, e.g. (time, rho_norm).

Coordinates
===========

All the ``Dataset`` objects in the output contains the following Coordinates.
In order for users of TORAX outputs to not have to worry about TORAX internals
such as interpolation routines or the grid on which a value is computed we
provide outputs on three different grids depending on the available data for an
output. Some TORAX outputs are only computed on the face grid (such as transport
coefficients), some only on the cell (such as source profiles) and some are
computed on both (like the core profiles evolved by the PDE).
In cases where both are computed we merge the computed quantities into a single
output on the cell grid as well as the left and right face values. These
different grids exist due to the finite-volume method.

* ``time`` (time)
    Times corresponding to each simulation timestep, in units of [:math:`s`].

* ``rho_norm`` (rho_cell + boundary values)
   Normalized toroidal flux coordinate (see :ref:`glossary`) on the fvm cell
   grid. The array size is set in the input config by ``geometry['nrho']+2``.

* ``rho_cell_norm`` (rho_cell)
    Normalized toroidal flux coordinate (see :ref:`glossary`) on the fvm cell
    grid. The array size is set in the input config by ``geometry['nrho']``.

* ``rho_face_norm`` (rho_face)
    Normalized toroidal flux coordinate (see :ref:`glossary`) on the fvm face
    grid. The array size is ``geometry['nrho']+1``.

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
The following datatrees are child nodes, the title of each section is the name
of the child ``DataTree``.

numerics
--------
The ``numerics`` dataset contains the following data variables.

``inner_solver_iterations`` (time)
  Number of inner solver iterations.

``outer_solver_iterations`` (time)
  Number of outer solver iterations. This will either be 1 or in the case of
  any adaptive steps being taken, 1+`num_adaptive_steps`

``sawtooth_crash`` (time)
  Boolean array with a length equal to the number of simulation timesteps,
  indicating whether the state at that timestep corresponds to a
  post-sawtooth-crash state.

``sim_error`` ()
  Indicator if the simulation completed successfully, 0 if successful, 1 if not.

profiles
--------

This dataset contains radial profiles of various plasma parameters at different
times. The radial coordinate is the normalized toroidal flux coordinate. Note
that the output structure is dependent on the input config for the
``geometry``, ``transport`` and ``sources`` fields.

Note that certain profiles are only output for specific input configurations.
These are called out in the list of profiles below, and generate relate to:

* ``sources`` profiles which are only output if the source is active.

* ``transport`` profiles which are only output if the ``bohm-gyrobohm`` model is
  used.

* ``geometry`` profiles which are only output for non ``circular`` geometries.

``area`` (time, rho_norm)
  Poloidal cross-sectional area of each flux surface [:math:`m^2`].

``chi_bohm_e`` (time, rho_face_norm) [:math:`m^2/s`]
  Bohm component of electron heat turbulent conductivity. Only output if active.

``chi_bohm_i`` (time, rho_face_norm) [:math:`m^2/s`]
  Bohm component of ion heat turbulent conductivity. Only output if active.

``chi_gyrobohm_e`` (time, rho_face_norm) [:math:`m^2/s`]
  Gyro-Bohm component of electron heat turbulent conductivity. Only output if
  active.

``chi_gyrobohm_i`` (time, rho_face_norm) [:math:`m^2/s`]
  Gyro-Bohm component of ion heat turbulent conductivity. Only output if active.

``chi_neo_e`` (time, rho_face_norm)
  Neoclassical electron heat conductivity [:math:`m^2/s`].

``chi_neo_i`` (time, rho_face_norm)
  Neoclassical ion heat conductivity [:math:`m^2/s`].

``chi_turb_e`` (time, rho_face_norm)
  Total turbulent electron heat conductivity [:math:`m^2/s`].

``chi_turb_i`` (time, rho_face_norm)
  Total turbulent ion heat conductivity [:math:`m^2/s`].

``D_neo_e`` (time, rho_face_norm)
  Neoclassical electron particle diffusivity [:math:`m^2/s`].

``D_turb_e`` (time, rho_face_norm)
  Total turbulent electron particle diffusivity [:math:`m^2/s`].

``ei_exchange`` (time, rho_cell_norm)
  Ion-electron heat exchange power density profile [:math:`W/m^3`]. Positive
  values mean heat source for ions, and heat sink for electrons.

``elongation`` (time, rho_norm)
  Elongation of each flux surface [dimensionless].

``epsilon`` (time, rho_norm)
  Local inverse aspect ratio at each flux surface [dimensionless].

``F`` (time, rho_norm)
  Flux function :math:`F=B_{tor}R` , constant on any given flux surface
  [:math:`T m`].

``FFprime`` (time, rho_face_norm)
  :math:`FF'`, where :math:`F'` is the derivative of the flux function with
  respect to poloidal flux [:math:`m^2 T^2 / Wb`].

``g0`` (time, rho_norm)
  Flux surface averaged :math:`\nabla V`, the radial derivative of the plasma
  volume [:math:`m^2`].

``g0_over_vpr`` (time, rho_face_norm)
  Ratio of g0 to vpr [dimensionless].

``g1`` (time, rho_norm)
  Flux surface averaged :math:`(\nabla V)^2` [:math:`m^4`].

``g1_over_vpr`` (time, rho_norm)
  Ratio of g1 to vpr [dimensionless].

``g1_over_vpr2`` (time, rho_norm)
  Ratio of g1 to vpr-squared [dimensionless].

``g2`` (time, rho_norm)
  Flux surface averaged :math:`\frac{(\nabla V)^2}{R^2}`, where R is the major
  radius along the flux surface being averaged [:math:`m^2`].

``g2g3_over_rhon`` (time, rho_norm)
  Ratio of g2g3 to the normalized toroidal flux coordinate rho_norm
  [dimensionless].

``g3`` (time, rho_norm)
  Flux surface averaged :math:`\frac{1}{R^2}` [:math:`m^{-2}`].

``Ip_profile`` (time, rho_face_norm)
  Total cumulative current profile [:math:`A`].

``j_bootstrap`` (time, rho_norm)
  Bootstrap current density [:math:`A/m^2`].

``j_ecrh`` (time, rho_cell_norm)
  Electron cyclotron heating current density [:math:`A/m^2`]. Only output if
  ``ecrh`` source is active.

``j_external`` (time, rho_cell_norm)
  Total external current density (including generic and ECRH current)
  [:math:`A/m^2`].

``j_generic_current`` (time, rho_cell_norm)
  Generic external non-inductive current density [:math:`A/m^2`]. Only output if
  ``generic_current`` source is active.

``j_ohmic`` (time, rho_cell_norm)
  Ohmic current density [:math:`A/m^2`].

``j_total`` (time, rho_norm)
  Total toroidal current density [:math:`A/m^2`].

``magnetic_shear`` (time, rho_face_norm)
  Magnetic shear [dimensionless], defined as
  :math:`-\frac{\hat{\rho}}{\iota}\frac{\partial\iota}{\partial\hat{\rho}}`,
  where :math:`\iota \equiv 1/q` .

``n_e`` (time, rho_norm)
  Electron density [:math:`m^{-3}`].

``n_i`` (time, rho_norm)
  Main ion density [:math:`m^{-3}`].

``n_impurity`` (time, rho_norm)
  Impurity density [:math:`m^{-3}`].

``p_alpha_e`` (time, rho_cell_norm)
  Fusion alpha heating power density to electrons [:math:`W/m^3`]. Only output
  if ``fusion`` source is active.

``p_alpha_i`` (time, rho_cell_norm)
  Fusion alpha heating power density to ions [:math:`W/m^3`]. Only output if
  ``fusion`` source is active.

``p_cyclotron_radiation_e`` (time, rho_cell_norm) [:math:`W/m^3`]
  Cyclotron radiation heat sink density (only relevant for electrons). Only
  output if ``cyclotron_radiation`` source is active.

``p_ecrh_e`` (time, rho_cell_norm)
  Electron cyclotron heating power density (only relevant for electrons)
  [:math:`W/m^3`]. Only output if ``ecrh`` source is active.

``p_generic_heat_e`` (time, rho_cell_norm)
  Generic external heat source power density to electrons [:math:`W/m^3`]. Only
  output if ``generic_heat`` source is active.

``p_generic_heat_i`` (time, rho_cell_norm)
  Generic external heat source power density to ions [:math:`W/m^3`]. Only
  output if ``generic_heat`` source is active.

``p_icrh_e`` (time, rho_cell_norm)
  Ion cyclotron heating power density to electrons [:math:`W/m^3`]. Only output
  if ``icrh`` source is active.

``p_icrh_i`` (time, rho_cell_norm)
  Ion cyclotron heating power density to ions [:math:`W/m^3`]. Only output if
  ``icrh`` source is active.

``p_impurity_radiation_e`` (time, rho_cell_norm)
  Impurity radiation heat sink density (only relevant for electrons)
  [:math:`W/m^3`]. Only output if ``impurity_radiation`` source is active.

``p_ohmic_e`` (time, rho_cell_norm)
  Ohmic heating power density [:math:`W/m^3`] (only relevant for electrons).
  Only output if ``ohmic`` source is active.

``Phi`` (time, rho_norm)
  Toroidal magnetic flux at each radial grid point [:math:`Wb`].

``pprime`` (time, rho_face_norm)
  Derivative of total pressure with respect to poloidal flux [:math:`Pa/Wb`].

``pressure_thermal_e`` (time, rho_norm)
  Electron thermal pressure [:math:`Pa`].

``pressure_thermal_i`` (time, rho_norm)
  Ion thermal pressure  [:math:`Pa`].

``pressure_thermal_total`` (time, rho_norm)
  Total thermal pressure [:math:`Pa`].

``psi`` (time, rho_norm)
  Poloidal flux profile :math:`\psi` [:math:`Wb`].

``psi_from_geo`` (time, rho_cell_norm)
  Poloidal flux provided by the input geometry file (NOT psi calculated
  self-consistently by the TORAX PDE) on the cell grid [:math:`Wb`].

``psi_from_Ip`` (time, rho_norm)
  Poloidal flux calculated from the current profile provided by the input
  geometry file (NOT psi calculated self-consistently by the TORAX PDE)
  [:math:`Wb`].

``psi_norm`` (time, rho_face_norm)
  Normalized poloidal flux profile [dimensionless].

``q`` (time, rho_face_norm)
  Safety factor profile on the face grid [dimensionless].

``R_in`` (time, rho_norm)
  Inner (minimum) radius of each flux surface [:math:`m`].

``r_mid`` (time, rho_norm)
  Mid-plane radius of each flux surface [:math:`m`].

``R_out`` (time, rho_norm)
  Outer (maximum) radius of each flux surface [:math:`m`].

``s_gas_puff`` (time, rho_cell_norm)
  Gas puff particle source density [:math:`s^{-1} m^{-3}`]. Only output if
  ``gas_puff`` source is active.

``s_generic_particle`` (time, rho_cell_norm)
  Generic particle source density [:math:`s^{-1} m^{-3}`]. Only output if
  ``generic_particle`` source is active.

``s_pellet`` (time, rho_cell_norm)
  Pellet particle source density [:math:`s^{-1} m^{-3}`]. Only output if
  ``pellet`` source is active.

``sigma_parallel`` (time, rho_cell_norm)
  Plasma conductivity parallel to the magnetic field [:math:`S/m`].

``spr`` (time, rho_norm)
  Derivative of plasma surface area enclosed by each flux surface, with respect
  to the normalized toroidal flux coordinate rho_norm [:math:`m^2`].

``T_e`` (time, rho_norm)
  Electron temperature [:math:`keV`].

``T_i`` (time, rho_norm)
  Ion temperature [:math:`keV`].

``v_loop`` (time, rho_norm)
  Loop voltage profile :math:`V_{loop}=\frac{\partial\psi}{\partial t}`
  [:math:`V`].

``volume`` (time, rho_norm)
  Plasma volume enclosed by each flux surface [:math:`m^3`].

``vpr`` (time, rho_norm)
  Derivative of plasma volume enclosed by each flux surface with respect to the
  normalized toroidal flux coordinate rho_norm [:math:`m^3`].

``V_neo_e`` (time, rho_face_norm)
  Neoclassical electron particle convection velocity [:math:`m/s`]. Contains
  all components apart from the Ware pinch, which is output separately.

``V_turb_e`` (time, rho_face_norm)
  Turbulent electron particle convection [:math:`m/s`].

``V_neo_ware_e`` (time, rho_face_norm)
  Ware pinch electron particle convection velocity [:math:`m/s`], i.e. the
  component of neoclassical convection arising from the parallel electric field.

``Z_eff`` (time, rho_norm)
  Effective charge profile defined as
  :math:`(Z_i^2n_i + Z_{impurity}^2n_{impurity})/n_e` [dimensionless].

``Z_i`` (time, rho_norm)
  Averaged main ion charge profile [dimensionless].

``Z_impurity`` (time, rho_norm)
  Averaged impurity charge profile [dimensionless].

scalars
-------

This dataset contains time-dependent scalar quantities describing global plasma
properties and characteristics.

``a_minor`` (time)
  Minor radius [:math:`m`].

``A_i`` (time)
  Averaged main ion mass [amu].

``A_impurity`` (time)
  Averaged impurity mass [amu].

``B_0`` (time)
  Magnetic field strength at the magnetic axis [:math:`T`].

``beta_N`` (time)
  Normalized beta (thermal) in percent [dimensionless]. Defined as
  :math:`\beta_N = 10^8\frac{a B_0}{I_p}\beta_t.`, with :math:`B_0` in T and
  :math:`I_p` in A.

``beta_pol`` (time)
  Volume-averaged plasma poloidal beta (thermal) [dimensionless]:
  :math:`\beta_p = \langle P_{th} \rangle_V/(B_{\theta,lcfs}^2/(2\mu_0))`
  Approximated by taking :math:`B_{\theta,lcfs} \approx \mu_0 I_p / (2\pi a_V)`,
  where :math:`a_V` is a minor radius definition satisfying the volume relation
  :math:`V = 2\pi^2R_0a_V^2`. This leads to:
  :math:`\beta_p = \frac{4V \langle P_{th} \rangle_V}{\mu_0 I_p^2 R_0}=\frac{8W_{th}}{3\mu_0 I_p^2 R_0}`.

``beta_tor`` (time)
  Volume-averaged plasma toroidal beta (thermal) [dimensionless]. Defined as
  :math:`\langle P_{th} \rangle_V/(B_0^2/(2\mu_0))`.

``dW_thermal_dt`` (time)
  Time derivative of the total thermal stored energy [:math:`W`].

``drho`` (time)
  Radial grid spacing in the unnormalized rho coordinate [:math:`m`].

``drho_norm`` ()
  Radial grid spacing in the normalized rho coordinate [dimensionless].

``E_aux`` (time)
  Total cumulative auxiliary injected energy (Ohmic + auxiliary heating)
  [:math:`J`].

``E_fusion`` (time)
  Total cumulative fusion energy produced [:math:`J`].

``fgw_n_e_line_avg`` (time)
  Greenwald fraction from line-averaged electron density [dimensionless].

``fgw_n_e_volume_avg`` (time)
  Greenwald fraction from volume-averaged electron density [dimensionless].

``H20`` (time)
  H-mode confinement quality factor with respect to the ITER20 scaling law
  [dimensionless].

``H89P`` (time)
  H-mode confinement quality factor with respect to the ITER89-P scaling law
  [dimensionless].

``H97L`` (time)
  L-mode confinement quality factor with respect to the ITER97L scaling law
  [dimensionless].

``H98`` (time)
  H-mode confinement quality factor with respect to the ITER98y2 scaling law
  [dimensionless].

``I_aux_generic`` (time)
  Total generic auxiliary current [:math:`A`].

``I_bootstrap`` (time)
  Total bootstrap current [:math:`A`].

``I_ecrh`` (time)
  Total electron cyclotron source current [:math:`A`].

``Ip`` (time)
  Plasma current [:math:`A`].

``li3`` (time)
  Normalized plasma internal inductance (ITER convention) [dimensionless].

``n_e_line_avg`` (time)
  Line-averaged electron density [dimensionless].

``n_e_min_P_LH`` (time)
  Electron density at which the minimum H-mode transition power occurs
  [:math:`m^{-3}`].

``n_e_volume_avg`` (time)
  Volume-averaged electron density [dimensionless].

``n_i_line_avg`` (time)
  Line-averaged main ion density [dimensionless].

``n_i_volume_avg`` (time)
  Volume-averaged main ion density [dimensionless].

``P_alpha_e`` (time)
  Total fusion alpha heating power to electrons [:math:`W`].

``P_alpha_i`` (time)
  Total fusion alpha heating power to ions [:math:`W`].

``P_alpha_total`` (time)
  Total fusion alpha heating power [:math:`W`].

``P_aux_e`` (time)
  Total auxiliary electron heating power [:math:`W`].

``P_aux_generic_e`` (time)
  Total generic auxiliary heating power to electrons [:math:`W`].

``P_aux_generic_i`` (time)
  Total generic auxiliary heating power to ions [:math:`W`].

``P_aux_generic_total`` (time)
  Total generic auxiliary heating power [:math:`W`].

``P_aux_i`` (time)
  Total auxiliary ion heating power [:math:`W`].

``P_aux_total`` (time)
  Total auxiliary heating power [:math:`W`] (sum of ion and electron auxiliary
  heating).

``P_bremsstrahlung_e`` (time)
  Total Bremsstrahlung electron heat sink power [:math:`W`].

``P_cyclotron_e`` (time)
  Total cyclotron radiation heat sink power (only relevant for electrons)
  [:math:`W`].

``P_ecrh_e`` (time)
  Total electron cyclotron source power (only relevant for electrons)
  [:math:`W`].

``P_ei_exchange_e`` (time)
  Total electron-ion heat exchange power to electrons [:math:`W`].

``P_ei_exchange_i`` (time)
  Total electron-ion heat exchange power to ions [:math:`W`].

``P_external_injected`` (time)
  Total externally injected power into the plasma [:math:`W`]. This will be
  larger than ``P_external_tot`` if any source has a value of
  ``absorption_fraction`` less than 1.

``P_icrh_e`` (time)
  Total ion cyclotron resonance heating power to electrons [:math:`W`].

``P_icrh_i`` (time)
  Total ion cyclotron resonance heating power to ions [:math:`W`].

``P_icrh_total`` (time)
  Total ion cyclotron resonance heating power [:math:`W`].

``P_LH`` (time)
  Calculated H-mode transition power, taken as the maximum of ``P_LH_min`` and
  ``P_LH_high_density``. This does not include an accurate calculation for the
  low density branch. [:math:`W`].

``P_LH_high_density`` (time)
  H-mode transition power for the high density branch, according to the Martin
  2008 scaling law [:math:`W`].

``P_LH_min`` (time)
  H-mode transition power at the density corresponding to the minimum transition
  power, from Ryter 2014. [:math:`W`].

``P_ohmic_e`` (time)
  Total Ohmic heating power (only relevant for electrons) [:math:`W`].

``P_radiation_e`` (time)
  Total radiative heat sink power (including Bremsstrahlung, Cyclotron, and
  other radiation). Only relevant for electrons [:math:`W`].

``P_SOL_e`` (time)
  Total electron heating power exiting the plasma across the LCFS [:math:`W`].

``P_SOL_i`` (time)
  Total ion heating power exiting the plasma across the LCFS [:math:`W`].

``P_SOL_total`` (time)
  Total heating power exiting the plasma across the LCFS [:math:`W`].

``Phi_b`` (time)
  Total toroidal magnetic flux [:math:`Wb`].

``Phi_b_dot`` (time)
  Time derivative of the total toroidal magnetic flux [:math:`Wb/s`].

``q95`` (time)
  Safety factor at 95% of the normalized poloidal flux coordinate
  [dimensionless].

``Q_fusion`` (time)
  Fusion power gain [dimensionless].

``q_min`` (time)
  Minimum safety factor [dimensionless].

``R_major`` (time)
  Major radius [:math:`m`].

``rho_b`` (time)
  Value of the unnormalized rho coordinate at the boundary [:math:`m`].

``rho_q_2_1_first`` (time)
  Normalized toroidal flux coordinate of the first surface where q = 2
  [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_2_1_second`` (time)
  Normalized toroidal flux coordinate of the second surface where q = 2
  [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_3_1_first`` (time)
  Normalized toroidal flux coordinate of the first surface where q = 3
  [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_3_1_second`` (time)
  Normalized toroidal flux coordinate of the second surface where q = 3
  [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_3_2_first`` (time)
  Normalized toroidal flux coordinate of the first surface where q = 3/2
  [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_3_2_second`` (time)
  Normalized toroidal flux coordinate of the second surface where q = 3/2
  [dimensionless]. Values of -inf indicate no such surface exists.

``rho_q_min`` (time)
  Normalized toroidal flux coordinate at which the minimum safety factor occurs
  [dimensionless].

``S_gas_puff`` (time)
  Integrated gas puff particle source rate [:math:`s^{-1}`].

``S_generic_particle`` (time)
  Integrated generic particle source rate [:math:`s^{-1}`].

``S_pellet`` (time)
  Integrated pellet particle source rate [:math:`s^{-1}`].

``S_total`` (time)
  Total particle source rate from all active sources [:math:`s^{-1}`].

``T_e_volume_avg`` (time)
  Volume-averaged electron temperature [:math:`keV`].

``T_i_volume_avg`` (time)
  Volume-averaged ion temperature [:math:`keV`].

``tau_E`` (time)
  Thermal energy confinement time [:math:`s`].

``v_loop_lcfs`` (time)
  Loop voltage at the last closed flux surface (LCFS) [:math:`Wb/s` or
  :math:`V`]. This is a scalar value derived from the `v_loop` profile.

``W_pol`` (time)
  Total poloidal magnetic energy [:math:`J`].

``W_thermal_e`` (time)
  Total electron thermal stored energy [:math:`J`].

``W_thermal_i`` (time)
  Total ion thermal stored energy [:math:`J`].

``W_thermal_total`` (time)
  Total thermal stored energy [:math:`J`].

.. _output_examples:

Working with output data
========================

To demonstrate xarray and numpy manipulations of output data, the following
code carries out volume integration of ``alpha_e`` and ``alpha_i`` at the time
closest to t=1. The result equals the input config
``sources['fusion']['P_total']`` at the time closest to t=1.

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
  print(config_dict['transport']['model_name'])
  # We can also use ToraxConfig to run the simulation again.
  torax_config = torax.ToraxConfig.from_dict(config_dict)
  new_output = torax.run_simulation(torax_config)

