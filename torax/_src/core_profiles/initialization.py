# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions used for initializing core profiles."""

import dataclasses

from absl import logging
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import getters
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.physics import psi_calculations
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def initial_core_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
) -> state.CoreProfiles:
  """Calculates the initial core profiles.

  Args:
    runtime_params: Runtime parameters at t=t_initial.
    geo: Torus geometry at t=t_initial.
    source_models: All models for TORAX sources/sinks.
    neoclassical_models: All models for neoclassical calculations.

  Returns:
    Initial core profiles.
  """
  T_i = getters.get_updated_ion_temperature(
      runtime_params.profile_conditions, geo
  )
  T_e = getters.get_updated_electron_temperature(
      runtime_params.profile_conditions, geo
  )
  n_e = getters.get_updated_electron_density(
      runtime_params.profile_conditions, geo
  )
  ions = getters.get_updated_ions(runtime_params, geo, n_e, T_e)
  toroidal_angular_velocity = getters.get_updated_toroidal_angular_velocity(
      runtime_params.profile_conditions, geo
  )
  # Set v_loop_lcfs. Two branches:
  # 1. Set the v_loop_lcfs from profile_conditions if using the v_loop BC option
  # 2. Initialize v_loop_lcfs to 0 if using the Ip boundary condition for psi.
  # In case 2, v_loop_lcfs will be updated every timestep based on the psi_lcfs
  # values across the time interval. Since there is is one more time value than
  # time intervals, the v_loop_lcfs time-series is underconstrained. Therefore,
  # we set v_loop_lcfs[0] to v_loop_lcfs[1] when creating the outputs.
  if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition:
    v_loop_lcfs = jnp.asarray(
        runtime_params.profile_conditions.v_loop_lcfs,
        dtype=jax_utils.get_dtype(),
    )
  else:
    v_loop_lcfs = jnp.asarray(0.0, dtype=jax_utils.get_dtype())

  # Initialise psi and derived quantities to zero before they are calculated.
  psidot = cell_variable.CellVariable(
      value=jnp.zeros_like(geo.rho, dtype=jax_utils.get_dtype()),
      face_centers=geo.rho_face_norm,
  )
  psi = cell_variable.CellVariable(
      value=jnp.zeros_like(geo.rho, dtype=jax_utils.get_dtype()),
      face_centers=geo.rho_face_norm,
  )

  core_profiles = state.CoreProfiles(
      T_i=T_i,
      T_e=T_e,
      n_e=n_e,
      n_i=ions.n_i,
      Z_i=ions.Z_i,
      Z_i_face=ions.Z_i_face,
      A_i=ions.A_i,
      n_impurity=ions.n_impurity,
      impurity_fractions=ions.impurity_fractions,
      main_ion_fractions=ions.main_ion_fractions,
      Z_impurity=ions.Z_impurity,
      Z_impurity_face=ions.Z_impurity_face,
      A_impurity=ions.A_impurity,
      A_impurity_face=ions.A_impurity_face,
      Z_eff=ions.Z_eff,
      Z_eff_face=ions.Z_eff_face,
      psi=psi,
      psidot=psidot,
      q_face=jnp.zeros_like(geo.rho_face, dtype=jax_utils.get_dtype()),
      s_face=jnp.zeros_like(geo.rho_face, dtype=jax_utils.get_dtype()),
      v_loop_lcfs=v_loop_lcfs,
      sigma=jnp.zeros_like(geo.rho, dtype=jax_utils.get_dtype()),
      sigma_face=jnp.zeros_like(geo.rho_face, dtype=jax_utils.get_dtype()),
      j_total=jnp.zeros_like(geo.rho, dtype=jax_utils.get_dtype()),
      j_total_face=jnp.zeros_like(geo.rho_face, dtype=jax_utils.get_dtype()),
      Ip_profile_face=jnp.zeros_like(geo.rho_face, dtype=jax_utils.get_dtype()),
      toroidal_angular_velocity=toroidal_angular_velocity,
      charge_state_info=ions.charge_state_info,
      charge_state_info_face=ions.charge_state_info_face,
  )

  return _init_psi_and_psi_derived(
      runtime_params,
      geo,
      core_profiles,
      source_models,
      neoclassical_models,
  )


def update_psi_from_j(
    Ip: array_typing.FloatScalar,
    geo: geometry.Geometry,
    j_total_hires: jax.Array,
    use_v_loop_lcfs_boundary_condition: bool = False,
) -> cell_variable.CellVariable:
  """Calculates poloidal flux (psi) consistent with plasma current.

  For increased accuracy of psi, a hi-res grid is used, due to the double
    integration. Presently used only for initialization. Therefore Ip is
    a valid source of truth for Ip, even if use_v_loop_lcfs_boundary_condition
    is True.

  Args:
    Ip: Total plasma current [A].
    geo: Torus geometry.
    j_total_hires: High resolution version of j_total [A/m^2].
    use_v_loop_lcfs_boundary_condition: Whether to set the loop voltage from Ip.

  Returns:
    psi: Poloidal flux cell variable.
  """
  y = j_total_hires * geo.spr_hires
  assert y.ndim == 1
  assert geo.rho_hires.ndim == 1
  Ip_profile = math_utils.cumulative_trapezoid(
      y=y, x=geo.rho_hires_norm, initial=0.0
  )
  scale = jnp.concatenate((
      jnp.zeros((1,)),
      (16 * jnp.pi**3 * constants.CONSTANTS.mu_0 * geo.Phi_b)
      / (geo.F_hires[1:] * geo.g2g3_over_rhon_hires[1:]),
  ))
  # dpsi_dr on hires cell grid
  dpsi_drhon_hires = scale * Ip_profile

  # psi on hires cell grid
  psi_hires = math_utils.cumulative_trapezoid(
      y=dpsi_drhon_hires, x=geo.rho_hires_norm, initial=0.0
  )

  psi_value = jnp.interp(geo.rho_norm, geo.rho_hires_norm, psi_hires)

  # Set the BCs for psi to ensure the correct Ip
  dpsi_drhonorm_edge = psi_calculations.calculate_psi_grad_constraint_from_Ip(
      Ip,
      geo,
  )

  if use_v_loop_lcfs_boundary_condition:
    # For v_loop_lcfs, we will prescribe a rate of change of psi at the LCFS
    # For the first timestep, we need an initial value for psi at the LCFS, so
    # we set it to match the desired plasma current.
    right_face_grad_constraint = None
    right_face_constraint = (
        psi_value[-1] + dpsi_drhonorm_edge * geo.drho_norm[-1] / 2
    )
  else:
    # Use the dpsi/drho calculated above as the right face gradient constraint
    right_face_grad_constraint = dpsi_drhonorm_edge
    right_face_constraint = None

  psi = cell_variable.CellVariable(
      value=psi_value,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=right_face_grad_constraint,
      right_face_constraint=right_face_constraint,
  )

  return psi


def _get_initial_psi_mode(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
) -> profile_conditions_lib.InitialPsiMode:
  """Returns the initial psi mode based on the runtime parameters.

  This allows us to support the legacy behavior of initial_psi_from_j, which
  is only available when using the standard geometry and initial psi is not
  provided. Moving forward the initial_psi_mode setting in the profile
  conditions should be preferred.

  Args:
    runtime_params: Runtime parameters.
    geo: Torus geometry.

  Returns:
    How to calculate the initial psi value.
  """
  psi_mode = runtime_params.profile_conditions.initial_psi_mode
  if psi_mode == profile_conditions_lib.InitialPsiMode.PROFILE_CONDITIONS:
    if runtime_params.profile_conditions.psi is None:
      logging.warning(
          'Falling back to legacy behavior as `profile_conditions.psi` is '
          'None. Future versions of TORAX will require `psi` to be provided '
          'if `initial_psi_mode` is PROFILE_CONDITIONS. Use '
          '`initial_psi_mode` to initialize psi from `j` or `geometry` and '
          'avoid this warning.'
      )
      if (
          isinstance(geo, standard_geometry.StandardGeometry)
          and not runtime_params.profile_conditions.initial_psi_from_j
      ):
        psi_mode = profile_conditions_lib.InitialPsiMode.GEOMETRY
      else:
        psi_mode = profile_conditions_lib.InitialPsiMode.J
  return psi_mode


def _init_psi_and_psi_derived(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
) -> state.CoreProfiles:
  """Initialises psi and currents in core profiles.

  There are three modes of doing this that are supported:
    1. Retrieving psi from the profile conditions.
    2. Retrieving psi from the standard geometry input.
    3. Calculating j according to the nu formula and then
    calculating psi from that. As we are calculating j using a guess for psi,
    this method is iterated to converge to the true psi.

  Args:
    runtime_params: Runtime parameters.
    geo: Torus geometry.
    core_profiles: Core profiles.
    source_models: All TORAX source/sink functions.
    neoclassical_models: All models for neoclassical calculations.

  Returns:
    Refined core profiles.
  """

  # Flag to track if sources have been calculated during psi initialization.
  sources_are_calculated = False

  # Initialize psi source profiles and bootstrap current to all zeros.
  source_profiles = source_profile_builders.build_all_zero_profiles(geo)

  initial_psi_mode = _get_initial_psi_mode(runtime_params, geo)

  match initial_psi_mode:
    # Case 1: retrieving psi from the profile conditions, using the prescribed
    # profile and Ip
    case profile_conditions_lib.InitialPsiMode.PROFILE_CONDITIONS:
      if runtime_params.profile_conditions.psi is None:
        raise ValueError(
            'psi is None, but initial_psi_mode is PROFILE_CONDITIONS.'
        )
      # Calculate the dpsi/drho necessary to achieve the given Ip
      dpsi_drhonorm_edge = (
          psi_calculations.calculate_psi_grad_constraint_from_Ip(
              runtime_params.profile_conditions.Ip,
              geo,
          )
      )

      # Set the BCs to ensure the correct Ip
      if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition:
        # Extrapolate the value of psi at the LCFS from the dpsi/drho constraint
        # to achieve the desired Ip
        right_face_grad_constraint = None
        right_face_constraint = (
            runtime_params.profile_conditions.psi[-1]
            + dpsi_drhonorm_edge * geo.drho_norm[-1] / 2
        )
      else:
        # Use the dpsi/drho calculated above as the right face gradient
        # constraint
        right_face_grad_constraint = dpsi_drhonorm_edge
        right_face_constraint = None

      psi = cell_variable.CellVariable(
          value=runtime_params.profile_conditions.psi,
          face_centers=geo.rho_face_norm,
          right_face_grad_constraint=right_face_grad_constraint,
          right_face_constraint=right_face_constraint,
      )

    # Case 2: retrieving psi from the standard geometry input.
    case profile_conditions_lib.InitialPsiMode.GEOMETRY:
      if not isinstance(geo, standard_geometry.StandardGeometry):
        raise ValueError(
            'GEOMETRY initial_psi_source is only supported for standard'
            ' geometry.'
        )
      # psi is already provided from a numerical equilibrium, so no need to
      # first calculate currents.
      dpsi_drhonorm_edge = (
          psi_calculations.calculate_psi_grad_constraint_from_Ip(
              runtime_params.profile_conditions.Ip,
              geo,
          )
      )
      # Use the psi from the equilibrium as the right face constraint
      # This has already been made consistent with the desired Ip
      # by make_ip_consistent
      psi = cell_variable.CellVariable(
          value=geo.psi_from_Ip,  # Use psi from equilibrium
          face_centers=geo.rho_face_norm,
          right_face_grad_constraint=None
          if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
          else dpsi_drhonorm_edge,
          right_face_constraint=geo.psi_from_Ip_face[-1]
          if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
          else None,
      )

    # Case 3: calculating j according to nu formula and psi from j.
    case profile_conditions_lib.InitialPsiMode.J:
      # calculate j and psi from the nu formula
      j_total_hires = _get_j_total_hires_with_no_external_sources(
          runtime_params, geo
      )
      psi = update_psi_from_j(
          runtime_params.profile_conditions.Ip,
          geo,
          j_total_hires,
          use_v_loop_lcfs_boundary_condition=runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition,
      )
      if not (runtime_params.profile_conditions.initial_j_is_total_current):
        # In this branch we require non-inductive currents to determine j_total.
        # The nu formula only provides the Ohmic component of the current.
        # However calculating non-inductive currents requires a non-zero psi.
        # We thus iterate between psi and source calculations, using j_total
        # and psi calculated purely with the nu formula as an initial guess

        # Initialize iterations
        core_profiles_initial = dataclasses.replace(
            core_profiles,
            psi=psi,
            q_face=psi_calculations.calc_q_face(geo, psi),
            s_face=psi_calculations.calc_s_face(geo, psi),
        )

        # TODO(b/440385263): add tunable iteration number or convergence
        # criteria, and modify python for loop to jax fori loop for the general
        # case.

        # Iterate with non-inductive current source calculations. Stop after 2.
        psi, source_profiles = _iterate_psi_and_sources(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles_initial,
            neoclassical_models=neoclassical_models,
            source_models=source_models,
            source_profiles=source_profiles,
            iterations=2,
        )

        # Mark that sources have been calculated to avoid redundant work.
        sources_are_calculated = True

  # Conclude with completing core_profiles with all psi-dependent profiles.
  core_profiles = _calculate_all_psi_dependent_profiles(
      runtime_params=runtime_params,
      geo=geo,
      psi=psi,
      core_profiles=core_profiles,
      source_profiles=source_profiles,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
      sources_are_calculated=sources_are_calculated,
  )

  return core_profiles


def _calculate_all_psi_dependent_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
    core_profiles: state.CoreProfiles,
    source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    sources_are_calculated: bool,
) -> state.CoreProfiles:
  """Supplements core profiles with all other profiles that depend on psi."""
  j_total, j_total_face, Ip_profile_face = psi_calculations.calc_j_total(
      geo, psi, runtime_params.numerics.min_rho_norm
  )

  core_profiles = dataclasses.replace(
      core_profiles,
      psi=psi,
      q_face=psi_calculations.calc_q_face(geo, psi),
      s_face=psi_calculations.calc_s_face(geo, psi),
      j_total=j_total,
      j_total_face=j_total_face,
      Ip_profile_face=Ip_profile_face,
  )
  # Calculate conductivity once we have a consistent set of core profiles
  conductivity = neoclassical_models.conductivity.calculate_conductivity(
      geo,
      core_profiles,
  )

  # Calculate sources if they have not already been calculated.
  if not sources_are_calculated:
    source_profiles = _get_bootstrap_and_standard_source_profiles(
        runtime_params,
        geo,
        core_profiles,
        neoclassical_models,
        source_models,
        source_profiles,
    )

  # psidot calculated here with phibdot=0 in geo, since this is initial
  # conditions and we don't yet have information on geo_t_plus_dt for the
  # phibdot calculation.

  if (
      not runtime_params.numerics.evolve_current
      and runtime_params.profile_conditions.psidot is not None
  ):
    # If psidot is prescribed and psi does not evolve, use prescribed value
    psidot_value = runtime_params.profile_conditions.psidot
  else:
    # Otherwise, calculate psidot from psi sources.
    psi_sources = source_profiles.total_psi_sources(geo)
    psidot_value = psi_calculations.calculate_psidot_from_psi_sources(
        psi_sources=psi_sources,
        sigma=conductivity.sigma,
        resistivity_multiplier=runtime_params.numerics.resistivity_multiplier,
        psi=psi,
        geo=geo,
    )

  # psidot boundary condition. If v_loop_lcfs is not prescribed then we set it
  # to the last calculated psidot for the initialisation since we have no
  # other information.
  v_loop_lcfs = (
      runtime_params.profile_conditions.v_loop_lcfs
      if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
      else psidot_value[-1]
  )
  psidot = dataclasses.replace(
      core_profiles.psidot,
      value=psidot_value,
      right_face_constraint=v_loop_lcfs,
      right_face_grad_constraint=None,
  )
  core_profiles = dataclasses.replace(
      core_profiles,
      psidot=psidot,
      sigma=conductivity.sigma,
      sigma_face=conductivity.sigma_face,
  )
  return core_profiles


def _get_bootstrap_and_standard_source_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    source_models: source_models_lib.SourceModels,
    source_profiles: source_profiles_lib.SourceProfiles,
) -> source_profiles_lib.SourceProfiles:
  """Calculates bootstrap current and updates source profiles."""
  source_profile_builders.build_standard_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      psi_only=True,
      calculate_anyway=True,
      calculated_source_profiles=source_profiles,
  )
  bootstrap_current = (
      neoclassical_models.bootstrap_current.calculate_bootstrap_current(
          runtime_params, geo, core_profiles
      )
  )
  source_profiles = dataclasses.replace(
      source_profiles, bootstrap_current=bootstrap_current
  )
  return source_profiles


def _iterate_psi_and_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    source_models: source_models_lib.SourceModels,
    source_profiles: source_profiles_lib.SourceProfiles,
    iterations: int,
) -> tuple[cell_variable.CellVariable, source_profiles_lib.SourceProfiles]:
  """Iterates psi and sources to converge to a consistent state."""

  for _ in range(iterations):
    source_profiles = _get_bootstrap_and_standard_source_profiles(
        runtime_params,
        geo,
        core_profiles,
        neoclassical_models,
        source_models,
        source_profiles,
    )
    j_total_hires = get_j_toroidal_total_hires_with_external_sources(
        runtime_params,
        geo,
        source_profiles.bootstrap_current,
        j_toroidal_external=psi_calculations.j_parallel_to_j_toroidal(
            sum(source_profiles.psi.values()),
            geo,
            runtime_params.numerics.min_rho_norm,
        ),
    )
    psi = update_psi_from_j(
        runtime_params.profile_conditions.Ip,
        geo,
        j_total_hires,
        use_v_loop_lcfs_boundary_condition=runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition,
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        psi=psi,
        q_face=psi_calculations.calc_q_face(geo, psi),
        s_face=psi_calculations.calc_s_face(geo, psi),
    )
  return core_profiles.psi, source_profiles


def _get_j_total_hires_with_no_external_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
) -> jax.Array:
  """Calculates j_total hires when the total current is given by a formula."""
  Ip = runtime_params.profile_conditions.Ip
  jformula_hires = (
      1 - geo.rho_hires_norm**2
  ) ** runtime_params.profile_conditions.current_profile_nu
  denom = _trapz(jformula_hires * geo.spr_hires, geo.rho_hires_norm)
  Ctot_hires = Ip / denom
  j_total_hires = jformula_hires * Ctot_hires
  return j_total_hires


def get_j_toroidal_total_hires_with_external_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent,
    j_toroidal_external: jax.Array,
) -> jax.Array:
  """Calculates j_total hires when the Ohmic current is given by a formula."""
  # Convert bootstrap current density to toroidal, and calculate high-resolution
  # version
  j_toroidal_bootstrap = psi_calculations.j_parallel_to_j_toroidal(
      bootstrap_current.j_parallel_bootstrap,
      geo,
      runtime_params.numerics.min_rho_norm,
  )
  j_toroidal_bootstrap_hires = jnp.interp(
      geo.rho_hires, geo.rho_face, bootstrap_current.j_parallel_bootstrap_face
  )

  # Calculate high-resolution version of external (eg ECCD) current density
  j_toroidal_external_face = math_utils.cell_to_face(
      j_toroidal_external,
      geo,
      preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
  )
  j_toroidal_external_hires = jnp.interp(
      geo.rho_hires, geo.rho_face, j_toroidal_external_face
  )

  # Calculate high resolution j_total and j_ohmic
  j_toroidal_ohmic_formula_hires = (
      1 - geo.rho_hires_norm**2
  ) ** runtime_params.profile_conditions.current_profile_nu
  denom = _trapz(
      j_toroidal_ohmic_formula_hires * geo.spr_hires, geo.rho_hires_norm
  )
  I_non_inductive = math_utils.area_integration(
      j_toroidal_external + j_toroidal_bootstrap, geo
  )
  I_ohmic = runtime_params.profile_conditions.Ip - I_non_inductive
  C_ohm_hires = I_ohmic / denom
  j_toroidal_ohmic_hires = j_toroidal_ohmic_formula_hires * C_ohm_hires
  j_toroidal_total_hires = (
      j_toroidal_ohmic_hires
      + j_toroidal_external_hires
      + j_toroidal_bootstrap_hires
  )
  return j_toroidal_total_hires
