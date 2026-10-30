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

from collections.abc import Mapping
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import getters
from torax._src.core_profiles import runtime_params as core_profile_runtime_params
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.neoclassical import neoclassical_model as neoclassical_model_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.physics import psi_calculations
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.sources import source_state as source_state_lib

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def initial_core_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    source_models: source_models_lib.SourceModels,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    source_states: Mapping[str, source_state_lib.SourceState],
) -> state.CoreProfiles:
  """Calculates the initial core profiles.

  Args:
    runtime_params: Runtime parameters at t=t_initial.
    geo: Torus geometry at t=t_initial.
    source_models: All models for TORAX sources/sinks.
    neoclassical_model: The neoclassical model.
    source_states: Initial source states mapping.

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
  # 2. Initialize v_loop_lcfs to 0 if using the I_p boundary condition for psi.
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
      fast_ions=(),
      psi=psi,
      psidot=psidot,
      q_face=jnp.zeros_like(geo.rho_face, dtype=jax_utils.get_dtype()),
      s_face=jnp.zeros_like(geo.rho_face, dtype=jax_utils.get_dtype()),
      v_loop_lcfs=v_loop_lcfs,
      sigma=jnp.zeros_like(geo.rho, dtype=jax_utils.get_dtype()),
      sigma_face=jnp.zeros_like(geo.rho_face, dtype=jax_utils.get_dtype()),
      j_toroidal_total=jnp.zeros_like(geo.rho, dtype=jax_utils.get_dtype()),
      j_toroidal_total_face=jnp.zeros_like(
          geo.rho_face, dtype=jax_utils.get_dtype()
      ),
      I_p_profile_face=jnp.zeros_like(
          geo.rho_face, dtype=jax_utils.get_dtype()
      ),
      toroidal_angular_velocity=toroidal_angular_velocity,
      poloidal_velocity=cell_variable.CellVariable(
          value=jnp.zeros_like(geo.rho, dtype=jax_utils.get_dtype()),
          face_centers=geo.rho_face_norm,
          right_face_constraint=jnp.zeros((), dtype=jax_utils.get_dtype()),
          right_face_grad_constraint=None,
      ),
      charge_state_info=ions.charge_state_info,
      charge_state_info_face=ions.charge_state_info_face,
  )

  return _init_psi_and_psi_derived(
      runtime_params,
      geo,
      core_profiles,
      source_models,
      neoclassical_model,
      source_states=source_states,
  )


def update_psi_from_j(
    I_p: array_typing.FloatScalar,
    geo: geometry.Geometry,
    j_toroidal_total_hires: jax.Array,
    use_v_loop_lcfs_boundary_condition: bool = False,
) -> cell_variable.CellVariable:
  """Calculates poloidal flux (psi) consistent with plasma current.

  For increased accuracy of psi, a hi-res grid is used, due to the double
    integration. Presently used only for initialization. Therefore I_p is
    a valid source of truth for I_p, even if use_v_loop_lcfs_boundary_condition
    is True.

  Args:
    I_p: Total plasma current [A].
    geo: Torus geometry.
    j_toroidal_total_hires: High resolution version of j_toroidal_total [A/m^2].
    use_v_loop_lcfs_boundary_condition: Whether to set the loop voltage from
      I_p.

  Returns:
    psi: Poloidal flux cell variable.
  """
  y = j_toroidal_total_hires * geo.spr_hires
  assert y.ndim == 1
  assert geo.rho_hires.ndim == 1
  I_p_profile = math_utils.cumulative_trapezoid(
      y=y, x=geo.rho_hires_norm, initial=0.0
  )
  scale = jnp.concatenate((
      jnp.zeros((1,)),
      (16 * jnp.pi**3 * constants.CONSTANTS.mu_0 * geo.Phi_b)
      / (geo.F_hires[1:] * geo.g2g3_over_rhon_hires[1:]),
  ))
  # dpsi_dr on hires cell grid
  dpsi_drhon_hires = scale * I_p_profile

  # psi on hires cell grid
  psi_hires = math_utils.cumulative_trapezoid(
      y=dpsi_drhon_hires, x=geo.rho_hires_norm, initial=0.0
  )

  psi_value = jnp.interp(geo.rho_norm, geo.rho_hires_norm, psi_hires)

  # Set the BCs for psi to ensure the correct I_p
  dpsi_drhonorm_edge = psi_calculations.calculate_psi_grad_constraint_from_I_p(
      I_p,
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


def _init_psi_and_psi_derived(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    source_states: Mapping[str, source_state_lib.SourceState],
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
    neoclassical_model: The neoclassical model.
    source_states: Initial source states mapping.

  Returns:
    Refined core profiles.
  """

  # Flag to track if sources have been calculated during psi initialization.
  sources_are_calculated = False

  # Initialize psi source profiles and bootstrap current to all zeros.
  source_profiles = source_profile_builders.build_all_zero_profiles(geo)

  match runtime_params.profile_conditions.initial_psi_source:
    # Case 1: retrieving psi from the profile conditions, using the prescribed
    # profile and I_p
    case core_profile_runtime_params.InitialPsiSource.PROFILE_CONDITIONS:
      if runtime_params.profile_conditions.psi is None:
        raise ValueError(
            'psi is None, but initial_psi_source is PROFILE_CONDITIONS.'
        )
      # Calculate the dpsi/drho necessary to achieve the given I_p
      dpsi_drhonorm_edge = (
          psi_calculations.calculate_psi_grad_constraint_from_I_p(
              runtime_params.profile_conditions.I_p,
              geo,
          )
      )

      # Set the BCs to ensure the correct I_p
      if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition:
        # Extrapolate the value of psi at the LCFS from the dpsi/drho constraint
        # to achieve the desired I_p
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
    case core_profile_runtime_params.InitialPsiSource.GEOMETRY:
      if not isinstance(geo, standard_geometry.StandardGeometry):
        raise ValueError(
            'GEOMETRY initial_psi_source is only supported for standard'
            ' geometry.'
        )
      # psi is already provided from a numerical equilibrium, so no need to
      # first calculate currents.
      dpsi_drhonorm_edge = (
          psi_calculations.calculate_psi_grad_constraint_from_I_p(
              runtime_params.profile_conditions.I_p,
              geo,
          )
      )
      # Use the psi from the equilibrium as the right face constraint
      # This has already been made consistent with the desired I_p
      # by make_I_p_consistent
      psi = cell_variable.CellVariable(
          value=geo.psi_from_I_p,  # Use psi from equilibrium
          face_centers=geo.rho_face_norm,
          right_face_grad_constraint=None
          if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
          else dpsi_drhonorm_edge,
          right_face_constraint=geo.psi_from_I_p_face[-1]
          if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
          else None,
      )

    # Case 3: calculating j according to nu formula and psi from j.
    case core_profile_runtime_params.InitialPsiSource.J:
      # calculate j and psi from the nu formula
      j_toroidal_total_hires = (
          _get_j_toroidal_total_hires_with_no_external_sources(
              runtime_params, geo
          )
      )
      psi = update_psi_from_j(
          runtime_params.profile_conditions.I_p,
          geo,
          j_toroidal_total_hires,
          use_v_loop_lcfs_boundary_condition=runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition,
      )
      if not (runtime_params.profile_conditions.initial_j_is_total_current):
        # In this branch we require non-inductive currents to determine
        # j_toroidal_total. The nu formula only provides the Ohmic component of
        # the current. However calculating non-inductive currents requires a
        # non-zero psi. We thus iterate between psi and source calculations,
        # using j_toroidal_total and psi calculated purely with the nu formula
        # as an initial guess

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
            neoclassical_model=neoclassical_model,
            source_models=source_models,
            source_profiles=source_profiles,
            source_states=source_states,
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
      neoclassical_model=neoclassical_model,
      sources_are_calculated=sources_are_calculated,
      source_states=source_states,
  )

  return core_profiles


def _calculate_all_psi_dependent_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
    core_profiles: state.CoreProfiles,
    source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    sources_are_calculated: bool,
    source_states: Mapping[str, source_state_lib.SourceState],
) -> state.CoreProfiles:
  """Supplements core profiles with all other profiles that depend on psi."""
  j_toroidal_total, j_toroidal_total_face, I_p_profile_face = (
      psi_calculations.calc_j_toroidal_total(
          geo, psi, runtime_params.numerics.min_rho_norm
      )
  )

  core_profiles = dataclasses.replace(
      core_profiles,
      psi=psi,
      q_face=psi_calculations.calc_q_face(geo, psi),
      s_face=psi_calculations.calc_s_face(geo, psi),
      j_toroidal_total=j_toroidal_total,
      j_toroidal_total_face=j_toroidal_total_face,
      I_p_profile_face=I_p_profile_face,
  )
  (
      conductivity,
      bootstrap_current,
      poloidal_velocity,
      _,
  ) = neoclassical_model.calculate_conductivity_current_and_rotation(
      runtime_params, geo, core_profiles
  )

  # Calculate sources if they have not already been calculated.
  if not sources_are_calculated:
    source_profile_builders.build_standard_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
        psi_only=True,
        calculate_anyway=True,
        calculated_source_profiles=source_profiles,
    )
    source_profiles = dataclasses.replace(
        source_profiles,
        bootstrap_current=bootstrap_current,
    )

  # psidot calculated here with phibdot=0 in geo, since this is initial
  # conditions and we don't yet have information on geo_t_plus_dt for the
  # phibdot calculation.
  v_loop_lcfs = (
      runtime_params.profile_conditions.v_loop_lcfs
      if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
      else None
  )
  psidot = psi_calculations.calculate_psidot(
      psi=psi,
      sigma=conductivity.sigma,
      psi_sources=source_profiles.total_psi_sources(geo),
      geo=geo,
      evolve_current=runtime_params.numerics.evolve_current,
      prescribed_psidot=runtime_params.profile_conditions.psidot,
      v_loop_lcfs=v_loop_lcfs,
  )
  core_profiles = dataclasses.replace(
      core_profiles,
      psidot=psidot,
      sigma=conductivity.sigma,
      sigma_face=conductivity.sigma_face,
      poloidal_velocity=poloidal_velocity.v_pol,
  )
  return core_profiles


def _iterate_psi_and_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    source_models: source_models_lib.SourceModels,
    source_profiles: source_profiles_lib.SourceProfiles,
    source_states: Mapping[str, source_state_lib.SourceState],
    iterations: int,
) -> tuple[cell_variable.CellVariable, source_profiles_lib.SourceProfiles]:
  """Iterates psi and sources to converge to a consistent state."""

  for _ in range(iterations):
    source_states = source_profile_builders.build_standard_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
        psi_only=True,
        calculate_anyway=True,
        calculated_source_profiles=source_profiles,
    )
    _, bootstrap_current, _, _ = (
        neoclassical_model.calculate_conductivity_current_and_rotation(
            runtime_params, geo, core_profiles
        )
    )
    source_profiles = dataclasses.replace(
        source_profiles,
        bootstrap_current=bootstrap_current,
    )
    j_toroidal_total_hires = get_j_toroidal_total_hires_with_external_sources(
        runtime_params,
        geo,
        source_profiles.bootstrap_current,
        j_toroidal_external=psi_calculations.j_parallel_to_j_toroidal(
            sum(source_profiles.psi.values(), start=jnp.zeros_like(geo.rho)),
            geo,
            runtime_params.numerics.min_rho_norm,
        ),
    )
    psi = update_psi_from_j(
        runtime_params.profile_conditions.I_p,
        geo,
        j_toroidal_total_hires,
        use_v_loop_lcfs_boundary_condition=runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition,
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        psi=psi,
        q_face=psi_calculations.calc_q_face(geo, psi),
        s_face=psi_calculations.calc_s_face(geo, psi),
    )
  return core_profiles.psi, source_profiles


def _get_j_toroidal_total_hires_with_no_external_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
) -> jax.Array:
  """Calculates j_toroidal_total hires when the total current is given by a formula."""
  I_p = runtime_params.profile_conditions.I_p
  jformula_hires = (
      1 - geo.rho_hires_norm**2
  ) ** runtime_params.profile_conditions.current_profile_nu
  denom = _trapz(jformula_hires * geo.spr_hires, geo.rho_hires_norm)
  Ctot_hires = I_p / denom
  j_toroidal_total_hires = jformula_hires * Ctot_hires
  return j_toroidal_total_hires


def get_j_toroidal_total_hires_with_external_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent,
    j_toroidal_external: array_typing.FloatVectorCell,
) -> jax.Array:
  """Calculates j_toroidal_total hires when the Ohmic current is given by a formula."""
  # Convert bootstrap current density to toroidal, and calculate high-resolution
  # version
  j_toroidal_bootstrap = psi_calculations.j_parallel_to_j_toroidal(
      bootstrap_current.j_parallel_bootstrap,
      geo,
      runtime_params.numerics.min_rho_norm,
  )
  j_toroidal_bootstrap_face = math_utils.cell_to_face(
      j_toroidal_bootstrap,
      geo,
      preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
  )
  j_toroidal_bootstrap_hires = jnp.interp(
      geo.rho_hires, geo.rho_face, j_toroidal_bootstrap_face
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

  # Calculate high resolution j_toroidal_total and j_toroidal_ohmic
  j_toroidal_ohmic_formula_hires = (
      1 - geo.rho_hires_norm**2
  ) ** runtime_params.profile_conditions.current_profile_nu
  denom = _trapz(
      j_toroidal_ohmic_formula_hires * geo.spr_hires, geo.rho_hires_norm
  )
  I_non_inductive = math_utils.area_integration(
      j_toroidal_external + j_toroidal_bootstrap, geo
  )
  I_ohmic = runtime_params.profile_conditions.I_p - I_non_inductive
  C_ohm_hires = I_ohmic / denom
  j_toroidal_ohmic_hires = j_toroidal_ohmic_formula_hires * C_ohm_hires
  j_toroidal_total_hires = (
      j_toroidal_ohmic_hires
      + j_toroidal_external_hires
      + j_toroidal_bootstrap_hires
  )
  return j_toroidal_total_hires
