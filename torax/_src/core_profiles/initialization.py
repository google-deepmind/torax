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
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import getters
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles import runtime_params as core_profile_runtime_params
from torax._src.core_profiles import (
    solver_profiles as solver_profiles_lib,
)
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_model as neoclassical_model_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.physics import psi_calculations
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def initial_solver_state(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    source_models: source_models_lib.SourceModels,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    source_states: Mapping[str, source_state_lib.SourceState],
) -> solver_state_lib.SolverState:
  """Calculates the initial SolverState.

  Args:
    runtime_params: Runtime parameters at t=t_initial.
    geo: Torus geometry at t=t_initial.
    source_models: All models for TORAX sources/sinks.
    neoclassical_model: The neoclassical model.
    source_states: Initial source states mapping.

  Returns:
    Initial SolverState.
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
  toroidal_angular_velocity = getters.get_updated_toroidal_angular_velocity(
      runtime_params.profile_conditions, geo
  )
  psi = _init_psi(
      runtime_params=runtime_params,
      geo=geo,
      T_i=T_i,
      T_e=T_e,
      n_e=n_e,
      toroidal_angular_velocity=toroidal_angular_velocity,
      source_models=source_models,
      neoclassical_model=neoclassical_model,
      source_states=source_states,
  )
  return solver_state_lib.SolverState(
      T_i=T_i,
      T_e=T_e,
      psi=psi,
      n_e=n_e,
      toroidal_angular_velocity=toroidal_angular_velocity,
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

  return getters.build_psi_cell_variable(
      psi_value=psi_value,
      I_p=I_p,
      geo=geo,
      use_v_loop_lcfs_boundary_condition=use_v_loop_lcfs_boundary_condition,
  )


def _init_psi(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    T_i: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    n_e: cell_variable.CellVariable,
    toroidal_angular_velocity: cell_variable.CellVariable,
    source_models: source_models_lib.SourceModels,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    source_states: Mapping[str, source_state_lib.SourceState],
) -> cell_variable.CellVariable:
  """Initialises psi consistent with profile conditions, geometry, or current.

  There are three modes of doing this that are supported:
    1. Retrieving psi from the profile conditions.
    2. Retrieving psi from the standard geometry input.
    3. Calculating j according to the nu formula and then
    calculating psi from that. As we are calculating j using a guess for psi,
    this method is iterated to converge to the true psi.

  Args:
    runtime_params: Runtime parameters.
    geo: Torus geometry.
    T_i: Ion temperature [keV].
    T_e: Electron temperature [keV].
    n_e: Electron density [m^-3].
    toroidal_angular_velocity: Toroidal angular velocity [rad/s].
    source_models: All TORAX source/sink functions.
    neoclassical_model: The neoclassical model.
    source_states: Initial source states mapping.

  Returns:
    Initial poloidal flux [Wb].
  """

  match runtime_params.profile_conditions.psi_source:
    # Case 1 & 2: retrieving prescribed psi from profile conditions or geometry.
    case (
        core_profile_runtime_params.PsiSource.PROFILE_CONDITIONS
        | core_profile_runtime_params.PsiSource.GEOMETRY
    ):
      return getters.get_prescribed_psi(runtime_params.profile_conditions, geo)

    # Case 3: calculating j according to nu formula and psi from j.
    case core_profile_runtime_params.PsiSource.J:
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
          use_v_loop_lcfs_boundary_condition=(
              runtime_params.profile_conditions.psi_bc_source
              == core_profile_runtime_params.PsiBoundaryConditionSource.V_LOOP
          ),
      )
      if not (runtime_params.profile_conditions.initial_j_is_total_current):
        # In this branch we require non-inductive currents to determine
        # j_toroidal_total. The nu formula only provides the Ohmic component of
        # the current. However calculating non-inductive currents requires a
        # non-zero psi. We thus iterate between psi and source calculations,
        # using j_toroidal_total and psi calculated purely with the nu formula
        # as an initial guess.
        solver_state = solver_state_lib.SolverState(
            T_i=T_i,
            T_e=T_e,
            psi=psi,
            n_e=n_e,
            toroidal_angular_velocity=toroidal_angular_velocity,
        )

        # TODO(b/440385263): add tunable iteration number or convergence
        # criteria, and modify python for loop to jax fori loop for the general
        # case.

        # Iterate with non-inductive current source calculations. Stop after 2.
        psi = _iterate_psi_and_sources(
            runtime_params=runtime_params,
            geo=geo,
            solver_state=solver_state,
            neoclassical_model=neoclassical_model,
            source_models=source_models,
            source_states=source_states,
            iterations=2,
        )
      return psi

    case _:
      raise ValueError(
          'Unknown psi_source:'
          f' {runtime_params.profile_conditions.psi_source}'
      )


def _iterate_psi_and_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: solver_state_lib.SolverState,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    iterations: int,
) -> cell_variable.CellVariable:
  """Iterates psi and sources to converge to a consistent state."""
  match runtime_params.profile_conditions.psi_bc_source:
    case core_profile_runtime_params.PsiBoundaryConditionSource.I_P:
      v_loop_lcfs = jnp.asarray(0.0, dtype=jax_utils.get_dtype())
    case core_profile_runtime_params.PsiBoundaryConditionSource.V_LOOP:
      v_loop_lcfs = jnp.asarray(
          runtime_params.profile_conditions.v_loop_lcfs,
          dtype=jax_utils.get_dtype(),
      )
    case _:
      raise ValueError(
          'Unknown psi_bc_source:'
          f' {runtime_params.profile_conditions.psi_bc_source}'
      )
  source_states_i = source_states
  for _ in range(iterations):
    base_profiles = solver_profiles_lib.SolverProfiles.from_solver_state(
        runtime_params=runtime_params,
        geo=geo,
        solver_state=solver_state,
        v_loop_lcfs=v_loop_lcfs,
    )
    basic_source_outputs, source_states_i = (
        source_profile_builders.evaluate_basic_sources(
            runtime_params=runtime_params,
            geo=geo,
            solver_profiles=base_profiles,
            source_models=source_models,
            source_states=source_states_i,
            explicit=True,
            calculate_anyway=True,
        )
    )
    if runtime_params.numerics.enable_fast_ions:
      fast_ions = profile_conditions_lib.apply_prescribed_fast_ions(
          source_profiles.extract_fast_ions(basic_source_outputs),
          runtime_params.profile_conditions.prescribed_fast_ions,
          geo.rho_face_norm,
      )
    else:
      fast_ions = ()
    _, bootstrap_current, _, _ = (
        neoclassical_model.calculate_conductivity_current_and_rotation(
            runtime_params, geo, base_profiles, fast_ions=fast_ions
        )
    )
    psi_sources = source_profiles.extract_psi_sources(
        basic_source_outputs, source_models
    )
    j_toroidal_external = psi_calculations.j_parallel_to_j_toroidal(
        sum(psi_sources.values(), start=jnp.zeros_like(geo.rho)),
        geo,
        runtime_params.numerics.min_rho_norm,
    )
    j_toroidal_total_hires = get_j_toroidal_total_hires_with_external_sources(
        runtime_params=runtime_params,
        geo=geo,
        bootstrap_current=bootstrap_current,
        j_toroidal_external=j_toroidal_external,
    )
    use_v_loop = (
        runtime_params.profile_conditions.psi_bc_source
        == core_profile_runtime_params.PsiBoundaryConditionSource.V_LOOP
    )
    psi = update_psi_from_j(
        runtime_params.profile_conditions.I_p,
        geo,
        j_toroidal_total_hires,
        use_v_loop_lcfs_boundary_condition=use_v_loop,
    )
    solver_state = dataclasses.replace(
        solver_state,
        psi=psi,
    )
  return solver_state.psi


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
