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

"""Update routines for core_profiles.

Set of routines that relate to updating/creating core profiles from existing
core profiles.

Includes:
- get_prescribed_core_profile_values: Updates core profiles which are not being
  evolved by PDE.
- update_core_profiles_during_step: Intra-step updates of the evolving variables
  in core profiles during solver iterations.
- update_all_core_profiles_after_step: Updates all core_profiles after a step.
  Includes the evolved variables and derived variables like q_face, psidot, etc.
- compute_boundary_conditions_for_t_plus_dt: Computes boundary conditions for
  the next timestep and returns updates to State.
"""

from collections.abc import Mapping
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import builders
from torax._src.core_profiles import convertors
from torax._src.core_profiles import getters
from torax._src.core_profiles import profile_conditions
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_intermediates as neoclassical_intermediates_lib
from torax._src.neoclassical import neoclassical_model as neoclassical_model_lib
from torax._src.physics import psi_calculations
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.sources import source_state as source_state_lib

# pylint: disable=invalid-name


# TODO(b/572586768): Remove once calc_coeffs accepts SolverState directly.
@jax.jit(static_argnames='evolving_names')
def update_core_profiles_during_step(
    x_new: tuple[cell_variable.CellVariable, ...],
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: state.SolverState,
    evolving_names: tuple[str, ...],
) -> state.CoreProfiles:
  """Returns the new core profiles after updating the evolving variables.

  Intended for use during iterative solves in the step function. Only updates
  the primary evolving profiles and quantities derived directly from them
  and geometry (e.g., ion density fractions, Z_eff, q_face, s_face,
  j_toroidal_total, and I_p_profile_face). Secondary quantities that require
  evaluating source profiles or spatial Ohm's law (such as psidot,
  j_toroidal_ohmic, and electron-ion heat exchange qei) are not recomputed
  here during solver iterations.

  Args:
    x_new: The new values of the evolving variables.
    runtime_params: The runtime params slice.
    geo: Magnetic geometry.
    solver_state: The primary profiles for this timestep.
    evolving_names: The names of the evolving variables.

  Returns:
    A newly constructed CoreProfiles object.
  """
  updated_solver_state = convertors.solver_x_tuple_to_solver_state(
      x_new, evolving_names, solver_state
  )

  base = builders.build_base_core_profiles(
      runtime_params, geo, updated_solver_state
  )

  # TODO(b/572586768): Remove transitional partial CoreProfiles once
  # physics_pipeline retires updaters.
  return state.CoreProfiles(
      kinetic=base.kinetic,
      ions=base.ions,
      flux=base.flux,
      toroidal_angular_velocity=base.toroidal_angular_velocity,
      fast_ions=(),
      conductivity=state.Conductivity.zeros(geo),
      poloidal_velocity=cell_variable.CellVariable(
          value=jnp.zeros_like(geo.rho_norm),
          face_centers=geo.rho_face_norm,
      ),
      psidot=cell_variable.CellVariable(
          value=jnp.zeros_like(geo.rho_norm),
          face_centers=geo.rho_face_norm,
      ),
  )


def update_core_and_source_profiles_after_step(
    dt: array_typing.FloatScalar,
    x_new: tuple[cell_variable.CellVariable, ...],
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
    solver_state_t_plus_dt: state.SolverState,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    evolving_names: tuple[str, ...],
    source_states: Mapping[str, source_state_lib.SourceState],
) -> tuple[
    state.CoreProfiles,
    source_profiles_lib.SourceProfiles,
    dict[str, source_state_lib.SourceState],
    neoclassical_intermediates_lib.NeoclassicalIntermediates,
]:
  """Updates core profiles, source profiles, and source states after a step.

  Updates the evolved variables and derived variables like q_face, psidot, etc.

  Args:
    dt: The size of the last timestep.
    x_new: The new values of the evolving variables.
    runtime_params_t_plus_dt: The runtime params slice at t=t_plus_dt.
    geo: Magnetic geometry.
    core_profiles_t: The old set of core plasma profiles.
    solver_state_t_plus_dt: The primary profiles at t + dt with prescribed
      quantities and boundary conditions.
    explicit_source_profiles: The explicit source profiles.
    source_models: The source models.
    neoclassical_model: The neoclassical model.
    evolving_names: The names of the evolving variables.
    source_states: Source state carried over from the previous timestep.

  Returns:
    Updated core profiles, source profiles, source states, and neoclassical
    intermediates.
  """

  updated_solver_state = convertors.solver_x_tuple_to_solver_state(
      x_new, evolving_names, solver_state_t_plus_dt
  )

  time_reduction_factor = (
      runtime_params_t_plus_dt.numerics.current_diffusion_time_reduction_factor
  )
  v_loop_lcfs = (
      runtime_params_t_plus_dt.profile_conditions.v_loop_lcfs  # pylint: disable=g-long-ternary
      if runtime_params_t_plus_dt.profile_conditions.use_v_loop_lcfs_boundary_condition
      else psi_calculations.calculate_v_loop_lcfs_from_psi(
          core_profiles_t.psi,
          updated_solver_state.psi,
          dt,
          current_diffusion_time_reduction_factor=time_reduction_factor,
      )
  )

  base = builders.build_base_core_profiles(
      runtime_params_t_plus_dt,
      geo,
      updated_solver_state,
      v_loop_lcfs=v_loop_lcfs,
  )
  # TODO(b/572586768): Remove transitional partial CoreProfiles once
  # physics_pipeline retires updaters.
  intermediate_core_profiles = state.CoreProfiles(
      kinetic=base.kinetic,
      ions=base.ions,
      flux=base.flux,
      toroidal_angular_velocity=base.toroidal_angular_velocity,
      fast_ions=(),
      conductivity=state.Conductivity.zeros(geo),
      poloidal_velocity=cell_variable.CellVariable(
          value=jnp.zeros_like(geo.rho_norm),
          face_centers=geo.rho_face_norm,
      ),
      psidot=cell_variable.CellVariable(
          value=jnp.zeros_like(geo.rho_norm),
          face_centers=geo.rho_face_norm,
      ),
  )
  partial_source_profiles, updated_source_states = (
      source_profile_builders.build_basic_sources(
          runtime_params=runtime_params_t_plus_dt,
          geo=geo,
          core_profiles=intermediate_core_profiles,
          source_models=source_models,
          source_states=source_states,
          explicit=False,
          explicit_source_profiles=explicit_source_profiles,
      )
  )
  if runtime_params_t_plus_dt.numerics.enable_fast_ions:
    fast_ions = profile_conditions.apply_prescribed_fast_ions(
        sum(partial_source_profiles.fast_ions.values(), ()),
        runtime_params_t_plus_dt.profile_conditions.prescribed_fast_ions,
        geo.rho_face_norm,
    )
  else:
    fast_ions = ()
  intermediate_core_profiles = dataclasses.replace(
      intermediate_core_profiles,
      fast_ions=fast_ions,
  )
  (
      conductivity,
      bootstrap_current,
      poloidal_velocity,
      neoclassical_intermediates,
  ) = neoclassical_model.calculate_conductivity_current_and_rotation(
      runtime_params_t_plus_dt, geo, intermediate_core_profiles
  )
  psidot = psi_calculations.calculate_psidot(
      psi=intermediate_core_profiles.psi,
      sigma=conductivity.sigma,
      psi_sources=dataclasses.replace(
          partial_source_profiles, bootstrap_current=bootstrap_current
      ).total_psi_sources(geo),
      geo=geo,
      evolve_current=runtime_params_t_plus_dt.numerics.evolve_current,
      prescribed_psidot=runtime_params_t_plus_dt.profile_conditions.psidot,
      v_loop_lcfs=v_loop_lcfs,
  )
  core_profiles_t_plus_dt = dataclasses.replace(
      intermediate_core_profiles,
      conductivity=conductivity,
      poloidal_velocity=poloidal_velocity.v_pol,
      psidot=psidot,
  )

  total_source_profiles, next_source_states = (
      source_profile_builders.build_remaining_sources(
          runtime_params=runtime_params_t_plus_dt,
          geo=geo,
          core_profiles=core_profiles_t_plus_dt,
          source_models=source_models,
          source_states=updated_source_states,
          partial_source_profiles=partial_source_profiles,
          explicit=False,
          conductivity=conductivity,
          bootstrap_current=bootstrap_current,
      )
  )
  return (
      core_profiles_t_plus_dt,
      total_source_profiles,
      next_source_states,
      neoclassical_intermediates,
  )


def provide_solver_state_t_plus_dt(
    dt: array_typing.FloatScalar,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> state.SolverState:
  """Provides primary solver state at t + dt with updated BCs and prescribed."""
  numerics = runtime_params_t.numerics
  profile_conditions_t_plus_dt = runtime_params_t_plus_dt.profile_conditions
  # If a profile is being evolved, only its boundary condition is updated while
  # the value is preserved from the previous timestep. Otherwise, both the value
  # and the boundary condition are updated from prescribed profile conditions.
  time_reduction_factor = (
      runtime_params_t_plus_dt.numerics.current_diffusion_time_reduction_factor
  )
  psi = getters.get_updated_psi(
      profile_conditions_t_plus_dt,
      geo_t_plus_dt,
      dt=dt,
      theta=runtime_params_t_plus_dt.solver.theta_implicit,
      only_boundary_condition=numerics.evolve_current,
      original_psi=core_profiles_t.psi,
      current_diffusion_time_reduction_factor=time_reduction_factor,
  )
  T_i = getters.get_updated_ion_temperature(
      profile_conditions_t_plus_dt,
      geo_t_plus_dt,
      only_boundary_condition=numerics.evolve_ion_heat,
      original_T_i_value=core_profiles_t.T_i,
  )
  T_e = getters.get_updated_electron_temperature(
      profile_conditions_t_plus_dt,
      geo_t_plus_dt,
      only_boundary_condition=numerics.evolve_electron_heat,
      original_T_e_value=core_profiles_t.T_e,
  )
  n_e = getters.get_updated_electron_density(
      profile_conditions_t_plus_dt,
      geo_t_plus_dt,
      only_boundary_condition=numerics.evolve_density,
      original_n_e_value=core_profiles_t.n_e,
  )
  toroidal_angular_velocity = getters.get_updated_toroidal_angular_velocity(
      profile_conditions_t_plus_dt, geo_t_plus_dt
  )

  return state.SolverState(
      T_i=T_i,
      T_e=T_e,
      psi=psi,
      n_e=n_e,
      toroidal_angular_velocity=toroidal_angular_velocity,
  )


# pylint: enable=invalid-name
