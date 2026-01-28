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

import dataclasses

import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import getters
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.physics import psi_calculations
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name


@jax.jit(static_argnames='evolving_names')
def update_core_profiles_during_step(
    x_new: tuple[cell_variable.CellVariable, ...],
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> state.CoreProfiles:
  """Returns the new core profiles after updating the evolving variables.

  Intended for use during iterative solves in the step function. Only updates
  the core profiles which are being evolved by the PDE and directly derivable
  quantities like q_face, s_face. core_profile calculations which require
  sources are not updated.

  Args:
    x_new: The new values of the evolving variables.
    runtime_params: The runtime params slice.
    geo: Magnetic geometry.
    core_profiles: The old set of core plasma profiles.
    evolving_names: The names of the evolving variables.
  """

  updated_core_profiles = convertors.solver_x_tuple_to_core_profiles(
      x_new, evolving_names, core_profiles
  )

  ions = getters.get_updated_ions(
      runtime_params,
      geo,
      updated_core_profiles.n_e,
      updated_core_profiles.T_e,
  )

  return dataclasses.replace(
      updated_core_profiles,
      n_i=ions.n_i,
      n_impurity=ions.n_impurity,
      impurity_fractions=ions.impurity_fractions,
      main_ion_fractions=ions.main_ion_fractions,
      Z_i=ions.Z_i,
      Z_i_face=ions.Z_i_face,
      Z_impurity=ions.Z_impurity,
      Z_impurity_face=ions.Z_impurity_face,
      A_i=ions.A_i,
      A_impurity=ions.A_impurity,
      A_impurity_face=ions.A_impurity_face,
      Z_eff=ions.Z_eff,
      Z_eff_face=ions.Z_eff_face,
      q_face=psi_calculations.calc_q_face(geo, updated_core_profiles.psi),
      s_face=psi_calculations.calc_s_face(geo, updated_core_profiles.psi),
      charge_state_info=ions.charge_state_info,
      charge_state_info_face=ions.charge_state_info_face,
  )


def update_core_and_source_profiles_after_step(
    dt: array_typing.FloatScalar,
    x_new: tuple[cell_variable.CellVariable, ...],
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    evolving_names: tuple[str, ...],
) -> tuple[state.CoreProfiles, source_profiles_lib.SourceProfiles]:
  """Returns a core profiles and source profiles after the solver has finished.

  Updates the evolved variables and derived variables like q_face, psidot, etc.

  Args:
    dt: The size of the last timestep.
    x_new: The new values of the evolving variables.
    runtime_params_t_plus_dt: The runtime params slice at t=t_plus_dt.
    geo: Magnetic geometry.
    core_profiles_t: The old set of core plasma profiles.
    core_profiles_t_plus_dt: The partially new set of core plasma profiles. On
      input into this function, all prescribed profiles and used boundary
      conditions are already set. But evolving values are not.
    explicit_source_profiles: The explicit source profiles.
    source_models: The source models.
    neoclassical_models: The neoclassical models.
    evolving_names: The names of the evolving variables.

  Returns:
    A tuple of the new core profiles and the source profiles.
  """

  updated_core_profiles_t_plus_dt = convertors.solver_x_tuple_to_core_profiles(
      x_new, evolving_names, core_profiles_t_plus_dt
  )

  ions = getters.get_updated_ions(
      runtime_params_t_plus_dt,
      geo,
      updated_core_profiles_t_plus_dt.n_e,
      updated_core_profiles_t_plus_dt.T_e,
  )

  v_loop_lcfs = (
      runtime_params_t_plus_dt.profile_conditions.v_loop_lcfs  # pylint: disable=g-long-ternary
      if runtime_params_t_plus_dt.profile_conditions.use_v_loop_lcfs_boundary_condition
      else psi_calculations.calculate_v_loop_lcfs_from_psi(
          core_profiles_t.psi,
          updated_core_profiles_t_plus_dt.psi,
          dt,
      )
  )

  j_total, j_total_face, Ip_profile_face = psi_calculations.calc_j_total(
      geo,
      updated_core_profiles_t_plus_dt.psi,
      runtime_params_t_plus_dt.numerics.min_rho_norm,
  )

  # A wholly new core profiles object is defined as a guard against neglecting
  # to update one of the attributes if doing dataclasses.replace
  intermediate_core_profiles = state.CoreProfiles(
      T_i=updated_core_profiles_t_plus_dt.T_i,
      T_e=updated_core_profiles_t_plus_dt.T_e,
      psi=updated_core_profiles_t_plus_dt.psi,
      n_e=updated_core_profiles_t_plus_dt.n_e,
      n_i=ions.n_i,
      n_impurity=ions.n_impurity,
      impurity_fractions=ions.impurity_fractions,
      main_ion_fractions=ions.main_ion_fractions,
      Z_i=ions.Z_i,
      Z_i_face=ions.Z_i_face,
      Z_impurity=ions.Z_impurity,
      Z_impurity_face=ions.Z_impurity_face,
      psidot=core_profiles_t_plus_dt.psidot,
      q_face=psi_calculations.calc_q_face(
          geo, updated_core_profiles_t_plus_dt.psi
      ),
      s_face=psi_calculations.calc_s_face(
          geo, updated_core_profiles_t_plus_dt.psi
      ),
      A_i=ions.A_i,
      A_impurity=ions.A_impurity,
      A_impurity_face=ions.A_impurity_face,
      Z_eff=ions.Z_eff,
      Z_eff_face=ions.Z_eff_face,
      v_loop_lcfs=v_loop_lcfs,
      sigma=core_profiles_t_plus_dt.sigma,  # Not yet updated
      sigma_face=core_profiles_t_plus_dt.sigma_face,  # Not yet updated
      j_total=j_total,
      j_total_face=j_total_face,
      Ip_profile_face=Ip_profile_face,
      toroidal_angular_velocity=updated_core_profiles_t_plus_dt.toroidal_angular_velocity,
      charge_state_info=ions.charge_state_info,
      charge_state_info_face=ions.charge_state_info_face,
  )

  conductivity = neoclassical_models.conductivity.calculate_conductivity(
      geo, intermediate_core_profiles
  )

  intermediate_core_profiles = dataclasses.replace(
      intermediate_core_profiles,
      sigma=conductivity.sigma,
      sigma_face=conductivity.sigma_face,
  )

  # build_source_profiles calculates the union with explicit + implicit
  total_source_profiles = source_profile_builders.build_source_profiles(
      runtime_params=runtime_params_t_plus_dt,
      geo=geo,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
      core_profiles=intermediate_core_profiles,
      explicit=False,
      explicit_source_profiles=explicit_source_profiles,
      conductivity=conductivity,
  )

  if (
      not runtime_params_t_plus_dt.numerics.evolve_current
      and runtime_params_t_plus_dt.profile_conditions.psidot is not None
  ):
    # If psidot is prescribed and current does not evolve, use prescribed value
    psidot_value = runtime_params_t_plus_dt.profile_conditions.psidot
  else:
    # Otherwise, calculate psidot from psi sources.
    psi_sources = total_source_profiles.total_psi_sources(geo)
    psidot_value = psi_calculations.calculate_psidot_from_psi_sources(
        psi_sources=psi_sources,
        sigma=intermediate_core_profiles.sigma,
        resistivity_multiplier=runtime_params_t_plus_dt.numerics.resistivity_multiplier,
        psi=intermediate_core_profiles.psi,
        geo=geo,
    )
  psidot = dataclasses.replace(
      core_profiles_t_plus_dt.psidot,
      value=psidot_value,
      right_face_constraint=v_loop_lcfs,
      right_face_grad_constraint=None,
  )

  core_profiles_t_plus_dt = dataclasses.replace(
      intermediate_core_profiles,
      psidot=psidot,
  )
  return core_profiles_t_plus_dt, total_source_profiles


def provide_core_profiles_t_plus_dt(
    dt: jax.Array,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> state.CoreProfiles:
  """Provides state at t_plus_dt with new boundary conditions and prescribed profiles."""
  numerics = runtime_params_t.numerics
  profile_conditions_t_plus_dt = runtime_params_t_plus_dt.profile_conditions
  # In each of the following if the profile condition is being evolved then we
  # only need to update the boundary condition and can get the value from the
  # previous timestep. Otherwise, we need to update the value and boundary
  # condition.
  psi = getters.get_updated_psi(
      profile_conditions_t_plus_dt,
      geo_t_plus_dt,
      dt=dt,
      theta=runtime_params_t_plus_dt.solver.theta_implicit,
      only_boundary_condition=numerics.evolve_current,
      original_psi=core_profiles_t.psi,
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

  # Update Z_face with boundary condition Z, needed for cases where T_
  # pylint: enable=invalid-name
  core_profiles_t_plus_dt = dataclasses.replace(
      core_profiles_t,
      T_i=T_i,
      T_e=T_e,
      psi=psi,
      n_e=n_e,
      toroidal_angular_velocity=toroidal_angular_velocity,
  )
  return core_profiles_t_plus_dt
