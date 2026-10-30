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

Routines for providing SolverState at candidate timesteps.
"""

from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import getters
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.geometry import geometry

# pylint: disable=invalid-name


def provide_solver_state_t_plus_dt(
    dt: array_typing.FloatScalar,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> solver_state_lib.SolverState:
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
      original_psi=core_profiles_t.flux.psi,
      current_diffusion_time_reduction_factor=time_reduction_factor,
  )
  T_i = getters.get_updated_ion_temperature(
      profile_conditions_t_plus_dt,
      geo_t_plus_dt,
      only_boundary_condition=numerics.evolve_ion_heat,
      original_T_i_value=core_profiles_t.kinetic.T_i,
  )
  T_e = getters.get_updated_electron_temperature(
      profile_conditions_t_plus_dt,
      geo_t_plus_dt,
      only_boundary_condition=numerics.evolve_electron_heat,
      original_T_e_value=core_profiles_t.kinetic.T_e,
  )
  n_e = getters.get_updated_electron_density(
      profile_conditions_t_plus_dt,
      geo_t_plus_dt,
      only_boundary_condition=numerics.evolve_density,
      original_n_e_value=core_profiles_t.kinetic.n_e,
  )
  toroidal_angular_velocity = getters.get_updated_toroidal_angular_velocity(
      profile_conditions_t_plus_dt, geo_t_plus_dt
  )

  return solver_state_lib.SolverState(
      T_i=T_i,
      T_e=T_e,
      psi=psi,
      n_e=n_e,
      toroidal_angular_velocity=toroidal_angular_velocity,
  )


# pylint: enable=invalid-name
