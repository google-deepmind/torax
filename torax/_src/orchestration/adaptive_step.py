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
"""Functions for calculating adaptive step for step function with whilei_loop."""
import dataclasses

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.edge import base as edge_base
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration import sim_state
from torax._src.solver import solver as solver_lib
from torax._src.sources import source_profiles as source_profiles_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AdaptiveStepState:
  x_new: tuple[cell_variable.CellVariable, ...]
  dt: chex.Numeric
  solver_numeric_outputs: state.SolverNumericOutputs
  runtime_params: runtime_params_lib.RuntimeParams
  geo: geometry.Geometry
  core_profiles: state.CoreProfiles


def create_initial_state(
    input_state: sim_state.ToraxSimState,
    evolving_names: tuple[str, ...],
    initial_dt: chex.Numeric,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    geo_t: geometry.Geometry,
) -> AdaptiveStepState:
  """Creates the initial state for the adaptive step."""
  initial_x_new = convertors.core_profiles_to_solver_x_tuple(
      input_state.core_profiles, evolving_names
  )
  initial_solver_numeric_outputs = state.SolverNumericOutputs(
      # The solver has not converged yet as we have not performed
      # any steps yet.
      solver_error_state=jnp.array(1, jax_utils.get_int_dtype()),
      outer_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
      inner_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
      sawtooth_crash=False,
  )
  return AdaptiveStepState(
      x_new=initial_x_new,
      dt=initial_dt,
      solver_numeric_outputs=initial_solver_numeric_outputs,
      runtime_params=runtime_params_t,
      geo=geo_t,
      core_profiles=input_state.core_profiles,
  )


def compute_state(
    i: chex.Numeric,
    loop_statistics: dict[str, array_typing.IntScalar],
    initial_dt: chex.Numeric,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    geo_t: geometry.Geometry,
    input_state: sim_state.ToraxSimState,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    edge_outputs: edge_base.EdgeModelOutputs | None,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    solver: solver_lib.Solver,
) -> tuple[AdaptiveStepState, dict[str, array_typing.IntScalar]]:
  """Computes the state for attempt i of the adaptive step."""
  dt = initial_dt / runtime_params_t.numerics.dt_reduction_factor**i
  runtime_params_t_plus_dt, geo_t_plus_dt = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=input_state.t + dt,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
          edge_outputs=edge_outputs,
      )
  )
  core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
      dt=dt,
      runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      core_profiles_t=input_state.core_profiles,
  )

  pedestal_policy = solver.physics_models.pedestal_model.pedestal_policy
  pedestal_policy_state_t = input_state.pedestal_policy_state
  pedestal_policy_state_t_plus_dt = pedestal_policy.update(
      t=input_state.t + dt,
      runtime_params=runtime_params_t_plus_dt.pedestal_policy,
  )

  # The solver returned state is still "intermediate" since the CoreProfiles
  # need to be updated by the evolved CellVariables in x_new
  x_new, solver_numeric_outputs = solver(
      t=input_state.t,
      dt=dt,
      runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t=geo_t,
      geo_t_plus_dt=geo_t_plus_dt,
      core_profiles_t=input_state.core_profiles,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      pedestal_policy_state_t=pedestal_policy_state_t,
      pedestal_policy_state_t_plus_dt=pedestal_policy_state_t_plus_dt,
  )
  loop_statistics[
      'inner_solver_iterations'
  ] += solver_numeric_outputs.inner_solver_iterations

  return (
      AdaptiveStepState(
          x_new,
          dt,
          solver_numeric_outputs,
          runtime_params_t_plus_dt,
          geo_t_plus_dt,
          core_profiles_t_plus_dt,
      ),
      loop_statistics,
  )


def cond_fun(
    inputs: AdaptiveStepState,
    unused_initial_dt: chex.Numeric,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    unused_geo_t: geometry.Geometry,
    input_state: sim_state.ToraxSimState,
    unused_explicit_source_profiles: source_profiles_lib.SourceProfiles,
    unused_edge_outputs: edge_base.EdgeModelOutputs | None,
    unused_runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    unused_geometry_provider: geometry_provider_lib.GeometryProvider,
) -> array_typing.BoolScalar:
  """Condition function for the adaptive step to keep stepping."""
  solver_outputs = inputs.solver_numeric_outputs
  next_dt = inputs.dt

  # Check for NaN in the next dt to avoid a recursive loop.
  is_nan_next_dt = jnp.isnan(next_dt)

  # If the solver did not converge we need to make a new step.
  solver_did_not_converge = solver_outputs.solver_error_state == 1

  # If t + dt  is exactly the final time we may need a smaller step than
  # min_dt to exactly reach the final time.
  if runtime_params_t.numerics.exact_t_final:
    at_exact_t_final = jnp.allclose(
        input_state.t + next_dt,
        runtime_params_t.numerics.t_final,
    )
  else:
    at_exact_t_final = jnp.array(False)

  next_dt_too_small = next_dt < runtime_params_t.numerics.min_dt

  take_another_step = jax.lax.cond(
      solver_did_not_converge,
      # If the solver did not converge then we check if we are at the exact
      # final time and should take a smaller step. If not we also check if
      # the next dt is too small, if so we should end the step.
      lambda: jax.lax.cond(
          at_exact_t_final, lambda: True, lambda: ~next_dt_too_small
      ),
      lambda: False,
  )

  return take_another_step & ~is_nan_next_dt
