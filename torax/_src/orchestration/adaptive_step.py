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
"""Functions for calculating adaptive step function with whilei_loop."""
import dataclasses

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import model_state as model_state_lib
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.core_profiles import updaters
from torax._src.edge import base as edge_base
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration import explicit_model_outputs as explicit_model_outputs_lib
from torax._src.orchestration import sim_state
from torax._src.solver import solver as solver_lib
from torax._src.time_step_calculator import time_step_calculator as time_step_calculator_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AdaptiveStepState:
  x_new: tuple[cell_variable.CellVariable, ...]
  dt: chex.Numeric
  solver_numeric_outputs: state.SolverNumericOutputs
  runtime_params: runtime_params_lib.RuntimeParams
  geo: geometry.Geometry
  solver_state: solver_state_lib.SolverState


def create_initial_state(
    input_state: sim_state.SimState,
    evolving_names: tuple[str, ...],
    initial_dt: chex.Numeric,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    geo_t: geometry.Geometry,
) -> AdaptiveStepState:
  """Creates the initial state for the adaptive step."""
  solver_state_t = solver_state_lib.SolverState.from_core_profiles(
      input_state.core_profiles
  )
  initial_x_new = convertors.solver_state_to_solver_x_tuple(
      solver_state_t, evolving_names
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
      solver_state=solver_state_t,
  )


def compute_state(
    i: array_typing.IntScalar,
    loop_statistics: dict[str, array_typing.IntScalar],
    initial_dt: array_typing.FloatScalar,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    geo_t: geometry.Geometry,
    input_state: sim_state.SimState,
    explicit_model_outputs: explicit_model_outputs_lib.ExplicitModelOutputs,
    edge_outputs: edge_base.EdgeModelOutputs | None,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    model_state_t: model_state_lib.ModelState,
    solver: solver_lib.Solver,
    time_step_calculator: time_step_calculator_lib.TimeStepCalculator,
) -> tuple[AdaptiveStepState, dict[str, array_typing.IntScalar]]:
  """Computes the state for attempt i of the adaptive step."""
  dt = time_step_calculator.backtrack_dt(
      initial_dt=initial_dt,
      attempt=i,
      runtime_params=runtime_params_t,
  )
  runtime_params_t_plus_dt, geo_t_plus_dt = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=input_state.t + dt,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
          edge_outputs=edge_outputs,
          core_profiles=input_state.core_profiles,
      )
  )
  solver_state_t_plus_dt = (
      updaters.provide_solver_state_t_plus_dt(
          dt=dt,
          runtime_params_t=runtime_params_t,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=input_state.core_profiles,
      )
  )
  x_new, solver_numeric_outputs = solver(
      t=input_state.t,
      dt=dt,
      runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t=geo_t,
      geo_t_plus_dt=geo_t_plus_dt,
      core_profiles_t=input_state.core_profiles,
      solver_state_t_plus_dt=solver_state_t_plus_dt,
      explicit_model_outputs=explicit_model_outputs,
      model_state=model_state_t,
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
          solver_state_t_plus_dt,
      ),
      loop_statistics,
  )


def cond_fun(
    inputs: AdaptiveStepState,
    unused_initial_dt: chex.Numeric,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    unused_geo_t: geometry.Geometry,
    input_state: sim_state.SimState,
    unused_explicit_model_outputs: (
        explicit_model_outputs_lib.ExplicitModelOutputs
    ),
    unused_edge_outputs: edge_base.EdgeModelOutputs | None,
    unused_runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    unused_geometry_provider: geometry_provider_lib.GeometryProvider,
    unused_model_state_t: model_state_lib.ModelState,
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
  if runtime_params_t.time_step_calculator.exact_t_final:
    at_exact_t_final = jnp.allclose(
        input_state.t + next_dt,
        runtime_params_t.time_step_calculator.t_final,
    )
  else:
    at_exact_t_final = jnp.array(False)

  next_dt_too_small = next_dt < runtime_params_t.time_step_calculator.min_dt

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
