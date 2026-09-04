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
"""Contains the main programmatic entry point for running a TORAX simulation.

The intended use is
```
torax_config = torax.ToraxConfig.from_dict(config_dict)
sim_outputs = torax.run_simulation(torax_config)

# Update the config to run a new simulation with different parameters.
torax_config.update(updated_fields)
new_sim_outputs = torax.run_simulation(torax_config)
```
"""

from absl import logging
from torax._src.config import build_runtime_params
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import jit_run_loop
from torax._src.orchestration import run_loop
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax._src.torax_pydantic import model_config
import xarray as xr

# Upper bound on the number of geometries to precompute, to bound memory usage
# (each geometry is ~13 kB at n_rho=25, scaling linearly with n_rho).
_MAX_PRECOMPUTED_GEOMETRIES: int = 10_000


def _maybe_precompute_geometry_provider(
    torax_config: model_config.ToraxConfig,
    geometry_provider: geometry_provider_lib.GeometryProvider,
) -> geometry_provider_lib.GeometryProvider:
  """Pre-interpolates a time-dependent geometry when all step times are known.

  This is only possible when the fixed time step calculator is used with a
  constant `fixed_dt` and nothing can alter the sequence of time steps
  (adaptive dt, sawtooth crashes, or a restart from a different start time).

  Args:
    torax_config: The TORAX config.
    geometry_provider: The geometry provider built from the config.

  Returns:
    A `PrecomputedGeometryProvider` if precomputation is possible, otherwise
    `geometry_provider` unchanged.
  """
  numerics = torax_config.numerics
  fixed_dt = numerics.fixed_dt
  if (
      isinstance(
          geometry_provider, geometry_provider_lib.ConstantGeometryProvider
      )
      or torax_config.time_step_calculator.calculator_type
      != time_step_calculator_pydantic_model.TimeStepCalculatorType.FIXED
      or numerics.adaptive_dt
      or torax_config.mhd.sawtooth is not None
      or (torax_config.restart is not None and torax_config.restart.do_restart)
      or len(fixed_dt.value) != 1
      or fixed_dt.value[0] <= 0.0
  ):
    return geometry_provider

  times = fixed_time_step_calculator.get_time_grid(
      t_initial=numerics.t_initial,
      t_final=numerics.t_final,
      fixed_dt=float(fixed_dt.value[0]),
      exact_t_final=numerics.exact_t_final,
      tolerance=torax_config.time_step_calculator.tolerance,
      max_num_times=_MAX_PRECOMPUTED_GEOMETRIES,
  )
  if times is None:
    logging.info(
        'Not precomputing geometries: the grid exceeds the maximum of %d.',
        _MAX_PRECOMPUTED_GEOMETRIES,
    )
    return geometry_provider

  logging.info('Precomputing geometries at %d fixed time steps.', len(times))
  return geometry_provider_lib.PrecomputedGeometryProvider.from_provider(
      geometry_provider, times
  )


def make_step_fn(
    torax_config: model_config.ToraxConfig,
) -> step_function.SimulationStepFn:
  """Prepare a TORAX step function from a config."""
  geometry_provider = _maybe_precompute_geometry_provider(
      torax_config, torax_config.geometry.build_provider
  )
  models = torax_config.build_models()

  solver = torax_config.solver.build_solver(
      models=models,
  )

  runtime_params_provider = (
      build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
  )

  return step_function.SimulationStepFn(
      solver=solver,
      time_step_calculator=models.time_step_calculator,
      geometry_provider=geometry_provider,
      runtime_params_provider=runtime_params_provider,
  )


def prepare_simulation(
    torax_config: model_config.ToraxConfig,
) -> tuple[
    sim_state.SimState,
    post_processing.PostProcessedOutputs,
    step_function.SimulationStepFn,
]:
  """Prepare a TORAX simulation returning the necessary inputs for the run loop.

  Args:
    torax_config: The TORAX config to use for the simulation.

  Returns:
    A tuple containing:
      - The initial state.
      - The initial post processed outputs.
      - The simulation step function.
  """
  step_fn = make_step_fn(torax_config)

  if torax_config.restart and torax_config.restart.do_restart:
    initial_state, post_processed_outputs = (
        initial_state_lib.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=torax_config.restart,
            step_fn=step_fn,
        )
    )
  else:
    initial_state, post_processed_outputs = (
        initial_state_lib.get_initial_state_and_post_processed_outputs(
            step_fn=step_fn,
        )
    )

  return (
      initial_state,
      post_processed_outputs,
      step_fn,
  )


def run_simulation(
    torax_config: model_config.ToraxConfig,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
    max_steps: int | None = None,
    _use_jitted_run_loop: bool = False,  # pylint: disable=invalid-name
) -> tuple[xr.DataTree, output.StateHistory]:
  """Runs a TORAX simulation using the config and returns the outputs.

  Args:
    torax_config: The TORAX config to use for the simulation.
    log_timestep_info: Whether to log the timestep information.
    progress_bar: Whether to show a progress bar.
    max_steps: The maximum number of steps to take, if not provided, then the
      simulation will run until the maximum time is reached.
    _use_jitted_run_loop: If True, then a jitted run loop will be used. A
      temporary private argument used for testing.

  Returns:
    A tuple of the simulation outputs in the form of a DataTree and the state
    history which is intended for helpful use with debugging as it contains
    the `CoreProfiles`, `CoreTransport`, `CoreSources`, `Geometry`, and
    `PostProcessedOutputs` dataclasses for each step of the simulation.
  """

  (
      initial_state,
      post_processed_outputs,
      step_fn,
  ) = prepare_simulation(torax_config)

  if _use_jitted_run_loop:
    state_history, post_processed_outputs_history, sim_error = (
        jit_run_loop.run_loop(
            step_fn,
            max_steps=max_steps,
            log_timestep_info=log_timestep_info,
            progress_bar=progress_bar,
        )
    )
  else:
    state_history, post_processed_outputs_history, sim_error = (
        run_loop.run_loop(
            initial_state=initial_state,
            initial_post_processed_outputs=post_processed_outputs,
            step_fn=step_fn,
            log_timestep_info=log_timestep_info,
            progress_bar=progress_bar,
            max_steps=max_steps,
        )
    )

  state_history = output.StateHistory(
      state_history=state_history,
      post_processed_outputs_history=post_processed_outputs_history,
      sim_error=sim_error,
      torax_config=torax_config,
  )

  return (
      state_history.simulation_output_to_xr(),
      state_history,
  )
