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

from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import jit_run_loop
from torax._src.orchestration import run_loop
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.torax_pydantic import model_config
import xarray as xr


def make_step_fn(
    torax_config: model_config.ToraxConfig,
) -> step_function.SimulationStepFn:
  """Prepare a TORAX step function from a config."""
  geometry_provider = torax_config.geometry.build_provider
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
) -> tuple[xr.DataTree, output.StateHistory]:
  """Runs a TORAX simulation using the config and returns the outputs.

  Args:
    torax_config: The TORAX config to use for the simulation.
    log_timestep_info: Whether to log the timestep information.
    progress_bar: Whether to show a progress bar.
    max_steps: The maximum number of steps to take, if not provided, then the
      simulation will run until the maximum time is reached.

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

  state_history, post_processed_outputs_history, sim_error = run_loop.run_loop(
      initial_state=initial_state,
      initial_post_processed_outputs=post_processed_outputs,
      step_fn=step_fn,
      log_timestep_info=log_timestep_info,
      progress_bar=progress_bar,
      max_steps=max_steps,
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


def run_simulation_jitted(
    torax_config: model_config.ToraxConfig,
    max_steps: int | None = None,
) -> tuple[xr.DataTree, output.StateHistory]:
  """Runs a TORAX simulation using the config and returns the outputs.

  NOTE: This function doesn't guarantee that the simulation will complete. If
  the simulation does not complete successfully, then the state history will
  contain the error state `SimError.DID_NOT_REACH_T_FINAL` and a truncated
  simulation history.

  Args:
    torax_config: The TORAX config to use for the simulation.
    max_steps: The maximum number of steps to take, if not provided, then the
      maximum number of steps will be determined by the numerics.t_final and
      numerics.min_dt.

  Returns:
    A tuple of the simulation outputs in the form of a DataTree and the state
    history which is intended for helpful use with debugging as it contains
    the `CoreProfiles`, `CoreTransport`, `CoreSources`, `Geometry`, and
    `PostProcessedOutputs` dataclasses for each step of the simulation.
  """
  step_fn = make_step_fn(torax_config)
  states_history, post_processed_outputs_history, sim_error = (
      jit_run_loop.run_loop(
          step_fn,
          max_steps=max_steps,
      )
  )
  state_history = output.StateHistory(
      state_history=states_history,
      post_processed_outputs_history=post_processed_outputs_history,
      sim_error=sim_error,
      torax_config=torax_config,
  )
  return (
      state_history.simulation_output_to_xr(),
      state_history,
  )
