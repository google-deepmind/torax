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
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.torax_pydantic import config


def make_step_fn(
    torax_config: config.ToraxConfig,
) -> step_function.SimulationStepFn:
  """Prepare a TORAX step function from a torax_config."""
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
    torax_config: config.ToraxConfig,
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
    torax_config: config.ToraxConfig,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
    log_n_steps: int = 20,
    max_steps: int | None = None,
) -> output.StateHistory:
  """Runs a TORAX simulation using the config and returns the outputs.

  Args:
    torax_config: The TORAX config to use for the simulation.
    log_timestep_info: Whether to log the timestep information.
    progress_bar: Whether to show a progress bar.
    log_n_steps: The number of steps between logging events. This affects both
      the `log_timestep_info` and `progress_bar` arguments.
    max_steps: The maximum number of steps to take. If None, uses
      `time_step_calculator.max_steps` from the config if set, or falls back to
      `time_step_calculator.default_max_steps`.

  Returns:
    The state history which contains the `CoreProfiles`, `CoreTransport`,
    `CoreSources`, `Geometry`, and `PostProcessedOutputs` dataclasses for each
    step of the simulation. A DataTree can be obtained via
    `state_history.simulation_output_to_xr()`.
  """

  # Only prepare the initial state if we are doing a restart, as this is more
  # expensive than doing this in the JITted function.
  if torax_config.restart and torax_config.restart.do_restart:
    initial_state, initial_post_processed_outputs, step_fn = prepare_simulation(
        torax_config
    )
  else:
    step_fn = make_step_fn(torax_config)
    initial_state = None
    initial_post_processed_outputs = None

  state_history, post_processed_outputs_history, sim_error = (
      jit_run_loop.run_loop(
          step_fn,
          max_steps=max_steps,
          log_timestep_info=log_timestep_info,
          progress_bar=progress_bar,
          log_n_steps=log_n_steps,
          initial_state=initial_state,
          initial_post_processed_outputs=initial_post_processed_outputs,
      )
  )

  state_history = output.StateHistory(
      state_history=state_history,
      post_processed_outputs_history=post_processed_outputs_history,
      sim_error=sim_error,
      torax_config=torax_config,
  )

  return state_history
