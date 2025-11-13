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
from torax._src.orchestration import run_loop
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.torax_pydantic import model_config
import xarray as xr


def prepare_simulation(
    torax_config: model_config.ToraxConfig,
) -> tuple[
    build_runtime_params.RuntimeParamsProvider,
    sim_state.ToraxSimState,
    post_processing.PostProcessedOutputs,
    step_function.SimulationStepFn,
]:
  """Prepare a TORAX simulation returning the necessary inputs for the run loop.

  Args:
    torax_config: The TORAX config to use for the simulation.

  Returns:
    A tuple containing:

      - The runtime parameters slice provider.
      - The initial state.
      - The initial post processed outputs.
      - The simulation step function.
  """
  geometry_provider = torax_config.geometry.build_provider
  physics_models = torax_config.build_physics_models()

  solver = torax_config.solver.build_solver(
      physics_models=physics_models,
  )

  runtime_params_provider = (
      build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
  )

  step_fn = step_function.SimulationStepFn(
      solver=solver,
      time_step_calculator=torax_config.time_step_calculator.time_step_calculator,
      geometry_provider=geometry_provider,
      runtime_params_provider=runtime_params_provider,
  )

  if torax_config.restart and torax_config.restart.do_restart:
    initial_state, post_processed_outputs = (
        initial_state_lib.get_initial_state_and_post_processed_outputs_from_file(
            t_initial=torax_config.numerics.t_initial,
            file_restart=torax_config.restart,
            runtime_params_provider=runtime_params_provider,
            geometry_provider=geometry_provider,
            step_fn=step_fn,
        )
    )
  else:
    initial_state, post_processed_outputs = (
        initial_state_lib.get_initial_state_and_post_processed_outputs(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=runtime_params_provider,
            geometry_provider=geometry_provider,
            step_fn=step_fn,
        )
    )

  return (
      runtime_params_provider,
      initial_state,
      post_processed_outputs,
      step_fn,
  )


def run_simulation(
    torax_config: model_config.ToraxConfig,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
) -> tuple[xr.DataTree, output.StateHistory]:
  """Runs a TORAX simulation using the config and returns the outputs.

  Args:
    torax_config: The TORAX config to use for the simulation.
    log_timestep_info: Whether to log the timestep information.
    progress_bar: Whether to show a progress bar.

  Returns:
    A tuple of the simulation outputs in the form of a DataTree and the state
    history which is intended for helpful use with debugging as it contains
    the `CoreProfiles`, `CoreTransport`, `CoreSources`, `Geometry`, and
    `PostProcessedOutputs` dataclasses for each step of the simulation.
  """

  (
      runtime_params_provider,
      initial_state,
      post_processed_outputs,
      step_fn,
  ) = prepare_simulation(torax_config)

  state_history, post_processed_outputs_history, sim_error = run_loop.run_loop(
      runtime_params_provider=runtime_params_provider,
      initial_state=initial_state,
      initial_post_processed_outputs=post_processed_outputs,
      step_fn=step_fn,
      log_timestep_info=log_timestep_info,
      progress_bar=progress_bar,
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
