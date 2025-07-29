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

"""JITTED run_loop for iterating over the simulation step function."""
from torax._src import xnp
from torax._src.config import build_runtime_params
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import post_processing


def run_loop(
    dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
    initial_state: sim_state.ToraxSimState,
    initial_post_processed_outputs: post_processing.PostProcessedOutputs,
    step_fn: step_function.SimulationStepFn,
) -> tuple[
    tuple[sim_state.ToraxSimState, ...],
    tuple[post_processing.PostProcessedOutputs, ...],
]:
  """Runs the simulation loop.

  Iterates over the step function until the time_step_calculator tells us we are
  done or the simulation hits an error state.

  Performs logging and updates the progress bar if requested.

  Args:
    dynamic_runtime_params_slice_provider: Provides a DynamicRuntimeParamsSlice
      to use as input for each time step. See static_runtime_params_slice and
      the runtime_params_slice module docstring for runtime_params_slice to
      understand why we need the dynamic and static config slices and what they
      control.
    initial_state: The starting state of the simulation. This includes both the
      state variables which the solver.Solver will evolve (like ion temp, psi,
      etc.) as well as other states that need to be be tracked, like time.
    initial_post_processed_outputs: The post-processed outputs at the start of
      the simulation. This is used to calculate cumulative quantities.
    step_fn: Callable which takes in ToraxSimState and outputs the ToraxSimState
      after one timestep. Note that step_fn determines dt (how long the timestep
      is). The state_history that run_simulation() outputs comes from these
      ToraxSimState objects.

  Returns:
    A tuple of:
      - the simulation history, consisting of a tuple of ToraxSimState objects,
        one for each time step. There are N+1 objects returned, where N is the
        number of simulation steps taken. The first object in the tuple is for
        the initial state. If the sim error state is 1, then a trunctated
        simulation history is returned up until the last valid timestep.
      - the post-processed outputs history, consisting of a tuple of
        PostProcessedOutputs objects, one for each time step. There are N+1
        objects returned, where N is the number of simulation steps taken. The
        first object in the tuple is for the initial state. If the sim error
        state is 1, then a trunctated simulation history is returned up until
        the last valid timestep.
  """
  # Some of the dynamic params are not time-dependent, so we can get them once
  # before the loop.
  initial_dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
      initial_state.t
  )
  time_step_calculator_dynamic_params = (
      initial_dynamic_runtime_params_slice.time_step_calculator
  )

  def _cond_fun(inputs):
    current_state, _ = inputs
    return step_fn.time_step_calculator.not_done(
        current_state.t,
        dynamic_runtime_params_slice_provider.numerics.t_final,
        time_step_calculator_dynamic_params,
    )

  final_state, final_post_processed_outputs = xnp.while_loop(
      _cond_fun, step_fn, (initial_state, initial_post_processed_outputs)
  )

  return final_state, final_post_processed_outputs
