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

"""JITted run_loop for iterating over the simulation step function."""
import functools

import chex
import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import post_processing


@functools.partial(jax.jit, static_argnames=['max_steps'])
def run_loop_jit(
    step_fn: step_function.SimulationStepFn,
    max_steps: int,
    runtime_params_overrides: (
        build_runtime_params.RuntimeParamsProvider | None
    ) = None,
) -> tuple[
    sim_state.ToraxSimState, post_processing.PostProcessedOutputs, chex.Numeric
]:
  """Runs the simulation loop under jax.jit."""
  runtime_params_provider = (
      runtime_params_overrides or step_fn.runtime_params_provider
  )
  initial_state, initial_post_processed_outputs = (
      initial_state_lib.get_initial_state_and_post_processed_outputs(
          t=runtime_params_provider.numerics.t_initial,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=step_fn.geometry_provider,
          step_fn=step_fn,
      )
  )
  # Some of the runtime params are not time-dependent, so we can get them once
  # before the loop.
  initial_runtime_params = runtime_params_provider(t=initial_state.t)
  time_step_calculator_params = initial_runtime_params.time_step_calculator

  # Pre-allocate history buffers
  states_history = jax.tree_util.tree_map(
      lambda x: jnp.zeros((max_steps + 1,) + x.shape, dtype=x.dtype),
      initial_state,
  )
  post_processed_outputs_history = jax.tree_util.tree_map(
      lambda x: jnp.zeros((max_steps + 1,) + x.shape, dtype=x.dtype),
      initial_post_processed_outputs,
  )

  # Store initial state
  states_history = jax.tree_util.tree_map(
      lambda hist, val: hist.at[0].set(val), states_history, initial_state
  )
  post_processed_outputs_history = jax.tree_util.tree_map(
      lambda hist, val: hist.at[0].set(val),
      post_processed_outputs_history,
      initial_post_processed_outputs,
  )

  def _cond_fun(inputs):
    i, current_state, _, _, _ = inputs
    not_done = step_fn.time_step_calculator.not_done(
        current_state.t,
        step_fn.runtime_params_provider.numerics.t_final,
        time_step_calculator_params,
    )
    return jnp.logical_and(i < max_steps, not_done)

  def _step_fn(inputs):
    (
        i,
        previous_state,
        previous_post_processed_outputs,
        states_hist,
        post_processed_outputs_hist,
    ) = inputs
    current_state, post_processed_outputs = step_fn(
        previous_state,
        previous_post_processed_outputs,
        runtime_params_overrides=runtime_params_overrides,
    )
    states_hist = jax.tree_util.tree_map(
        lambda hist, val: hist.at[i + 1].set(val), states_hist, current_state
    )
    post_processed_outputs_hist = jax.tree_util.tree_map(
        lambda hist, val: hist.at[i + 1].set(val),
        post_processed_outputs_hist,
        post_processed_outputs,
    )
    return (
        i + 1,
        current_state,
        post_processed_outputs,
        states_hist,
        post_processed_outputs_hist,
    )

  final_i, _, _, states_history, post_processed_outputs_history = (
      jax_utils.while_loop_bounded(
          _cond_fun,
          _step_fn,
          (
              0,
              initial_state,
              initial_post_processed_outputs,
              states_history,
              post_processed_outputs_history,
          ),
          max_steps,
      )
  )

  return states_history, post_processed_outputs_history, final_i


def _unstack_pytree_history(
    states_history,
    post_processed_outputs_history,
    final_i,
):
  """Unstacks stacked JIT output into a list of unstacked outputs.

  Args:
    states_history: A PyTree where each leaf is a JAX array with shape
      (max_steps + 1, ...) representing the history of that component over time.
    post_processed_outputs_history: A PyTree where each leaf is a JAX array with
      shape (max_steps + 1, ...) representing the history of that component over
      time.
    final_i: The actual number of steps taken in the simulation.

  Returns:
    A list of PyTrees, where each element of the list corresponds to
    a time step [0, max_steps]. Each element of the list has the same
    structure and leaf types as the original initial_state.
  """
  states_history_list = []
  post_processed_outputs_history_list = []
  for time_index in range(final_i + 1):
    # Use tree_map to slice each leaf array at the current time step `i`
    current_state = jax.tree_util.tree_map(
        lambda x, i=time_index: x[i], states_history
    )
    states_history_list.append(current_state)
    post_processed_output = jax.tree_util.tree_map(
        lambda x, i=time_index: x[i], post_processed_outputs_history
    )
    post_processed_outputs_history_list.append(post_processed_output)
  return states_history_list, post_processed_outputs_history_list


def run_loop(
    step_fn: step_function.SimulationStepFn,
    runtime_params_overrides: (
        build_runtime_params.RuntimeParamsProvider | None
    ) = None,
) -> tuple[
    list[sim_state.ToraxSimState],
    tuple[post_processing.PostProcessedOutputs, ...],
    state.SimError,
]:
  """Version of torax._src.orchestration.run_loop that loops with jax.jit.

  Unlike the `run_loop` function, This does not support logging or progress bar.

  Args:
    step_fn: Callable which takes in ToraxSimState and outputs the ToraxSimState
      after one timestep. Note that step_fn determines dt (how long the timestep
      is). The state_history that run_simulation() outputs comes from these
      ToraxSimState objects.
    runtime_params_overrides: Optional runtime params overrides to use.

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
      - The sim error state.
  """
  numerics = step_fn.runtime_params_provider.numerics
  max_steps = int(
      ((numerics.t_final - numerics.t_initial) / numerics.min_dt) / 1e5
  )
  states_history, post_processed_outputs_history, final_i = run_loop_jit(
      step_fn,
      max_steps,
      runtime_params_overrides=runtime_params_overrides,
  )
  unstacked_states, unstacked_post_processed_outputs = _unstack_pytree_history(
      states_history, post_processed_outputs_history, final_i
  )
  sim_error = step_function.check_for_errors(
      step_fn.runtime_params_provider.numerics,
      unstacked_states[-1],
      unstacked_post_processed_outputs[-1],
  )
  return (
      unstacked_states,
      tuple(unstacked_post_processed_outputs),
      sim_error,
  )
