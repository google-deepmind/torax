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

from typing import Any, TypeAlias
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import post_processing

PyTree: TypeAlias = Any


@jax.jit(static_argnames='max_steps')
def run_loop_jit(
    step_fn: step_function.SimulationStepFn,
    max_steps: int,
    runtime_params_overrides: (
        build_runtime_params.RuntimeParamsProvider | None
    ) = None,
) -> tuple[
    sim_state.SimState, post_processing.PostProcessedOutputs, chex.Numeric
]:
  """Runs the simulation loop under jax.jit."""
  initial_state, initial_post_processed_outputs = (
      initial_state_lib.get_initial_state_and_post_processed_outputs(
          step_fn=step_fn,
          runtime_params_overrides=runtime_params_overrides,
      )
  )

  def _cond_fun(inputs):
    current_state, _ = inputs
    is_done = step_fn.is_done(current_state.t)
    return jnp.logical_not(is_done)

  def _step_fn(inputs):
    previous_state, previous_post_processed_outputs = inputs
    current_state, post_processed_outputs = step_fn(
        previous_state,
        previous_post_processed_outputs,
        runtime_params_overrides=runtime_params_overrides,
    )
    return current_state, post_processed_outputs

  _, final_i, history = jax_utils.while_loop_bounded(
      _cond_fun,
      _step_fn,
      (initial_state, initial_post_processed_outputs),
      max_steps,
      implementation='while_loop',
  )

  # Prepend initial state to give (max_steps + 1, ...) output.
  history = jax.tree_util.tree_map(
      lambda init, stacked: jnp.concatenate(
          [jnp.expand_dims(init, axis=0), stacked], axis=0
      ),
      (initial_state, initial_post_processed_outputs),
      history,
  )

  states_history, post_processed_outputs_history = history

  return states_history, post_processed_outputs_history, final_i


def _unstack_array(x: jax.Array, i: int) -> tuple[np.ndarray, ...]:
  x = np.asarray(x[:i], copy=False)
  unstacked = np.unstack(x)
  # If the array is 1D, then unstack returns a list of scalars. Convert these
  # to a tuple of scalar NumPy arrays.
  if x.ndim == 1:
    return tuple(np.asarray(val) for val in unstacked)
  return unstacked


def _unstack_pytree_history(
    history: PyTree,
    final_i: int,
) -> list[PyTree]:
  """Unstacks stacked JIT output into a list of unstacked outputs.

  Args:
    history: A PyTree where each leaf is a JAX array with shape (max_steps + 1,
      ...) representing the history of that component over time.
    final_i: The actual number of steps taken in the simulation.

  Returns:
    A list of PyTrees, where each element of the list corresponds to
    a time step [0, max_steps]. Each element of the list has the same
    structure and leaf types as the original initial_state.
  """
  # This is the number of steps taken in the while_loop + the initial state.
  num_states = final_i + 1
  history_list = []
  vals, treedef = jax.tree.flatten(history)
  vals = [_unstack_array(x, num_states) for x in vals]

  for time_index in range(num_states):
    sub_vals = [val[time_index] for val in vals]
    new_tree = jax.tree.unflatten(treedef, sub_vals)
    history_list.append(new_tree)

  assert len(history_list) == num_states
  return history_list


def run_loop(
    step_fn: step_function.SimulationStepFn,
    runtime_params_overrides: (
        build_runtime_params.RuntimeParamsProvider | None
    ) = None,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
    max_steps: int | None = None,
) -> tuple[
    list[sim_state.SimState],
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
    log_timestep_info: If True, logs basic timestep info, like time, dt, on
      every step.
    progress_bar: If True, displays a progress bar.
    max_steps: Optional maximum number of steps to take. If not provided, then
      the maximum number of steps will be determined by the numerics.t_final and
      numerics.min_dt.

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

  if progress_bar:
    raise NotImplementedError(
        'Progress bar is not supported with the jitted run loop.'
    )
  if log_timestep_info:
    raise NotImplementedError(
        'Log timestep info is not supported with the jitted run loop.'
    )

  numerics = step_fn.runtime_params_provider.numerics
  if max_steps is None:
    max_steps = int(
        ((numerics.t_final - numerics.t_initial) / numerics.min_dt) / 1e5
    )
  states_history, post_processed_outputs_history, final_i = run_loop_jit(
      step_fn,
      max_steps,
      runtime_params_overrides=runtime_params_overrides,
  )

  unstacked_states = _unstack_pytree_history(states_history, final_i)
  unstacked_post_processed_outputs = _unstack_pytree_history(
      post_processed_outputs_history, final_i
  )

  sim_error = step_fn.check_for_errors(
      unstacked_states[-1],
      unstacked_post_processed_outputs[-1],
  )
  if sim_error == state.SimError.NO_ERROR:
    if not step_fn.is_done(unstacked_states[-1].t):
      sim_error = state.SimError.DID_NOT_REACH_T_FINAL
  return (
      unstacked_states,
      tuple(unstacked_post_processed_outputs),
      sim_error,
  )
