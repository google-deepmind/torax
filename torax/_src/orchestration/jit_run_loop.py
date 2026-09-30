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

import contextlib
import dataclasses
import threading
from typing import Any, TypeAlias
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import post_processing
import tqdm

PyTree: TypeAlias = Any


type Counter = jax.Array  # An integer scalar JAX array.


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LoggingInfo:
  """Logging info for the jit run loop.

  Attributes:
    progress_bar_ref: Integer reference ID for the active progress bar in
      `TQDM_REF`.
    progress_bar: Whether to update the progress bar.
    log_timestep_info: Whether to log timestep information.
    log_n_steps: Number of steps between logging/progress bar updates.
    step: Current iteration count within the simulation loop.
  """

  progress_bar_ref: array_typing.IntScalar = 0
  progress_bar: array_typing.BoolScalar = False
  log_timestep_info: array_typing.BoolScalar = False
  log_n_steps: array_typing.IntScalar = 20
  step: array_typing.IntScalar = 0


LOCK = threading.Lock()
COUNTER: int = 0
TQDM_REF: dict[int, tqdm.tqdm] = {}
# The progress bar measures percentage of simulated time.
_PROGRESS_BAR_TOTAL = 100


def _logging_callback(
    progress_bar_ref: int,
    progress_bar: bool,
    log_timestep_info: bool,
    progress: float,
    t: float,
    dt: float,
    outer_solver_iterations: int,
    solver_error_state: int,
) -> None:
  """Host-side callback executed during simulation steps."""
  with LOCK:
    if progress_bar:
      pbar = TQDM_REF.get(int(progress_bar_ref))
      if pbar is not None:
        pbar.n = round(float(progress) * _PROGRESS_BAR_TOTAL)
        pbar.set_description(f'Simulating (t={t:.5f})')
        pbar.refresh()

    if log_timestep_info:
      log_str = (
          f'Simulation time: {t:.5f}, previous dt:'
          f' {dt:.6f}, previous solver iterations:'
          f' {outer_solver_iterations}'
      )
      match solver_error_state:
        case 0:
          pass
        case 1:
          log_str += ' Solver did not converge in previous step.'
        case 2:
          log_str += (
              ' Solver converged only within coarse tolerance in previous step.'
          )
      tqdm.tqdm.write(log_str)


@jax.jit(static_argnames='max_steps')
def run_loop_jit(
    step_fn: step_function.SimulationStepFn,
    max_steps: int,
    runtime_params_overrides: (
        build_runtime_params.RuntimeParamsProvider | None
    ) = None,
    logging_info: LoggingInfo | None = None,
) -> tuple[sim_state.SimState, post_processing.PostProcessedOutputs, Counter]:
  """Runs the simulation loop under jax.jit.

  Args:
    step_fn: Callable taking the current state and returning the next state.
    max_steps: Maximum number of steps to perform in the bounded while loop.
    runtime_params_overrides: Optional overrides for runtime parameters.
    logging_info: Logging configuration and state to pass through the JIT.

  Returns:
    A tuple of (states_history, post_processed_outputs_history, final_i).
  """
  initial_state, initial_post_processed_outputs = (
      initial_state_lib.get_initial_state_and_post_processed_outputs(
          step_fn=step_fn,
          runtime_params_overrides=runtime_params_overrides,
      )
  )

  if logging_info is None:
    logging_info = LoggingInfo()
  logging_info = LoggingInfo(
      progress_bar_ref=jnp.asarray(
          logging_info.progress_bar_ref, dtype=jax_utils.get_int_dtype()
      ),
      progress_bar=jnp.asarray(logging_info.progress_bar, dtype=jnp.bool_),
      log_timestep_info=jnp.asarray(
          logging_info.log_timestep_info, dtype=jnp.bool_
      ),
      log_n_steps=jnp.asarray(
          logging_info.log_n_steps, dtype=jax_utils.get_int_dtype()
      ),
      step=jnp.asarray(logging_info.step, dtype=jax_utils.get_int_dtype()),
  )

  numerics = (
      runtime_params_overrides
      if runtime_params_overrides is not None
      else step_fn.runtime_params_provider
  ).numerics
  t_initial = numerics.t_initial
  duration = numerics.t_final - t_initial

  def _cond_fun(inputs):
    current_state, _, _ = inputs
    is_done = step_fn.is_done(current_state.t)
    return jnp.logical_not(is_done)

  def _step_fn(inputs):
    previous_state, previous_post_processed_outputs, loop_logging_info = inputs
    current_state, post_processed_outputs = step_fn(
        previous_state,
        previous_post_processed_outputs,
        runtime_params_overrides=runtime_params_overrides,
    )
    is_done = step_fn.is_done(current_state.t)
    is_logging_enabled = (
        loop_logging_info.progress_bar | loop_logging_info.log_timestep_info
    )
    is_n_step = (
        loop_logging_info.step
        % jnp.maximum(loop_logging_info.log_n_steps, 1)
    ) == 0
    should_log = is_logging_enabled & (is_n_step | is_done)

    def _do_callback():
      progress = jnp.where(
          duration > 0,
          jnp.clip((current_state.t - t_initial) / duration, 0.0, 1.0),
          1.0,
      )
      # Not `ordered=True`: ordered effects are unsupported for multi-device
      # jit, and strict ordering is unnecessary as each update sets the
      # absolute progress.
      jax.debug.callback(
          _logging_callback,
          loop_logging_info.progress_bar_ref,
          loop_logging_info.progress_bar,
          loop_logging_info.log_timestep_info,
          progress,
          current_state.t,
          current_state.dt,
          current_state.solver_numeric_outputs.outer_solver_iterations,
          current_state.solver_numeric_outputs.solver_error_state,
      )

    jax.lax.cond(should_log, _do_callback, lambda: None)

    new_logging_info = dataclasses.replace(
        loop_logging_info, step=loop_logging_info.step + 1
    )
    return current_state, post_processed_outputs, new_logging_info

  _, final_i, history = jax_utils.while_loop_bounded(
      _cond_fun,
      _step_fn,
      (initial_state, initial_post_processed_outputs, logging_info),
      max_steps,
      implementation='while_loop',
  )

  states_stacked, post_processed_outputs_stacked, _ = history

  # Prepend initial state to give (max_steps + 1, ...) output.
  history = jax.tree_util.tree_map(
      lambda init, stacked: jnp.concatenate(
          [jnp.expand_dims(init, axis=0), stacked], axis=0
      ),
      (initial_state, initial_post_processed_outputs),
      (states_stacked, post_processed_outputs_stacked),
  )

  states_history, post_processed_outputs_history = history

  return states_history, post_processed_outputs_history, final_i


def _unstack_array(x: jax.Array, i: int) -> tuple[np.ndarray, ...]:
  x = np.asarray(x[:i], copy=False)  # pyrefly: ignore[bad-assignment]
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
    progress_bar: bool = False,
    progress_bar_ref: int = 0,
    log_n_steps: int = 20,
    max_steps: int | None = None,
) -> tuple[
    list[sim_state.SimState],
    tuple[post_processing.PostProcessedOutputs, ...],
    state.SimError,
]:
  """Version of torax._src.orchestration.run_loop that loops with jax.jit.

  Performs logging and updates the progress bar if requested.

  Args:
    step_fn: Callable which takes in ToraxSimState and outputs the ToraxSimState
      after one timestep. Note that step_fn determines dt (how long the timestep
      is). The state_history that run_simulation() outputs comes from these
      ToraxSimState objects.
    runtime_params_overrides: Optional runtime params overrides to use.
    log_timestep_info: If True, logs basic timestep info, like time, dt, on
      every step.
    progress_bar: If True, displays a progress bar.
    progress_bar_ref: Integer reference to an active progress bar in TQDM_REF.
      If 0 and progress_bar is True, a new progress bar is created locally.
    log_n_steps: Frequency in steps for logging and progress bar updates.
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
  numerics = step_fn.runtime_params_provider.numerics
  if max_steps is None:
    max_steps = int(
        ((numerics.t_final - numerics.t_initial) / numerics.min_dt) / 1e5
    )

  pbar_context = (
      tqdm.tqdm(total=_PROGRESS_BAR_TOTAL, desc='Simulating', leave=True)
      if progress_bar and progress_bar_ref == 0
      else contextlib.nullcontext()
  )
  with pbar_context as pbar:
    if pbar is not None:
      with LOCK:
        global COUNTER
        COUNTER += 1
        bar_ref = COUNTER
        TQDM_REF[bar_ref] = pbar
    else:
      bar_ref = progress_bar_ref

    try:
      states_history, post_processed_outputs_history, final_i = run_loop_jit(
          step_fn,
          max_steps,
          runtime_params_overrides=runtime_params_overrides,
          logging_info=LoggingInfo(
              progress_bar_ref=bar_ref,
              progress_bar=progress_bar,
              log_timestep_info=log_timestep_info,
              log_n_steps=log_n_steps,
          ),
      )
      # JAX dispatch is asynchronous: wait for pending logging callbacks
      # before the progress bar is unregistered and closed.
      jax.effects_barrier()
    finally:
      if pbar is not None:
        with LOCK:
          TQDM_REF.pop(bar_ref, None)

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
