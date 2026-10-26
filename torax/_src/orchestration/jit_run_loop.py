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
import math
import threading
from typing import Any, Final, TypeAlias
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
from torax._src.time_step_calculator import fixed_time_step_calculator
import tqdm

PyTree: TypeAlias = Any

type Counter = array_typing.IntScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LoggingInfo:
  """Logging info for the JITted run loop.

  Attributes:
    progress_bar_ref: Integer reference ID for the active progress bar in
      `_TQDM_REF`.
    log_n_steps: Number of steps between logging/progress bar updates.
    step: Current iteration count within the simulation loop.
    progress_bar: Whether to update the progress bar.
    log_timestep_info: Whether to log timestep information.
  """

  progress_bar_ref: array_typing.IntScalar
  log_n_steps: array_typing.IntScalar
  step: array_typing.IntScalar
  progress_bar: bool = jax.tree.static()
  log_timestep_info: bool = jax.tree.static()


_LOCK: threading.Lock = threading.Lock()
_counter_global: int = 0
# This stores references  active tqdm progress bar instances.
_TQDM_REF: dict[int, tqdm.tqdm] = {}
# The progress bar measures percentage of simulated time.
_PROGRESS_BAR_TOTAL: Final[int] = 100
_ADAPTIVE_DT_MAX_STEPS_MULTIPLIER: Final[int] = 10


def _progress_bar_callback(
    logging_info: LoggingInfo,
    progress: float,
    t: float,
) -> None:
  """Host-side callback to update the progress bar during simulation steps."""
  with _LOCK:
    pbar = _TQDM_REF.get(int(logging_info.progress_bar_ref))
    if pbar is not None:
      pbar.n = round(float(progress) * _PROGRESS_BAR_TOTAL)
      pbar.set_description(f'Simulating (t={t:.5f})')
      pbar.refresh()


def _log_timestep_callback(
    t: float,
    dt: float,
    outer_solver_iterations: int,
    solver_error_state: int,
) -> None:
  """Host-side callback to log timestep info during simulation steps."""
  with _LOCK:
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
    step_fn: A callable `SimulationStepFn` taking the current state and
      returning the next state.
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

  int_dtype = jax_utils.get_int_dtype()
  if logging_info is None:
    logging_info = LoggingInfo(
        progress_bar_ref=jnp.asarray(0, dtype=int_dtype),
        log_n_steps=jnp.asarray(1, dtype=int_dtype),
        step=jnp.asarray(0, dtype=int_dtype),
        progress_bar=False,
        log_timestep_info=False,
    )
  else:
    logging_info = dataclasses.replace(
        logging_info,
        progress_bar_ref=jnp.asarray(
            logging_info.progress_bar_ref, dtype=int_dtype
        ),
        log_n_steps=jnp.asarray(logging_info.log_n_steps, dtype=int_dtype),
        step=jnp.asarray(logging_info.step, dtype=int_dtype),
    )

  runtime_params = (
      runtime_params_overrides
      if runtime_params_overrides is not None
      else step_fn.runtime_params_provider
  )
  numerics = runtime_params.numerics
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
    is_log_step = (
        loop_logging_info.step % jnp.maximum(loop_logging_info.log_n_steps, 1)
    ) == 0
    should_log = is_log_step | is_done

    if loop_logging_info.progress_bar:

      def _do_progress_bar_callback():
        progress = jnp.where(
            duration > 0,
            jnp.clip((current_state.t - t_initial) / duration, 0.0, 1.0),
            1.0,
        )
        jax.debug.callback(
            _progress_bar_callback,
            loop_logging_info,
            progress,
            current_state.t,
        )

      jax.lax.cond(should_log, _do_progress_bar_callback, lambda: None)

    if loop_logging_info.log_timestep_info:

      def _do_log_timestep_callback():
        jax.debug.callback(
            _log_timestep_callback,
            current_state.t,
            current_state.dt,
            current_state.solver_numeric_outputs.outer_solver_iterations,
            current_state.solver_numeric_outputs.solver_error_state,
        )

      jax.lax.cond(should_log, _do_log_timestep_callback, lambda: None)

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


def _get_max_steps(
    step_fn: step_function.SimulationStepFn,
) -> int:
  """Calculates a heuristic for the maximum number of simulation steps.

  Args:
    step_fn: Callable which takes in ToraxSimState and outputs the ToraxSimState
      after one timestep.

  Returns:
    The maximum number of steps to pre-allocate for the JITted run loop.

  Raises:
    ValueError: If `chi_time_step_calculator` or
      `from_previous_time_step_calculator` (or any non-fixed time step
      calculator) is used, as a default `max_steps` cannot be determined
      statically.
  """
  if not isinstance(
      step_fn.time_step_calculator,
      fixed_time_step_calculator.FixedTimeStepCalculator,
  ):
    # TODO(b/564475000): Support max_steps heuristics for other time step
    # calculators.
    raise ValueError(
        'max_steps must be specified when using '
        f'{type(step_fn.time_step_calculator).__name__}. Please provide an '
        'explicit max_steps argument.'
    )

  numerics = step_fn.runtime_params_provider.numerics
  min_fixed_dt = float(np.min(numerics.fixed_dt.value))
  max_steps = math.ceil((numerics.t_final - numerics.t_initial) / min_fixed_dt)
  if numerics.adaptive_dt:
    max_steps *= _ADAPTIVE_DT_MAX_STEPS_MULTIPLIER
  return max_steps


def run_loop(
    step_fn: step_function.SimulationStepFn,
    runtime_params_overrides: (
        build_runtime_params.RuntimeParamsProvider | None
    ) = None,
    log_timestep_info: bool = False,
    progress_bar: bool = False,
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
    log_n_steps: Frequency in steps for logging and progress bar updates.
    max_steps: Optional maximum number of steps to take. When using
      `chi_time_step_calculator` or `from_previous_time_step_calculator`,
      `max_steps` must be explicitly specified or a `ValueError` is raised. When
      using `fixed_time_step_calculator`, if not provided, `max_steps` is
      deterministically set to `ceil((t_final - t_initial) / min(fixed_dt))`
      when `adaptive_dt=False`, or 10x that value when `adaptive_dt=True`. Users
      can explicitly override `max_steps` (e.g. to reduce pre-allocated buffer
      memory when `adaptive_dt=True`).

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
  if max_steps is None:
    max_steps = _get_max_steps(step_fn)

  # Use a nullcontext when `progress_bar` is False to manage the progress bar
  # lifecycle with a single `with` block.
  pbar_context = (
      tqdm.tqdm(total=_PROGRESS_BAR_TOTAL, desc='Simulating', leave=True)
      if progress_bar
      else contextlib.nullcontext()
  )
  with pbar_context as pbar:
    # `tqdm` instances cannot be traced by JAX, and passing them as static
    # arguments would trigger recompilation on every run. Instead, register the
    # active bar in `_TQDM_REF` under a unique integer ID and pass that ID
    # into `run_loop_jit`. `_LOCK` protects the global registry and counter
    # across concurrent threads and asynchronous JAX callbacks.
    if pbar is not None:
      with _LOCK:
        global _counter_global
        _counter_global += 1
        bar_ref = _counter_global
        _TQDM_REF[bar_ref] = pbar
    else:
      bar_ref = 0

    int_dtype = jax_utils.get_int_dtype()
    try:
      states_history, post_processed_outputs_history, final_i = run_loop_jit(
          step_fn,
          max_steps,
          runtime_params_overrides=runtime_params_overrides,
          logging_info=LoggingInfo(
              progress_bar_ref=jnp.asarray(bar_ref, dtype=int_dtype),
              log_n_steps=jnp.asarray(log_n_steps, dtype=int_dtype),
              step=jnp.asarray(0, dtype=int_dtype),
              progress_bar=progress_bar,
              log_timestep_info=log_timestep_info,
          ),
      )
      # JAX dispatch is asynchronous: wait for pending logging callbacks
      # before the progress bar is unregistered and closed.
      jax.effects_barrier()
    finally:
      if pbar is not None:
        with _LOCK:
          _TQDM_REF.pop(bar_ref, None)

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
