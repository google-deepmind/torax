# Copyright 2026 DeepMind Technologies Limited
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

"""Backtracking line search for use in solving functions."""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import jaxtyping as jt


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LinesearchState:
  """State and result of the backtracking line search.

  Attributes are the values at the accepted step size, or the last value
  tried if the search failed.

  Attributes:
    iteration: Current iteration of the linesearch.
    step_size: Current step size.
    next_step_size: Next step size to try.
    x: Current location.
    residual: Current residual.
    residual_norm: Norm of current residual.
    step_found: Whether a step has been found.
    done: Whether the linesearch is done.
  """

  iteration: jnp.ndarray
  step_size: jnp.ndarray
  next_step_size: jnp.ndarray
  x: jt.PyTree
  residual: jt.PyTree
  residual_norm: jnp.ndarray
  step_found: jt.Bool[jax.Array, ""]
  done: jt.Bool[jax.Array, ""]


def backtracking_linesearch(
    residual_fn: Callable[[jt.PyTree], jt.PyTree],
    x_init: jt.PyTree,
    direction: jt.PyTree,
    accept_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    norm_fn: Callable[[jt.PyTree], jnp.ndarray],
    initial_residual: jt.PyTree,
    initial_residual_norm: jnp.ndarray,
    delta_reduction_factor: float,
    max_steps: int,
    min_step_norm: float = 0.0,
) -> LinesearchState:
  """Performs backtracking line search.

  A backtracking linesearch seeks a value for step_size such that
      x_trial = x_init + step_size * direction
  meets the condition specified by accept_fn. It performs the search by starting
  step_size at 1.0, and decreasing step_size until either accept_fn is true or
  the maximum number of iterations is reached.

  Args:
    residual_fn: Accepts the location x, and returns the residual R(x).
    x_init: Starting location.
    direction: Search direction, a PyTree with the same shape as x.
    accept_fn: Accepts (step_size, trial_residual_norm) and returns True if the
      trial point is acceptable, and false otherwise.
    norm_fn: Function compute the norm of the residual.
    initial_residual: Residual vector at input x_init.
    initial_residual_norm: Norm of initial_residual.
    delta_reduction_factor: Factor by which step_size is reduced each step.
    max_steps: Maximum number of backtracking steps.
    min_step_norm: Minimum value of max(abs(step_size * direction)) allowed.

  Returns:
    LinesearchState with the accepted (or last tried) trial point.
  """

  init_step_size = 1.0
  init_state = LinesearchState(
      iteration=jnp.array(0, dtype=jnp.int32),
      step_size=jnp.array(
          init_step_size,
          dtype=x_init.dtype if hasattr(x_init, "dtype") else jnp.float32,
      ),
      next_step_size=jnp.array(
          init_step_size,
          dtype=x_init.dtype if hasattr(x_init, "dtype") else jnp.float32,
      ),
      x=x_init,
      residual=initial_residual,
      residual_norm=initial_residual_norm,
      step_found=jnp.array(False),
      done=jnp.array(False),
  )

  def cond_fun(state: LinesearchState) -> jt.Bool[jax.Array, ""]:
    return jnp.logical_not(state.done)

  def body_fun(state: LinesearchState) -> LinesearchState:
    new_iter = state.iteration + 1
    step_size = state.next_step_size

    new_x = jax.tree.map(lambda a, b: a + step_size * b, x_init, direction)
    new_res = residual_fn(new_x)
    new_norm = norm_fn(new_res)

    new_step_found = accept_fn(step_size, new_norm)
    is_max_iter = new_iter >= max_steps

    # Check if step is too small.
    max_abs_dir = jnp.max(
        jnp.array(
            [jnp.max(jnp.abs(leaf)) for leaf in jax.tree.leaves(direction)]
        )
    )
    step_too_small = (step_size * max_abs_dir) <= min_step_norm

    new_done = new_step_found | is_max_iter | step_too_small
    next_step_size = step_size * delta_reduction_factor

    return LinesearchState(
        iteration=new_iter,
        step_size=step_size,
        next_step_size=next_step_size,
        x=new_x,
        residual=new_res,
        residual_norm=new_norm,
        step_found=new_step_found,
        done=new_done,
    )

  # TODO(b/515250945): Use whilei_loop for autodiff compatibility.
  return jax.lax.while_loop(cond_fun, body_fun, init_state)


def vmapped_backtracking_linesearch(
    residual_fn: Callable[[jt.PyTree], jt.PyTree],
    x_init: jt.PyTree,
    direction: jt.PyTree,
    accept_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    norm_fn: Callable[[jt.PyTree], jnp.ndarray],
    initial_residual: jt.PyTree,
    initial_residual_norm: jnp.ndarray,
    delta_reduction_factor: float,
    max_steps: int,
    min_step_norm: float = 0.0,
) -> LinesearchState:
  """Performs vmapped backtracking line search by evaluating multiple step sizes in parallel.

  Args:
    residual_fn: Accepts the location x, and returns the residual R(x).
    x_init: Starting location.
    direction: Search direction, a PyTree with the same shape as x.
    accept_fn: Accepts (step_size, trial_residual_norm) and returns True if the
      trial point is acceptable, and false otherwise.
    norm_fn: Function compute the norm of the residual.
    initial_residual: Residual vector at input x_init.
    initial_residual_norm: Norm of initial_residual.
    delta_reduction_factor: Factor by which step_size is reduced each step.
    max_steps: Number of backtracking steps to try in parallel.
    min_step_norm: Minimum value of max(abs(step_size * direction)) allowed.

  Returns:
    LinesearchState with the accepted (or last tried) trial point.
  """
  del initial_residual
  del initial_residual_norm
  # Generate step sizes: [1.0, eta, eta^2, ..., eta^(max_steps-1)]
  step_sizes = delta_reduction_factor ** jnp.arange(
      max_steps,
      dtype=x_init.dtype if hasattr(x_init, "dtype") else jnp.float32,
  )

  def single_attempt(step_size):
    new_x = jax.tree.map(lambda a, b: a + step_size * b, x_init, direction)
    new_res = residual_fn(new_x)
    new_norm = norm_fn(new_res)
    step_found = accept_fn(step_size, new_norm)

    # Check if step is too small.
    max_abs_dir = jnp.max(
        jnp.array(
            [jnp.max(jnp.abs(leaf)) for leaf in jax.tree.leaves(direction)]
        )
    )
    step_too_small = (step_size * max_abs_dir) <= min_step_norm

    return new_x, new_res, new_norm, step_found, step_too_small

  # vmap over step_sizes
  vmapped_fn = jax.vmap(single_attempt)
  xs, residuals, norms, step_founds, step_too_smalls = vmapped_fn(step_sizes)

  # Find the first index where step_found is True.
  # We want the minimum index `i` such that `step_founds[i]` is True.
  indices = jnp.arange(max_steps)
  masked_indices = jnp.where(step_founds, indices, max_steps)
  selected_index = jnp.min(masked_indices)

  # If none are True, fall back to the last one (index max_steps - 1).
  none_found = selected_index == max_steps
  final_index = jnp.where(none_found, max_steps - 1, selected_index)

  # Extract the selected state
  selected_x = jax.tree.map(lambda arr: arr[final_index], xs)
  selected_res = jax.tree.map(lambda arr: arr[final_index], residuals)
  selected_norm = norms[final_index]
  selected_step_size = step_sizes[final_index]
  selected_step_found = step_founds[final_index]
  selected_step_too_small = step_too_smalls[final_index]

  selected_iteration = final_index + 1
  selected_done = (
      selected_step_found
      | (final_index >= max_steps - 1)
      | selected_step_too_small
  )

  return LinesearchState(
      iteration=selected_iteration,
      step_size=selected_step_size,
      next_step_size=selected_step_size * delta_reduction_factor,
      x=selected_x,
      residual=selected_res,
      residual_norm=selected_norm,
      step_found=selected_step_found,
      done=selected_done,
  )
