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

  Attributes:
    iteration: Current iteration of the linesearch.
    x: Current location.
    residual: Current residual.
    residual_norm: Norm of current residual.
    step_size: Current step size.
    accepted: Whether the current step satisfies accept_fn.
  """

  iteration: jnp.ndarray
  x: jt.PyTree
  residual: jt.PyTree
  residual_norm: jnp.ndarray
  step_size: jnp.ndarray
  accepted: jt.Bool[jax.Array, ""]


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
    vmap: bool = False,
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
    vmap: If True, evaluates trial steps in parallel using jax.vmap instead of
      sequentially with a while loop.

  Returns:
    LinesearchState with the accepted (or last tried) trial point.
  """
  dtype = (
      x_init.dtype
      if hasattr(x_init, "dtype")
      else jnp.float32
  )

  def evaluate_step(step_size):
    x = jax.tree.map(lambda a, b: a + step_size * b, x_init, direction)
    res = residual_fn(x)
    norm = norm_fn(res)
    accepted = accept_fn(step_size, norm)

    return x, res, norm, accepted

  if vmap:
    return _vmapped_backtracking_linesearch(
        evaluate_step=evaluate_step,
        delta_reduction_factor=delta_reduction_factor,
        max_steps=max_steps,
        dtype=dtype,
    )

  return _sequential_backtracking_linesearch(
      evaluate_step=evaluate_step,
      x_init=x_init,
      initial_residual=initial_residual,
      initial_residual_norm=initial_residual_norm,
      delta_reduction_factor=delta_reduction_factor,
      max_steps=max_steps,
      dtype=dtype,
  )


def _vmapped_backtracking_linesearch(
    evaluate_step: Callable[
        [jnp.ndarray],
        tuple[jt.PyTree, jt.PyTree, jnp.ndarray, jt.Bool[jax.Array, ""]],
    ],
    delta_reduction_factor: float,
    max_steps: int,
    dtype: jnp.dtype,
) -> LinesearchState:
  """Performs backtracking line search in parallel using jax.vmap."""
  # Generate step sizes: [1.0, eta, eta^2, ..., eta^(max_steps-1)]
  step_sizes = delta_reduction_factor ** jnp.arange(
      max_steps,
      dtype=dtype,
  )

  vmapped_trial_step = jax.vmap(evaluate_step)
  xs, residuals, norms, accepteds = vmapped_trial_step(step_sizes)

  # Extract the first state where accept is True.
  # If no step was found, return the smallest trial step (last in the array).
  i = jnp.where(jnp.any(accepteds), jnp.argmax(accepteds), -1)
  x = jax.tree.map(lambda arr: arr[i], xs)
  res = jax.tree.map(lambda arr: arr[i], residuals)
  norm = norms[i]
  step_size = step_sizes[i]
  accepted = accepteds[i]
  iteration = i + 1

  return LinesearchState(
      iteration=iteration,
      step_size=step_size,
      x=x,
      residual=res,
      residual_norm=norm,
      accepted=accepted,
  )


def _sequential_backtracking_linesearch(
    evaluate_step: Callable[
        [jnp.ndarray],
        tuple[jt.PyTree, jt.PyTree, jnp.ndarray, jt.Bool[jax.Array, ""]],
    ],
    x_init: jt.PyTree,
    initial_residual: jt.PyTree,
    initial_residual_norm: jnp.ndarray,
    delta_reduction_factor: float,
    max_steps: int,
    dtype: jnp.dtype,
) -> LinesearchState:
  """Performs backtracking line search sequentially using jax.lax.while_loop."""
  init_state = LinesearchState(
      iteration=jnp.array(0, dtype=jnp.int32),
      step_size=jnp.array(1.0, dtype=dtype),
      x=x_init,
      residual=initial_residual,
      residual_norm=initial_residual_norm,
      accepted=jnp.array(False),
  )

  def cond_fun(state: LinesearchState) -> jt.Bool[jax.Array, ""]:
    return jnp.logical_and(
        state.iteration < max_steps,
        jnp.logical_not(state.accepted),
    )

  def body_fun(state: LinesearchState) -> LinesearchState:
    step_size = (delta_reduction_factor ** state.iteration).astype(dtype)
    x, res, norm, accepted = evaluate_step(step_size)

    return LinesearchState(
        iteration=state.iteration + 1,
        step_size=step_size,
        x=x,
        residual=res,
        residual_norm=norm,
        accepted=accepted,
    )

  # TODO(b/515250945): Use whilei_loop for autodiff compatibility.
  return jax.lax.while_loop(cond_fun, body_fun, init_state)

