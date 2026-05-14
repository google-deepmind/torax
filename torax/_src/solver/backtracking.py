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

"""Backtracking line search.

Given a proposed step `delta` from a current point `x`, the line search
repeatedly halves `delta` until the residual at `x + delta` is smaller than
at `x`, or until the step fraction `tau` drops below a minimum threshold.

This prevents steps that increase the residual or produce NaN values from
being accepted, which is critical for stability on stiff transport systems.
"""

import dataclasses
import functools
from typing import Callable, Final

import jax
import jax.numpy as jnp

# If no entry of delta is above this magnitude, we terminate the loop.
# This prevents getting stuck in an infinite loop in edge cases with bad
# numerics where the residual never decreases.
MIN_DELTA: Final[float] = 1e-7


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DeltaState:
  """State carried through the backtracking line search.

  Attributes:
    x: Current state vector.
    delta: Proposed step from `x`.
    residual_old: Residual vector at the current state, R(x).
    residual_new: Residual vector at the trial point, R(x + delta). Initialised
      to `residual_old` and updated during the search.
    tau: Step fraction relative to the initial delta. Starts at 1.0 and is
      multiplied by `delta_reduction_factor` on each failed attempt.
  """

  x: jax.Array
  delta: jax.Array
  residual_old: jax.Array
  residual_new: jax.Array
  tau: jax.Array


def compute_backtracked_delta(
    initial_state: DeltaState,
    residual_fun: Callable[[jax.Array], jax.Array],
    loss_fun: Callable[[jax.Array], jax.Array],
    delta_reduction_factor: float,
    tau_min: float = 0.0,
) -> DeltaState:
  """Backtracking line search on the step `delta`.

  If `x + delta` produces NaN in the residual or increases the residual
  magnitude, the step is repeatedly halved until the residual decreases,
  the step magnitude drops below `MIN_DELTA`, or the step fraction `tau`
  drops below `tau_min`.

  Args:
    initial_state: Initial state for the line search.
    residual_fun: R(x) residual function.
    loss_fun: A function that computes a scalar value from the residual vector.
      This is used to check for increases in the residual. It must be monotonic
      with the residual, and should ideally be convex.
    delta_reduction_factor: Multiplicative factor to reduce delta by on each
      failed attempt (typically 0.5).
    tau_min: Minimum allowed step fraction. When tau drops below this, the
      search stops regardless of whether the residual decreased.

  Returns:
    Updated state with the accepted `delta`, `tau`, and `residual_new`.
    The caller should compute `x_new = state.x + state.delta`.
  """
  delta_body_fun = functools.partial(
      _delta_body,
      delta_reduction_factor=delta_reduction_factor,
  )
  delta_cond_fun = functools.partial(
      _delta_cond,
      residual_fun=residual_fun,
      loss_fun=loss_fun,
      tau_min=tau_min,
  )
  output_state = jax.lax.while_loop(
      delta_cond_fun, delta_body_fun, initial_state
  )
  x_new = output_state.x + output_state.delta
  residual_vec_x_new = residual_fun(x_new)
  return dataclasses.replace(output_state, residual_new=residual_vec_x_new)


def _delta_cond(
    delta_state: DeltaState,
    residual_fun: Callable[[jax.Array], jax.Array],
    loss_fun: Callable[[jax.Array], jax.Array],
    tau_min: float = 0.0,
) -> bool:
  """Check if the step obtained from Newton is valid.

  Args:
    delta_state: See `DeltaState`.
    residual_fun: Residual function.
    loss_fun: A function that computes a scalar value from the residual vector.
      This is used to check for increases in the residual. It must be monotonic
      with the residual, and should ideally be convex.
    tau_min: Minimum allowed step fraction. 0.0 disables the tau check.

  Returns:
    True if the new value of `x` causes any NaNs or has increased the residual
    relative to the old value of `x`, AND the step is still large enough to
    be worth reducing further.
  """
  x_new = delta_state.x + delta_state.delta
  residual_scalar_x_old = loss_fun(delta_state.residual_old)
  # If x_new contains NaN (e.g. from bad numerics), skip the residual
  # evaluation entirely and force delta reduction by setting the residual
  # scalar to inf.  When x_new is valid the residual function runs normally
  # with all error_if checks enabled.
  # TODO(b/326577625) consider checking for negative temperatures/densities
  # explicitly, rather than just checking for NaNs.
  x_new_has_nan = jnp.any(jnp.isnan(x_new))
  residual_vec_x_new = jax.lax.cond(
      x_new_has_nan,
      lambda: delta_state.residual_old,
      lambda: residual_fun(x_new),
  )
  residual_scalar_x_new = jnp.where(
      x_new_has_nan,
      jnp.array(jnp.inf, dtype=residual_scalar_x_old.dtype),
      loss_fun(residual_vec_x_new),
  )
  # The step is still worth reducing if:
  # (a) delta is still large enough in magnitude, AND
  # (b) tau hasn't dropped below the minimum fraction, AND
  # (c) the residual increased or has NaN.
  step_large_enough = jnp.logical_and(
      jnp.max(jnp.abs(delta_state.delta)) > MIN_DELTA,
      delta_state.tau > tau_min,
  )
  return jnp.bool_(
      jnp.logical_and(
          step_large_enough,
          jnp.logical_or(
              residual_scalar_x_old < residual_scalar_x_new,
              jnp.isnan(residual_scalar_x_new),
          ),
      ),
  )


def _delta_body(
    state: DeltaState,
    delta_reduction_factor: float,
) -> DeltaState:
  """Reduces step size for this iteration."""
  return dataclasses.replace(
      state,
      delta=state.delta * delta_reduction_factor,
      tau=state.tau * delta_reduction_factor,
  )
