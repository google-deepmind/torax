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

"""Anderson acceleration with safeguarding.

References:
  [1] D.G. Anderson, "Iterative procedures for nonlinear integral
      equations," J. ACM, 12(4):547-560, 1965.
  [2] H.F. Walker and P. Ni, "Anderson acceleration for fixed-point
      iterations," SIAM J. Numer. Anal., 49(4):1715-1735, 2011.
  [3] A. Toth and C.T. Kelley, "Convergence analysis for Anderson
      acceleration," SIAM J. Numer. Anal., 53(2):805-819, 2015.
  [4] J. Zhang, B. O'Donoghue, and S. Boyd, "Globally convergent type-I
      Anderson acceleration for non-smooth fixed-point iterations,"
      SIAM J. Optim., 30(4):3170-3197, 2020.
  [5] "Safeguarded Anderson acceleration for parametric nonexpansive
      operators".
"""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class AndersonSettings:
  """Configuration for Anderson acceleration.

  Attributes:
    window_size: Number of retained iterates (m). Set to 0 to disable Anderson
      acceleration.
    safeguard_eta: Accept the AA candidate only if the residual norm is at most
      eta times the current residual norm. eta=1 means "don't make it worse".
      Values > 1 allow temporary increases.
    regularization: Tikhonov regularization for the least-squares solve, scaled
      by Gram matrix average diagonal.
    beta: Relaxation parameter in (0, 1]
  """

  window_size: int = 5
  safeguard_eta: float = 1.0
  regularization: float = 1e-10
  beta: float = 1.0


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AndersonHistory:
  """Circular buffer of past iterates and their fixed-point residuals."""

  x_history: jnp.ndarray
  f_history: jnp.ndarray
  count: jnp.ndarray

  @classmethod
  def create(cls, n: int, window_size: int, dtype):
    """Creates an empty Anderson history buffer."""
    return cls(
        x_history=jnp.zeros((window_size, n), dtype=dtype),
        f_history=jnp.zeros((window_size, n), dtype=dtype),
        count=jnp.array(0, dtype=jnp.int32),
    )

  def push(self, x_k: jnp.ndarray, f_k: jnp.ndarray, window_size: int):
    """Pushes a new (x, f) pair into the circular buffer."""
    idx = self.count % window_size
    return AndersonHistory(
        x_history=self.x_history.at[idx].set(x_k),
        f_history=self.f_history.at[idx].set(f_k),
        count=self.count + 1,
    )

  def update(
      self,
      accepted: jnp.ndarray,
      x: jnp.ndarray,
      picard_step: jnp.ndarray,
      settings: AndersonSettings,
  ) -> 'AndersonHistory':
    """Updates Anderson history based on acceptance.

    If accepted, pushes the current step to the history. If rejected, resets
    the history and pushes the current step to the fresh history.

    Args:
      accepted: True if the Anderson step was accepted.
      x: Current iterate.
      picard_step: Current fixed-point residual (or Picard step).
      settings: Anderson configuration.

    Returns:
      The updated history.
    """
    base_history = jax.lax.cond(
        accepted,
        lambda _: self,
        lambda _: AndersonHistory.create(
            x.shape[0], settings.window_size, dtype=x.dtype
        ),
        operand=None,
    )
    return base_history.push(x, picard_step, settings.window_size)

  def get_deltas(
      self, x_k: jnp.ndarray, f_k: jnp.ndarray, window_size: int
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes Delta_F and Delta_X matrices from history."""

    def _get_deltas(i):
      hist_idx = (self.count - 1 - i) % window_size
      df = f_k - self.f_history[hist_idx]
      dx = x_k - self.x_history[hist_idx]
      return df, dx

    indices = jnp.arange(window_size)
    all_df, all_dx = jax.vmap(_get_deltas)(indices)
    return all_df, all_dx


def _compute_candidate(
    history: AndersonHistory,
    x_k: jnp.ndarray,
    f_k: jnp.ndarray,
    settings: AndersonSettings,
) -> jnp.ndarray:
  """Computes the Anderson acceleration candidate."""
  # Following Walker and Ni [2], section 1 and section 3.
  m = settings.window_size
  m_actual = jnp.minimum(history.count, m)
  beta = settings.beta

  # Following (1.2): Compute residual differences and step differences.
  # history class puts them in the correct temporal order.
  all_df, all_dx = history.get_deltas(x_k, f_k, m)
  # Mask unused columns.
  indices = jnp.arange(m)
  mask = (indices < m_actual).astype(f_k.dtype)
  masked_df = all_df * mask[:, None]

  # Following Eq 3.1, finding the least squares solution of
  #   ||f_k - Delta_F @ gamma||^2
  rhs = masked_df @ f_k
  lhs_raw = masked_df @ masked_df.T  # (m, m)
  # We want to regularize the least squares solution, as the problem can be
  # ill-conditioned. Parametrize with the average of the trace.
  trace = jnp.trace(lhs_raw)
  regularizer = (
      settings.regularization * (trace / jnp.maximum(m_actual, 1)) + 1e-14
  )
  lhs = lhs_raw + regularizer * jnp.eye(m, dtype=f_k.dtype)
  gamma = jnp.linalg.solve(lhs, rhs)
  gamma = gamma * mask  # Zero out unused coefficients.

  # From eq. 3.1, subtract a weighted sum of the iterates.
  damped_picard_step = x_k + beta * f_k
  correction = (all_dx + beta * all_df).T @ gamma
  candidate = damped_picard_step - correction
  return candidate


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Result:
  """Result of attempting a safeguarded Anderson acceleration step."""

  candidate: jnp.ndarray
  residual: jnp.ndarray
  residual_norm: jnp.ndarray
  accepted: jnp.ndarray


def try_step(
    x: jnp.ndarray,
    picard_step: jnp.ndarray,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    current_residual_norm: jnp.ndarray,
    current_history: AndersonHistory,
    settings: AndersonSettings,
) -> Result:
  """Attempts an Anderson acceleration step with safeguarding."""

  m_actual = jnp.minimum(current_history.count, settings.window_size)
  candidate = jax.lax.cond(
      m_actual >= 1,
      lambda _: _compute_candidate(current_history, x, picard_step, settings),
      lambda _: x + settings.beta * picard_step,
      operand=None,
  )
  res = residual_fn(candidate)
  res_norm = jnp.linalg.norm(res)

  # Safeguard ([5], Eq. 13): accept only if the residual norm does not
  # increase by more than a factor eta.
  safeguard_threshold = settings.safeguard_eta * current_residual_norm
  accepted = res_norm <= safeguard_threshold

  return Result(
      candidate=candidate,
      residual=res,
      residual_norm=res_norm,
      accepted=accepted,
  )
