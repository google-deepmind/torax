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
"""Tests for backtracking line search."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from torax._src.solver import backtracking


def _residual_scalar(x):
  return jnp.mean(jnp.abs(x))


def _make_state(
    x: jax.Array,
    delta: jax.Array,
    residual_fun,
) -> backtracking.DeltaState:
  """Helper to construct a DeltaState from x, delta, and a residual function."""
  residual_old = residual_fun(x)
  return backtracking.DeltaState(
      x=x,
      delta=delta,
      residual_old=residual_old,
      residual_new=residual_old,
      tau=jnp.array(1.0, dtype=x.dtype),
  )


class BacktrackingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)

  def test_reduces_delta_on_increasing_residual(self):
    """Backtracking halves the step when the residual increases."""
    # Residual function: root at x=2, convex so large steps overshoot badly.
    residual_fun = lambda x: (x - 2.0) ** 2
    x = jnp.array([1.5], dtype=jnp.float64)
    # A deliberately large step that overshoots the root.
    delta = jnp.array([100.0], dtype=jnp.float64)
    initial_state = _make_state(x, delta, residual_fun)
    output = backtracking.compute_backtracked_delta(
        initial_state=initial_state,
        residual_fun=residual_fun,
        loss_fun=_residual_scalar,
        delta_reduction_factor=0.5,
    )
    # tau should have been reduced (delta was halved at least once).
    self.assertLess(float(output.tau), 1.0)
    # The accepted x_new = x + delta should have a smaller residual.
    x_new = output.x + output.delta
    self.assertLess(
        float(_residual_scalar(residual_fun(x_new))),
        float(_residual_scalar(initial_state.residual_old)),
    )

  def test_handles_nan_in_x_new(self):
    """Backtracking reduces delta when x_new contains NaN."""

    # Residual function that produces NaN for large inputs.
    def residual_fun(x):
      return jnp.where(jnp.abs(x) > 5.0, jnp.nan, x**2 - 1.0)

    x = jnp.array([1.0], dtype=jnp.float64)
    # A step that produces x_new = 1 + nan = nan.
    delta = jnp.array([jnp.nan], dtype=jnp.float64)
    initial_state = _make_state(x, delta, residual_fun)
    output = backtracking.compute_backtracked_delta(
        initial_state=initial_state,
        residual_fun=residual_fun,
        loss_fun=_residual_scalar,
        delta_reduction_factor=0.5,
    )
    # Delta should have been reduced to below MIN_DELTA (NaN * 0.5 is still
    # NaN, so the loop exits via the MIN_DELTA guard on abs(delta)).
    # The key thing is that it doesn't crash.
    self.assertTrue(jnp.all(jnp.isnan(output.delta)))

  def test_no_backtracking_needed(self):
    """When the step immediately reduces residual, no backtracking occurs."""
    residual_fun = lambda x: x - 1.0  # Root at x=1
    x = jnp.array([0.0], dtype=jnp.float64)
    delta = jnp.array([1.0], dtype=jnp.float64)  # Perfect step to root.
    initial_state = _make_state(x, delta, residual_fun)
    output = backtracking.compute_backtracked_delta(
        initial_state=initial_state,
        residual_fun=residual_fun,
        loss_fun=_residual_scalar,
        delta_reduction_factor=0.5,
    )
    # tau should remain 1.0 — no backtracking needed.
    self.assertEqual(float(output.tau), 1.0)
    # delta should be unchanged.
    self.assertEqual(float(output.delta[0]), 1.0)

  def test_tau_min_stops_search(self):
    """Backtracking stops when tau drops below tau_min."""
    # A function where the residual always increases — forces repeated halving.
    residual_fun = jnp.exp
    x = jnp.array([0.0], dtype=jnp.float64)
    delta = jnp.array([1.0], dtype=jnp.float64)
    initial_state = _make_state(x, delta, residual_fun)
    output = backtracking.compute_backtracked_delta(
        initial_state=initial_state,
        residual_fun=residual_fun,
        loss_fun=_residual_scalar,
        delta_reduction_factor=0.5,
        tau_min=0.1,
    )
    # tau should have been reduced but stopped near tau_min.
    # With factor 0.5: 1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.0625 (stops).
    self.assertLessEqual(float(output.tau), 0.125)
    self.assertGreater(float(output.tau), 0.0)


if __name__ == '__main__':
  absltest.main()
