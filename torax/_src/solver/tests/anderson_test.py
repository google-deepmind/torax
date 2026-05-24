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

"""Tests for Anderson acceleration."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from torax._src.solver import anderson


def _default_settings(window_size=3, beta=1.0):
  return anderson.AndersonSettings(
      window_size=window_size,
      safeguard_eta=1.0,
      regularization=1e-10,
      beta=beta,
  )


class AndersonTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_enable_x64", True)


class HistoryTest(AndersonTest):

  def test_init_history(self):
    n = 5
    h = anderson.AndersonHistory.create(n, window_size=3, dtype=jnp.float64)
    self.assertEqual(h.x_history.shape, (3, n))
    self.assertEqual(h.f_history.shape, (3, n))
    self.assertEqual(h.count, 0)

  def test_push_history(self):
    n = 3
    window_size = 2
    h = anderson.AndersonHistory.create(
        n, window_size=window_size, dtype=jnp.float64
    )
    x = jnp.array([1.0, 2.0, 3.0])
    f = jnp.array([0.1, 0.2, 0.3])

    h = h.push(x, f, window_size=window_size)
    self.assertEqual(h.count, 1)
    chex.assert_trees_all_close(h.x_history[0], x)
    chex.assert_trees_all_close(h.f_history[0], f)

    x2 = jnp.array([4.0, 5.0, 6.0])
    f2 = jnp.array([0.4, 0.5, 0.6])
    h = h.push(x2, f2, window_size=window_size)
    self.assertEqual(h.count, 2)
    chex.assert_trees_all_close(h.x_history[1], x2)

    # Third push should wrap around to index 0.
    x3 = jnp.array([7.0, 8.0, 9.0])
    f3 = jnp.array([0.7, 0.8, 0.9])
    h = h.push(x3, f3, window_size=window_size)
    self.assertEqual(h.count, 3)
    chex.assert_trees_all_close(h.x_history[0], x3)
    chex.assert_trees_all_close(h.f_history[0], f3)


class ComputeCandidateTest(AndersonTest):

  def test_no_history_returns_picard(self):
    """With empty history, returns the damped Picard step."""
    n = 2
    beta = 0.5
    settings = _default_settings(beta=beta)
    h = anderson.AndersonHistory.create(
        n, settings.window_size, dtype=jnp.float64
    )

    x = jnp.array([1.0, 2.0])
    f = jnp.array([0.1, 0.2])

    candidate = anderson._compute_candidate(h, x, f, settings)
    expected = x + beta * f
    chex.assert_trees_all_close(candidate, expected)

  def test_with_history_differs_from_picard(self):
    """With history, the Anderson candidate differs from Picard."""
    n = 3
    settings = _default_settings()
    history = anderson.AndersonHistory.create(
        n, settings.window_size, dtype=jnp.float64
    )
    history = history.push(
        jnp.array([1.0, 2.0, 0.0]),
        jnp.array([0.5, -0.3, 0.1]),
        settings.window_size,
    )
    history = history.push(
        jnp.array([1.5, 1.7, 0.1]),
        jnp.array([0.2, -0.1, -0.2]),
        settings.window_size,
    )

    x = jnp.array([1.7, 1.6, 0.2])
    f = jnp.array([0.1, -0.05, 0.3])

    candidate = anderson._compute_candidate(history, x, f, settings)
    picard = x + f
    self.assertFalse(jnp.allclose(candidate, picard, atol=1e-6))

    # Test with damping as well.
    settings_damped = _default_settings(beta=0.5)
    candidate_damped = anderson._compute_candidate(
        history, x, f, settings_damped
    )
    picard_damped = x + 0.5 * f
    self.assertFalse(jnp.allclose(candidate_damped, picard_damped, atol=1e-6))

    # Damped candidate should differ from undamped candidate.
    self.assertFalse(jnp.allclose(candidate, candidate_damped, atol=1e-6))

  def test_exact_on_affine(self):
    """Anderson should exactly solve an affine fixed-point in one step."""

    def g(x):
      # Fixed point of function is x = [0.5, 0.5].
      return 0.5 * x + jnp.array([0.25, 0.25])

    n = 2
    settings = _default_settings(window_size=3)
    h = anderson.AndersonHistory.create(
        n, settings.window_size, dtype=jnp.float64
    )

    # Run multiple iterations of Picard to build history.
    x0 = jnp.array([0.0, 0.0])
    f0 = g(x0) - x0
    h = h.push(x0, f0, settings.window_size)

    x1 = g(x0)
    f1 = g(x1) - x1
    h = h.push(x1, f1, settings.window_size)

    x2 = g(x1)
    f2 = g(x2) - x2

    # Constructed candidate should be close to the exact answer.
    candidate = anderson._compute_candidate(h, x2, f2, settings)
    chex.assert_trees_all_close(candidate, jnp.array([0.5, 0.5]), atol=1e-10)

  def test_jit_compatible(self):
    """try_step works under jit."""
    n = 2
    settings = _default_settings()

    def residual_fn(x):
      return x - jnp.array([0.5, 0.5])

    @jax.jit
    def step(x, picard_step, current_residual_norm, current_history):
      return anderson.try_step(
          x,
          picard_step,
          residual_fn,
          current_residual_norm,
          current_history,
          settings,
      )

    h = anderson.AndersonHistory.create(
        n, settings.window_size, dtype=jnp.float64
    )
    x = jnp.array([1.0, 2.0])
    picard_step = jnp.array([0.1, 0.2])
    current_residual_norm = jnp.linalg.norm(residual_fn(x))

    result = step(x, picard_step, current_residual_norm, h)
    self.assertEqual(result.candidate.shape, (n,))
    self.assertIsInstance(result, anderson.Result)


class TryStepTest(AndersonTest):

  @parameterized.named_parameters(
      ("improving", lambda c: jnp.array([0.01, 0.02]), True),
      ("worse", lambda c: jnp.array([10.0, 20.0]), False),
  )
  def test_try_step_safeguard(self, residual_fn, expected_accepted):
    """Tests that try_step accepts improving steps and rejects worse steps."""
    n = 2
    settings = _default_settings()
    h = anderson.AndersonHistory.create(
        n, settings.window_size, dtype=jnp.float64
    )
    x = jnp.array([1.0, 2.0])
    picard_step = jnp.array([0.1, 0.2])
    current_residual_norm = jnp.array(1.0)

    result = anderson.try_step(
        x=x,
        picard_step=picard_step,
        residual_fn=residual_fn,
        current_residual_norm=current_residual_norm,
        current_history=h,
        settings=settings,
    )
    self.assertEqual(result.accepted, expected_accepted)


if __name__ == "__main__":
  absltest.main()
