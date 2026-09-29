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
"""Tests for trapped_fraction module."""

# pylint: disable=invalid-name

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.geometry import trapped_fraction


class TrappedFractionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig().build_geometry()

  def test_calculate_sauter_trapped_fraction_positive_triangularity(self):
    result = trapped_fraction.calculate_sauter_trapped_fraction(
        epsilon=np.array(0.1), delta=np.array(0.2)
    )
    expected = 0.4362384616678634
    np.testing.assert_allclose(result, expected)

  def test_calculate_sauter_trapped_fraction_negative_triangularity(self):
    result = trapped_fraction.calculate_sauter_trapped_fraction(
        epsilon=np.array(0.1), delta=np.array(-0.2)
    )
    expected = 0.45134158459680895
    np.testing.assert_allclose(result, expected)

  def test_calculate_sauter_trapped_fraction_gradient_on_axis(self):
    grad_fn = jax.grad(
        lambda geo: jnp.sum(
            trapped_fraction.calculate_sauter_trapped_fraction(
                geo.epsilon_face, geo.delta_face
            )
        ),
        allow_int=True,
    )
    grad_geo = grad_fn(self.geo)

    for leaf in jax.tree_util.tree_leaves(grad_geo):
      if isinstance(leaf, (jax.Array, np.ndarray)) and jnp.issubdtype(
          leaf.dtype, jnp.inexact
      ):
        chex.assert_tree_all_finite(leaf)

  def test_extrapolate_exact_trapped_fraction_to_axis(self):
    rhon = np.linspace(0.0, 1.0, 21)
    true_ftrap = 0.6 * np.sqrt(rhon)
    # Simulate missing/unreliable inner surfaces close to axis
    test_ftrap = true_ftrap.copy()
    test_ftrap[0] = np.nan
    test_ftrap[1] = np.nan
    test_ftrap[2] = np.nan

    extrapolated = trapped_fraction.extrapolate_exact_trapped_fraction_to_axis(
        trapped_fraction=test_ftrap,
        rhon=rhon,
    )
    # Axis must be exactly zero
    self.assertEqual(extrapolated[0], 0.0)
    # All values must be finite and within [0, 1]
    self.assertTrue(np.all(np.isfinite(extrapolated)))
    self.assertTrue(np.all(extrapolated >= 0.0))
    self.assertTrue(np.all(extrapolated <= 1.0))
    # Must preserve valid points exactly
    np.testing.assert_allclose(extrapolated[3:], true_ftrap[3:])
    # Must be strictly increasing
    self.assertTrue(np.all(np.diff(extrapolated) > 0.0))
    # Interpolated values should be close to true sqrt(rhon) curve
    np.testing.assert_allclose(extrapolated[:3], true_ftrap[:3], atol=0.01)

  def test_extrapolate_exact_trapped_fraction_fallback(self):
    rhon = np.linspace(0.0, 1.0, 10)
    all_nan = np.full_like(rhon, np.nan)
    sauter = 0.5 * rhon

    result = trapped_fraction.extrapolate_exact_trapped_fraction_to_axis(
        trapped_fraction=all_nan,
        rhon=rhon,
        sauter_trapped_fraction=sauter,
    )
    np.testing.assert_allclose(result, sauter)

  def test_bounce_averaged_trapped_fraction_uniform_field_is_zero(self):
    B = jnp.full(64, 5.0)
    dl_over_Bp = jnp.ones(64)
    result = trapped_fraction.calculate_bounce_averaged_trapped_fraction(
        B=B, dl_over_Bp=dl_over_Bp
    )
    np.testing.assert_allclose(result, 0.0, atol=1e-6)

  def test_bounce_averaged_trapped_fraction_small_epsilon_asymptotic(self):
    epsilon = 1e-3
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 512, endpoint=False)
    B = 5.0 / (1.0 + epsilon * jnp.cos(theta))
    dl_over_Bp = 1.0 + epsilon * jnp.cos(theta)
    result = trapped_fraction.calculate_bounce_averaged_trapped_fraction(
        B=B, dl_over_Bp=dl_over_Bp
    )
    # Analytical small-epsilon expansion: f_t = 1.46024 * sqrt(epsilon) + O(eps)
    expected = 1.46024 * np.sqrt(epsilon)
    np.testing.assert_allclose(result, expected, rtol=2e-2)


if __name__ == '__main__':
  absltest.main()
