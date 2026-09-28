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


if __name__ == '__main__':
  absltest.main()
