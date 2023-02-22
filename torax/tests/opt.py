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

"""Unit tests for torax.opt."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import torax  # This sets the precision (float64 or float32)

opt = torax.opt


class OptTest(parameterized.TestCase):
  """Unit tests for the `torax.opt` module."""

  def test_interp_bad_shape(self):
    """Test that `opt.interp` raises when called on bad shapes."""
    good_tree = jnp.zeros(2)
    good_coords = jnp.array([0.0, 1.0])
    good_desired_coord = jnp.array(0.5)
    # Wrong size of tree
    with self.assertRaises(AssertionError):
      opt.interp(jnp.zeros(3), good_coords, good_desired_coord)
    # Wrong rank of coords
    with self.assertRaises(AssertionError):
      opt.interp(good_tree, jnp.zeros(()), good_desired_coord)
    # Wrong rank of desired_coord
    with self.assertRaises(AssertionError):
      opt.interp(good_tree, good_coords, jnp.zeros(2))

  def test_interp_unsorted(self):
    """Test that `opt.interp` raises when called on unsorted coords."""
    tree = jnp.zeros(2)
    coords = jnp.array([1, 0])
    desired_coord = jnp.array(0.5)
    with self.assertRaises(jax.lib.xla_extension.XlaRuntimeError):
      opt.interp(tree, coords, desired_coord)

  def test_interp_nonincreases(self):
    """Test that `opt.interp` raises when called on nonincreasing coords."""
    tree = jnp.zeros(2)
    coords = jnp.zeros(2)
    desired_coord = jnp.array(0.0)
    with self.assertRaises(jax.lib.xla_extension.XlaRuntimeError):
      opt.interp(tree, coords, desired_coord)

  def test_interp_out_of_bounds(self):
    """Test that `opt.interp` raises on out of bounds desired_coord."""
    tree = jnp.zeros(2)
    coords = jnp.array([1.0, 2.0])
    # Too low
    with self.assertRaises(jax.lib.xla_extension.XlaRuntimeError):
      opt.interp(tree, coords, jnp.array(0.9))
    # Too high
    with self.assertRaises(jax.lib.xla_extension.XlaRuntimeError):
      opt.interp(tree, coords, jnp.array(2.1))

  def test_interp_canonical(self):
    """Test that `opt.interp` gets the right answer for a canonical input."""

    tree = jnp.array([
        [0.1, 0.2, 0.3],
        [1.1, 1.2, 1.3],
        [2.1, 2.2, 2.3],
    ])
    coords = jnp.array([1.4, 2.4, 3.4])
    desired_coord = jnp.array(1.9)
    result = opt.interp(tree, coords, desired_coord)
    ground_truth = jnp.array([0.6, 0.7, 0.8])
    np.testing.assert_allclose(result, ground_truth)

  def test_interp_exact(self):
    """Test that `opt.interp` for `exact match` edge case."""

    tree = jnp.array([
        [0.1, 0.2, 0.3],
        [1.1, 1.2, 1.3],
        [2.1, 2.2, 2.3],
    ])
    coords = jnp.array([1.4, 2.4, 3.4])

    # Cover the edge case where desired_coord exactly matches a coordinate in
    # the array.
    # Further, cover the edge case where the desired_coord exactly matches the
    # first or last coordinate in the array.
    for i in range(3):
      result = opt.interp(tree, coords, coords[i])
      ground_truth = tree[i]
      np.testing.assert_allclose(result, ground_truth)


if __name__ == '__main__':
  absltest.main()
