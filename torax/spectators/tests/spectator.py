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

"""Tests for spectator.py."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax.spectators import spectator


class SpectatorTest(parameterized.TestCase):
  """Tests the simulation spectators."""

  def test_in_mem_jax_array_observer_collects_arrays(self):
    """Test the jax array observer adds arrays."""
    observer = spectator.InMemoryJaxArraySpectator()
    self.assertEmpty(observer.arrays)
    observer.observe('foo', jnp.zeros((1,)))
    observer.observe('bar', jnp.zeros((2,)))
    observer.observe('foo', jnp.ones((3,)))

    self.assertIn('foo', observer.arrays)
    self.assertLen(observer.arrays['foo'], 2)
    np.testing.assert_allclose(
        jnp.concatenate(observer.arrays['foo']), [0.0, 1.0, 1.0, 1.0]
    )

    self.assertIn('bar', observer.arrays)
    self.assertLen(observer.arrays['bar'], 1)
    np.testing.assert_allclose(observer.arrays['bar'][0], [0.0, 0.0])

    self.assertNotIn('baz', observer.arrays)

    observer.reset()
    self.assertEmpty(observer.arrays)


if __name__ == '__main__':
  absltest.main()
