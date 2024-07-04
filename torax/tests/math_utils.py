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

"""Unit tests for torax.math_utils."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import scipy.integrate
from torax import math_utils


class MathUtilsTest(parameterized.TestCase):
  """Unit tests for the `torax.math_utils` module."""

  @parameterized.parameters([
      dict(seed=20221007, initial=None),
      dict(seed=20221007, initial=0.0),
      dict(seed=20221007, initial=1.0),
  ])
  def test_cumulative_trapz(self, seed, initial):
    """Test that cumulative_trapezoid matches the scipy implementation."""
    rng_state = jax.random.PRNGKey(seed)
    del seed  # Make sure seed isn't accidentally re-used

    rng_use_dim, rng_use_y, rng_use_x, _ = jax.random.split(
        rng_state, 4
    )
    dim = int(jax.random.randint(rng_use_dim, (1,), 1, 100)[0])
    y = jax.random.normal(rng_use_y, (dim,))
    del rng_use_y  # Make sure rng_use_y isn't accidentally re-used
    x = jax.random.normal(rng_use_x, (dim,))
    del rng_use_x  # Make sure rng_use_x isn't accidentally re-used

    cumulative = math_utils.cumulative_trapezoid(x, y, initial=initial)

    ref = scipy.integrate.cumulative_trapezoid(y, x, initial=initial)

    np.testing.assert_allclose(cumulative, ref)


if __name__ == '__main__':
  absltest.main()
