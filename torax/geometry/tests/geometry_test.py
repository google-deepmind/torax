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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax.geometry import circular_geometry
from torax.geometry import geometry


class GeometryTest(parameterized.TestCase):
  """Unit tests for the `geometry` module."""

  @parameterized.parameters([
      dict(n_rho=25, seed=20220930),
  ])
  def test_face_to_cell(self, n_rho, seed):
    """Compare `face_to_cell` to a PINT reference."""

    rng_state = jax.random.PRNGKey(seed)
    del seed  # Make sure seed isn't accidentally re-used

    # Generate face variables
    rng_use, _ = jax.random.split(rng_state)
    face = jax.random.normal(rng_use, (n_rho + 1,))
    del rng_use  # Make sure rng_use isn't accidentally re-used

    # Convert face values to cell values using jax code being tested
    cell_jax = geometry.face_to_cell(jnp.array(face))

    # Make ground truth
    cell_np = _pint_face_to_cell(n_rho, face)

    np.testing.assert_allclose(cell_jax, cell_np)

  def test_none_z_magnetic_axis_raises_an_error(self):
    geo = circular_geometry.build_circular_geometry()
    geo = dataclasses.replace(geo, _z_magnetic_axis=None)

    with self.subTest('non_jitted_function'):
      with self.assertRaisesRegex(
          ValueError, 'Geometry does not have a z magnetic axis.'
      ):
        geo.z_magnetic_axis()

    with self.subTest('jitted_function'):
      with self.assertRaisesRegex(
          ValueError, 'Geometry does not have a z magnetic axis.'
      ):
        foo = jax.jit(geo.z_magnetic_axis)
        _ = foo()


def _pint_face_to_cell(n_rho, face):
  cell = np.zeros(n_rho)
  cell[:] = 0.5 * (face[1:] + face[:-1])
  return cell


if __name__ == '__main__':
  absltest.main()
