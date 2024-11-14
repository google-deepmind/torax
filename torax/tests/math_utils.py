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
import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
from torax import geometry
from torax import math_utils

jax.config.update('jax_enable_x64', True)


class MathUtilsTest(parameterized.TestCase):
  """Unit tests for the `torax.math_utils` module."""

  @parameterized.product(
      initial=(None, 0.),
      axis=(-1, 1, -1),
      array_x=(False, True),
      dtype=(jnp.float32, jnp.float64),
      shape=((13,), (2, 4, 1, 3)),
  )
  def test_cumulative_trapezoid(self, axis, array_x, initial, dtype, shape):
    """Test that cumulative_trapezoid matches the scipy implementation."""
    rng_state = jax.random.PRNGKey(20221007)
    rng_use_y, rng_use_x, _ = jax.random.split(rng_state, 3)

    if axis == 1 and len(shape) == 1:
      self.skipTest('Axis out of range.')

    dx = 0.754
    y = jax.random.normal(rng_use_y, shape=shape, dtype=dtype)
    del rng_use_y  # Make sure rng_use_y isn't accidentally re-used
    if array_x:
      x = jax.random.normal(rng_use_x, (shape[axis],), dtype=dtype)
    else:
      x = None
    del rng_use_x  # Make sure rng_use_x isn't accidentally re-used

    cumulative = math_utils.cumulative_trapezoid(
        y, x, dx=dx, axis=axis, initial=initial
    )

    self.assertEqual(cumulative.dtype, y.dtype)

    ref = scipy.integrate.cumulative_trapezoid(
        y, x, dx=dx, axis=axis, initial=initial
    )

    atol = 3e-7 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(cumulative, ref, atol=atol)

  @parameterized.parameters(5, 50, 500)
  def test_cell_integration(self, num_cell_grid_points: int):
    """Test that the cell_integration method works as expected."""
    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(num_cell_grid_points + 1,)
    )
    geo = geometry.build_circular_geometry(n_rho=num_cell_grid_points)

    np.testing.assert_allclose(
        math_utils.cell_integration(geometry.face_to_cell(x), geo),
        jax.scipy.integrate.trapezoid(x, geo.rho_face_norm),
        rtol=1e-6,  # 1e-7 rtol is too tight for this test to pass.
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='with_equally_spaced_cell_values_value_preserved',
          cell_values=[1.0, 2.0, 3.0, 4.0],
          expected_face_values_except_right=np.array([0.5, 1.5, 2.5, 3.5]),
          preserved_quantity=math_utils.IntegralPreservationQuantity.VALUE,
      ),
      dict(
          testcase_name='with_sawtooth_cell_values_value_preserved',
          cell_values=[-1.0, 2.0, -3.0, 4.0],
          expected_face_values_except_right=np.array([-2.5, 0.5, -0.5, 0.5]),
          preserved_quantity=math_utils.IntegralPreservationQuantity.VALUE,
      ),
      dict(
          testcase_name='with_unevenly_spaced_cell_values_value_preserved',
          cell_values=[10, 6, 0, 20],
          expected_face_values_except_right=np.array([12, 8, 3, 10]),
          preserved_quantity=math_utils.IntegralPreservationQuantity.VALUE,
      ),
      dict(
          testcase_name='with_equally_spaced_cell_values_surface_preserved',
          cell_values=[1.0, 2.0, 3.0, 4.0],
          expected_face_values_except_right=np.array([0.5, 1.5, 2.5, 3.5]),
          preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
      ),
      dict(
          testcase_name='with_sawtooth_cell_values_surface_preserved',
          cell_values=[-1.0, 2.0, -3.0, 4.0],
          expected_face_values_except_right=np.array([-2.5, 0.5, -0.5, 0.5]),
          preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
      ),
      dict(
          testcase_name='with_unevenly_spaced_cell_values_surface_preserved',
          cell_values=[10, 6, 0, 20],
          expected_face_values_except_right=np.array([12, 8, 3, 10]),
          preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
      ),
      dict(
          testcase_name='with_equally_spaced_cell_values_volume_preserved',
          cell_values=[1.0, 2.0, 3.0, 4.0],
          expected_face_values_except_right=np.array([0.5, 1.5, 2.5, 3.5]),
          preserved_quantity=math_utils.IntegralPreservationQuantity.VOLUME,
      ),
      dict(
          testcase_name='with_sawtooth_cell_values_volume_preserved',
          cell_values=[-1.0, 2.0, -3.0, 4.0],
          expected_face_values_except_right=np.array([-2.5, 0.5, -0.5, 0.5]),
          preserved_quantity=math_utils.IntegralPreservationQuantity.VOLUME,
      ),
      dict(
          testcase_name='with_unevenly_spaced_cell_values_volume_preserved',
          cell_values=[10, 6, 0, 20],
          expected_face_values_except_right=np.array([12, 8, 3, 10]),
          preserved_quantity=math_utils.IntegralPreservationQuantity.VOLUME,
      ),
  )
  def test_cell_to_face(
      self,
      cell_values: list[float],
      expected_face_values_except_right: np.ndarray,
      preserved_quantity: math_utils.IntegralPreservationQuantity,
  ):
    """Test that the cell_to_face method works as expected."""
    geo = geometry.build_circular_geometry(n_rho=len(cell_values))
    cell_values = jnp.array(cell_values, dtype=jnp.float32)

    face_values = math_utils.cell_to_face(cell_values, geo, preserved_quantity)
    chex.assert_shape(face_values, (len(cell_values) + 1,))

    np.testing.assert_array_equal(
        face_values[:-1], expected_face_values_except_right
    )
    # Check the integral is preserved.
    match preserved_quantity:
      case math_utils.IntegralPreservationQuantity.VALUE:
        np.testing.assert_allclose(
            math_utils.cell_integration(cell_values, geo),
            jax.scipy.integrate.trapezoid(face_values, geo.rho_face_norm),
        )
      case math_utils.IntegralPreservationQuantity.SURFACE:
        np.testing.assert_allclose(
            math_utils.cell_integration(cell_values * geo.spr_cell, geo),
            jax.scipy.integrate.trapezoid(
                face_values * geo.spr_face, geo.rho_face_norm
            ),
        )
      case math_utils.IntegralPreservationQuantity.VOLUME:
        np.testing.assert_allclose(
            math_utils.cell_integration(cell_values * geo.vpr, geo),
            jax.scipy.integrate.trapezoid(
                face_values * geo.vpr_face, geo.rho_face_norm
            ),
        )

  def test_cell_to_face_raises_when_too_few_values(self,):
    """Test that the cell_to_face method raises when too few values are provided."""
    geo = geometry.build_circular_geometry(n_rho=1)
    with self.assertRaises(ValueError):
      math_utils.cell_to_face(jnp.array([1.0], dtype=np.float32), geo)


if __name__ == '__main__':
  absltest.main()
