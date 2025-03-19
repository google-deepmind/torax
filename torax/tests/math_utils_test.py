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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
from torax import math_utils
from torax.geometry import geometry
from torax.geometry import pydantic_model as geometry_pydantic_model

jax.config.update('jax_enable_x64', True)


class MathUtilsTest(parameterized.TestCase):

  @parameterized.product(
      initial=(None, 0.0),
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
    """Test that cell_integration is equivalent to scipy.integrate.trapezoid."""
    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(num_cell_grid_points + 1,)
    )
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=num_cell_grid_points
    ).build_geometry()

    np.testing.assert_allclose(
        math_utils.cell_integration(geometry.face_to_cell(x), geo),
        jax.scipy.integrate.trapezoid(x, geo.rho_face_norm),
        rtol=1e-6,  # 1e-7 rtol is too tight for this test to pass.
    )

  def test_area_integration(self):
    """Test that area_integration is equivalent to an analytical formula."""
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=200,
        elongation_LCFS=1.0,
    ).build_geometry()

    x = geo.rho_norm
    # For circular geometry, spr = 2*pi*rho*Rmin. Do the integration by hand.
    expected_area_integration = 2 / 3 * np.pi * geo.Rmin**2

    np.testing.assert_allclose(
        math_utils.area_integration(x, geo),
        expected_area_integration,
        rtol=1e-5,
    )

  def test_volume_integration(self):
    """Test that volume_integration is equivalent to an analytical formula."""
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=200,
        elongation_LCFS=1.0,
    ).build_geometry()

    x = geo.rho_norm
    # For circular geometry, vpr = 4*pi^2*rho*Rmin*Rmaj. Do integration by hand.
    expected_volume_integration = 4 / 3 * np.pi**2 * geo.Rmin**2 * geo.Rmaj

    np.testing.assert_allclose(
        math_utils.volume_integration(x, geo),
        expected_volume_integration,
        rtol=1e-5,
    )

  def test_line_average(self):
    """Test that line_average is equivalent to an analytical formula."""
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=200,
        elongation_LCFS=1.0,
    ).build_geometry()

    x = geo.rho_norm
    expected_line_average = 0.5

    np.testing.assert_allclose(
        math_utils.line_average(x, geo),
        expected_line_average,
        rtol=1e-5,
    )

  def test_volume_average(self):
    """Test that volume_average is equivalent to an analytical formula."""
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=200,
        elongation_LCFS=1.0,
    ).build_geometry()

    x = geo.rho_norm
    # For circular geometry, vpr = 4*pi^2*rho*Rmin*Rmaj. Do integration by hand.
    expected_volume_average = 2 / 3

    np.testing.assert_allclose(
        math_utils.volume_average(x, geo),
        expected_volume_average,
        rtol=1e-5,
    )

  def test_cell_integration_raises_when_shape_mismatch(
      self,
  ):
    geo = geometry_pydantic_model.CircularConfig(n_rho=10).build_geometry()
    with self.assertRaises(ValueError):
      math_utils.cell_to_face(jnp.array([1.0]), geo)

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
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=len(cell_values)
    ).build_geometry()
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
            math_utils.cell_integration(cell_values * geo.spr, geo),
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


if __name__ == '__main__':
  absltest.main()
