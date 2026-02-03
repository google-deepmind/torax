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
from torax._src import math_utils
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry

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
    geo = circular_geometry.CircularConfig(
        n_rho=num_cell_grid_points
    ).build_geometry()

    np.testing.assert_allclose(
        math_utils.cell_integration(geometry.face_to_cell(x), geo),
        jax.scipy.integrate.trapezoid(x, geo.rho_face_norm),
        rtol=1e-6,  # 1e-7 rtol is too tight for this test to pass.
    )

  def test_area_integration(self):
    """Test that area_integration is equivalent to an analytical formula."""
    geo = circular_geometry.CircularConfig(
        n_rho=200,
        elongation_LCFS=1.0,
    ).build_geometry()

    x = geo.rho_norm
    # For circular geometry, spr = 2*pi*rho*a_minor. Do the integration by hand.
    expected_area_integration = 2 / 3 * np.pi * geo.a_minor**2

    np.testing.assert_allclose(
        math_utils.area_integration(x, geo),
        expected_area_integration,
        rtol=1e-5,
    )

  def test_volume_integration(self):
    """Test that volume_integration is equivalent to an analytical formula."""
    geo = circular_geometry.CircularConfig(
        n_rho=200,
        elongation_LCFS=1.0,
    ).build_geometry()

    x = geo.rho_norm
    # For circular geometry, vpr = 4*pi^2*rho*a_minor*R_major.
    # Do integration by hand.
    expected_volume_integration = (
        4 / 3 * np.pi**2 * geo.a_minor**2 * geo.R_major
    )

    np.testing.assert_allclose(
        math_utils.volume_integration(x, geo),
        expected_volume_integration,
        rtol=1e-5,
    )

  def test_line_average(self):
    """Test that line_average is equivalent to an analytical formula."""
    geo = circular_geometry.CircularConfig(
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
    geo = circular_geometry.CircularConfig(
        n_rho=200,
        elongation_LCFS=1.0,
    ).build_geometry()

    x = geo.rho_norm
    # For circular geometry, vpr = 4*pi^2*rho*a_minor*R_major.
    # Do integration by hand.
    expected_volume_average = 2 / 3

    np.testing.assert_allclose(
        math_utils.volume_average(x, geo),
        expected_volume_average,
        rtol=1e-5,
    )

  def test_cell_integration_raises_when_shape_mismatch(
      self,
  ):
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
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
    geo = circular_geometry.CircularConfig(
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

  @parameterized.parameters(5, 50)
  def test_cumulative_cell_integration(self, num_cell_grid_points: int):
    """Tests cumulative_cell_integration against cell_integration."""
    geo = circular_geometry.CircularConfig(
        n_rho=num_cell_grid_points
    ).build_geometry()
    x = jax.random.uniform(jax.random.PRNGKey(1), shape=(num_cell_grid_points,))

    cumulative_result = math_utils.cumulative_cell_integration(x, geo)
    expected = np.zeros(num_cell_grid_points)

    for i in range(len(cumulative_result)):
      expected[i] = np.sum(x[: i + 1] * geo.drho_norm[: i + 1])

    np.testing.assert_allclose(
        cumulative_result,
        expected,
    )

  @parameterized.parameters(5, 50)
  def test_cumulative_area_integration(self, num_cell_grid_points: int):
    """Tests cumulative_area_integration against area_integration."""
    geo = circular_geometry.CircularConfig(
        n_rho=num_cell_grid_points
    ).build_geometry()
    x = jax.random.uniform(jax.random.PRNGKey(2), shape=(num_cell_grid_points,))

    cumulative_result = math_utils.cumulative_area_integration(x, geo)
    expected = np.zeros(num_cell_grid_points)

    for i in range(len(cumulative_result)):
      expected[i] = np.sum(
          x[: i + 1] * geo.spr[: i + 1] * geo.drho_norm[: i + 1]
      )

    np.testing.assert_allclose(
        cumulative_result,
        expected,
    )

  @parameterized.parameters(5, 50)
  def test_cumulative_volume_integration(self, num_cell_grid_points: int):
    """Tests cumulative_volume_integration against volume_integration."""
    geo = circular_geometry.CircularConfig(
        n_rho=num_cell_grid_points
    ).build_geometry()
    x = jax.random.uniform(jax.random.PRNGKey(3), shape=(num_cell_grid_points,))

    cumulative_result = math_utils.cumulative_volume_integration(x, geo)
    expected = np.zeros(num_cell_grid_points)

    for i in range(len(cumulative_result)):
      expected[i] = np.sum(
          x[: i + 1] * geo.vpr[: i + 1] * geo.drho_norm[: i + 1]
      )

    np.testing.assert_allclose(
        cumulative_result,
        expected,
    )

  @parameterized.parameters(1e-14, 1e-6, 1e-4, 0.1)
  def test_inverse_softplus_small_values(self, value):
    x_val = jnp.array(value)
    y_val = math_utils.inverse_softplus(x_val)
    x_rec = jax.nn.softplus(y_val)
    np.testing.assert_allclose(x_val, x_rec, rtol=1e-6)

  @parameterized.parameters(1.0, 5.0, 10.0)
  def test_inverse_softplus_medium_values(self, value):
    x_val = jnp.array(value)
    y_val = math_utils.inverse_softplus(x_val)
    x_rec = jax.nn.softplus(y_val)
    np.testing.assert_allclose(x_val, x_rec, rtol=1e-6)

  @parameterized.parameters(25.0, 50.0, 100.0)
  def test_inverse_softplus_large_values(self, value):
    x_val = jnp.array(value)
    y_val = math_utils.inverse_softplus(x_val)
    np.testing.assert_allclose(x_val, y_val, rtol=1e-6)
    x_rec = jax.nn.softplus(y_val)
    np.testing.assert_allclose(x_val, x_rec, rtol=1e-6)

  @parameterized.parameters(-20, -10, -1, 1e-10, 1e-6, 0.1, 1.0, 10.0, 100.0)
  def test_softplus_round_trip(self, value):
    x = jnp.array(value)
    y = jax.nn.softplus(x)
    x_rec = math_utils.inverse_softplus(y)
    np.testing.assert_allclose(x, x_rec, rtol=1e-6)

  def test_smooth_sqrt_c1_continuity(self):
    """Tests C1 continuity at x=epsilon."""
    epsilon = 1.0
    # Value at epsilon
    val = math_utils.smooth_sqrt(jnp.array(epsilon), epsilon=epsilon)
    expected_val = jnp.sqrt(epsilon)
    np.testing.assert_allclose(val, expected_val, rtol=1e-14)

    # Gradient at epsilon
    grad_fn = jax.grad(lambda x: math_utils.smooth_sqrt(x, epsilon=epsilon))
    grad = grad_fn(jnp.array(epsilon))
    expected_grad = 0.5 / jnp.sqrt(epsilon)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-14)

    # Check left/right gradients numerically close to epsilon
    delta = 1e-6
    grad_left = grad_fn(epsilon - delta)
    grad_right = grad_fn(epsilon + delta)
    # Allow some tolerance due to curvature changes
    np.testing.assert_allclose(grad_left, expected_grad, rtol=1e-4)
    np.testing.assert_allclose(grad_right, expected_grad, rtol=1e-4)

  @parameterized.parameters(1.0, 10.0, 100.0, 1e5, 1e10, 1e15)
  def test_smooth_sqrt_equivalence_positive(self, value):
    """Tests equivalence to sqrt(x) for x >= epsilon."""
    epsilon = 1.0
    x = jnp.array(value, dtype=jnp.float64)
    res = math_utils.smooth_sqrt(x, epsilon=epsilon)
    expected = jnp.sqrt(x)
    np.testing.assert_allclose(res, expected, rtol=1e-14)

  @parameterized.parameters(0.0, -1e-15, -1e-10, -1.0, -1e2, -1e5, -1e10, -1e15)
  def test_smooth_sqrt_negative_branch(self, value):
    """Tests positive values and gradients for x < epsilon."""
    epsilon = 1.0
    x = jnp.array(value, dtype=jnp.float64)

    # Value check
    result = math_utils.smooth_sqrt(x, epsilon=epsilon)
    self.assertGreater(result, 0.0)

    # Gradient check
    grad_fn = jax.grad(lambda u: math_utils.smooth_sqrt(u, epsilon=epsilon))
    grad = grad_fn(x)
    self.assertGreater(grad, 0.0)
    self.assertFalse(jnp.isnan(grad))
    self.assertFalse(jnp.isinf(grad))

  @parameterized.named_parameters(
      dict(
          testcase_name='linear_scale',
          x=np.array([1.0, 2.0, 3.5, 5.0, 6.0]),
          smoothing_start=2.0,
          smoothing_end=5.0,
          y_left=1.0,
          y_right=10.0,
          log_scale=False,
          expected=np.array([1.0, 1.0, 5.5, 10.0, 10.0]),
      ),
      dict(
          testcase_name='log_scale',
          x=np.array([0.5, 1.0, np.sqrt(10.0), 10.0, 11.0]),
          smoothing_start=1.0,
          smoothing_end=10.0,
          y_left=1.0,
          y_right=10.0,
          log_scale=True,
          expected=np.array([1.0, 1.0, 5.5, 10.0, 10.0]),
      ),
  )
  def test_sigmoid_transition(
      self,
      x,
      smoothing_start,
      smoothing_end,
      y_left,
      y_right,
      log_scale,
      expected,
  ):
    got = math_utils.smoothstep_transition(
        x, smoothing_start, smoothing_end, y_left, y_right, log_scale
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
