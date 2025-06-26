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
from jax import numpy as jnp
import numpy as np

from torax._src.output_tools import safety_factor_fit


class SafetyFactorFitTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.existing_x64_flag = jax.config.values.get("jax_enable_x64", False)
    jax.config.update("jax_enable_x64", True)

  def tearDown(self):
    jax.config.update("jax_enable_x64", self.existing_x64_flag)
    super().tearDown()

  @parameterized.named_parameters([
      dict(
          testcase_name="3x^2+0.5",
          transform=lambda x: 3 * x**2 + 0.5,
          expected_q_min=0.5,
          expected_rho_q_min=0.0,
      ),
      dict(
          testcase_name="2x^2+1",
          transform=lambda x: 2 * x**2 + 1,
          expected_q_min=1.0,
          expected_rho_q_min=0.0,
      ),
      dict(
          testcase_name="20*(x-0.5)^2",
          transform=lambda x: 20 * (x - 0.5) ** 2,
          expected_q_min=0.0,
          expected_rho_q_min=0.5,
      ),
      dict(
          testcase_name="-3x^2-0.5",
          transform=lambda x: -3 * x**2 - 0.5,
          expected_q_min=-3.5,
          expected_rho_q_min=1.0,
      ),
      dict(
          testcase_name="-2x^2-1",
          transform=lambda x: -2 * x**2 - 1,
          expected_q_min=-3.0,
          expected_rho_q_min=1.0,
      ),
      dict(
          testcase_name="-20*(x-0.5)^2",
          transform=lambda x: -20 * (x - 0.5) ** 2,
          expected_q_min=-5.0,
          expected_rho_q_min=0.0,
      ),
  ])
  def test_find_min_q_and_q_surface_intercepts_finds_minimum_q(
      self, transform, expected_q_min, expected_rho_q_min
  ):
    rho_norm_face = jnp.linspace(0.0, 1.0, 1000)
    q_face = transform(rho_norm_face)
    outputs = safety_factor_fit.find_min_q_and_q_surface_intercepts(
        rho_norm_face, q_face
    )
    np.testing.assert_almost_equal(
        outputs.rho_q_min,
        np.array(expected_rho_q_min),
    )
    np.testing.assert_almost_equal(
        outputs.q_min,
        np.array(expected_q_min),
    )

  @parameterized.named_parameters([
      dict(
          testcase_name="3x^2+0.5",
          transform=lambda x: 3 * x**2 + 0.5,
          expected_q_3_2=[-np.inf, 1 / np.sqrt(3)],
          expected_q_2_1=[-np.inf, 1 / np.sqrt(2)],
          expected_q_3_1=[-np.inf, np.sqrt(5 / 6)],
      ),
      dict(
          testcase_name="2x^2+1",
          transform=lambda x: 2 * x**2 + 1,
          expected_q_3_2=[-np.inf, 0.5],
          expected_q_2_1=[-np.inf, 1 / np.sqrt(2)],
          expected_q_3_1=[
              -np.inf,
              -np.inf,
          ],  # We don't find the root on the border.
      ),
      dict(
          testcase_name="20*(x-0.5)^2",
          transform=lambda x: 20 * (x - 0.5) ** 2,
          expected_q_3_2=[(10 - np.sqrt(30)) / 20, (10 + np.sqrt(30)) / 20],
          expected_q_2_1=[0.5 - 1 / np.sqrt(10), 0.5 + 1 / np.sqrt(10)],
          expected_q_3_1=[(5 - np.sqrt(15)) / 10, (5 + np.sqrt(15)) / 10],
      ),
      dict(
          testcase_name="-3x^2-0.5",
          transform=lambda x: -3 * x**2 - 0.5,
          expected_q_3_2=[-np.inf, -np.inf],
          expected_q_2_1=[-np.inf, -np.inf],
          expected_q_3_1=[-np.inf, -np.inf],
      ),
      dict(
          testcase_name="-2x^2-1",
          transform=lambda x: -2 * x**2 - 1,
          expected_q_3_2=[-np.inf, -np.inf],
          expected_q_2_1=[-np.inf, -np.inf],
          expected_q_3_1=[-np.inf, -np.inf],
      ),
      dict(
          testcase_name="-20*(x-0.5)^2",
          transform=lambda x: -20 * (x - 0.5) ** 2,
          expected_q_3_2=[-np.inf, -np.inf],
          expected_q_2_1=[-np.inf, -np.inf],
          expected_q_3_1=[-np.inf, -np.inf],
      ),
  ])
  def test_find_min_q_and_q_surface_intercepts_finds_q_surface_intercepts(
      self, transform, expected_q_3_2, expected_q_2_1, expected_q_3_1
  ):
    rho_norm_face = jnp.linspace(0.0, 1.0, 1000)
    q_face = transform(rho_norm_face)
    outputs = safety_factor_fit.find_min_q_and_q_surface_intercepts(
        rho_norm_face, q_face
    )
    q_3_2 = np.array([outputs.rho_q_3_2_first, outputs.rho_q_3_2_second])
    q_2_1 = np.array([outputs.rho_q_2_1_first, outputs.rho_q_2_1_second])
    q_3_1 = np.array([outputs.rho_q_3_1_first, outputs.rho_q_3_1_second])
    with self.subTest("q_3_2"):
      np.testing.assert_allclose(q_3_2, np.array(expected_q_3_2))
    with self.subTest("q_2_1"):
      np.testing.assert_allclose(q_2_1, np.array(expected_q_2_1))
    with self.subTest("q_3_1"):
      np.testing.assert_allclose(q_3_1, np.array(expected_q_3_1))

  def test_polynomial_fit_to_threes_on_exact_quadratic(self):
    """Test to give us an idea of the error on the quadratic fit."""
    rho_norm = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64)
    random_coeffs = jax.random.normal(
        key=jax.random.PRNGKey(0), shape=(3,), dtype=jnp.float64
    )
    q_face = (
        random_coeffs[0] * rho_norm**2
        + random_coeffs[1] * rho_norm
        + random_coeffs[2]
    )
    polyfit, rho_norm_3, q_face_3 = (
        safety_factor_fit._fit_polynomial_to_intervals_of_three(
            rho_norm, q_face
        )
    )
    chex.assert_shape(polyfit, (1, 3))
    chex.assert_shape(rho_norm_3, (1, 3))
    chex.assert_shape(q_face_3, (1, 3))
    np.testing.assert_allclose(polyfit[0], random_coeffs)

  def test_root_in_interval_finds_correct_roots(self):
    """Test correct roots found in each interval.

    y=4      *
      |
      |
    y=1  *       *
      |
      *--1---2---3----> x
    """
    # Given two polynomials that fit the above points.
    a_1, b_1, c_1 = 1.0, 0.0, 0.0
    a_2, b_2, c_2 = -3.0, 12.0, -8.0
    coeffs = np.array([[a_1, b_1, c_1], [a_2, b_2, c_2]])
    rho_norm = np.array([[0.0, 2.0], [2.0, 3.0]])
    # When we calculate roots for intercepts of the q=3/2 plane.
    intercept = 1.5
    intercepts = safety_factor_fit._root_in_interval(
        coeffs, rho_norm, intercept
    )
    # Then expect only one root to be found in first interval.
    np.testing.assert_allclose(
        intercepts[0], np.array([np.sqrt(intercept), -np.inf])
    )
    # And expect only one root in second interval (as only over [2.0, 3.0]]).
    np.testing.assert_allclose(intercepts[1], np.array([-np.inf, 2.912871]))


if __name__ == "__main__":
  absltest.main()
