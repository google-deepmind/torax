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
    rho_norm_face = jnp.linspace(0.0, 1.0, 1001)
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
          expected_q_3_2=[-np.inf, 0.5773500721500722],
          expected_q_2_1=[-np.inf, 0.7071067137809187],
          expected_q_3_1=[-np.inf, 0.91287086758],
      ),
      dict(
          testcase_name="2x^2+1",
          transform=lambda x: 2 * x**2 + 1,
          expected_q_3_2=[-np.inf, 0.5],
          expected_q_2_1=[-np.inf, 0.7071067137809187],
          expected_q_3_1=[
              -np.inf,
              1.0,
          ],
      ),
      dict(
          testcase_name="20*(x-0.5)^2",
          transform=lambda x: 20 * (x - 0.5) ** 2,
          expected_q_3_2=[0.2261389396709324, 0.7738610603290677],
          expected_q_2_1=[0.18377251184834123, 0.8162274881516588],
          expected_q_3_1=[0.11270193548387099, 0.887298064516129],
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
    rho_norm_face = jnp.linspace(0.0, 1.0, 1001)
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

  def test_linear_intercepts(self):
    rho_norm = jnp.array([0.0, 0.5, 1.0])
    q_face = jnp.array([1.0, 2.0, 3.0])

    with self.subTest("target_1_5"):
      intercepts = safety_factor_fit._linear_intercepts(rho_norm, q_face, 1.5)
      np.testing.assert_allclose(intercepts, np.array([-np.inf, 0.25, -np.inf]))

    with self.subTest("target_2_0"):
      intercepts = safety_factor_fit._linear_intercepts(rho_norm, q_face, 2.0)
      np.testing.assert_allclose(intercepts, np.array([-np.inf, 0.5, -np.inf]))

    with self.subTest("target_1_0"):
      intercepts = safety_factor_fit._linear_intercepts(rho_norm, q_face, 1.0)
      np.testing.assert_allclose(intercepts, np.array([0.0, -np.inf, -np.inf]))


if __name__ == "__main__":
  absltest.main()
