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
from jax import numpy as jnp
import numpy as np
from torax import math_utils
from torax.fvm import cell_variable
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.mhd.sawtooth import flatten_profile


_NRHO = 20  # Define grid size for tests


class FlattenProfileTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = geometry_pydantic_model.CircularConfig(
        n_rho=_NRHO
    ).build_geometry()

  def _create_profile(self, values: chex.Array) -> cell_variable.CellVariable:
    """Helper to create a CellVariable for testing."""
    return cell_variable.CellVariable(
        value=jnp.array(values),
        dr=self.geo.drho_norm,
        left_face_grad_constraint=jnp.array(0.0),
        left_face_constraint=None,
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(values[-1]),
    )

  # pylint: disable=g-unreachable-test-method
  def _check_conservation_within_mixing_radius(
      self,
      profile_before: cell_variable.CellVariable,
      profile_after: cell_variable.CellVariable,
      rho_norm_mixing: float,
  ):
    """Checks volume integral conservation within the mixing radius."""
    rho_norm = self.geo.rho_norm
    idx_mixing = jnp.searchsorted(rho_norm, rho_norm_mixing)
    redistribution_mask = jnp.arange(rho_norm.shape[0]) < idx_mixing

    integrand_before_masked = jnp.where(
        redistribution_mask, profile_before.value, 0.0
    )
    integral_before = math_utils.volume_integration(
        integrand_before_masked, self.geo
    )

    integrand_after_masked = jnp.where(
        redistribution_mask, profile_after.value, 0.0
    )
    integral_after = math_utils.volume_integration(
        integrand_after_masked, self.geo
    )

    np.testing.assert_allclose(
        integral_before,
        integral_after,
        rtol=1e-6,
        err_msg='Integral conservation within mixing radius failed',
    )

  def _check_total_conservation(
      self,
      profile_before: cell_variable.CellVariable,
      profile_after: cell_variable.CellVariable,
  ):
    """Checks total volume integral conservation."""

    integral_before = math_utils.volume_integration(
        profile_before.value, self.geo
    )

    integral_after = math_utils.volume_integration(
        profile_after.value, self.geo
    )

    np.testing.assert_allclose(
        integral_before,
        integral_after,
        rtol=1e-6,
        err_msg='Integral conservation failed',
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='monotonic_rising_profile_flattening=1.01',
          rho_norm_q1=0.3,
          rho_norm_mixing=0.5,
          flatten_factor=1.01,
          initial_values=1.0 + 4.0 * np.linspace(0, 1, _NRHO) ** 2,
      ),
      dict(
          testcase_name='monotonic_falling_profile_flattening=1.01',
          rho_norm_q1=0.3,
          rho_norm_mixing=0.5,
          flatten_factor=1.01,
          initial_values=5.0 - 4.0 * np.linspace(0, 1, _NRHO) ** 2,
      ),
      dict(
          testcase_name='monotonic_falling_profile_flattening=1.0',
          rho_norm_q1=0.3,
          rho_norm_mixing=0.5,
          flatten_factor=1.0,
          initial_values=5.0 - 4.0 * np.linspace(0, 1, _NRHO) ** 2,
      ),
      dict(
          testcase_name='hollow_profile_c=1.01',
          rho_norm_q1=0.4,
          rho_norm_mixing=0.6,
          flatten_factor=1.01,
          initial_values=2.0
          - 1.5 * np.exp(-(((np.linspace(0, 1, _NRHO) - 0.2) / 0.1) ** 2)),
      ),
      dict(
          testcase_name='q1_near_axis_c=1.01',
          rho_norm_q1=0.08,  # Close to axis
          rho_norm_mixing=0.2,
          flatten_factor=1.01,
          initial_values=5.0 - 4.0 * np.linspace(0, 1, _NRHO) ** 2,
      ),
      dict(
          testcase_name='mix_near_edge_c=1.01',
          rho_norm_q1=0.5,
          rho_norm_mixing=0.98,  # Close to edge
          flatten_factor=1.01,
          initial_values=5.0 - 4.0 * np.linspace(0, 1, _NRHO) ** 2,
      ),
      dict(
          testcase_name='large_flatten_factor',
          rho_norm_q1=0.3,
          rho_norm_mixing=0.5,
          flatten_factor=1.2,
          initial_values=5.0 - 4.0 * np.linspace(0, 1, _NRHO) ** 2,
      ),
      dict(
          testcase_name='mix_equals_q1',
          rho_norm_q1=0.4,
          rho_norm_mixing=0.4,
          flatten_factor=1.01,
          initial_values=5.0 - 4.0 * np.linspace(0, 1, _NRHO) ** 2,
      ),
  )
  def test_flatten_profile_logic_and_conservation(
      self,
      rho_norm_q1: float,
      rho_norm_mixing: float,
      flatten_factor: float,
      initial_values: np.ndarray,
  ):
    initial_profile = self._create_profile(initial_values)

    flattened_profile = flatten_profile.flatten_density_profile(
        rho_norm_q1=jnp.array(rho_norm_q1),
        rho_norm_mixing=jnp.array(rho_norm_mixing),
        flattening_factor=jnp.array(flatten_factor),
        original_density_profile=initial_profile,
        geo=self.geo,
    )

    # Basic verifications
    self.assertIsInstance(flattened_profile, cell_variable.CellVariable)
    self.assertEqual(flattened_profile.value.shape, initial_profile.value.shape)
    self.assertEqual(flattened_profile.value.shape, initial_profile.value.shape)
    self.assertFalse(
        np.allclose(initial_profile.value, flattened_profile.value)
    )

    with self.subTest('conservation_within_mixing_radius'):
      self._check_conservation_within_mixing_radius(
          initial_profile, flattened_profile, rho_norm_mixing
      )

    with self.subTest('total_conservation'):
      self._check_total_conservation(initial_profile, flattened_profile)

    # Detailed checks on profile shape
    rho_norm = self.geo.rho_norm
    idx_mixing = np.searchsorted(rho_norm, rho_norm_mixing)
    val_after = flattened_profile.value

    with self.subTest('outer_region_unchanged'):
      if idx_mixing < _NRHO:
        np.testing.assert_allclose(
            val_after[idx_mixing:],
            initial_profile.value[idx_mixing:],
            err_msg='Profile changed outside mixing radius',
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='peaked_density_hollow_temperature',
          initial_density_values=3.0
          + 1.5 * np.exp(-((np.linspace(0, 1, _NRHO) / 0.2) ** 2)),
          initial_temperature_values=6.0
          - 3.0 * np.exp(-(((np.linspace(0, 1, _NRHO) - 0.3) / 0.15) ** 2)),
      ),
      dict(
          testcase_name='hollow_density_hollow_temperature',
          initial_density_values=3.0
          - 1.5 * np.exp(-(((np.linspace(0, 1, _NRHO) - 0.3) / 0.15) ** 2)),
          initial_temperature_values=6.0
          - 3.0 * np.exp(-(((np.linspace(0, 1, _NRHO) - 0.3) / 0.15) ** 2)),
      ),
      dict(
          testcase_name='peaked_density_peaked_temperature',
          initial_density_values=3.0
          + 1.5 * np.exp(-((np.linspace(0, 1, _NRHO) / 0.2) ** 2)),
          initial_temperature_values=6.0
          + 3.0 * np.exp(-((np.linspace(0, 1, _NRHO) / 0.2) ** 2)),
      ),
      dict(
          testcase_name='hollow_density_peaked_temperature',
          initial_density_values=3.0
          - 1.5 * np.exp(-(((np.linspace(0, 1, _NRHO) - 0.3) / 0.15) ** 2)),
          initial_temperature_values=6.0
          + 3.0 * np.exp(-((np.linspace(0, 1, _NRHO) / 0.2) ** 2)),
      ),
  )
  def test_temperature_profile_flattening_and_energy_conservation(
      self,
      initial_density_values: np.ndarray,
      initial_temperature_values: np.ndarray,
  ):
    initial_density_profile = self._create_profile(initial_density_values)
    initial_temperature_profile = self._create_profile(
        initial_temperature_values
    )
    rho_norm_q1 = 0.3
    rho_norm_mixing = 0.5
    flatten_factor = 1.01

    flattened_density_profile = flatten_profile.flatten_density_profile(
        rho_norm_q1=jnp.array(rho_norm_q1),
        rho_norm_mixing=jnp.array(rho_norm_mixing),
        flattening_factor=jnp.array(flatten_factor),
        original_density_profile=initial_density_profile,
        geo=self.geo,
    )

    flattened_temperature_profile = flatten_profile.flatten_temperature_profile(
        rho_norm_q1=jnp.array(rho_norm_q1),
        rho_norm_mixing=jnp.array(rho_norm_mixing),
        flattening_factor=jnp.array(flatten_factor),
        original_temperature_profile=initial_temperature_profile,
        original_density_profile=initial_density_profile,
        flattened_density_profile=flattened_density_profile,
        geo=self.geo,
    )

    initial_pressure_profile = self._create_profile(
        initial_temperature_profile.value * initial_density_profile.value
    )
    flattened_pressure_profile = self._create_profile(
        flattened_temperature_profile.value * flattened_density_profile.value
    )

    with self.subTest('conservation_within_mixing_radius'):
      self._check_conservation_within_mixing_radius(
          initial_pressure_profile, flattened_pressure_profile, rho_norm_mixing
      )

    with self.subTest('total_conservation'):
      self._check_total_conservation(
          initial_pressure_profile, flattened_pressure_profile
      )


if __name__ == '__main__':
  absltest.main()
