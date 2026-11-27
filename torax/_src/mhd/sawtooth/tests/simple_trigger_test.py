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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.geometry import circular_geometry
from torax._src.mhd import runtime_params as mhd_runtime_params
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.mhd.sawtooth import simple_trigger
from torax._src.mhd.sawtooth import trigger_base
from torax._src.test_utils import core_profile_helpers

jax.config.update('jax_enable_x64', jax_utils.get_dtype() == jnp.float64)

_NRHO = 20
_RHO_FACE = np.linspace(0.0, 1.0, _NRHO + 1)


class SimpleTriggerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = (
        circular_geometry.CircularConfig(n_rho=_NRHO)
        .build_runtime_params(0.0)
        .geometry
    )
    self.trigger = simple_trigger.SimpleTrigger()

  def _get_core_profiles(
      self, q_face: np.ndarray, s_face: np.ndarray
  ) -> core_profiles_lib.CoreProfiles:
    core_profiles = core_profile_helpers.make_zero_core_profiles(self.geo)
    dtype = jax_utils.get_dtype()
    return dataclasses.replace(
        core_profiles,
        poloidal_flux=dataclasses.replace(
            core_profiles.poloidal_flux,
            q_face=jnp.asarray(q_face, dtype=dtype),
            s_face=jnp.asarray(s_face, dtype=dtype),
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='q_always_above_1',
          q_profile=np.linspace(1.1, 3.0, _NRHO + 1),
          s_profile=np.linspace(0.0, 1.0, _NRHO + 1),
          expected_trigger=False,
      ),
      dict(
          testcase_name='q_always_below_1',
          q_profile=0.9 * np.ones(_NRHO + 1),
          s_profile=np.linspace(0.0, 1.0, _NRHO + 1),
          expected_trigger=False,
      ),
      dict(
          testcase_name='q_equals_1_at_axis_only',
          q_profile=1.0 + np.linspace(0, 1, _NRHO + 1) ** 2,
          s_profile=np.linspace(-0.5, 1.0, _NRHO + 1),
          expected_trigger=False,
      ),
      dict(
          testcase_name='q1_surface_below_min_radius',
          q_profile=0.9 + np.linspace(0, 1, _NRHO + 1) ** 2,
          s_profile=np.linspace(1.0, 2.0, _NRHO + 1),
          minimum_radius=0.4,
          expected_trigger=False,
      ),
      dict(
          testcase_name='shear_below_critical',
          q_profile=0.9 + np.linspace(0, 1, _NRHO + 1),
          s_profile=np.linspace(0.0, 0.2, _NRHO + 1),
          s_critical=0.3,
          minimum_radius=0.05,
          expected_trigger=False,
      ),
      dict(
          testcase_name='reversed_shear_q1_exists_shear_low',
          q_profile=1.5 - 4.0 * (np.linspace(0, 1, _NRHO + 1) - 0.5) ** 2,
          s_profile=np.linspace(0.5, -0.5, _NRHO + 1),
          s_critical=0.3,
          minimum_radius=0.1,
          expected_trigger=False,
      ),
      dict(
          testcase_name='monotonic_q_shear_high',
          q_profile=0.9 + np.linspace(0, 1, _NRHO + 1),
          s_profile=np.linspace(0.0, 1.0, _NRHO + 1),
          s_critical=0.05,
          minimum_radius=0.05,
          expected_trigger=True,
      ),
      dict(
          testcase_name='reversed_shear_q1_exists_shear_high',
          q_profile=1.5 - 4.0 * (np.linspace(0, 1, _NRHO + 1) - 0.5) ** 2,
          s_profile=np.linspace(-0.5, 0.5, _NRHO + 1),
          s_critical=0.3,
          minimum_radius=0.1,
          expected_trigger=True,
      ),
  )
  def test_simple_trigger(
      self,
      q_profile: np.ndarray,
      s_profile: np.ndarray,
      expected_trigger: bool,
      s_critical: float = 0.3,
      minimum_radius: float = 0.1,
  ):
    core_profiles = self._get_core_profiles(q_profile, s_profile)

    mock_params = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        instance=True,
        mhd=mhd_runtime_params.RuntimeParams(
            discrete_sawtooth=mock.create_autospec(
                sawtooth_runtime_params.RuntimeParams,
                instance=True,
                trigger_params=simple_trigger.RuntimeParams(
                    s_critical=s_critical,
                    minimum_radius=minimum_radius,
                ),
            )
        ),
    )

    trigger_result, _ = self.trigger(
        mock_params,
        self.geo,
        core_profiles,
    )
    self.assertEqual(trigger_result, expected_trigger)


class FindRhoNormQ1Test(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='upward_crossing',
          q_face=0.55 + 2.0 * _RHO_FACE,
          expected=0.225,
      ),
      dict(
          testcase_name='downward_crossing',
          q_face=1.45 - 2.0 * _RHO_FACE,
          expected=0.225,
      ),
      dict(
          testcase_name='multiple_crossings_outermost',
          q_face=0.8 + 1.6 * np.abs(_RHO_FACE - 0.4),
          expected=0.525,
      ),
  )
  def test_find_rho_norm_q1_standard_cases(
      self,
      q_face: np.ndarray,
      expected: float,
  ):
    dtype = jax_utils.get_dtype()
    result = float(
        trigger_base.find_rho_norm_q1(
            jnp.asarray(q_face, dtype=dtype),
            jnp.asarray(_RHO_FACE, dtype=dtype),
        )
    )
    atol = 1e-6 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=atol)

  @parameterized.named_parameters(
      dict(
          testcase_name='flat_q_below_1_within_eps',
          q_face=np.full(_NRHO + 1, 1.0 - 5e-8),
          expected=0.0,
      ),
      dict(
          testcase_name='tangent_from_above',
          q_face=1.0 + (_RHO_FACE - 0.5) ** 2,
          expected=0.0,
      ),
      dict(
          testcase_name='exact_face_hit_from_below',
          q_face=0.5 + _RHO_FACE,
          expected=0.5,
      ),
      dict(
          testcase_name='small_slope_downward_crossing',
          q_face=np.where(_RHO_FACE < 0.5, 1.0 + 4e-8, 1.0 - 4e-8),
          expected=0.475,
          f32_expected=0.45,
      ),
  )
  def test_find_rho_norm_q1_edge_cases(
      self,
      q_face: np.ndarray,
      expected: float,
      f32_expected: float | None = None,
  ):
    dtype = jax_utils.get_dtype()
    result = float(
        trigger_base.find_rho_norm_q1(
            jnp.asarray(q_face, dtype=dtype),
            jnp.asarray(_RHO_FACE, dtype=dtype),
        )
    )
    if dtype == jnp.float32 and f32_expected is not None:
      expected = f32_expected
    atol = 1e-6 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=atol)

  @parameterized.named_parameters(
      dict(
          testcase_name='interior_crossing',
          q_face=0.55 + 2.0 * _RHO_FACE,
          expected_nonzero={4: -0.25, 5: -0.25},
      ),
      dict(
          testcase_name='flat_q_1',
          q_face=np.ones(_NRHO + 1),
          expected_nonzero={},
      ),
  )
  def test_find_rho_norm_q1_grad(
      self,
      q_face: np.ndarray,
      expected_nonzero: dict[int, float],
  ):
    dtype = jax_utils.get_dtype()
    # In `flat_q_1`, every interval has q1 == q2 == 1.0, exercising the safe
    # denominator so reverse-mode AD through `jnp.where` avoids 0/0 NaNs.
    grad = jax.grad(trigger_base.find_rho_norm_q1)(
        jnp.asarray(q_face, dtype=dtype),
        jnp.asarray(_RHO_FACE, dtype=dtype),
    )
    expected_grad = np.zeros(_NRHO + 1)
    for idx, val in expected_nonzero.items():
      expected_grad[idx] = val
    atol = 1e-5 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(grad, expected_grad, rtol=0.0, atol=atol)


if __name__ == '__main__':
  absltest.main()
