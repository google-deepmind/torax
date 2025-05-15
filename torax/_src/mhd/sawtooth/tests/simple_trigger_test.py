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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.mhd import runtime_params as mhd_runtime_params
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.mhd.sawtooth import simple_trigger

_NRHO = 20


class SimpleTriggerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = geometry_pydantic_model.CircularConfig(
        n_rho=_NRHO
    ).build_geometry()
    self.mock_static_params = mock.ANY
    self.trigger = simple_trigger.SimpleTrigger()

  def _get_mock_core_profiles(
      self, q_face: np.ndarray, s_face: np.ndarray
  ) -> state.CoreProfiles:
    return mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        q_face=jnp.asarray(q_face),
        s_face=jnp.asarray(s_face),
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
    mock_core_profiles = self._get_mock_core_profiles(q_profile, s_profile)

    mock_dynamic_params = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        instance=True,
        mhd=mhd_runtime_params.DynamicMHDParams(
            sawtooth=mock.create_autospec(
                sawtooth_runtime_params.DynamicRuntimeParams,
                instance=True,
                trigger_params=simple_trigger.DynamicRuntimeParams(
                    s_critical=s_critical,
                    minimum_radius=minimum_radius,
                ),
            )
        ),
    )

    trigger_result, _ = self.trigger(
        self.mock_static_params,
        mock_dynamic_params,
        self.geo,
        mock_core_profiles,
    )
    self.assertEqual(trigger_result, expected_trigger)


if __name__ == '__main__':
  absltest.main()
