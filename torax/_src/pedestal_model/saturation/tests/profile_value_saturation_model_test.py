# Copyright 2026 DeepMind Technologies Limited
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
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model.saturation import profile_value_saturation_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class FromPedestalModelSaturationModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    self.torax_config = model_config.ToraxConfig.from_dict(config)
    self.provider = build_runtime_params.RuntimeParamsProvider.from_config(
        self.torax_config
    )
    self.runtime_params = self.provider(t=0.0)
    self.geo = self.torax_config.geometry.build_provider(t=0.0)
    self.source_models = self.torax_config.sources.build_models()
    self.neoclassical_models = self.torax_config.neoclassical.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        self.runtime_params,
        self.geo,
        self.source_models,
        self.neoclassical_models,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='active',
          # T_current >> T_target -> saturation is active.
          T_target_over_T_current=1e-3,
      ),
      dict(
          testcase_name='inactive',
          # T_current << T_target -> no saturation.
          T_target_over_T_current=1e3,
      ),
  )
  def test_saturation_multiplier(
      self,
      T_target_over_T_current,
  ):
    saturation_model = (
        profile_value_saturation_model.ProfileValueSaturationModel()
    )

    # For this test, we put the pedestal top at the last grid point.
    ped_top_idx = -1
    current_T_e_ped = self.core_profiles.T_e.face_value()[ped_top_idx]

    # Construct a pedestal output that is asking for a pedestal with
    # target temperature.
    pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=self.geo.rho_face[ped_top_idx],
        rho_norm_ped_top_idx=ped_top_idx,
        T_i_ped=1.0,
        T_e_ped=current_T_e_ped * T_target_over_T_current,
        n_e_ped=1.0,
    )

    transport_multipliers = saturation_model(
        self.runtime_params,
        self.geo,
        self.core_profiles,
        pedestal_output,
    )

    if T_target_over_T_current > 1.0:
      # If the target temperature is above the current temperature, we expect
      # the multiplier to be equal to 1.0 - the pedestal is not saturated.
      np.testing.assert_allclose(transport_multipliers.chi_e_multiplier, 1.0)
    else:
      # If the target temperature is below the current temperature, we expect
      # the multiplier to be greater than 1.0 - the pedestal is saturated.
      self.assertGreater(transport_multipliers.chi_e_multiplier, 1.0)


if __name__ == '__main__':
  absltest.main()
