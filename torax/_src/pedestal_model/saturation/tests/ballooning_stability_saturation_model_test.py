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
import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model.saturation import ballooning_stability_saturation_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class BallooningStabilitySaturationModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    # Add ballooning stability config with a default value for alpha_crit.
    config['pedestal']['saturation_model'] = {
        'model_name': 'ballooning_stability',
        'alpha_crit': 1.0,  # Will be overridden in specific tests.
    }
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
          # Low alpha_crit -> saturation is active.
          alpha_crit=1e-3,
      ),
      dict(
          testcase_name='inactive',
          # High alpha_crit -> saturation is inactive.
          alpha_crit=1e3,
      ),
  )
  def test_saturation_multiplier(
      self,
      alpha_crit: float,
  ):
    assert isinstance(
        self.runtime_params.pedestal.saturation,
        ballooning_stability_saturation_model.BallooningStabilitySaturationRuntimeParams,
    )
    saturation_model = (
        ballooning_stability_saturation_model.BallooningStabilitySaturationModel()
    )

    # For this test, we put the pedestal top at the last grid point.
    pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=1.0,
        rho_norm_ped_top_idx=-1,
        # The following values are not used in the saturation model.
        T_i_ped=1.0,
        T_e_ped=1.0,
        n_e_ped=1.0,
    )

    # Set alpha_crit in the runtime params.
    new_saturation_params = dataclasses.replace(
        self.runtime_params.pedestal.saturation,
        alpha_crit=alpha_crit,
        steepness=self.runtime_params.pedestal.saturation.steepness,
        offset=self.runtime_params.pedestal.saturation.offset,
        base_multiplier=self.runtime_params.pedestal.saturation.base_multiplier,
    )
    new_pedestal_params = dataclasses.replace(
        self.runtime_params.pedestal, saturation=new_saturation_params
    )
    runtime_params = dataclasses.replace(
        self.runtime_params, pedestal=new_pedestal_params
    )

    # Get actual alpha in the pedestal.
    alpha = ballooning_stability_saturation_model.calculate_normalized_pressure_gradient(
        self.core_profiles, self.geo
    )
    max_alpha_ped = alpha[-1]

    # Calculate the multiplier.
    transport_multipliers = saturation_model(
        runtime_params,
        self.geo,
        self.core_profiles,
        pedestal_output,
    )

    if max_alpha_ped < alpha_crit:
      np.testing.assert_allclose(transport_multipliers.chi_e_multiplier, 1.0)
    else:
      self.assertGreater(transport_multipliers.chi_e_multiplier, 1.0)


if __name__ == '__main__':
  absltest.main()
