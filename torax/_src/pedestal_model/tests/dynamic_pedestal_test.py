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
from torax._src.pedestal_model import dynamic_pedestal
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class DynamicPedestalTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    self.torax_config = model_config.ToraxConfig.from_dict(config)

    self.runtime_params_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(
            self.torax_config
        )
    )
    # Build geo and runtime params at t=0.0
    self.geo = self.torax_config.geometry.build_provider(0.0)
    self.runtime_params = self.runtime_params_provider(t=0.0)

    # Build initial core profiles
    self.source_models = self.torax_config.sources.build_models()
    self.neoclassical_models = self.torax_config.neoclassical.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        self.runtime_params,
        self.geo,
        self.source_models,
        self.neoclassical_models,
    )

  def test_suppression_when_above_PLH(self):
    suppression_factor = 1e-3
    pedestal_params = dynamic_pedestal.RuntimeParams(
        set_pedestal=True,
        mode=pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT,
        suppression_factor=suppression_factor,
        suppression_rate=1.0,
        augmentation_factor=0.0,
        augmentation_rate=1.0,
        alpha_crit=1e6,  # Set high so we don't trigger augmentation
        rho_norm_ped_top=0.9,
    )
    runtime_params = dataclasses.replace(
        self.runtime_params, pedestal=pedestal_params
    )

    pedestal_model = dynamic_pedestal.DynamicPedestal()
    output = pedestal_model._call_implementation(
        runtime_params, self.geo, self.core_profiles
    )

    # Check the output type.
    assert isinstance(
        output,
        dynamic_pedestal.pedestal_model.AdaptiveTransportPedestalModelOutput,
    )

    # Check the transport multiplier decreases the transport.
    self.assertLess(output.chi_e_multiplier, 1.0)

    # Check the transport multiplier is as expected.
    # h_mode_weight = sigmoid(0) = 0.5
    expected_decrease = 0.5 * 1.0 + 0.5 * suppression_factor
    # Since alpha_crit is high, we should not trigger the transport-increasing
    # effect of being above the alpha_crit threshold.
    expected_increase = 1.0
    expected_multiplier = expected_decrease * expected_increase
    np.testing.assert_allclose(output.chi_e_multiplier, expected_multiplier)

  def test_augmentation_when_above_alpha_crit(self):
    augmentation_factor = 1e3
    alpha_crit = 1e-6  # Very low threshold, to ensure we trigger it

    pedestal_params = dynamic_pedestal.RuntimeParams(
        set_pedestal=True,
        mode=pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT,
        suppression_factor=1.0,
        suppression_rate=1.0,
        augmentation_factor=augmentation_factor,
        augmentation_rate=0.1,
        alpha_crit=alpha_crit,
        rho_norm_ped_top=0.9,
    )
    runtime_params = dataclasses.replace(
        self.runtime_params, pedestal=pedestal_params
    )

    pedestal_model = dynamic_pedestal.DynamicPedestal()
    pedestal_model_output = pedestal_model._call_implementation(
        runtime_params, self.geo, self.core_profiles
    )

    # Check the output type.
    assert isinstance(
        pedestal_model_output,
        dynamic_pedestal.pedestal_model.AdaptiveTransportPedestalModelOutput,
    )

    # Check the transport multiplier increases the transport.
    self.assertGreater(pedestal_model_output.chi_e_multiplier, 1.0)


if __name__ == '__main__':
  absltest.main()
