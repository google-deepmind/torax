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
from jax import numpy as jnp
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class NoPedestalTest(absltest.TestCase):

  def test_build_and_call_model(self):
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'no_pedestal',
        'set_pedestal': True,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    source_models = torax_config.sources.build_models()

    neoclassical_models = torax_config.neoclassical.build_models()
    geo = torax_config.geometry.build_provider(t=0.0)
    runtime_params = provider(t=0.0)
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models,
        neoclassical_models,
    )
    source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        explicit=True,
    )
    pedestal_model_output = pedestal_model(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_profiles=source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )
    self.assertEqual(pedestal_model_output.rho_norm_ped_top, jnp.inf)


if __name__ == '__main__':
  absltest.main()
