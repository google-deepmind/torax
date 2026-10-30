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
from torax._src.sources import source_profile_builders
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config as torax_pydantic_config


class NoPedestalTest(absltest.TestCase):

  def test_build_and_call_model(self):
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'structure_model': {'model_name': 'no_pedestal'},
    }
    torax_config = torax_pydantic_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    pedestal_model = torax_config.pedestal.build_pedestal_model()

    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    source_states = source_models.initial_state(runtime_params)
    source_profiles, _ = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
        explicit=True,
    )
    _, pedestal_state = pedestal_model.initialize(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_profiles=source_profiles,
    )
    pedestal_model_output, updated_pedestal_state = (
        pedestal_model(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            source_profiles=source_profiles,
            pedestal_state=pedestal_state,
        )
    )
    self.assertEqual(pedestal_model_output.rho_norm_ped_top, jnp.inf)
    self.assertEqual(updated_pedestal_state, pedestal_state)


if __name__ == '__main__':
  absltest.main()
