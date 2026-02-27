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
import jax
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class EPEDNNmitPedestalModelTest(parameterized.TestCase):

  def test_build_and_call_pedestal_model(self):
    """Tests the EPEDNN-mit pedestal model.

    Note that the EPEDNN-mit is only valid for SPARC parameter space, but we're
    testing here with a generic config. Hence, we don't perform checks on
    the values of the model outputs.
    """
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'epednn_mit',
        'set_pedestal': True,
        'n_e_ped': 0.7e20,
        'n_e_ped_is_fGW': False,
        'T_i_T_e_ratio': 1.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(
            torax_config
        )
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    jitted_pedestal_model = jax.jit(pedestal_model)

    geo = torax_config.geometry.build_provider(0.0)
    runtime_params = provider(t=0.0)
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models,
        neoclassical_models,
    )
    pedestal_model_output = jitted_pedestal_model(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )

    np.testing.assert_allclose(pedestal_model_output.n_e_ped, 0.7e20)
    np.testing.assert_allclose(
        pedestal_model_output.T_i_ped / pedestal_model_output.T_e_ped, 1.0
    )


if __name__ == '__main__':
  absltest.main()
