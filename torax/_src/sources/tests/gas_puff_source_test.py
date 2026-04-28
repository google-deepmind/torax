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
import chex
from torax._src import math_utils
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.sources import gas_puff_source
from torax._src.sources.tests import test_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class GasPuffSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for GasPuffSource."""

  def setUp(self):
    super().setUp(
        source_config_class=gas_puff_source.GasPuffSourceConfig,
        source_name=gas_puff_source.GasPuffSource.SOURCE_NAME,
    )

  def test_feedback_mode(self):
    """Tests calc_puff_feedback_source with real objects."""
    config = default_configs.get_default_config_dict()
    config['sources'] = {
        'gas_puff': {
            'model_name': 'feedback',
            'feedback_gain': 10.0,
        }
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)

    runtime_params_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
    )
    runtime_params = runtime_params_provider(t=torax_config.numerics.t_initial)

    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    initial_line_avg = math_utils.line_average(core_profiles.n_e.value, geo)

    # Rebuild with specific requested value
    config['sources']['gas_puff']['model_name'] = 'feedback'
    config['sources']['gas_puff']['target_line_average_n_e'] = float(
        initial_line_avg + 1e19
    )
    config['sources']['gas_puff']['feedback_gain'] = 10.0
    torax_config = model_config.ToraxConfig.from_dict(config)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)

    gas_puff_params = runtime_params.sources['gas_puff']
    chex.assert_trees_all_close(
        gas_puff_params.target_line_average_n_e, float(initial_line_avg + 1e19)
    )
    chex.assert_trees_all_close(gas_puff_params.feedback_gain, 10.0)

    profile = gas_puff_source.calc_puff_feedback_source(
        runtime_params=runtime_params,
        geo=geo,
        source_name='gas_puff',
        core_profiles=core_profiles,
        unused_calculated_source_profiles=None,
        unused_conductivity=None,
    )[0]

    total_particles = math_utils.volume_integration(profile, geo)

    # Expected error = (initial_line_avg + 1e19) - initial_line_avg = 1e19
    # Expected S_total = 10.0 * 1e19 = 1e20
    chex.assert_trees_all_close(total_particles, 1e20, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
