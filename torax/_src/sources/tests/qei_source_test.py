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
import pydantic
from torax._src.sources import qei_source
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


class QeiSourceTest(absltest.TestCase):
  """Tests for QeiSource."""

  def test_build_runtime_params(self):
    cfg = qei_source.QeiSourceConfig(
        mode=sources_runtime_params_lib.Mode.MODEL_BASED,
        Qei_multiplier=1.5,
    )
    runtime_params = cfg.build_runtime_params(t=0.0)
    self.assertEqual(
        runtime_params.mode, sources_runtime_params_lib.Mode.MODEL_BASED
    )
    self.assertEqual(runtime_params.Qei_multiplier, 1.5)
    self.assertFalse(runtime_params.is_explicit)

  def test_source_value(self):
    """Checks that the default implementation from Sources gives values."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['sources'] = {qei_source.QeiSource.SOURCE_ID: {}}
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.qei_source
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    qei = source.get_qei(
        runtime_params,
        geo,
        core_profiles,
    )
    self.assertIsNotNone(qei)
    self.assertIsNotNone(qei.p_ei)
    self.assertEqual(qei.p_ei.shape, geo.rho_norm.shape)

  def test_prescribed_mode_disallowed(self):
    """Checks that Mode.PRESCRIBED is disallowed for QeiSource."""
    with self.assertRaisesRegex(pydantic.ValidationError, 'Mode.PRESCRIBED'):
      qei_source.QeiSourceConfig.from_dict({'mode': 'prescribed'})


if __name__ == '__main__':
  absltest.main()
