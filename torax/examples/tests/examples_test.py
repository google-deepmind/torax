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

"""Tests for the example configs in the example folder."""

from absl.testing import absltest
from absl.testing import parameterized
from torax._src.config import config_loader
from torax._src.torax_pydantic import model_config


class ExamplesTest(parameterized.TestCase):
  """Smoke tests for the configs in the example folder."""

  @parameterized.parameters([
      'basic_config',
      'iterhybrid_predictor_corrector',
      'iterhybrid_rampup',
  ])
  def test_validation_of_configs(self, config_name_no_py: str):
    example_config_paths = config_loader.example_config_paths()
    example_config_path = example_config_paths[config_name_no_py]
    cfg = config_loader.build_torax_config_from_file(example_config_path)
    self.assertIsInstance(cfg, model_config.ToraxConfig)


if __name__ == '__main__':
  absltest.main()
