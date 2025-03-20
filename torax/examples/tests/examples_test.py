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
import importlib
import os

from absl.testing import absltest
from absl.testing import parameterized
from torax.tests.test_lib import paths
from torax.torax_pydantic import model_config

PYTHON_MODULE_PREFIX = '.examples.'
PYTHON_CONFIG_PACKAGE = 'torax'


class ExamplesTest(parameterized.TestCase):
  """Smoke tests for the configs in the example folder."""

  @parameterized.parameters([
      'basic_config',
      'iterhybrid_predictor_corrector',
      'iterhybrid_rampup',
  ])
  def test_validation_of_configs(self, config_name_no_py: str):
    config_path = os.path.join(paths.examples_dir(), config_name_no_py + '.py')
    assert os.path.exists(config_path), config_path
    python_config_module = PYTHON_MODULE_PREFIX + config_name_no_py
    config_module = importlib.import_module(
        python_config_module, PYTHON_CONFIG_PACKAGE
    )
    model_config.ToraxConfig.from_dict(config_module.CONFIG)


if __name__ == '__main__':
  absltest.main()
