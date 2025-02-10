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

"""Unit tests for the `torax.config` module."""

from absl.testing import absltest
from absl.testing import parameterized
from torax.config import config_loader
from torax.torax_pydantic import model_config


class ConfigTest(parameterized.TestCase):
  """Unit tests for the `torax.config` module."""

  def test_full_config_construction(self):
    """Test for basic config construction."""

    module = config_loader.import_module(
        ".tests.test_data.test_iterhybrid_newton",
        config_package="torax",
    )

    # Test only the subset of config fields that are currently supported.
    module_config = {
        key: module.CONFIG[key]
        for key in model_config.ToraxConfig.model_fields.keys()
    }
    config_pydantic = model_config.ToraxConfig.from_dict(module_config)

    self.assertEqual(
        config_pydantic.time_step_calculator.calculator_type.value,
        module_config["time_step_calculator"]["calculator_type"],
    )

    # The full model should always be serializable.
    with self.subTest("json_serialization"):
      config_json = config_pydantic.model_dump_json()
      config_pydantic_roundtrip = model_config.ToraxConfig.model_validate_json(
          config_json
      )
      self.assertEqual(config_pydantic, config_pydantic_roundtrip)


if __name__ == "__main__":
  absltest.main()
