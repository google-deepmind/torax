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
from unittest import mock

from absl.testing import absltest
from torax.config import config_loader


class ConfigLoaderTest(absltest.TestCase):

  def test_import_module_and_cache_reload(self):
    """test the import_module function uses cache."""
    test_value = "test_value"
    with mock.patch(
        "importlib.import_module", return_value=test_value
    ) as mock_import_module:
      value = config_loader.import_module("test_module")
      mock_import_module.assert_called_once()
      mock_import_module.assert_called_with("test_module", None)
      self.assertEqual(value, test_value)
      self.assertIn("test_module", config_loader._ALL_MODULES)
      self.assertEqual(config_loader._ALL_MODULES["test_module"], test_value)

    with mock.patch("importlib.reload", return_value=test_value) as mock_reload:
      value = config_loader.import_module("test_module")
      mock_reload.assert_called_once()
      mock_reload.assert_called_with(test_value)
      self.assertEqual(value, test_value)
      self.assertEqual(config_loader._ALL_MODULES["test_module"], test_value)


if __name__ == "__main__":
  absltest.main()
