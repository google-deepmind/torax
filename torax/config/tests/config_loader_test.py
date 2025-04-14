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
import pathlib
import typing
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from torax.config import config_loader


class ConfigLoaderTest(parameterized.TestCase):

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

  def test_example_config_paths(self):
    self.assertLen(
        config_loader.example_config_paths(),
        len(typing.get_args(config_loader.ExampleConfig)),
    )

  @parameterized.product(
      use_string=[True, False],
      path=list(config_loader.example_config_paths().values()),
  )
  def test_import_config_dict(self, use_string: bool, path: pathlib.Path):
    path = str(path) if use_string else path
    config_dict = config_loader.import_config_dict(path)

    with self.subTest("is_valid_dict"):
      self.assertIsInstance(config_dict, dict)
      self.assertNotEmpty(config_dict, dict)

    with self.subTest("mutation_safe"):
      config_dict["new_invalid_key"] = True
      config_dict_2 = config_loader.import_config_dict(path)
      self.assertNotIn("new_invalid_key", config_dict_2)

  def test_import_config_dict_invalid_path(self):
    with self.assertRaises(ValueError):
      config_loader.import_config_dict("/invalid/path/not_a_file.py")


if __name__ == "__main__":
  absltest.main()
