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
import os
import pathlib
import typing

from absl.testing import absltest
from absl.testing import parameterized
from torax._src import path_utils
from torax._src.config import config_loader
from torax._src.plotting import plotruns_lib
from torax._src.torax_pydantic import model_config

class ConfigLoaderTest(parameterized.TestCase):

  def test_example_config_paths(self):
    self.assertLen(
        config_loader.example_config_paths(),
        len(typing.get_args(config_loader.ExampleConfig)),
    )

  @parameterized.product(
      use_string=[True, False],
      relative_to=[None, "working", "torax"],
      path=list(config_loader.example_config_paths().values()),
  )
  def test_import_module(
      self, use_string: bool, relative_to: str | None, path: pathlib.Path
  ):
    if relative_to == "working":
      path = path.relative_to(os.getcwd())
    if relative_to == "torax":
      path = path.relative_to(path_utils.torax_path())

    if use_string:
      path = str(path)

    config_dict = config_loader.import_module(path)

    with self.subTest("is_valid_dict"):
      self.assertIsInstance(config_dict, dict)
      self.assertNotEmpty(config_dict, dict)

    with self.subTest("mutation_safe"):
      config_dict["new_invalid_key"] = True
      config_dict_2 = config_loader.import_module(path)
      self.assertNotIn("new_invalid_key", config_dict_2)

  def test_import_config_invalid_path(self):
    fake_file = pathlib.Path("/invalid/path/not_a_file.py")
    self.assertFalse(fake_file.is_file())

    with self.assertRaises(ValueError):
      config_loader.import_module(fake_file)

  @parameterized.product(
      path=list(config_loader.example_config_paths().values()),
  )
  def test_build_torax_config_from_file(self, path: pathlib.Path):
    config = config_loader.build_torax_config_from_file(path)
    self.assertIsInstance(config, model_config.ToraxConfig)

  @parameterized.product(
      use_string=[True, False],
      name=list(config_loader.example_plot_config_paths().keys()),
  )
  def test_get_plot_config_from_file(self, name: str, use_string: bool):
    path = config_loader.example_plot_config_paths()[name]
    path = str(path) if use_string else path
    cfg = config_loader.get_plot_config_from_file(path)
    self.assertIsInstance(cfg, plotruns_lib.FigureProperties)

  def test_build_torax_config_from_file_low_q_warning(self):
      # Create a dictionary representing a minimal ToraxConfig with unphysical values
      config_dict = {
          "global_parameters": {
              "plasma_current": {"value": 1e7},  # High plasma current
              "toroidal_field": {"value": 1.0},  # Low toroidal field
          },
          "geometry": {
              "minor_radius": 0.5,
              "major_radius": 2.0,
          },
          "sources": {
              "power_source_density": {"value": 2e8},  # High power source density
              "particle_source_density": {
                  "value": 3e20
              },  # High particle source density
          },
          "runtime_params": {},
          "physics_options": {},
          "numerics_options": {},
      }

      # Create a temporary config file with the unphysical values
      with open("temp_config.py", "w") as f:
          f.write("CONFIG = " + str(config_dict))

      # Capture log messages
      with self.assertLogs(level="WARNING") as captured:
          # Load the config from the temporary file
          config = config_loader.build_torax_config_from_file("temp_config.py")

      # Clean up the temporary file
      os.remove("temp_config.py")

      # Assert that a warning message was logged
      self.assertTrue(
          any(
              "Q-profile estimate is very low" in record.getMessage()
              for record in captured.records
          )
      )

      self.assertTrue(
          any(
              "Power source density is very high" in record.getMessage()
              for record in captured.records
          )
      )
      self.assertTrue(
          any(
              "Particle source density is very high" in record.getMessage()
              for record in captured.records
          )
      )

if __name__ == "__main__":
  absltest.main()
