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

"""Unit tests for torax.plotting.plotruns_lib."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from matplotlib import figure
import matplotlib.pyplot as plt
from torax._src import path_utils
from torax._src.config import config_loader
from torax._src.plotting import plotruns_lib
from torax._src.test_utils import paths


def _generate_all_test_cases():
  """Generates test cases for all configs and all test data files."""
  torax_base_path = path_utils.torax_path()
  test_data_dir = torax_base_path / "tests" / "test_data"

  data_files = [f.name for f in test_data_dir.glob("*.nc")]

  config_names = [
      "default_plot_config",
      "global_params_plot_config",
      "simple_plot_config",
      "sources_plot_config",
  ]
  test_cases = []
  for config_name in config_names:
    for data_file in data_files:
      test_cases.append({
          "testcase_name": f"_{config_name}_{data_file}",
          "config_name": config_name,
          "data_file": data_file,
      })
  return test_cases


class PlotrunsLibTest(parameterized.TestCase):

  def test_data_loading(self):
    test_data_dir = paths.test_data_dir()
    data_file = "test_iterhybrid_rampup.nc"
    test_data_path = os.path.join(test_data_dir, data_file)
    plotruns_lib.load_data(test_data_path)

  @parameterized.product(
      config_name=[
          "default_plot_config",
          "global_params_plot_config",
          "simple_plot_config",
          "sources_plot_config",
      ],
      data_file=[
          "test_iterhybrid_rampup.nc",
      ],
  )
  def test_plot_config_smoke_test(self, config_name: str, data_file: str):
    test_data_dir = paths.test_data_dir()
    test_data_path = os.path.join(test_data_dir, data_file)
    config_path = path_utils.torax_path().joinpath(
        "plotting", "configs", config_name + ".py"
    )
    assert config_path.is_file(), f"Path {config_path} is not a file."
    plot_config = config_loader.import_module(config_path)["PLOT_CONFIG"]
    fig = plotruns_lib.plot_run(plot_config, test_data_path, interactive=False)
    assert isinstance(fig, figure.Figure)
    plt.close(fig)

  @parameterized.named_parameters(_generate_all_test_cases())
  def test_plot_config_all_test(self, config_name: str, data_file: str):
    test_data_dir = paths.test_data_dir()
    config_path = path_utils.torax_path().joinpath(
        "plotting", "configs", config_name + ".py"
    )
    assert config_path.is_file(), f"Path {config_path} is not a file."
    plot_config = config_loader.import_module(config_path)["PLOT_CONFIG"]
    test_data_path = test_data_dir / data_file
    fig = plotruns_lib.plot_run(
        plot_config, str(test_data_path), interactive=False
    )
    assert isinstance(
        fig, figure.Figure
    ), f"Plotting of {test_data_path.name} failed"
    plt.close(fig)


if __name__ == "__main__":
  absltest.main()
