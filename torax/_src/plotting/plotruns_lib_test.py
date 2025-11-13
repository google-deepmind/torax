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
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torax._src import path_utils
from torax._src.config import config_loader
from torax._src.plotting import plotruns_lib
from torax._src.test_utils import paths


class PlotrunsLibTest(parameterized.TestCase):
  skip_data_files_list = [
      "test_iterhybrid_predictor_corrector_zeffprofile_qualikiz.nc",
  ]

  def setUp(self):
    super().setUp()

  def test_data_loading(self):
    test_data_dir = paths.test_data_dir()
    data_file = "test_iterhybrid_rampup.nc"
    test_data_path = os.path.join(
        test_data_dir, data_file
    )
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
    ]
  )
  def test_plot_config_smoke_test(self, config_name: str, data_file: str):
    test_data_dir = paths.test_data_dir()
    test_data_path = os.path.join(
       test_data_dir, data_file
    )
    config_path = path_utils.torax_path().joinpath(
        "plotting", "configs", config_name + ".py"
    )
    assert config_path.is_file(), f"Path {config_path} is not a file."
    plot_config = config_loader.import_module(config_path)["PLOT_CONFIG"]
    fig = plotruns_lib.plot_run(plot_config, test_data_path, interactive=False)
    assert isinstance(fig, Figure)
    plt.close(fig)

  @parameterized.product(
    config_name=[
      "default_plot_config",
      "global_params_plot_config",
      "simple_plot_config",
      "sources_plot_config",
    ]
  )
  def test_plot_config_all_test(self, config_name: str):
    test_data_dir = paths.test_data_dir()
    config_path = path_utils.torax_path().joinpath(
        "plotting", "configs", config_name + ".py"
    )
    assert config_path.is_file(), f"Path {config_path} is not a file."
    plot_config = config_loader.import_module(config_path)["PLOT_CONFIG"]
    data_files = test_data_dir.glob("*.nc")
    for test_data_path in data_files:
        # TODO: Allow plotting of runs without 'profiles' and remove exception
        # or filter out runs without 'profiles' apriori
        if test_data_path.name not in self.skip_data_files_list:
            fig = plotruns_lib.plot_run(plot_config, test_data_path, interactive=False)
            assert isinstance(fig, Figure), f"Plotting of {data_files.name} failed"
            plt.close(fig)


if __name__ == "__main__":
  absltest.main()
