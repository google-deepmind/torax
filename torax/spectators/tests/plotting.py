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

"""Tests for plotting.py."""

from absl.testing import absltest
from absl.testing import parameterized
import torax  # We want this import to make sure jax gets set to float64
from torax import config as config_lib
from torax import geometry
from torax.spectators import plotting
from torax.spectators import spectator
from torax.stepper import linear_theta_method
from torax.time_step_calculator import chi_time_step_calculator


class PlottingTest(parameterized.TestCase):
  """Tests the plotting library."""

  def test_default_plot_config_has_valid_keys(self):
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    plot_config = plotting.get_default_plot_config(geo)

    observer = spectator.InMemoryJaxArraySpectator()
    _run_sim(config, geo, observer)

    # Make sure all the keys in plot_config are collected by the observer.
    for plot in plot_config:
      for key in plot.keys:
        self.assertIn(key.key, observer.arrays)

  def test_plot_observer_runs_with_sim(self):
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    observer = plotting.PlotSpectator(
        plots=plotting.get_default_plot_config(geo),
    )
    _run_sim(config, geo, observer)


def _run_sim(
    config: config_lib.Config,
    geo: geometry.Geometry,
    observer: spectator.Spectator,
):
  torax.build_sim_from_config(
      config,
      geo,
      linear_theta_method.LinearThetaMethod,
      chi_time_step_calculator.ChiTimeStepCalculator(),
  ).run(
      spectator=observer,
  )


if __name__ == '__main__':
  absltest.main()
