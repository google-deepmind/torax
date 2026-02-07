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
import numpy as np
import plotly.graph_objects as go
from torax._src import path_utils
from torax._src.config import config_loader
from torax._src.plotting import plotruns_lib
from torax._src.test_utils import paths
import xarray as xr


def _generate_all_test_cases():
  """Generates test cases for all configs and all test data files."""
  torax_base_path = path_utils.torax_path()
  test_data_dir = torax_base_path / 'tests' / 'test_data'

  data_files = [f.name for f in test_data_dir.glob('*.nc')]

  config_names = [
      'default_plot_config',
      'global_params_plot_config',
      'simple_plot_config',
      'sources_plot_config',
  ]
  test_cases = []
  for config_name in config_names:
    for data_file in data_files:
      test_cases.append({
          'testcase_name': f'_{config_name}_{data_file}',
          'config_name': config_name,
          'data_file': data_file,
      })
  return test_cases


class PlotrunsLibTest(parameterized.TestCase):

  def test_data_loading(self):
    test_data_dir = paths.test_data_dir()
    data_file = 'test_iterhybrid_rampup.nc'
    test_data_path = os.path.join(test_data_dir, data_file)
    plotruns_lib.load_data(test_data_path)

  @parameterized.named_parameters(_generate_all_test_cases())
  def test_plot_config_all(self, config_name: str, data_file: str):
    test_data_dir = paths.test_data_dir()
    config_path = path_utils.torax_path().joinpath(
        'plotting', 'configs', config_name + '.py'
    )
    self.assertTrue(
        config_path.is_file(), msg=f'Path {config_path} is not a file.'
    )
    plot_config = config_loader.import_module(config_path)['PLOT_CONFIG']
    test_data_path = test_data_dir / data_file
    fig = plotruns_lib.plot_run(
        plot_config, str(test_data_path), interactive=False
    )
    self.assertIsInstance(
        fig, go.Figure, msg=f'Plotting of {test_data_path.name} failed'
    )


class FigurePropertiesTest(absltest.TestCase):

  def test_validation_raises_error(self):
    """Tests that a ValueError is raised if there are more axes than grid spots."""
    # Create a dummy PlotProperties object
    dummy_plot_properties = plotruns_lib.PlotProperties(
        attrs=('temp',), labels=('Temperature',), ylabel='T'
    )

    with self.assertRaisesRegex(ValueError, r'more than rows \* columns'):
      plotruns_lib.FigureProperties(
          rows=1,
          cols=1,
          font_family='Arial',
          title_size=16,
          subplot_title_size=12,
          tick_size=8,
          height=None,
          # 2 plots for a 1x1 grid
          axes=(dummy_plot_properties, dummy_plot_properties),
      )

  def test_contains_spatial_plot_type(self):
    spatial_plot = plotruns_lib.PlotProperties(
        attrs=('temp',),
        labels=('T',),
        ylabel='T',
        plot_type=plotruns_lib.PlotType.SPATIAL,
    )
    time_plot = plotruns_lib.PlotProperties(
        attrs=('time_var',),
        labels=('t',),
        ylabel='t',
        plot_type=plotruns_lib.PlotType.TIME_SERIES,
    )

    config_spatial = plotruns_lib.FigureProperties(
        rows=1,
        cols=1,
        font_family='Arial',
        title_size=16,
        subplot_title_size=12,
        tick_size=8,
        height=None,
        axes=(spatial_plot,),
    )
    self.assertTrue(config_spatial.contains_spatial_plot_type)

    config_time = plotruns_lib.FigureProperties(
        rows=1,
        cols=1,
        font_family='Arial',
        title_size=16,
        subplot_title_size=12,
        tick_size=8,
        height=None,
        axes=(time_plot,),
    )
    self.assertFalse(config_time.contains_spatial_plot_type)


class PlotDataTest(absltest.TestCase):
  """Unit tests for the PlotData wrapper class."""

  def setUp(self):
    super().setUp()

    # Create standard profile variables
    self.profiles_ds = xr.Dataset({
        'chi_turb_i': (('time', 'rho'), [[1.0, 1.0], [1.0, 1.0]]),
        'chi_neo_i': (('time', 'rho'), [[2.0, 2.0], [2.0, 2.0]]),
        'rho_cell_norm': (
            'rho',
            [0.1, 0.5],
        ),  # Needed for zero-fill logic
    })

    # Create scalar variables
    self.scalars_ds = xr.Dataset({
        'P_bremsstrahlung_e': ('time', [-10.0, -10.0]),
        'P_radiation_e': ('time', [-5.0, -5.0]),
        'P_cyclotron_e': ('time', [-2.0, -2.0]),
        'Ip': ('time', [15.0, 15.0]),
    })

    self.top_level_ds = xr.Dataset({'time': [0.0, 1.0]})
    self.numerics_ds = xr.Dataset({})

    self.data_tree = xr.DataTree(
        dataset=self.top_level_ds,
        children={
            'profiles': xr.DataTree(dataset=self.profiles_ds),
            'scalars': xr.DataTree(dataset=self.scalars_ds),
            'numerics': xr.DataTree(dataset=self.numerics_ds),
        },
    )

    self.plot_data = plotruns_lib.PlotData(
        data_tree=self.data_tree,
    )

  def test_derived_properties_summation(self):
    """Test that properties like chi_total_i sum their components correctly."""
    expected = np.array([[3.0, 3.0], [3.0, 3.0]])
    np.testing.assert_array_equal(self.plot_data.chi_total_i, expected)

  def test_p_sink_summation(self):
    expected = np.array([-17.0, -17.0])
    np.testing.assert_array_equal(self.plot_data.P_sink, expected)

  def test_legacy_ip_profile_redirect(self):
    """Test that requesting 'Ip_profile' returns 'Ip' from scalars."""
    np.testing.assert_array_equal(
        self.plot_data.Ip_profile, self.scalars_ds['Ip'].to_numpy()
    )

  def test_optional_attributes_default_to_zero(self):
    """Test that missing optional attributes return a zero array of correct shape."""
    # 's_pellet' is in _OPTIONAL_PROFILE_ATTRS but not in our mock datasets
    result = self.plot_data.s_pellet

    # Should be shape (time_steps, spatial_steps) -> (2, 2) based on mock data
    expected_shape = (2, 2)
    self.assertEqual(result.shape, expected_shape)
    np.testing.assert_array_equal(result, np.zeros(expected_shape))

  def test_missing_attribute_raises_error(self):
    """Test that accessing a truly non-existent variable raises AttributeError."""
    with self.assertRaisesRegex(
        AttributeError, "has no attribute 'non_existent_var'"
    ):
      _ = self.plot_data.non_existent_var


if __name__ == '__main__':
  absltest.main()
