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

"""Unit tests for the `torax.config.config_args` module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax.config import config_args
import xarray as xr


class ConfigArgsTest(parameterized.TestCase):
  """Unit tests for config arg parsing."""

  @parameterized.parameters(
      (True,),
  )
  def test_bool_single_value_param_always_return_constant(
      self, expected_output
  ):
    """Tests that when passed a single value this is always returned."""
    single_value_param = config_args.get_interpolated_var_single_axis(
        expected_output
    )
    np.testing.assert_allclose(
        single_value_param.get_value(-1), expected_output
    )
    np.testing.assert_allclose(single_value_param.get_value(0), expected_output)
    np.testing.assert_allclose(single_value_param.get_value(1), expected_output)

  def test_dict_range_input_must_have_values(self):
    with self.assertRaises(ValueError):
      config_args.get_interpolated_var_single_axis({})

  @parameterized.parameters(
      (
          (7.0, 'step'),
          1.0,
          7.0,
      ),
      (
          ({0.0: 1.0, 2.0: 7.0, 3.0: -1.0}, 'step'),
          -1.0,
          1.0,
      ),
      (
          ({0.0: 1.0, 2.0: 7.0, 3.0: -1.0}, 'step'),
          1.0,
          1.0,
      ),
      (
          ({0.0: 1.0, 2.0: 7.0, 3.0: -1.0}, 'step'),
          2.6,
          7.0,
      ),
      (
          ({0.0: 1.0, 2.0: 7.0, 3.0: -1.0}, 'step'),
          4.0,
          -1.0,
      ),
      (
          ({0.0: False, 2.0: True, 3.0: False}, 'step'),
          1.0,
          False,
      ),
      (
          ({0.0: False, 2.0: True, 3.0: False}, 'step'),
          2.5,
          True,
      ),
      (
          (
              (
                  np.array([0.0, 1.0]),
                  np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
              ),
              'step',
          ),
          0.5,
          np.array([3.0, 4.0, 5.0]),
      ),
  )
  def test_interpolated_var_returns_expected_output_for_step_interpolation(
      self,
      values,
      x,
      expected_output,
  ):
    """Tests that the range returns the expected output."""
    multi_val_range = config_args.get_interpolated_var_single_axis(
        values,
    )
    if isinstance(expected_output, bool):
      self.assertEqual(multi_val_range.get_value(x=x), expected_output)
    else:
      np.testing.assert_allclose(
          multi_val_range.get_value(x=x),
          expected_output,
      )

  def test_interpolated_var_time_rho_parses_float_input(self):
    """Tests that InterpolatedVarTimeRho parses float inputs correctly."""
    interpolated_var_time_rho = config_args.get_interpolated_var_2d(
        time_rho_interpolated_input=1.0, rho_norm=np.array([0.0, 0.5, 1.0])
    )
    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(x=0.0),
        np.array([1.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(x=5.0),
        np.array([1.0, 1.0, 1.0]),
    )
    self.assertLen(interpolated_var_time_rho.sorted_indices, 1)
    self.assertIn(0.0, interpolated_var_time_rho.sorted_indices)

  def test_interpolated_var_time_rho_parses_single_dict_input(self):
    """Tests that InterpolatedVarTimeRho parses dict inputs correctly."""
    interpolated_var_time_rho = config_args.get_interpolated_var_2d(
        time_rho_interpolated_input={
            0: 18.0,
            0.95: 5.0,
        },
        rho_norm=np.array(0.0),
    )
    np.testing.assert_allclose(interpolated_var_time_rho.get_value(x=0.0), 18.0)
    np.testing.assert_allclose(interpolated_var_time_rho.get_value(x=0.5), 18.0)

    interpolated_var_time_rho = config_args.get_interpolated_var_2d(
        time_rho_interpolated_input={
            0: 18.0,
            0.95: 5.0,
        },
        rho_norm=np.array(0.95),
    )
    np.testing.assert_allclose(interpolated_var_time_rho.get_value(x=0.0), 5.0)
    np.testing.assert_allclose(interpolated_var_time_rho.get_value(x=0.5), 5.0)

  def test_interpolated_var_time_rho_parses_xr_array_input(self):
    """Tests that InterpolatedVarTimeRho parses xr.DataArray inputs correctly."""
    array = xr.DataArray(
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        coords={'time': [0.0, 1.0], 'rho_norm': [0.25, 0.5, 0.75]},
    )
    interpolated_var_time_rho = config_args.get_interpolated_var_2d(
        time_rho_interpolated_input=array,
        rho_norm=np.array([0.25, 0.5, 0.75]),
    )

    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(
            x=0.0,
        ),
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(
            x=1.0,
        ),
        np.array([4.0, 5.0, 6.0]),
    )
    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(
            x=0.5,
        ),
        np.array([2.5, 3.5, 4.5]),
    )

  def test_interpolated_var_time_rho_parses_array_3_tuple_input(self):
    """Tests that InterpolatedVarTimeRho parses Array of 3 inputs correctly."""
    arrays = (
        np.array([0.0, 1.0]),
        np.array([0.25, 0.5, 0.75]),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    interpolated_var_time_rho = config_args.get_interpolated_var_2d(
        time_rho_interpolated_input=arrays,
        rho_norm=np.array([0.25, 0.5, 0.75]),
    )

    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(
            x=0.0,
        ),
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(
            x=1.0,
        ),
        np.array([4.0, 5.0, 6.0]),
    )
    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(
            x=0.5,
        ),
        np.array([2.5, 3.5, 4.5]),
    )

  def test_interpolated_var_time_rho_parses_array_2_tuple_input(self):
    """Tests that InterpolatedVarTimeRho parses Array of 2 inputs correctly."""
    arrays = (
        np.array([0.25, 0.5, 0.75]),
        np.array([1.0, 2.0, 3.0]),
    )
    interpolated_var_time_rho = config_args.get_interpolated_var_2d(
        time_rho_interpolated_input=arrays,
        rho_norm=np.array([0.25, 0.5, 0.75]),
    )

    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(
            x=0.0,
        ),
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(
            x=1.0,
        ),
        np.array([1.0, 2.0, 3.0]),
    )


if __name__ == '__main__':
  absltest.main()
