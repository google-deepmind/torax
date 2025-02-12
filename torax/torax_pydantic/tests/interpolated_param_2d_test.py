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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax.geometry import circular_geometry
from torax.torax_pydantic import interpolated_param_2d
import xarray as xr

RHO_NORM = 'rho_norm'
TIME_INTERPOLATION_MODE = 'time_interpolation_mode'
RHO_INTERPOLATION_MODE = 'rho_interpolation_mode'


class InterpolatedParam2dTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='2_tuple_input_t=0',
          time_rho_interpolated_input=(
              np.array([0.25, 0.5, 0.75]),
              np.array([1.0, 2.0, 3.0]),
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=0.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='2_tuple_input_t=1',
          time_rho_interpolated_input=(
              np.array([0.25, 0.5, 0.75]),
              np.array([1.0, 2.0, 3.0]),
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=1.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=0',
          time_rho_interpolated_input=(
              np.array([0.0, 1.0]),
              np.array([0.25, 0.5, 0.75]),
              np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=0.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=1',
          time_rho_interpolated_input=(
              np.array([0.0, 1.0]),
              np.array([0.25, 0.5, 0.75]),
              np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=1.0,
          expected_output=np.array([4.0, 5.0, 6.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=0.5',
          time_rho_interpolated_input=(
              np.array([0.0, 1.0]),
              np.array([0.25, 0.5, 0.75]),
              np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=0.5,
          expected_output=np.array([2.5, 3.5, 4.5]),
      ),
      dict(
          testcase_name='3_tuple_input_t=0_time_step_interpolation',
          time_rho_interpolated_input=(
              (
                  np.array([0.0, 1.0]),
                  np.array([0.25, 0.5, 0.75]),
                  np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              ),
              {
                  TIME_INTERPOLATION_MODE: 'step',
                  RHO_INTERPOLATION_MODE: 'piecewise_linear',
              },
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=0.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=1_time_step_interpolation',
          time_rho_interpolated_input=(
              (
                  np.array([0.0, 1.0]),
                  np.array([0.25, 0.5, 0.75]),
                  np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              ),
              {
                  TIME_INTERPOLATION_MODE: 'step',
                  RHO_INTERPOLATION_MODE: 'piecewise_linear',
              },
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=1.01,
          expected_output=np.array([4.0, 5.0, 6.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=0.5_time_step_interpolation',
          time_rho_interpolated_input=(
              (
                  np.array([0.0, 1.0]),
                  np.array([0.25, 0.5, 0.75]),
                  np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              ),
              {
                  TIME_INTERPOLATION_MODE: 'step',
                  RHO_INTERPOLATION_MODE: 'piecewise_linear',
              },
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=0.5,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='xarray_input_t=0.0',
          time_rho_interpolated_input=xr.DataArray(
              data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              coords={'time': [0.0, 1.0], 'rho_norm': [0.25, 0.5, 0.75]},
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=0.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='xarray_input_t=1',
          time_rho_interpolated_input=xr.DataArray(
              data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              coords={'time': [0.0, 1.0], 'rho_norm': [0.25, 0.5, 0.75]},
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=1.0,
          expected_output=np.array([4.0, 5.0, 6.0]),
      ),
      dict(
          testcase_name='xarray_input_t=0.5',
          time_rho_interpolated_input=xr.DataArray(
              data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              coords={'time': [0.0, 1.0], 'rho_norm': [0.25, 0.5, 0.75]},
          ),
          rho_norm=np.array([0.25, 0.5, 0.75]),
          time=0.5,
          expected_output=np.array([2.5, 3.5, 4.5]),
      ),
      dict(
          testcase_name='single_dict_t=0',
          time_rho_interpolated_input={
              0: 18.0,
              0.95: 5.0,
          },
          rho_norm=np.array(0.0),
          time=0.0,
          expected_output=18.0,
      ),
      # Single dict represents a constant (in time) radial profile.
      dict(
          testcase_name='single_dict_t=0.5',
          time_rho_interpolated_input={
              0: 18.0,
              0.95: 5.0,
          },
          rho_norm=np.array([0.0, 0.95]),
          time=0.5,
          expected_output=np.array([18.0, 5.0]),
      ),
      dict(
          testcase_name='single_dict_t=0.0',
          time_rho_interpolated_input={
              0: 18.0,
              0.95: 5.0,
          },
          rho_norm=np.array([0.0, 0.95]),
          time=0.0,
          expected_output=np.array([18.0, 5.0]),
      ),
      # Single float represents a constant (in time and rho) profile.
      dict(
          testcase_name='float_t=0.0',
          time_rho_interpolated_input=1.0,
          rho_norm=np.array([0.0, 0.5, 1.0]),
          time=0.0,
          expected_output=np.array([1.0, 1.0, 1.0]),
      ),
      dict(
          testcase_name='float_t=5.0',
          time_rho_interpolated_input=1.0,
          rho_norm=np.array([0.0, 0.5, 1.0]),
          time=5.0,
          expected_output=np.array([1.0, 1.0, 1.0]),
      ),
  )
  def test_interpolated_var_time_rho_parses_inputs_correctly(
      self, time_rho_interpolated_input, rho_norm, time, expected_output
  ):
    """Tests that the creation of InterpolatedVarTimeRho from config works."""
    interpolated = interpolated_param_2d.TimeVaryingArray.model_validate(
        time_rho_interpolated_input
    )
    interpolated.set_rho_norm_grid(rho_norm)

    np.testing.assert_allclose(
        interpolated.get_value(x=time),
        expected_output,
    )

    self.assertEqual(interpolated, interpolated)

  def test_mutation_behavior(self):
    v1 = 1.0
    interpolated = interpolated_param_2d.TimeVaryingArray.model_validate(v1)

    # Directly setting the grid is banned due to immutability.
    with self.assertRaises(ValueError):
      interpolated.rho_norm_grid = np.array([0.0, 0.5, 1.0])

    # The grid is not set, so we should raise an error as there is not enough
    # information to interpolate.
    with self.assertRaises(ValueError):
      interpolated.get_value(x=0.0)

    geo = circular_geometry.build_circular_geometry()
    interpolated.set_rho_norm_grid(geo.torax_mesh)

    # Setting the grid twice should raise an error.
    with self.assertRaises(RuntimeError):
      interpolated.set_rho_norm_grid(geo.torax_mesh)

    out1 = interpolated.get_value(x=0.0)
    self.assertEqual(out1.tolist(), [v1] * len(interpolated.rho_norm_grid))


if __name__ == '__main__':
  absltest.main()
