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
import chex
import numpy as np
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.torax_pydantic import interpolated_param_2d
from torax.torax_pydantic import model_base
import xarray as xr

RHO_NORM = 'rho_norm'
TIME_INTERPOLATION_MODE = 'time_interpolation_mode'
RHO_INTERPOLATION_MODE = 'rho_interpolation_mode'


class InterpolatedParam2dTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='2_tuple_input_t=0',
          time_rho_interpolated_input=(
              np.array([0.125, 0.375, 0.625]),
              np.array([1.0, 2.0, 3.0]),
          ),
          nx=3,
          dx=0.25,
          time=0.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='2_tuple_input_t=1',
          time_rho_interpolated_input=(
              np.array([0.125, 0.375, 0.625]),
              np.array([1.0, 2.0, 3.0]),
          ),
          nx=3,
          dx=0.25,
          time=1.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=0',
          time_rho_interpolated_input=(
              np.array([0.0, 1.0]),
              np.array([0.125, 0.375, 0.625]),
              np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          ),
          nx=3,
          dx=0.25,
          time=0.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=1',
          time_rho_interpolated_input=(
              np.array([0.0, 1.0]),
              np.array([0.125, 0.375, 0.625]),
              np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          ),
          nx=3,
          dx=0.25,
          time=1.0,
          expected_output=np.array([4.0, 5.0, 6.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=0.5',
          time_rho_interpolated_input=(
              np.array([0.0, 1.0]),
              np.array([0.125, 0.375, 0.625]),
              np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          ),
          nx=3,
          dx=0.25,
          time=0.5,
          expected_output=np.array([2.5, 3.5, 4.5]),
      ),
      dict(
          testcase_name='3_tuple_input_t=0_time_step_interpolation',
          time_rho_interpolated_input=(
              (
                  np.array([0.0, 1.0]),
                  np.array([0.125, 0.375, 0.625]),
                  np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              ),
              {
                  TIME_INTERPOLATION_MODE: 'step',
                  RHO_INTERPOLATION_MODE: 'piecewise_linear',
              },
          ),
          nx=3,
          dx=0.25,
          time=0.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=1_time_step_interpolation',
          time_rho_interpolated_input=(
              (
                  np.array([0.0, 1.0]),
                  np.array([0.125, 0.375, 0.625]),
                  np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              ),
              {
                  TIME_INTERPOLATION_MODE: 'step',
                  RHO_INTERPOLATION_MODE: 'piecewise_linear',
              },
          ),
          nx=3,
          dx=0.25,
          time=1.01,
          expected_output=np.array([4.0, 5.0, 6.0]),
      ),
      dict(
          testcase_name='3_tuple_input_t=0.5_time_step_interpolation',
          time_rho_interpolated_input=(
              (
                  np.array([0.0, 1.0]),
                  np.array([0.125, 0.375, 0.625]),
                  np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              ),
              {
                  TIME_INTERPOLATION_MODE: 'step',
                  RHO_INTERPOLATION_MODE: 'piecewise_linear',
              },
          ),
          nx=3,
          dx=0.25,
          time=0.5,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='xarray_input_t=0.0',
          time_rho_interpolated_input=xr.DataArray(
              data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              coords={'time': [0.0, 1.0], 'rho_norm': [0.125, 0.375, 0.625]},
          ),
          nx=3,
          dx=0.25,
          time=0.0,
          expected_output=np.array([1.0, 2.0, 3.0]),
      ),
      dict(
          testcase_name='xarray_input_t=1',
          time_rho_interpolated_input=xr.DataArray(
              data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              coords={'time': [0.0, 1.0], 'rho_norm': [0.125, 0.375, 0.625]},
          ),
          nx=3,
          dx=0.25,
          time=1.0,
          expected_output=np.array([4.0, 5.0, 6.0]),
      ),
      dict(
          testcase_name='xarray_input_t=0.5',
          time_rho_interpolated_input=xr.DataArray(
              data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
              coords={'time': [0.0, 1.0], 'rho_norm': [0.125, 0.375, 0.625]},
          ),
          nx=3,
          dx=0.25,
          time=0.5,
          expected_output=np.array([2.5, 3.5, 4.5]),
      ),
      dict(
          testcase_name='single_dict_t=0',
          time_rho_interpolated_input={
              0.25: 18.0,
              0.95: 5.0,
          },
          nx=1,
          dx=0.5,
          time=0.0,
          expected_output=18.0,
      ),
      # Single dict represents a constant (in time) radial profile.
      dict(
          testcase_name='single_dict_t=0.5',
          time_rho_interpolated_input={
              0.475: 18.0,
              1.425: 5.0,
          },
          nx=2,
          dx=0.95,
          time=0.5,
          expected_output=np.array([18.0, 5.0]),
      ),
      dict(
          testcase_name='single_dict_t=0.0',
          time_rho_interpolated_input={
              0.475: 18.0,
              1.425: 5.0,
          },
          nx=2,
          dx=0.95,
          time=0.0,
          expected_output=np.array([18.0, 5.0]),
      ),
      # Single float represents a constant (in time and rho) profile.
      dict(
          testcase_name='float_t=0.0',
          time_rho_interpolated_input=1.0,
          nx=3,
          dx=0.5,
          time=0.0,
          expected_output=np.array([1.0, 1.0, 1.0]),
      ),
      dict(
          testcase_name='float_t=5.0',
          time_rho_interpolated_input=1.0,
          nx=3,
          dx=0.5,
          time=5.0,
          expected_output=np.array([1.0, 1.0, 1.0]),
      ),
  )
  def test_time_varying_array_parses_inputs_correctly(
      self, time_rho_interpolated_input, nx, dx, time, expected_output
  ):
    """Tests that the creation of TimeVaryingArray from config works."""
    interpolated = interpolated_param_2d.TimeVaryingArray.model_validate(
        time_rho_interpolated_input
    )
    grid = interpolated_param_2d.Grid1D.construct(nx=nx, dx=dx)
    interpolated_param_2d.set_grid(interpolated, grid=grid)

    np.testing.assert_allclose(
        interpolated.get_value(t=time),
        expected_output,
    )

    self.assertEqual(interpolated, interpolated)

  def test_right_boundary_conditions_defined(self):
    """Tests that right_boundary_conditions_defined works correctly."""

    with self.subTest('float_input'):
      # A single float is interpreted as defined at rho=0.
      self.assertFalse(
          interpolated_param_2d.TimeVaryingArray.model_validate(
              1.0
          ).right_boundary_conditions_defined
      )

    with self.subTest('xarray'):
      value = xr.DataArray(
          data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          coords={'time': [0.0, 1.0], 'rho_norm': [0.25, 0.5, 1.0]},
      )
      self.assertTrue(
          interpolated_param_2d.TimeVaryingArray.model_validate(
              value
          ).right_boundary_conditions_defined
      )

  def test_set_grid(self):

    class Test1(model_base.BaseModelFrozen):
      x: float
      y: interpolated_param_2d.TimeVaryingArray

    class Test2(model_base.BaseModelFrozen):
      x: Test1  # pytype: disable=invalid-annotation
      y: interpolated_param_2d.TimeVaryingArray
      z: int

    m1 = Test1(
        x=1.0, y=interpolated_param_2d.TimeVaryingArray.model_validate(1.0)
    )
    m2 = Test2(
        x=m1, y=interpolated_param_2d.TimeVaryingArray.model_validate(2.0), z=5
    )
    grid = geometry_pydantic_model.CircularConfig().build_geometry().torax_mesh

    with self.subTest('set_grid_success'):
      interpolated_param_2d.set_grid(m2, grid)
      chex.assert_trees_all_equal(m2.x.y.grid.face_centers, grid.face_centers)  # pytype: disable=attribute-error
      chex.assert_trees_all_equal(m2.x.y.grid.cell_centers, grid.cell_centers)  # pytype: disable=attribute-error
      chex.assert_trees_all_equal(m2.y.grid.face_centers, grid.face_centers)  # pytype: disable=attribute-error
      chex.assert_trees_all_equal(m2.y.grid.cell_centers, grid.cell_centers)  # pytype: disable=attribute-error

    with self.subTest('set_grid_already_set'):
      with self.assertRaisesRegex(RuntimeError, '`grid` is already set'):
        interpolated_param_2d.set_grid(m2, grid)

    with self.subTest('set_grid_already_set_force'):
      grid._update_fields({'nx': grid.nx + 1})
      interpolated_param_2d.set_grid(m2, grid, mode='force')
      chex.assert_trees_all_equal(m2.y.grid.face_centers, grid.face_centers)  # pytype: disable=attribute-error

    with self.subTest('set_grid_already_set_relaxed'):
      interpolated_param_2d.set_grid(m2, grid, mode='relaxed')


if __name__ == '__main__':
  absltest.main()
