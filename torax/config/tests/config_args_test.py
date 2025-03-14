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

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax.config import config_args
import xarray as xr


class ConfigArgsTest(parameterized.TestCase):

  def test_recursive_replace(self):
    """Basic test of recursive replace."""

    # Make an nested dataclass with 3 layers of nesting, to make sure we can
    # handle root, internal, and leaf nodes. Make sure we can run a
    # recursive_replace call that is able to update some values but leave other
    # values untouched, at every level of the nested hierarchy.

    instance = A()

    # Change all values so we notice if `replace` fills them in with constructor
    # defaults
    instance.a1 = 7
    instance.a2 = 8
    instance.a3.b1 = 9
    instance.a3.b2 = 10
    instance.a3.b3.c1 = 11
    instance.a3.b3.c2 = 12
    instance.a3.b4.c1 = 13
    instance.a3.b4.c2 = 14
    instance.a4.b1 = 15
    instance.a4.b2 = 16
    instance.a4.b3.c1 = 17
    instance.a4.b3.c2 = 18
    instance.a4.b4.c1 = 19
    instance.a4.b4.c2 = 20

    changes = {
        'a1': -1,
        # Don't update a2, to test that it is untouched
        'a3': {
            # Don't update b1, to test that it is untouched
            'b2': -2,
            # Don't update b3, to test that it is untouched
            'b4': {
                'c1': -3,
                # Don't update c2, to test that it is untouched
            },
        },
        # Don't update a4, to test that it is untouched
    }

    result = config_args.recursive_replace(instance, **changes)

    self.assertIsInstance(result, A)
    self.assertEqual(result.a1, -1)
    self.assertEqual(result.a2, 8)
    self.assertIsInstance(result.a3, B)
    self.assertEqual(result.a3.b1, 9)
    self.assertEqual(result.a3.b2, -2)
    self.assertIsInstance(result.a3.b3, C)
    self.assertEqual(result.a3.b3.c1, 11)
    self.assertEqual(result.a3.b3.c2, 12)
    self.assertIsInstance(result.a3.b4, C)
    self.assertEqual(result.a3.b4.c1, -3)
    self.assertEqual(result.a3.b4.c2, 14)
    self.assertIsInstance(result.a4, B)
    self.assertEqual(result.a4.b1, 15)
    self.assertEqual(result.a4.b2, 16)
    self.assertIsInstance(result.a4.b3, C)
    self.assertEqual(result.a4.b3.c1, 17)
    self.assertEqual(result.a4.b3.c2, 18)
    self.assertIsInstance(result.a4.b4, C)
    self.assertEqual(result.a4.b4.c1, 19)
    self.assertEqual(result.a4.b4.c2, 20)

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

  @parameterized.named_parameters(
      # 2 tuple inputs represent a constant (in time) radial profile.
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
      # 3 tuples are a general profile.
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
                  config_args.TIME_INTERPOLATION_MODE: 'step',
                  config_args.RHO_INTERPOLATION_MODE: 'piecewise_linear',
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
                  config_args.TIME_INTERPOLATION_MODE: 'step',
                  config_args.RHO_INTERPOLATION_MODE: 'piecewise_linear',
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
                  config_args.TIME_INTERPOLATION_MODE: 'step',
                  config_args.RHO_INTERPOLATION_MODE: 'piecewise_linear',
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
    interpolated_var_time_rho = config_args.get_interpolated_var_2d(
        time_rho_interpolated_input=time_rho_interpolated_input,
        rho_norm=rho_norm,
    )
    np.testing.assert_allclose(
        interpolated_var_time_rho.get_value(x=time),
        expected_output,
    )


@dataclasses.dataclass
class C:
  c1: int = 1
  c2: int = 2


@dataclasses.dataclass
class B:
  b1: int = 3
  b2: int = 4
  b3: C = dataclasses.field(default_factory=C)
  b4: C = dataclasses.field(default_factory=C)


@dataclasses.dataclass
class A:
  a1: int = 5
  a2: int = 6
  a3: B = dataclasses.field(default_factory=B)
  a4: B = dataclasses.field(default_factory=B)


if __name__ == '__main__':
  absltest.main()
