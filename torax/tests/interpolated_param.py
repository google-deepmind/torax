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

"""Unit tests for torax.interpolated_param."""
import random

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import interpolated_param


class InterpolatedParamTest(parameterized.TestCase):
  """Unit tests for the `torax.interpolated_param` module."""

  @parameterized.parameters(
      (42.0,),
      (True,),
  )
  def test_single_value_param_always_return_constant(self, expected_output):
    single_value_param = interpolated_param.InterpolatedVar1d(expected_output)
    np.testing.assert_allclose(
        single_value_param.get_value(-1), expected_output
    )
    np.testing.assert_allclose(single_value_param.get_value(0), expected_output)
    np.testing.assert_allclose(single_value_param.get_value(1), expected_output)

  @parameterized.parameters(
      (
          {0.0: 0.0, 1.0: 1.0, 2.0: 2.0, 3.0: 3.0},
          1.5,
          1.5,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      ),
      (
          {0.0: 1.0, 1.0: 2.0, 2.0: 4, 3.0: 8},
          0.5,
          1.5,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      ),
      (
          {0.0: 1.0, 1.0: 2.0, 2.0: 4, 3.0: 8},
          1.5,
          3,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      ),
      (
          {1.0: 1.0, 3.0: 2.0, 5.0: 4, 7.0: 8},
          6.5,
          7,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      ),
      # outside range uses last value.
      (
          {12.0: 10.0, 14.0: 9.0, 18: 8.0, 19.0: 4.0},
          20,
          4,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      ),
      # sorts the keys.
      (
          {0.0: 1.0, 5.0: 0.0, 2.0: 4.0, 3.0: 4.0},
          1.0,
          2.5,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      ),
      (
          {0.0: 1.0, 5.0: 0.0, 2.0: 4.0, 3.0: 4.0},
          2.5,
          4.0,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      ),
      (
          7.0,
          1.0,
          7.0,
          interpolated_param.InterpolationMode.STEP,
      ),
      (
          {0.0: 1.0, 2.0: 7.0, 3.0: -1.0},
          -1.0,
          1.0,
          interpolated_param.InterpolationMode.STEP,
      ),
      (
          {0.0: 1.0, 2.0: 7.0, 3.0: -1.0},
          1.0,
          1.0,
          interpolated_param.InterpolationMode.STEP,
      ),
      (
          {0.0: 1.0, 2.0: 7.0, 3.0: -1.0},
          2.6,
          7.0,
          interpolated_param.InterpolationMode.STEP,
      ),
      (
          {0.0: 1.0, 2.0: 7.0, 3.0: -1.0},
          4.0,
          -1.0,
          interpolated_param.InterpolationMode.STEP,
      ),
      (
          {0.0: False, 2.0: True, 3.0: False},
          1.5,
          True,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      ),
      (
          {0.0: False, 2.0: True, 3.0: False},
          1.0,
          False,
          interpolated_param.InterpolationMode.STEP,
      ),
      (
          {0.0: False, 2.0: True, 3.0: False},
          2.5,
          True,
          interpolated_param.InterpolationMode.STEP,
      ),
      (
          (np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0, 8.0])),
          1.5,
          3.0,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      ),
      (
          (np.array([0.0, 2.0, 1.0, 3.0]), np.array([1.0, 4.0, 2.0, 8.0])),
          1.5,
          3.0,
          interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
      )
  )
  def test_multi_value_range_returns_expected_output(
      self,
      values,
      x,
      expected_output,
      interpolation_mode,
  ):
    """Tests that the range returns the expected output."""
    multi_val_range = interpolated_param.InterpolatedVar1d(
        values, interpolation_mode
    )
    if isinstance(expected_output, bool):
      self.assertEqual(multi_val_range.get_value(x=x), expected_output)
    else:
      np.testing.assert_allclose(
          multi_val_range.get_value(x=x),
          expected_output,
      )

  def test_dict_range_input_must_have_values(self):
    with self.assertRaises(ValueError):
      interpolated_param.InterpolatedVar1d({})

  @parameterized.parameters(
      (interpolated_param.PiecewiseLinearInterpolatedParam,),
      (interpolated_param.StepInterpolatedParam,),
  )
  def test_interpolated_param_needs_rank_one_values(self, range_class):
    """Tests that the interpolated_param only take 1D inputs."""
    range_class(
        xs=jnp.array([1.0, 2.0, 3.0, 4.0]),
        ys=jnp.array([1.0, 2.0, 3.0, 4.0]),
    )
    with self.assertRaises(AssertionError):
      range_class(
          xs=jnp.array(1.0),
          ys=jnp.array(2.0),
      )
    with self.assertRaises(AssertionError):
      range_class(
          xs=jnp.arange(6).reshape((2, 3)),
          ys=jnp.arange(6).reshape((2, 3)),
      )

  @parameterized.parameters(
      (interpolated_param.PiecewiseLinearInterpolatedParam,),
      (interpolated_param.StepInterpolatedParam,),
  )
  def test_interpolated_param_need_xs_ys_same_shape(self, range_class):
    """Tests the xs and ys inputs have to have the same shape."""
    range_class(
        xs=jnp.array([1.0, 2.0, 3.0, 4.0]),
        ys=jnp.array([1.0, 2.0, 3.0, 4.0]),
    )
    with self.assertRaises(AssertionError):
      range_class(
          xs=jnp.array([1.0, 2.0, 3.0, 4.0]),
          ys=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
      )

  @parameterized.parameters(
      (interpolated_param.PiecewiseLinearInterpolatedParam,),
      (interpolated_param.StepInterpolatedParam,),
  )
  def test_interpolated_param_need_xs_to_be_sorted(self, range_class):
    """Tests the xs inputs have to be sorted."""
    range_class(
        xs=jnp.array([1.0, 2.0, 3.0, 4.0]),
        ys=jnp.array([1.0, 2.0, 3.0, 4.0]),
    )
    with self.assertRaises(jax.lib.xla_extension.XlaRuntimeError):
      range_class(
          xs=jnp.array([4.0, 2.0, 1.0, 3.0]),
          ys=jnp.array([1.0, 2.0, 3.0, 4.0]),
      )

  @parameterized.named_parameters(
      # One line cases.
      {
          'testcase_name': 'one_line_case_1',
          'values': {1.0: {0: 0.0, 0.3: 0.5, 0.9: 1, 1: 1.0}},
          'x': random.uniform(0, 2),
          'y': 0.0,
          'expected_output': 0.0,
      },
      {
          'testcase_name': 'one_line_case_2',
          'values': {1.0: {0: 0.0, 0.3: 0.5, 0.9: 1.0, 1: 1.0}},
          'x': random.uniform(0, 2),
          'y': 0.95,
          'expected_output': 1,
      },
      {
          'testcase_name': 'one_line_case_3',
          'values': {1.0: {0: 0.0, 0.3: 0.5, 0.8: 1.0, 1: 1.0}},
          'x': random.uniform(0, 2),
          'y': 0.7,
          'expected_output': 0.9,
      },
      # Two lines cases, constant at x=0, linear between 0 and 1 at x=1.
      {
          'testcase_name': 'two_line_case_1',
          'values': {0.0: 0.0, 1.0: {0: 0.0, 1: 1.0}},
          'x': 0.0,
          'y': random.randrange(0, 1),
          'expected_output': 0.0,
      },
      {
          'testcase_name': 'two_line_case_2',
          'values': {0.0: 0.0, 1.0: {0: 0.0, 1: 1.0}},
          'x': 1.0,
          'y': 0.0,
          'expected_output': 0.0,
      },
      {
          'testcase_name': 'two_line_case_3',
          'values': {0.0: 0.0, 1.0: {0: 0.0, 1: 1.0}},
          'x': 1.0,
          'y': 1.0,
          'expected_output': 1.0,
      },
      {
          'testcase_name': 'two_line_case_4',
          'values': {0.0: 0.0, 1.0: {0: 0.0, 1: 1.0}},
          'x': 1.0,
          'y': 0.5,
          'expected_output': 0.5,
      },
      {
          'testcase_name': 'two_line_case_5',
          'values': {0.0: 0.0, 1.0: {0: 0.0, 1: 1.0}},
          'x': 0.5,
          'y': 0.5,
          'expected_output': 0.25,
      },
      {
          'testcase_name': 'two_line_case_6',
          'values': {0.0: 0.0, 1.0: {0: 0.0, 1: 1.0}},
          'x': 0.5,
          'y': 0.75,
          'expected_output': 0.375,
      },
      # Test cases with 4 interpolated lines along the x axis.
      # Case in between the first two lines.
      {
          'testcase_name': 'four_line_case_1',
          'values': {
              0.0: {0: 0.0, 1: 0.0},
              1.0: {0: 0.0, 0.3: 0.5, 0.8: 1.0, 1: 1.0},
              2.0: {0: 1.0, 0.5: 0},
              3.0: {0: 1.0, 1: 0.0},
          },
          'x': 0.5,
          'y': 0.5,
          'expected_output': 0.35
      },
      # Case in between the second and third lines.
      {
          'testcase_name': 'four_line_case_2',
          'values': {
              0.0: {0: 0.0, 1: 0.0},
              1.0: {0: 0.0, 0.3: 0.5, 0.8: 1.0, 1: 1.0},
              2.0: {0: 1.0, 0.5: 0},
              3.0: {0: 1.0, 1: 0.0},
          },
          'x': 1.2,
          'y': 0.5,
          'expected_output': 0.56
      },
      # Case in between the third and fourth lines.
      {
          'testcase_name': 'four_line_case_3',
          'values': {
              0.0: {0: 0.0, 1: 0.0},
              1.0: {0: 0.0, 0.3: 0.5, 0.8: 1.0, 1: 1.0},
              2.0: {0: 1.0, 0.5: 0},
              3.0: {0: 1.0, 1: 0.0},
          },
          'x': 2.8,
          'y': 0.5,
          'expected_output': 0.4
      },
      # Case where y is an array.
      {
          'testcase_name': 'y_array_case',
          'values': {
              2.0: {0: 1.0, 0.5: 0},
              3.0: {0: 1.0, 1: 0.0},
          },
          'x': 2.8,
          'y': np.array([0.1, 0.5, 0.6]),
          'expected_output': np.array([0.88, 0.4, 0.32,])
      }
  )
  def test_interpolated_var_2d(self, values, x, y, expected_output):
    """Tests the doubly interpolated param gives correct outputs on 2D mesh."""
    interpolated_var_2d = interpolated_param.InterpolatedVar2d(
        values
    )

    output = interpolated_var_2d.get_value(time=x, rho=y)
    np.testing.assert_allclose(output, expected_output)

  def test_interpolated_var_2d_parses_float_input(self):
    """Tests that InterpolatedVar2d parses float inputs correctly."""
    interpolated_var_2d = interpolated_param.InterpolatedVar2d(
        values=1.0,
    )
    np.testing.assert_allclose(
        interpolated_var_2d.get_value(time=0.0, rho=0.0), 1.0
    )
    np.testing.assert_allclose(
        interpolated_var_2d.get_value(time=0.0, rho=1.0), 1.0
    )
    self.assertLen(interpolated_var_2d.values, 1)
    self.assertIn(0.0, interpolated_var_2d.values)

  def test_interpolated_var_2d_parses_single_dict_input(self):
    """Tests that InterpolatedVar2d parses float inputs correctly."""
    interpolated_var_2d = interpolated_param.InterpolatedVar2d(
        values={0: 18.0, 0.95: 5.0,},
    )
    np.testing.assert_allclose(
        interpolated_var_2d.get_value(time=0.0, rho=0.0), 18.0
    )
    np.testing.assert_allclose(
        interpolated_var_2d.get_value(time=0.5, rho=0.0), 18.0
    )

    np.testing.assert_allclose(
        interpolated_var_2d.get_value(time=0.0, rho=0.95), 5.0
    )
    np.testing.assert_allclose(
        interpolated_var_2d.get_value(time=0.5, rho=0.95), 5.0
    )


if __name__ == '__main__':
  absltest.main()
