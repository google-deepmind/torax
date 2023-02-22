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
    single_value_param = interpolated_param.InterpolatedParam(expected_output)
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
  )
  def test_multi_value_range_returns_expected_output(
      self,
      values,
      x,
      expected_output,
      interpolation_mode,
  ):
    """Tests that the range returns the expected output."""
    multi_val_range = interpolated_param.InterpolatedParam(
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
      interpolated_param.InterpolatedParam({})

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


if __name__ == '__main__':
  absltest.main()
