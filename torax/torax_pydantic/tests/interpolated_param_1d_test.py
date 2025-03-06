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
import pydantic
from torax import interpolated_param
from torax.torax_pydantic import torax_pydantic
import xarray as xr

RHO_NORM = 'rho_norm'
TIME_INTERPOLATION_MODE = 'time_interpolation_mode'
RHO_INTERPOLATION_MODE = 'rho_interpolation_mode'


class InterpolatedParam1dTest(parameterized.TestCase):

  def test_time_varying_model_basic(
      self,
  ):
    """Check the properties of the interpolated var are set correctly."""

    default_value = 1.53

    class TestModel(torax_pydantic.BaseModelFrozen):
      a: torax_pydantic.TimeVaryingScalar
      b: torax_pydantic.TimeVaryingScalar = pydantic.Field(
          default_factory=lambda: default_value, validate_default=True
      )

    a_expected = torax_pydantic.TimeVaryingScalar(
        time=np.array([0.0, 1.0, 2.0]),
        value=np.array([1.0, 2.0, 4.0]),
        interpolation_mode=interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
    )
    b_expected = torax_pydantic.TimeVaryingScalar(
        time=np.array([0.0]),
        value=np.array([default_value]),
        interpolation_mode=interpolated_param.InterpolationMode.PIECEWISE_LINEAR,
    )

    user_config = dict(
        a=xr.DataArray(
            data=a_expected.value,
            coords={'time': a_expected.time},
        ),
    )
    with self.subTest('model_validate'):
      model_pydantic = TestModel.model_validate(user_config)
      self.assertEqual(a_expected, model_pydantic.a)
      self.assertEqual(b_expected, model_pydantic.b)

    with self.subTest('json_dump_and_load'):
      model_json = model_pydantic.model_dump_json()
      self.assertEqual(
          model_pydantic, TestModel.model_validate_json(model_json)
      )

  def test_bool_single_value_param_always_return_constant(self):
    """Tests that when passed a single value this is always returned."""
    expected_output = True
    single_value_param = torax_pydantic.TimeVaryingScalar.model_validate(
        expected_output
    )
    np.testing.assert_allclose(
        single_value_param.get_value(-1), expected_output
    )
    np.testing.assert_allclose(single_value_param.get_value(0), expected_output)
    np.testing.assert_allclose(single_value_param.get_value(1), expected_output)

  def test_dict_range_input_must_have_values(self):
    with self.assertRaises(ValueError):
      torax_pydantic.TimeVaryingScalar.model_validate({})

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
    multi_val_range = torax_pydantic.TimeVaryingScalar.model_validate(
        values
    )

    if isinstance(expected_output, bool):
      self.assertEqual(multi_val_range.get_value(t=x), expected_output)
    else:
      np.testing.assert_allclose(
          multi_val_range.get_value(t=x),
          expected_output,
      )


if __name__ == '__main__':
  absltest.main()
