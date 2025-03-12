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

"""Classes and functions for defining interpolated parameters."""

import functools
from typing import Any

import chex
import pydantic
from torax import interpolated_param
from torax.torax_pydantic import model_base
from torax.torax_pydantic import pydantic_types


class TimeVaryingScalar(model_base.BaseModelFrozen):
  """Base class for time interpolated scalar types.

  The Pydantic `.model_validate` constructor can accept a variety of input types
  defined by the `TimeInterpolatedInput` type. See
  https://torax.readthedocs.io/en/latest/configuration.html#time-varying-scalars
  for more details.

  Attributes:
    time: A 1-dimensional NumPy array of times.
    value: A NumPy array specifying the values to interpolate.
    is_bool_param: If True, the input value is assumed to be a bool and is
      converted to a float.
    interpolation_mode: An InterpolationMode enum specifying the interpolation
      mode to use.
  """

  time: pydantic_types.NumpyArray1D
  value: pydantic_types.NumpyArray
  is_bool_param: bool = False
  interpolation_mode: interpolated_param.InterpolationMode = (
      interpolated_param.InterpolationMode.PIECEWISE_LINEAR
  )

  def get_value(self, t: chex.Numeric) -> chex.Array:
    """Returns the value of this parameter interpolated at x=time.

    Args:
      t: An array of times to interpolate at.

    Returns:
      An array of interpolated values.
    """
    return self._get_cached_interpolated_param.get_value(t)

  def __eq__(self, other):
    try:
      chex.assert_trees_all_equal(vars(self), vars(other))
      return True
    except AssertionError:
      return False

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(
      cls, data: interpolated_param.TimeInterpolatedInput | dict[str, Any]
  ) -> dict[str, Any]:

    if isinstance(data, dict):
      # A workaround for https://github.com/pydantic/pydantic/issues/10477.
      data.pop('_get_cached_interpolated_param', None)

      # This is the standard constructor input. No conforming required.
      if set(data.keys()).issubset(cls.model_fields.keys()):
        return data  # pytype: disable=bad-return-type

    time, value, interpolation_mode, is_bool_param = (
        interpolated_param.convert_input_to_xs_ys(data)
    )
    return dict(
        time=time,
        value=value,
        interpolation_mode=interpolation_mode,
        is_bool_param=is_bool_param,
    )

  @functools.cached_property
  def _get_cached_interpolated_param(
      self,
  ) -> interpolated_param.InterpolatedVarSingleAxis:
    """Interpolates the input param at time t."""

    return interpolated_param.InterpolatedVarSingleAxis(
        value=(self.time, self.value),
        interpolation_mode=self.interpolation_mode,
        is_bool_param=self.is_bool_param,
    )
