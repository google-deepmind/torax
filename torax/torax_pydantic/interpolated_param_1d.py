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
from typing import Any, TypeAlias

import chex
import numpy as np
import pydantic
from torax import interpolated_param
from torax.torax_pydantic import model_base
from torax.torax_pydantic import pydantic_types
from typing_extensions import Annotated
from typing_extensions import Self


class TimeVaryingScalar(model_base.BaseModelFrozen):
  """Base class for time interpolated scalar types.

  The Pydantic `.model_validate` constructor can accept a variety of input types
  defined by the `TimeInterpolatedInput` type. See
  https://torax.readthedocs.io/en/latest/configuration.html#time-varying-scalars
  for more details.

  Attributes:
    time: A 1-dimensional NumPy array of times sorted in ascending time order.
    value: A NumPy array specifying the values to interpolate. The same length
      as `time`.
    is_bool_param: If True, the input value is assumed to be a bool and is
      converted to a float.
    interpolation_mode: An InterpolationMode enum specifying the interpolation
      mode to use.
  """

  time: pydantic_types.NumpyArray1DSorted
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
    return (
        np.array_equal(self.time, other.time)
        and np.array_equal(self.value, other.value)
        and self.is_bool_param == other.is_bool_param
        and self.interpolation_mode == other.interpolation_mode
    )

  @pydantic.model_validator(mode='after')
  def _ensure_consistent_arrays(self) -> Self:

    if not np.issubdtype(self.time.dtype, np.floating):
      raise ValueError('The time array must be a float array.')

    if self.time.dtype != self.value.dtype:
      raise ValueError('The time and value arrays must have the same dtype.')

    if len(self.time) != len(self.value):
      raise ValueError('The value and time arrays must be the same length.')

    return self

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

    # Ensure that the time is sorted.
    sort_order = np.argsort(time)
    time = time[sort_order]
    value = value[sort_order]
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


def _is_positive(time_varying_scalar: TimeVaryingScalar) -> TimeVaryingScalar:
  if not np.all(time_varying_scalar.value > 0):
    raise ValueError('All values must be positive.')
  return time_varying_scalar


def _interval(
    time_varying_scalar: TimeVaryingScalar,
    lower_bound: float,
    upper_bound: float,
) -> TimeVaryingScalar:
  if not np.all(lower_bound <= time_varying_scalar.value <= upper_bound):
    raise ValueError(
        'All values must be less than %f and greater than %f.'
        % (upper_bound, lower_bound)
    )
  return time_varying_scalar


PositiveTimeVaryingScalar: TypeAlias = Annotated[
    TimeVaryingScalar, pydantic.AfterValidator(_is_positive)
]
UnitIntervalTimeVaryingScalar: TypeAlias = Annotated[
    TimeVaryingScalar,
    pydantic.AfterValidator(
        functools.partial(_interval, lower_bound=0.0, upper_bound=1.0)
    ),
]
