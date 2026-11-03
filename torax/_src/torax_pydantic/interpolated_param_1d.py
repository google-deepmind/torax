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

import dataclasses
import functools
from typing import Annotated, Any, Self, TypeAlias

import chex
import equinox as eqx
import jax
import jaxtyping as jt
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import interpolated_param
from torax._src.torax_pydantic import config_base
from torax._src.torax_pydantic import pydantic_types


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TimeVaryingScalarUpdate:
  """Replacement values for updating a TimeVaryingScalar under `jax.jit`."""

  value: jt.Float[jax.Array, 't'] | None = None
  time: jt.Float[jax.Array, 't'] | None = None


class TimeVaryingScalar(config_base.ConfigBase):
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
  is_bool_param: Annotated[bool, config_base.JAX_STATIC] = (
      False
  )
  interpolation_mode: Annotated[
      interpolated_param.InterpolationMode, config_base.JAX_STATIC
  ] = interpolated_param.InterpolationMode.PIECEWISE_LINEAR

  def get_value(self, t: chex.Numeric) -> array_typing.Array:
    """Returns the value of this parameter interpolated at time t.

    Args:
      t: An array of times to interpolate at.

    Returns:
      An array of interpolated values.
    """
    return self._get_cached_interpolated_param.get_value(t)

  def update(
      self, replacements: TimeVaryingScalarUpdate
  ) -> Self:
    """Updates the TimeVaryingScalar with new values or times under `jax.jit`.

    Args:
      replacements: A `TimeVaryingScalarUpdate` specifying replacement `value`
        and/or `time` arrays.

    Returns:
      A new `TimeVaryingScalar` with the updated arrays.

    Raises:
      ValueError: If the replacement `value` and `time` arrays do not have the
        same shape.
    """
    value = replacements.value if replacements.value is not None else self.value
    time = replacements.time if replacements.time is not None else self.time
    if time.shape != value.shape:
      raise ValueError(
          'The value and time arrays to update this `TimeVaryingScalar` must'
          f' be the same length. Got value: {value.shape}, time: {time.shape}.'
      )

    def get_leaves(x: Self) -> tuple[array_typing.Array, array_typing.Array]:
      return (x.time, x.value)

    return eqx.tree_at(get_leaves, self, (time, value),)

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
      data.pop('_get_cached_interpolated_param', None)  # pyrefly: ignore[no-matching-overload]

      # This is the standard constructor input. No conforming required.
      if set(data.keys()).issubset(cls.model_fields.keys()):
        return data  # pyrefly: ignore[bad-return]

    default_interpolation_mode = cls.model_fields['interpolation_mode'].default

    time, value, interpolation_mode, is_bool_param = (
        interpolated_param.convert_input_to_xs_ys(
            data, default_interpolation_mode  # pyrefly: ignore[bad-argument-type]
        )
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


class TimeVaryingScalarStep(TimeVaryingScalar):
  """TimeVaryingScalar with STEP interpolation mode by default."""
  interpolation_mode: Annotated[
      interpolated_param.InterpolationMode, config_base.JAX_STATIC
  ] = interpolated_param.InterpolationMode.STEP


def _validate_scalar_bounds(
    time_varying_scalar: TimeVaryingScalar,
    *,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
) -> TimeVaryingScalar:
  """Validates that values in a TimeVaryingScalar satisfy bound constraints."""
  if gt is not None and ge is not None:
    raise ValueError('At most one of `gt` and `ge` can be provided.')
  if lt is not None and le is not None:
    raise ValueError('At most one of `lt` and `le` can be provided.')
  if gt is not None and not np.all(time_varying_scalar.value > gt):
    raise ValueError(f'All values must be greater than {gt}.')
  if ge is not None and not np.all(time_varying_scalar.value >= ge):
    raise ValueError(f'All values must be greater than or equal to {ge}.')
  if lt is not None and not np.all(time_varying_scalar.value < lt):
    raise ValueError(f'All values must be less than {lt}.')
  if le is not None and not np.all(time_varying_scalar.value <= le):
    raise ValueError(f'All values must be less than or equal to {le}.')
  return time_varying_scalar


def scalar_bounds_validator(
    *,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
) -> pydantic.AfterValidator:
  """Returns an AfterValidator that validates scalar bounds.

  Args:
    gt: Exclusive lower bound (`value > gt`), if specified.
    ge: Inclusive lower bound (`value >= ge`), if specified.
    lt: Exclusive upper bound (`value < lt`), if specified.
    le: Inclusive upper bound (`value <= le`), if specified.

  Returns:
    A `pydantic.AfterValidator` that checks the specified bounds on a
    `TimeVaryingScalar`.

  Raises:
    ValueError: If both `gt` and `ge` are provided, or if both `lt` and `le` are
      provided.
  """
  if gt is not None and ge is not None:
    raise ValueError('At most one of `gt` and `ge` can be provided.')
  if lt is not None and le is not None:
    raise ValueError('At most one of `lt` and `le` can be provided.')
  return pydantic.AfterValidator(
      functools.partial(
          _validate_scalar_bounds,
          gt=gt,
          ge=ge,
          lt=lt,
          le=le,
      )
  )


PositiveTimeVaryingScalar: TypeAlias = Annotated[
    TimeVaryingScalar, scalar_bounds_validator(gt=0.0)
]
NonNegativeTimeVaryingScalar: TypeAlias = Annotated[
    TimeVaryingScalar, scalar_bounds_validator(ge=0.0)
]
NonNegativeTimeVaryingScalarStep: TypeAlias = Annotated[
    TimeVaryingScalarStep, scalar_bounds_validator(ge=0.0)
]
UnitIntervalTimeVaryingScalar: TypeAlias = Annotated[
    TimeVaryingScalar, scalar_bounds_validator(ge=0.0, le=1.0)
]
