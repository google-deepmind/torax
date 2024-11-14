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

"""Functions for building arguments for configs and runtime input params."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import enum
import types
import typing
from typing import Any
from typing import TypeVar

import chex
from torax import interpolated_param
import xarray as xr

# TypeVar for generic dataclass types.
_T = TypeVar('_T')
RHO_NORM = 'rho_norm'
TIME_INTERPOLATION_MODE = 'time_interpolation_mode'
RHO_INTERPOLATION_MODE = 'rho_interpolation_mode'


def input_is_an_interpolated_var_single_axis(
    field_name: str,
    input_config_fields_to_types: dict[str, Any],
) -> bool:
  """Returns True if the input config field is an InterpolatedVarSingleAxis."""
  if field_name not in input_config_fields_to_types:
    return False

  def _check(ft):
    """Checks if the input field type is an InterpolatedVarSingleAxis."""
    try:
      return (
          # If the type comes as a string rather than an object, the Union check
          # below won't work, so we check for the full name here.
          ft == 'TimeInterpolatedInput'
          or
          # Common alias for TimeInterpolatedInput in a few files.
          (isinstance(ft, str) and 'TimeInterpolatedInput' in ft)
          or
          # Otherwise, check if it is actually the InterpolatedVarSingleAxis.
          ft == 'interpolated_param.InterpolatedVarSingleAxis'
          or issubclass(ft, interpolated_param.InterpolatedParamBase)
      )
    except:  # pylint: disable=bare-except
      # issubclass does not play nicely with generics, but if a type is a
      # generic at this stage, it is not an InterpolatedVarSingleAxis.
      return False

  field_type = input_config_fields_to_types[field_name]
  if isinstance(field_type, types.UnionType):
    # Look at all the args of the union and see if any match properly
    for arg in typing.get_args(field_type):
      if _check(arg):
        return True
  else:
    return _check(field_type)  # pytype: disable=bad-return-type


def _is_bool(
    interp_input: interpolated_param.InterpolatedVarSingleAxisInput,
) -> bool:
  if isinstance(interp_input, dict):
    if not interp_input:
      raise ValueError('InterpolatedVarSingleAxisInput must include values.')
    value = list(interp_input.values())[0]
    return isinstance(value, bool)
  return isinstance(interp_input, bool)


def _convert_value_to_floats(
    interp_input: interpolated_param.InterpolatedVarSingleAxisInput,
) -> interpolated_param.InterpolatedVarSingleAxisInput:
  if isinstance(interp_input, dict):
    return {key: float(value) for key, value in interp_input.items()}
  return float(interp_input)


def get_interpolated_var_single_axis(
    interpolated_var_single_axis_input: interpolated_param.InterpolatedVarSingleAxisInput,
) -> interpolated_param.InterpolatedVarSingleAxis:
  """Interpolates the input param at time t.

  Args:
    interpolated_var_single_axis_input: Input that can be used to construct a
      `interpolated_param.InterpolatedVarSingleAxis` object. Can be either:
      Python primitives, an xr.DataArray, a tuple(axis_array, values_array).
      See torax.readthedocs.io/en/latest/configuration.html#time-varying-scalars
      for more information on the supported inputs.

  Returns:
    A constructed interpolated var.
  """
  interpolation_mode = interpolated_param.InterpolationMode.PIECEWISE_LINEAR
  # The param is a InterpolatedVarSingleAxisInput, so we need to convert it to
  # an InterpolatedVarSingleAxis first.
  if isinstance(interpolated_var_single_axis_input, tuple):
    if len(interpolated_var_single_axis_input) != 2:
      raise ValueError(
          'Single axis interpolated var tuple length must be 2. The first '
          'element are the values and the second element is the '
          'interpolation mode or both values should be arrays to be directly '
          f'interpolated. Given: {interpolated_var_single_axis_input}.'
      )
    if isinstance(interpolated_var_single_axis_input[1], str):
      interpolation_mode = interpolated_param.InterpolationMode[
          interpolated_var_single_axis_input[1].upper()
      ]
      interpolated_var_single_axis_input = interpolated_var_single_axis_input[0]

  if _is_bool(interpolated_var_single_axis_input):
    interpolated_var_single_axis_input = _convert_value_to_floats(
        interpolated_var_single_axis_input
    )
    is_bool_param = True
  else:
    is_bool_param = False

  xs, ys = interpolated_param.convert_input_to_xs_ys(
      interpolated_var_single_axis_input
  )

  interpolated_var_single_axis = interpolated_param.InterpolatedVarSingleAxis(
      value=(xs, ys),
      interpolation_mode=interpolation_mode,
      is_bool_param=is_bool_param,
  )
  return interpolated_var_single_axis


def input_is_an_interpolated_var_time_rho(
    field_name: str,
    input_config_fields_to_types: dict[str, Any],
) -> bool:
  """Returns True if the input config field is a TimeRhoInterpolated."""

  def _check(ft):
    """Checks if the input field type is an InterpolatedVarTimeRho."""
    try:
      return isinstance(ft, str) and 'InterpolatedVarTimeRhoInput' in ft
    except:  # pylint: disable=bare-except
      # issubclass does not play nicely with generics, but if a type is a
      # generic at this stage, it is not an InterpolatedVarTimeRhoInput.
      return False

  field_type = input_config_fields_to_types[field_name]
  if isinstance(field_type, types.UnionType):
    # Look at all the args of the union and see if any match properly
    for arg in typing.get_args(field_type):
      if _check(arg):
        return True
  else:
    return _check(field_type)  # pytype: disable=bad-return-type


def _load_from_primitives(
    primitive_values: (
        Mapping[float, interpolated_param.InterpolatedVarSingleAxisInput]
        | float
    ),
) -> Mapping[float, tuple[chex.Array, chex.Array]]:
  """Loads the data from primitives.

  Three cases are supported:
  1. A float is passed in, describes constant initial condition profile.
  2. A non-nested dict is passed in, it will describe the radial profile for
     the initial condition.
  3. A nested dict is passed in, it will describe the time-dependent radial
     profile for the initial condition.

  Args:
    primitive_values: The python primitive values to load.

  Returns:
    A mapping from time to (rho_norm, values) where rho_norm and values are both
    arrays of equal length.
  """
  # Float case.
  if isinstance(primitive_values, (float, int)):
    primitive_values = {0.0: {0.0: primitive_values}}
  # Non-nested dict.
  if isinstance(primitive_values, Mapping) and all(
      isinstance(v, float) for v in primitive_values.values()
  ):
    primitive_values = {0.0: primitive_values}

  if len(set(primitive_values.keys())) != len(primitive_values):
    raise ValueError('Indicies in values mapping must be unique.')
  if not primitive_values:
    raise ValueError('Values mapping must not be empty.')

  primitive_values = {
      t: interpolated_param.convert_input_to_xs_ys(v)
      for t, v in primitive_values.items()
  }
  return primitive_values


def _load_from_xr_array(
    xr_array: xr.DataArray,
) -> Mapping[float, tuple[chex.Array, chex.Array]]:
  """Loads the data from an xr.DataArray."""
  if 'time' not in xr_array.coords:
    raise ValueError('"time" must be a coordinate in given dataset.')
  if RHO_NORM not in xr_array.coords:
    raise ValueError(f'"{RHO_NORM}" must be a coordinate in given dataset.')
  values = {
      t: (
          xr_array.rho_norm.data,
          xr_array.sel(time=t).values,
      )
      for t in xr_array.time.data
  }
  return values


def _load_from_arrays(
    arrays: tuple[chex.Array, ...],
) -> Mapping[float, tuple[chex.Array, chex.Array]]:
  """Loads the data from numpy arrays.

  Args:
    arrays: A tuple of (times, rho_norm, values) or (rho_norm, values). - In the
      former case times and rho_norm are assumed to be 1D arrays of equal
      length, values is a 2D array with shape (len(times), len(rho_norm)). - In
      the latter case rho_norm and values are assumed to be 1D arrays of equal
      length (shortcut for initial condition profile).

  Returns:
    A mapping from time to (rho_norm, values)
  """
  if len(arrays) == 2:
    # Shortcut for initial condition profile.
    rho_norm, values = arrays
    if len(rho_norm.shape) != 1:
      raise ValueError(f'rho_norm must be a 1D array. Given: {rho_norm.shape}.')
    if len(values.shape) != 1:
      raise ValueError(f'values must be a 1D array. Given: {values.shape}.')
    if rho_norm.shape != values.shape:
      raise ValueError(
          'rho_norm and values must be of the same shape. Given: '
          f'{rho_norm.shape} and {values.shape}.'
      )
    return {0.0: (rho_norm, values)}
  if len(arrays) == 3:
    times, rho_norm, values = arrays
    if len(times.shape) != 1:
      raise ValueError(f'times must be a 1D array. Given: {times.shape}.')
    if len(rho_norm.shape) != 1:
      raise ValueError(f'rho_norm must be a 1D array. Given: {rho_norm.shape}.')
    if values.shape != (len(times), len(rho_norm)):
      raise ValueError(
          'values must be of shape (len(times), len(rho_norm)). Given: '
          f'{values.shape}.'
      )
    return {t: (rho_norm, values[i, :]) for i, t in enumerate(times)}
  else:
    raise ValueError(f'arrays must be length 2 or 3. Given: {len(arrays)}.')


def get_interpolated_var_2d(
    time_rho_interpolated_input: interpolated_param.InterpolatedVarTimeRhoInput,
    rho_norm: chex.Array,
) -> interpolated_param.InterpolatedVarTimeRho:
  """Constructs an InterpolatedVarTimeRho from the given input.

  Three cases are supported:
  1. Python primitives are passed in, see _load_from_primitives for details.
  2. An xr.DataArray is passed in, see _load_from_xr_array for details.
  3. A tuple of arrays is passed in, see _load_from_arrays for details.

  Additionally the interpolation mode for rhon and time can be specified as
  strings by passing a 3-tuple with the first element being the input, the
  second element being the time interpolation mode and the third element
  being the rhon interpolation mode.

  Args:
    time_rho_interpolated_input: An input that can be used to construct a
      InterpolatedVarTimeRho object.
    rho_norm: The rho_norm values to interpolate at (usually the TORAX mesh).

  Returns:
    An InterpolatedVarTimeRho object which has been preinterpolated onto the
    provided rho_norm values.
  """
  # Potentially parse the interpolation modes from the input.
  time_interpolation_mode = (
      interpolated_param.InterpolationMode.PIECEWISE_LINEAR
  )
  rho_interpolation_mode = interpolated_param.InterpolationMode.PIECEWISE_LINEAR
  if isinstance(time_rho_interpolated_input, tuple):
    if (
        len(time_rho_interpolated_input) == 2
        and isinstance(time_rho_interpolated_input[1], dict)
    ):
      # Second and third elements in tuple are interpreted as interpolation
      # modes.
      time_interpolation_mode = interpolated_param.InterpolationMode[
          time_rho_interpolated_input[1][TIME_INTERPOLATION_MODE].upper()
      ]
      rho_interpolation_mode = interpolated_param.InterpolationMode[
          time_rho_interpolated_input[1][RHO_INTERPOLATION_MODE].upper()
      ]
      # First element in tuple assumed to be the input.
      time_rho_interpolated_input = time_rho_interpolated_input[0]

  if isinstance(time_rho_interpolated_input, xr.DataArray):
    values = _load_from_xr_array(
        time_rho_interpolated_input,
    )
  elif isinstance(time_rho_interpolated_input, tuple) and all(
      isinstance(v, chex.Array) for v in time_rho_interpolated_input
  ):
    values = _load_from_arrays(
        time_rho_interpolated_input,
    )
  elif isinstance(time_rho_interpolated_input, Mapping) or isinstance(
      time_rho_interpolated_input, (float, int)
  ):
    values = _load_from_primitives(
        time_rho_interpolated_input,
    )
  else:
    raise ValueError('Input for interpolated var not recognised.')

  time_rho_interpolated = interpolated_param.InterpolatedVarTimeRho(
      values=values,
      rho_norm=rho_norm,
      time_interpolation_mode=time_interpolation_mode,
      rho_interpolation_mode=rho_interpolation_mode,
  )
  return time_rho_interpolated


def recursive_replace(
    obj: _T, ignore_extra_kwargs: bool = False, **changes
) -> _T:
  """Recursive version of `dataclasses.replace`.

  This allows updating of nested dataclasses.
  Assumes all dict-valued keys in `changes` are themselves changes to apply
  to fields of obj.

  Args:
    obj: Any dataclass instance.
    ignore_extra_kwargs: If True, any kwargs from `changes` are ignored if they
      do not apply to `obj`.
    **changes: Dict of updates to apply to fields of `obj`.

  Returns:
    A copy of `obj` with the changes applied.
  """

  flattened_changes = {}
  if dataclasses.is_dataclass(obj):
    keys_to_types = {
        field.name: field.type for field in dataclasses.fields(obj)
    }
  else:
    # obj is another dict-like object that does not have typed fields.
    keys_to_types = None
  for key, value in changes.items():
    if (
        ignore_extra_kwargs
        and keys_to_types is not None
        and key not in keys_to_types
    ):
      continue
    if isinstance(value, dict):
      if dataclasses.is_dataclass(getattr(obj, key)):
        # If obj[key] is another dataclass, recurse and populate that dataclass
        # with the input changes.
        flattened_changes[key] = recursive_replace(
            getattr(obj, key), ignore_extra_kwargs=ignore_extra_kwargs, **value
        )
      elif keys_to_types is not None:
        # obj[key] is likely just a dict, and each key needs to be treated
        # separately.
        # In order to support this, there needs to be some added type
        # information for what the values of the dict should be.
        typing_args = typing.get_args(keys_to_types[key])
        if len(typing_args) == 2:  # the keys type, the values type.
          inner_dict = {}
          value_type = typing_args[1]
          for inner_key, inner_value in value.items():
            if dataclasses.is_dataclass(value_type):
              inner_dict[inner_key] = recursive_replace(
                  value_type(),
                  ignore_extra_kwargs=ignore_extra_kwargs,
                  **inner_value,
              )
            else:
              inner_dict[inner_key] = value_type(inner_value)
          flattened_changes[key] = inner_dict
        else:
          # If we don't have additional type information, just try using the
          # value as is.
          flattened_changes[key] = value
      else:
        # keys_to_types is None, so again, we don't have additional information.
        flattened_changes[key] = value
    else:
      # For any value that should be an enum value but is not an enum already
      # (could come a YAML file for instance and might be a string or int),
      # this converts that value to an enum.
      try:
        if (
            # if obj is a dataclass
            keys_to_types is not None
            and
            # and this param should be an enum
            issubclass(keys_to_types[key], enum.Enum)
            and
            # but it is not already one.
            not isinstance(value, enum.Enum)
        ):
          if isinstance(value, str):
            value = keys_to_types[key][value.upper()]
          else:
            value = keys_to_types[key](value)
      except TypeError:
        # Ignore these errors. issubclass doesn't work with typing.Optional
        # types. Note that this means that optional enum fields might not be
        # cast properly, so avoid these when defining configs.
        pass
      flattened_changes[key] = value
  return dataclasses.replace(obj, **flattened_changes)
