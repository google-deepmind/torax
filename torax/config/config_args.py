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

import dataclasses
import enum
import types
import typing
from typing import Any
from typing import TypeVar

import chex
from jax import numpy as jnp
from torax import geometry
from torax import interpolated_param

# TypeVar for generic dataclass types.
_T = TypeVar('_T')


def _input_is_a_float_field(
    field_name: str,
    input_config_fields_to_types: dict[str, Any],
) -> bool:
  try:
    return field_name in input_config_fields_to_types and issubclass(
        input_config_fields_to_types[field_name], float
    )
  except:  # pylint: disable=bare-except
    # issubclass does not play nicely with generics, but if a type is a
    # generic at this stage, it is not a float.
    return False


def _input_is_an_interpolated_var_1d(
    field_name: str,
    input_config_fields_to_types: dict[str, Any],
) -> bool:
  """Returns True if the input config field is an InterpolatedVar1d."""
  if field_name not in input_config_fields_to_types:
    return False

  def _check(ft):
    """Checks if the input field type is an InterpolatedVar1d."""
    try:
      return (
          # If the type comes as a string rather than an object, the Union check
          # below won't work, so we check for the full name here.
          ft == 'TimeInterpolatedScalar'
          or
          # Common alias for TimeInterpolatedScalar in a few files.
          (isinstance(ft, str) and 'TimeInterpolatedScalar' in ft)
          or
          # Otherwise, only check if it is actually the InterpolatedVar1d.
          ft == 'interpolated_param.InterpolatedVar1d'
          or issubclass(ft, interpolated_param.InterpolatedParamBase)
      )
    except:  # pylint: disable=bare-except
      # issubclass does not play nicely with generics, but if a type is a
      # generic at this stage, it is not an InterpolatedVar1d.
      return False

  field_type = input_config_fields_to_types[field_name]
  if isinstance(field_type, types.UnionType):
    # Look at all the args of the union and see if any match properly
    for arg in typing.get_args(field_type):
      if _check(arg):
        return True
  else:
    return _check(field_type)


def _interpolate_var_1d(
    param_or_param_input: interpolated_param.TimeInterpolatedScalar,
    t: chex.Numeric,
) -> jnp.ndarray:
  """Interpolates the input param at time t."""
  if not isinstance(param_or_param_input, interpolated_param.InterpolatedVar1d):
    # The param is a InterpolatedVar1dInput, so we need to convert it to an
    # InterpolatedVar1d first.
    if isinstance(param_or_param_input, tuple):
      if len(param_or_param_input) != 2:
        raise ValueError(
            '1D interpolated var tuple length must be 2. The first element are'
            ' the values and the second element is the interpolation mode.'
            f' Given: {param_or_param_input}.'
        )
      param_or_param_input = interpolated_param.InterpolatedVar1d(
          value=param_or_param_input[0],
          interpolation_mode=interpolated_param.InterpolationMode[
              param_or_param_input[1].upper()
          ],
      )
    else:
      param_or_param_input = interpolated_param.InterpolatedVar1d(
          value=param_or_param_input,
      )
  return param_or_param_input.get_value(t)


def _input_is_an_interpolated_var_2d(
    field_name: str,
    input_config_fields_to_types: dict[str, Any],
) -> bool:
  """Returns True if the input config field is a TimeInterpolatedArray."""
  if field_name not in input_config_fields_to_types:
    return False

  def _check(ft):
    """Checks if the input field type is an InterpolatedVar2d."""
    try:
      return (
          # If the type comes as a string rather than an object, the Union check
          # below won't work, so we check for the full name here.
          ft == 'TimeInterpolatedArray'
          or
          # Common alias for TimeInterpolatedArray in a few files.
          (isinstance(ft, str) and 'TimeInterpolatedArray' in ft)
          or
          # Otherwise, only check if it is actually the InterpolatedVar2d.
          ft == 'interpolated_param.InterpolatedVar2d'
          or issubclass(ft, interpolated_param.InterpolatedVar2d)
      )
    except:  # pylint: disable=bare-except
      # issubclass does not play nicely with generics, but if a type is a
      # generic at this stage, it is not an InterpolatedVar2d.
      return False

  field_type = input_config_fields_to_types[field_name]
  if isinstance(field_type, types.UnionType):
    # Look at all the args of the union and see if any match properly
    for arg in typing.get_args(field_type):
      if _check(arg):
        return True
  else:
    return _check(field_type)


def _interpolate_var_2d(
    param_or_param_input: interpolated_param.TimeInterpolatedArray,
    t: chex.Numeric,
    geo: geometry.Geometry,
) -> jnp.ndarray:
  """Interpolates the input param at time t and rho_norm for the current geo."""
  if not isinstance(param_or_param_input, interpolated_param.InterpolatedVar2d):
    # Dealing with a param input so convert it first.
    param_or_param_input = interpolated_param.InterpolatedVar2d(
        values=param_or_param_input,
    )
  return param_or_param_input.get_value(t, geo.mesh.face_centers)


def get_init_kwargs(
    input_config: ...,
    output_type: ...,
    t: chex.Numeric | None = None,
    geo: geometry.Geometry | None = None,
    skip: tuple[str, ...] = (),
) -> dict[str, Any]:
  """Builds init() kwargs based on the input config for all non-dict fields."""
  kwargs = {}
  input_config_fields_to_types = {
      field.name: field.type for field in dataclasses.fields(input_config)
  }
  for field in dataclasses.fields(output_type):
    if field.name in skip:
      continue
    if not hasattr(input_config, field.name):
      raise ValueError(f'Missing field {field.name}')
    config_val = getattr(input_config, field.name)
    # If the input config type is an InterpolatedVar1d, we need to interpolate
    # it at time t to populate the correct values in the output config.
    # dataclass fields can either be the actual type OR the string name of the
    # type. Check for both.
    if _input_is_an_interpolated_var_1d(
        field.name, input_config_fields_to_types
    ):
      if t is None:
        raise ValueError('t must be specified for interpolated params')
      if config_val is not None:
        config_val = _interpolate_var_1d(config_val, t)
    elif _input_is_an_interpolated_var_2d(
        field.name, input_config_fields_to_types
    ):
      if config_val is not None:
        if t is None:
          raise ValueError('t must be specified for interpolated params')
        if geo is None:
          raise ValueError('geo must be specified for interpolated params')
        config_val = _interpolate_var_2d(config_val, t, geo)
    elif _input_is_a_float_field(field.name, input_config_fields_to_types):
      config_val = float(config_val)
    elif isinstance(config_val, enum.Enum):
      config_val = config_val.value
    elif hasattr(config_val, 'build_dynamic_params'):
      config_val = config_val.build_dynamic_params(t)
    kwargs[field.name] = config_val
  return kwargs


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
