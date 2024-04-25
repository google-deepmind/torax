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

"""Functions to help build the arguments to config slice object constructors."""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from typing import Any

import chex
from jax import numpy as jnp
from torax import interpolated_param


def input_is_a_float_field(
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


def input_is_an_interpolated_param(
    field_name: str,
    input_config_fields_to_types: dict[str, Any],
) -> bool:
  """Returns True if the input config field is an InterpolatedParam."""
  if field_name not in input_config_fields_to_types:
    return False

  def _check(ft):
    """Checks if the input field type is an InterpolatedParam."""
    try:
      return (
          # If the type comes as a string rather than an object, the Union check
          # below won't work, so we check for the full name here.
          ft == 'InterpParamOrInterpParamInput'
          or
          # Common alias for InterpParamOrInterpParamInput in a few files.
          (isinstance(ft, str) and 'TimeDependentField' in ft)
          or
          # Otherwise, only check if it is actually the InterpolatedParam.
          ft == 'interpolated_param.InterpolatedParam'
          or issubclass(ft, interpolated_param.InterpolatedParamBase)
      )
    except:  # pylint: disable=bare-except
      # issubclass does not play nicely with generics, but if a type is a
      # generic at this stage, it is not an InterpolatedParam.
      return False

  field_type = input_config_fields_to_types[field_name]
  if isinstance(field_type, types.UnionType):
    # Look at all the args of the union and see if any match properly
    for arg in typing.get_args(field_type):
      if _check(arg):
        return True
  else:
    return _check(field_type)


def interpolate_param(
    param_or_param_input: interpolated_param.InterpParamOrInterpParamInput,
    t: chex.Numeric,
) -> jnp.ndarray:
  if not isinstance(param_or_param_input, interpolated_param.InterpolatedParam):
    # The param is a InterpolatedParamInput, so we need to convert it to an
    # InterpolatedParam first.
    param_or_param_input = interpolated_param.InterpolatedParam(
        value=param_or_param_input,
    )
  return param_or_param_input.get_value(t)


def get_init_kwargs(
    input_config: ...,
    output_type: ...,
    t: chex.Numeric | None = None,
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
    # If the input config type is an InterpolatedParam, we need to interpolate
    # it at time t to populate the correct values in the output config.
    # dataclass fields can either be the actual type OR the string name of the
    # type. Check for both.
    if input_is_an_interpolated_param(field.name, input_config_fields_to_types):
      if t is None:
        raise ValueError('t must be specified for interpolated params')
      config_val = interpolate_param(config_val, t)
    elif input_is_a_float_field(field.name, input_config_fields_to_types):
      config_val = float(config_val)
    elif isinstance(config_val, enum.Enum):
      config_val = config_val.value
    elif hasattr(config_val, 'build_dynamic_params'):
      config_val = config_val.build_dynamic_params(t)
    kwargs[field.name] = config_val
  return kwargs
