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
import typing
from typing import TypeVar


# TypeVar for generic dataclass types.
_T = TypeVar('_T')
RHO_NORM = 'rho_norm'
TIME_INTERPOLATION_MODE = 'time_interpolation_mode'
RHO_INTERPOLATION_MODE = 'rho_interpolation_mode'


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
