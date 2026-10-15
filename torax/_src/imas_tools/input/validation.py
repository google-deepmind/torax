# Copyright 2025 DeepMind Technologies Limited
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
"""Validation functions for IMAS loading tools."""

import functools
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Collection

import numpy as np

from imas import ids_toplevel
from torax._src import constants


def validate_finite_values(value: Any, *, context: str) -> None:
  """Raises when nested numeric input data contains NaN or infinity.

  Args:
    value: Nested mappings/sequences/scalars to validate.
    context: Human-readable description included in the error message.
  """

  def _walk(item: Any, path: str) -> None:
    if item is None or isinstance(item, (str, bytes)):
      return
    if isinstance(item, Mapping):
      for key, child in item.items():
        _walk(child, f"{path}.{key}" if path else str(key))
      return
    if isinstance(item, Sequence) and not isinstance(item, np.ndarray):
      for index, child in enumerate(item):
        _walk(child, f"{path}[{index}]")
      return

    try:
      array = np.asarray(item)
    except (TypeError, ValueError):
      return
    if array.dtype.kind in "biufc" and not np.all(np.isfinite(array)):
      location = path or "<root>"
      raise ValueError(f"{context} contains non-finite values at {location}.")

  _walk(value, "")


# pylint: disable=invalid-name
def _get_nested_attr(obj: Any, path: str) -> Any:
  """Gets a nested attribute using a dot-separated path."""
  return functools.reduce(getattr, path.split("."), obj)


def _validate_paths(
    obj: Any,
    required_paths: tuple[str, ...],
    warning_paths: tuple[str, ...],
) -> None:
  """Validates required and optional paths on an object."""
  for path in required_paths:
    if not _get_nested_attr(obj, path).has_value:
      raise ValueError(f"The IDS is missing the {path} quantity.")
  for path in warning_paths:
    if not _get_nested_attr(obj, path).has_value:
      logging.warning("The IDS is missing the %s quantity.", path)


def validate_profile_conditions_from_IMAS(
    ids: ids_toplevel.IDSToplevel,
) -> None:
  """Validates core_profiles IDS for profile_conditions."""
  # Assume the initial profile has the same structure as the rest.
  # This is a reasonable assumption on structure and avoids spamming warning
  # messages for many profiles.
  initial_profile = ids.profiles_1d[0]
  _validate_paths(
      initial_profile,
      required_paths=("grid.rho_tor_norm",),
      warning_paths=(
          "grid.psi",
          "time",
          "electrons.temperature",
          "electrons.density",
          "t_i_average",
      ),
  )

  _validate_paths(
      ids,
      required_paths=(),
      warning_paths=(
          "global_quantities.v_loop",
          "global_quantities.ip",
      ),
  )


def validate_plasma_composition_from_IMAS(
    ids: ids_toplevel.IDSToplevel,
) -> None:
  """Validates core_profiles IDS for plasma_composition."""
  # Assume the initial profile has the same structure as the rest.
  # This is a reasonable assumption on structure and avoids spamming warning
  # messages for many profiles.
  initial_profile = ids.profiles_1d[0]
  _validate_paths(
      initial_profile,
      required_paths=(
          "grid.rho_tor_norm",
          "electrons.density",
          "ion",
      ),
      warning_paths=(
          "time",
          "zeff",
      ),
  )
  # TODO(b/459479939): i/539): add name to required paths when updating
  # supported dd_versions.
  for ion in initial_profile.ion:
    _validate_paths(
        ion,
        required_paths=("density",),
        warning_paths=(
            "name",
            "temperature",
        ),
    )


def validate_core_profiles_ions(
    parsed_ions: list[str],
) -> None:
  """Checks if all parsed ions are recognized."""
  for ion in parsed_ions:
    # ion is casted to str to avoid issues with imas string types.
    if str(ion) not in constants.ION_PROPERTIES_DICT.keys():
      raise (
          KeyError(
              f"{ion} is present in the IDS but not a valid TORAX ion. Check"
              "typing or add the ion to the excluded_impurities."
          )
      )


def validate_main_ions_presence(
    parsed_ions: list[str],
    main_ion_symbols: Collection[str],
) -> None:
  """Check if all main ions are present in the parsed ions."""
  for ion in main_ion_symbols:
    if ion not in constants.ION_PROPERTIES_DICT.keys():
      raise (
          KeyError(
              f"{ion} is not a valid symbol of a TORAX valid ion. Please"
              " check typing of main_ion_symbols."
          )
      )
    if ion not in parsed_ions:
      raise (
          ValueError(
              f"The expected main ion {ion} cannot be found in the input"
              " IDS or has no valid data. \n Please check that the IDS is"
              " properly filled"
          )
      )
