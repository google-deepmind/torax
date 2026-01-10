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
from typing import Any

from imas import ids_toplevel


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


def validate_core_profiles_ids(ids: ids_toplevel.IDSToplevel) -> None:
  """Validates core_profiles IDS for profiles_conditions."""
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
