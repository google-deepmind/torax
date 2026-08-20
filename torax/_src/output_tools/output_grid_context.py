# Copyright 2026 DeepMind Technologies Limited
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

"""Context providing simulation grids and packaging arrays for fast xr.Dataset creation."""

from collections.abc import Mapping
import enum
from typing import Any, TypeAlias
import chex
import numpy as np
from torax._src import array_typing
from torax._src.output_tools import output_keys
import xarray as xr

# Lightweight tuple format: (dimensions, data array, attributes dictionary)
OutputVar: TypeAlias = tuple[tuple[str, ...], np.ndarray, dict[str, Any]]


class OutputGridContext:
  """Provides time and spatial grids for converting arrays into xr.Dataset data variables."""

  def __init__(
      self,
      times: array_typing.Array,
      rho_face_norm: array_typing.Array,
      rho_cell_norm: array_typing.Array,
      rho_cell_plus_boundaries_norm: array_typing.Array,
  ):
    self._times: np.ndarray = np.asarray(times)
    self._rho_face_norm: np.ndarray = np.asarray(rho_face_norm)
    self._rho_cell_norm: np.ndarray = np.asarray(rho_cell_norm)
    self._rho_cell_plus_boundaries_norm: np.ndarray = np.asarray(
        rho_cell_plus_boundaries_norm
    )
    self.coords: dict[str, np.ndarray] = {
        output_keys.TIME: self._times,
        output_keys.RHO_FACE_NORM: self._rho_face_norm,
        output_keys.RHO_CELL_NORM: self._rho_cell_norm,
        output_keys.RHO_NORM: self._rho_cell_plus_boundaries_norm,
    }

  @property
  def times(self) -> np.ndarray:
    """Returns the time coordinate array."""
    return self._times

  def pack(
      self,
      key: output_keys.OutputKey,
      data: chex.Numeric | enum.Enum,
  ) -> OutputVar:
    """Validates array shape against key.grid_type and returns an OutputVar tuple."""
    if key.grid_type == output_keys.GridType.NOT_APPLICABLE:
      raise ValueError(
          f"OutputKey '{key}' must have a valid spatial/temporal grid_type"
          " specified."
      )

    if isinstance(data, enum.Enum):
      data = data.value

    data_array = np.asarray(data)
    n_time = len(self._times)
    match key.grid_type:
      case output_keys.GridType.SCALAR:
        expected_shape = (n_time,)
        dims = (output_keys.TIME,)
      case output_keys.GridType.FACE:
        expected_shape = (n_time, len(self._rho_face_norm))
        dims = (output_keys.TIME, output_keys.RHO_FACE_NORM)
      case output_keys.GridType.CELL:
        expected_shape = (n_time, len(self._rho_cell_norm))
        dims = (output_keys.TIME, output_keys.RHO_CELL_NORM)
      case output_keys.GridType.CELL_PLUS_BOUNDARIES:
        expected_shape = (n_time, len(self._rho_cell_plus_boundaries_norm))
        dims = (output_keys.TIME, output_keys.RHO_NORM)
      case _:
        raise ValueError(f"Unsupported GridType: {key.grid_type}")

    if data_array.shape != expected_shape:
      raise ValueError(
          f"Shape mismatch for '{key}': expected {expected_shape} for grid"
          f" '{key.grid_type}', but got {data_array.shape}."
      )

    return (dims, data_array, output_keys.get_units(key))

  def build_dataset(
      self,
      data_vars: Mapping[str, OutputVar],
      coords: Mapping[str, Any] | None = None,
  ) -> xr.Dataset:
    """Builds an xr.Dataset from a mapping of variables and coordinates in one fast step."""
    if coords is None:
      coords = self.coords
    return xr.Dataset(data_vars=data_vars, coords=coords)
