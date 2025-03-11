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

from collections.abc import Mapping
import functools
from typing import Any, Literal
import chex
import numpy as np
import pydantic
from torax import interpolated_param
from torax.torax_pydantic import model_base
from torax.torax_pydantic import pydantic_types
from typing_extensions import Self
import xarray as xr


class Grid1D(model_base.BaseModelFrozen):
  """Data structure defining a 1-D grid of cells with faces.

  Construct via `construct` classmethod.

  Attributes:
    nx: Number of cells.
    dx: Distance between cell centers.
    face_centers: Coordinates of face centers.
    cell_centers: Coordinates of cell centers.
  """

  nx: pydantic.PositiveInt
  dx: pydantic.PositiveFloat
  face_centers: pydantic_types.NumpyArray1D
  cell_centers: pydantic_types.NumpyArray1D

  def __eq__(self, other: Self) -> bool:
    return (
        self.nx == other.nx
        and self.dx == other.dx
        and np.array_equal(self.face_centers, other.face_centers)
        and np.array_equal(self.cell_centers, other.cell_centers)
    )

  def __hash__(self) -> int:
    return hash((self.nx, self.dx))

  @classmethod
  def construct(cls, nx: int, dx: float) -> Self:
    """Constructs a Grid1D.

    Args:
      nx: Number of cells.
      dx: Distance between cell centers.

    Returns:
      grid: A Grid1D with the remaining fields filled in.
    """

    # Note: nx needs to be an int so that the shape `nx + 1` is not a Jax
    # tracer.

    return Grid1D(
        nx=nx,
        dx=dx,
        face_centers=np.linspace(0, nx * dx, nx + 1),
        cell_centers=np.linspace(dx * 0.5, (nx - 0.5) * dx, nx),
    )


class TimeVaryingArray(model_base.BaseModelFrozen):
  """Base class for time interpolated array types.

  The Pydantic `.model_validate` constructor can accept a variety of input types
  defined by the `TimeRhoInterpolatedInput` type. See
  https://torax.readthedocs.io/en/latest/configuration.html#time-varying-arrays
  for more details.

  Attributes:
    value: A mapping of the form `{time: (rho_norm, values), ...}`, where
      `rho_norm` and `values` are 1D NumPy arrays of equal length.
    rho_interpolation_mode: The interpolation mode to use for the rho axis.
    time_interpolation_mode: The interpolation mode to use for the time axis.
    grid_face_centers: The face centers of the grid to use for the
      interpolation. This is optional, as this value is often not known at
      construction time, and is set later.
    grid_cell_centers: The cell centers of the grid to use for the
      interpolation. This is optional, as this value is often not known at
      construction time, and is set later.
  """

  value: Mapping[
      float, tuple[pydantic_types.NumpyArray1D, pydantic_types.NumpyArray1D]
  ]
  rho_interpolation_mode: interpolated_param.InterpolationMode = (
      interpolated_param.InterpolationMode.PIECEWISE_LINEAR
  )
  time_interpolation_mode: interpolated_param.InterpolationMode = (
      interpolated_param.InterpolationMode.PIECEWISE_LINEAR
  )
  grid: Grid1D | None = None

  @functools.cached_property
  def right_boundary_conditions_defined(self) -> bool:
    """Checks if the boundary condition at rho=1.0 is always defined."""

    for rho_norm, _ in self.value.values():
      if 1.0 not in rho_norm:
        return False
    return True

  def get_value(
      self,
      t: chex.Numeric,
      grid_type: Literal['cell', 'face', 'face_right'] = 'cell',
  ) -> chex.Array:
    """Returns the value of this parameter interpolated at x=time.

    Args:
      t: An array of times to interpolate at.
      grid_type: One of 'cell', 'face', or 'face_right'. For 'face_right', the
        element `self.grid_face_centers[-1]` is used as the grid.

    Raises:
      RuntimeError: If `self.grid` is None.

    Returns:
      An array of interpolated values.
    """
    match grid_type:
      case 'cell':
        return self._get_cached_interpolated_param_cell.get_value(t)
      case 'face':
        return self._get_cached_interpolated_param_face.get_value(t)
      case 'face_right':
        return self._get_cached_interpolated_param_face_right.get_value(t)
      case _:
        raise ValueError(f'Unknown grid type: {grid_type}')

  def __eq__(self, other: Self):
    try:
      chex.assert_trees_all_equal(vars(self), vars(other))
      return True
    except AssertionError:
      return False

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(
      cls, data: interpolated_param.TimeRhoInterpolatedInput | dict[str, Any]
  ) -> dict[str, Any]:

    if isinstance(data, dict):
      # A workaround for https://github.com/pydantic/pydantic/issues/10477.
      data.pop('_get_cached_interpolated_param_cell_centers', None)
      data.pop('_get_cached_interpolated_param_face_centers', None)
      data.pop('_get_cached_interpolated_param_face_right_centers', None)

      # This is the standard constructor input. No conforming required.
      if set(data.keys()).issubset(cls.model_fields.keys()):
        return data

    # Potentially parse the interpolation modes from the input.
    time_interpolation_mode = (
        interpolated_param.InterpolationMode.PIECEWISE_LINEAR
    )
    rho_interpolation_mode = (
        interpolated_param.InterpolationMode.PIECEWISE_LINEAR
    )

    if isinstance(data, tuple):
      if len(data) == 2 and isinstance(data[1], dict):
        time_interpolation_mode = interpolated_param.InterpolationMode[
            data[1]['time_interpolation_mode'].upper()
        ]
        rho_interpolation_mode = interpolated_param.InterpolationMode[
            data[1]['rho_interpolation_mode'].upper()
        ]
        # First element in tuple assumed to be the input.
        data = data[0]

    if isinstance(data, xr.DataArray):
      value = _load_from_xr_array(data)
    elif isinstance(data, tuple) and all(
        isinstance(v, chex.Array) for v in data
    ):
      value = _load_from_arrays(
          data,
      )
    elif isinstance(data, Mapping) or isinstance(data, (float, int)):
      value = _load_from_primitives(data)
    else:
      raise ValueError('Input to TimeVaryingArray unsupported.')

    return dict(
        value=value,
        time_interpolation_mode=time_interpolation_mode,
        rho_interpolation_mode=rho_interpolation_mode,
    )

  @functools.cached_property
  def _get_cached_interpolated_param_cell(
      self,
  ) -> interpolated_param.InterpolatedVarTimeRho:
    if self.grid is None:
      raise RuntimeError('grid must be set.')

    return interpolated_param.InterpolatedVarTimeRho(
        self.value,
        rho_norm=self.grid.cell_centers,
        time_interpolation_mode=self.time_interpolation_mode,
        rho_interpolation_mode=self.rho_interpolation_mode,
    )

  @functools.cached_property
  def _get_cached_interpolated_param_face(
      self,
  ) -> interpolated_param.InterpolatedVarTimeRho:
    if self.grid is None:
      raise RuntimeError('grid must be set.')

    return interpolated_param.InterpolatedVarTimeRho(
        self.value,
        rho_norm=self.grid.face_centers,
        time_interpolation_mode=self.time_interpolation_mode,
        rho_interpolation_mode=self.rho_interpolation_mode,
    )

  @functools.cached_property
  def _get_cached_interpolated_param_face_right(
      self,
  ) -> interpolated_param.InterpolatedVarTimeRho:
    if self.grid is None:
      raise RuntimeError('grid must be set.')

    return interpolated_param.InterpolatedVarTimeRho(
        self.value,
        rho_norm=self.grid.face_centers[-1],
        time_interpolation_mode=self.time_interpolation_mode,
        rho_interpolation_mode=self.rho_interpolation_mode,
    )


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
  3. A nested dict is passed in, it will describe a time-dependent radial
  profile providing both initial condition and prescribed values at times beyond

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

  loaded_values = {}
  for t, v in primitive_values.items():
    x, y, _, _ = interpolated_param.convert_input_to_xs_ys(v)
    loaded_values[t] = (x, y)

  return loaded_values


def _load_from_xr_array(
    xr_array: xr.DataArray,
) -> Mapping[float, tuple[chex.Array, chex.Array]]:
  """Loads the data from an xr.DataArray."""
  if 'time' not in xr_array.coords:
    raise ValueError('"time" must be a coordinate in given dataset.')
  if interpolated_param.RHO_NORM not in xr_array.coords:
    raise ValueError(
        f'"{interpolated_param.RHO_NORM}" must be a coordinate in given'
        ' dataset.'
    )
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


def set_grid(
    model: model_base.BaseModelFrozen,
    grid: Grid1D,
    mode: Literal['strict', 'force', 'relaxed'] = 'strict',
):
  """Sets the grid for for the model and all its submodels.

  Args:
    model: The model to set the geometry mesh for.
    grid: A `Grid1D` object.
    mode: The update mode. With `'strict'` (default), an error will be raised if
      the `grid` is already set in `model` or any of its submodels. With
      `'force'`, `grid` will be overwritten in a potentially unsafe way (no
      cache invalidation). With `'relaxed'`, `grid` will only be set if it is
      not already set.

  Raises:
    RuntimeError: If `force_update=False` and `grid` is already set.
  """

  def _update_rule(submodel):
    if submodel.grid is None:
      submodel.__dict__['grid'] = grid
    else:
      match mode:
        case 'force':
          submodel.__dict__['grid'] = grid
        case 'relaxed':
          pass
        case 'strict':
          raise RuntimeError(
              '`grid` is already set and using strict update mode.'
          )

  for submodel in model.submodels:
    if isinstance(submodel, TimeVaryingArray):
      _update_rule(submodel)
