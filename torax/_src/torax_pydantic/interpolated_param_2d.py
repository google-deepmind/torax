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
import dataclasses
import functools
from typing import Any, Literal, TypeAlias

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import interpolated_param
from torax._src import jax_utils
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import pydantic_types
import typing_extensions
import xarray as xr


ValueType: TypeAlias = dict[
    float,
    tuple[pydantic_types.NumpyArray1DUnitInterval, pydantic_types.NumpyArray1D],
]


class Grid1D(model_base.BaseModelFrozen):
  """Data structure defining a 1-D grid of cells with faces.

  The grid is defined by the face_centers array, which specifies the locations
  of all faces (including boundary faces). For a grid with N cells, there are
  N+1 faces.
  """
  face_centers: pydantic_types.NumpyArray1DSorted

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: Any) -> Any:
    if not isinstance(data, dict):
      return data
    if 'nx' in data:
      nx = data['nx']
      face_centers = get_face_centers(nx)
      data['face_centers'] = face_centers
      del data['nx']
    return data

  @pydantic.field_validator('face_centers')
  @classmethod
  def _validate_face_centers(cls, v: np.ndarray) -> np.ndarray:
    """Validates that face_centers has at least 5 elements (4 cells)."""
    if len(v) < 5:
      raise ValueError(
          'face_centers must have at least 5 elements (4 cells) but got'
          f' {len(v)}'
      )
    if not np.isclose(v[0], 0.0) or not np.isclose(v[-1], 1.0):
      raise ValueError(f'face_centers must include 0.0 and 1.0 but got {v}')
    return v

  @property
  def nx(self) -> int:
    """Number of cells in the grid."""
    return len(self.face_centers) - 1

  @property
  def cell_centers(self) -> np.ndarray:
    """Coordinates of cell centers (midpoints between faces)."""
    return (self.face_centers[1:] + self.face_centers[:-1]) / 2.0

  @functools.cached_property
  def cell_widths(self) -> jax.Array:
    """Widths of cells."""
    return jnp.diff(self.face_centers)

  def __eq__(self, other: typing_extensions.Self) -> bool:
    """Custom equality to handle numpy array comparison."""
    if not isinstance(other, Grid1D):
      return False
    return np.array_equal(self.face_centers, other.face_centers)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TimeVaryingArrayUpdate:
  """Replacements for TimeVaryingArray."""

  value: jt.Float[jax.Array, 't rhon'] | None = None
  rho_norm: jt.Float[jax.Array, 'rhon'] | None = None
  time: jt.Float[jax.Array, 't'] | None = None

  def __post_init__(self):
    """Consistency checks for the provided values."""
    if (self.rho_norm is None and self.value is not None) or (
        self.rho_norm is not None and self.value is None
    ):
      raise ValueError(
          'Either both or neither of rho_norm and value must be provided.'
      )

    if self.rho_norm is not None and self.value is not None:
      rho_norm_shape = self.rho_norm.shape
      if rho_norm_shape[0] != self.value.shape[1]:
        raise ValueError(
            'rho_norm and value must have the same trailing dimension. '
            f'Got rho_norm shape: {rho_norm_shape} and value shape: '
            f'{self.value.shape}'
        )

    if self.value is not None and self.time is not None:
      if self.value.shape[0] != self.time.shape[0]:
        raise ValueError(
            'value and time arrays must have same leading dimension.'
            f'Got time: {self.time.shape}, value: {self.value.shape}'
        )


_vmap_interp = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 0)))


class TimeVaryingArray(model_base.BaseModelFrozen):
  """Base class for time interpolated array types.

  The Pydantic `.model_validate` constructor can accept a variety of input types
  defined by the `TimeRhoInterpolatedInput` type. See
  https://torax.readthedocs.io/en/latest/configuration.html#time-varying-arrays
  for more details.

  Attributes:
    value: A dict of the form `{time: (rho_norm, values), ...}`, where
      `rho_norm` and `values` are 1D NumPy arrays, and each pair is of equal
      length. Note that all `rho_norm` values `x` are in the range `0 <= x <=
      1.0`, and the mapping is ordered by increasing `time`.
    rho_interpolation_mode: The interpolation mode to use for the rho axis.
    time_interpolation_mode: The interpolation mode to use for the time axis.
    grid: The grid to use for the interpolation. This is optional, as this value
      is often not known at construction time, and is set later. This grid is
      required to call `get_value`.
  """

  value: ValueType
  rho_interpolation_mode: typing_extensions.Annotated[
      interpolated_param.InterpolationMode, model_base.JAX_STATIC
  ] = interpolated_param.InterpolationMode.PIECEWISE_LINEAR
  time_interpolation_mode: typing_extensions.Annotated[
      interpolated_param.InterpolationMode, model_base.JAX_STATIC
  ] = interpolated_param.InterpolationMode.PIECEWISE_LINEAR
  grid: Grid1D | None = None

  def tree_flatten(self):
    children = (
        self.value,
        # Save out the cached interpolated params.
        self.get_cached_interpolated_param_cell,
        self.get_cached_interpolated_param_face,
        self.get_cached_interpolated_param_face_right,
    )
    aux_data = (
        self.rho_interpolation_mode,
        self.time_interpolation_mode,
        self.grid,
    )
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    # Avoid calling model_validate as validation should be done already.
    obj = cls.model_construct(
        value=children[0],
        rho_interpolation_mode=aux_data[0],
        time_interpolation_mode=aux_data[1],
        grid=aux_data[2],
    )
    # Plug back in the cached interpolated params to avoid losing the cache.
    # pylint: disable=protected-access
    obj.get_cached_interpolated_param_cell = children[1]
    obj.get_cached_interpolated_param_face = children[2]
    obj.get_cached_interpolated_param_face_right = children[3]
    # pylint: enable=protected-access
    return obj

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
  ) -> array_typing.Array:
    """Returns the value of this parameter interpolated at x=time.

    Args:
      t: An array of times to interpolate at.
      grid_type: One of 'cell', 'face', or 'face_right'. For 'face_right', the
        element `self.grid.face_centers[-1]` is used as the grid.

    Raises:
      RuntimeError: If `self.grid` is None.

    Returns:
      An array of interpolated values.
    """
    match grid_type:
      case 'cell':
        return self.get_cached_interpolated_param_cell.get_value(t)
      case 'face':
        return self.get_cached_interpolated_param_face.get_value(t)
      case 'face_right':
        return self.get_cached_interpolated_param_face_right.get_value(t)
      case _:
        raise ValueError(f'Unknown grid type: {grid_type}')

  def update(
      self, replace_value: TimeVaryingArrayUpdate
  ) -> typing_extensions.Self:
    """This method can be used under `jax.jit`."""
    assert self.grid is not None, 'grid must be set to update.'

    time = (
        replace_value.time
        if replace_value.time is not None
        else self.get_cached_interpolated_param_cell.xs
    )
    if replace_value.rho_norm is not None and replace_value.value is not None:
      # NOTE: This assumes that the provided value covers the entire grid.
      # Constant interpolation is used outside of the provided range.
      cell_value = _vmap_interp(
          self.grid.cell_centers, replace_value.rho_norm, replace_value.value
      )
      face_value = _vmap_interp(
          self.grid.face_centers, replace_value.rho_norm, replace_value.value
      )
      face_right_value = _vmap_interp(
          self.grid.face_centers[-1],
          replace_value.rho_norm,
          replace_value.value,
      )
    else:
      cell_value = self.get_cached_interpolated_param_cell.ys
      face_value = self.get_cached_interpolated_param_face.ys
      face_right_value = self.get_cached_interpolated_param_face_right.ys

    # All of these should have a leading `time` dimension.
    chex.assert_tree_shape_prefix(
        (time, cell_value, face_value, face_right_value), time.shape
    )

    def get_leaves(
        x: typing_extensions.Self,
    ) -> tuple[
        chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array
    ]:
      # We need to update the time (xs) and value (ys) arrays for all
      # cached interpolated params.
      return (
          x.get_cached_interpolated_param_cell.xs,
          x.get_cached_interpolated_param_cell.ys,
          x.get_cached_interpolated_param_face.xs,
          x.get_cached_interpolated_param_face.ys,
          x.get_cached_interpolated_param_face_right.xs,
          x.get_cached_interpolated_param_face_right.ys,
      )

    return eqx.tree_at(
        get_leaves,
        self,
        (time, cell_value, time, face_value, time, face_right_value),
    )

  def __eq__(self, other: typing_extensions.Self):
    try:
      chex.assert_trees_all_equal(self.value, other.value)
      return (
          self.rho_interpolation_mode == other.rho_interpolation_mode
          and self.time_interpolation_mode == other.time_interpolation_mode
          and self.grid == other.grid
      )
    except AssertionError:
      return False

  @pydantic.field_validator('value', mode='after')
  @classmethod
  def _valid_value(cls, value: ValueType) -> ValueType:
    # Ensure the keys are sorted.
    value = dict(sorted(value.items()))

    for t, (rho_norm, values) in value.items():
      if not isinstance(t, float):
        raise ValueError(f'Time values must be a float, but got {t}.')

      if not np.issubdtype(rho_norm.dtype, np.floating):
        raise ValueError(
            f'rho_norm must be a float array, but got {rho_norm.dtype}.'
        )
      if rho_norm.dtype != values.dtype:
        raise ValueError(
            'rho_norm and values must have the same dtype. Given: '
            f'{rho_norm.dtype} and {values.dtype}.'
        )
      if len(rho_norm) != len(values):
        raise ValueError(
            'rho_norm and values must be of the same length. Given: '
            f'{len(rho_norm)} and {len(values)}.'
        )
      if np.any(rho_norm < 0.0) or np.any(rho_norm > 1.0):
        raise ValueError(
            f'rho_norm values must be in the range [0, 1], but got {rho_norm}.'
        )

    return value

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
      value = _load_from_arrays(data)
    elif isinstance(data, tuple):
      values = []
      for v in data:
        if isinstance(v, array_typing.Array):
          values.append(v)
        elif isinstance(v, list):
          values.append(np.asarray(v))
        else:
          raise ValueError(
              'Input to TimeVaryingArray unsupported. Input was of type:'
              f' {type(v)}. Expected array_typing.Array or list of'
              ' floats/ints/bools.'
          )
      value = _load_from_arrays(tuple(values))
    elif isinstance(data, Mapping) or isinstance(data, (float, int)):
      value = _load_from_primitives(data)
    else:
      raise ValueError(
          'Input to TimeVaryingArray unsupported. Input was of type:'
          f' {type(data)}'
      )

    return dict(
        value=value,
        time_interpolation_mode=time_interpolation_mode,
        rho_interpolation_mode=rho_interpolation_mode,
    )

  @functools.cached_property
  def get_cached_interpolated_param_cell(
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
  def get_cached_interpolated_param_face(
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
  def get_cached_interpolated_param_face_right(
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


class SparseTimeVaryingArray(model_base.BaseModelFrozen):
  """A class for specifying profiles that are zero except at particular locations.

  This class conforms data from the format: {time: {point_or_range: value}}.
  It evaluates to a full profile on the grid, but is zero except at the
  specified points or ranges.

  Note that internally, this is stored as a list of tuples
  (point_or_range, TimeVaryingArray). This is for ease of interpolation and
  to avoid JAX sorting errors with mixed dict keys.

  Attributes:
    values: A list of tuples mapping a point (float) or range (tuple of floats)
      to a TimeVaryingArray.
    grid: The grid to use for evaluating the profile.
  """

  values: list[tuple[float | tuple[float, float], TimeVaryingArray]]
  grid: Grid1D | None = None

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: Any) -> Any:
    # If a single scalar value is passed, we assume it is a constant profile for
    # all times and locations.
    if isinstance(data, (float, int)):
      return {'values': [((0.0, 1.0), TimeVaryingArray.model_validate(data))]}

    # If the data is not a dict, we assume it is already in the final format.
    if not isinstance(data, dict):
      return data

    # Support creating from {'values': ..., 'grid': } dict, which is used in
    # Pydantic serialization and deserialization
    if set(data.keys()).issubset(cls.model_fields.keys()):
      return data

    # The standard input format is {time: {location: value}}.
    # To convert to [(location, TimeVaryingArray), ...], we first group all
    # time-dependent values for a specific spatial location together.
    grouped_by_location: dict[float | tuple[float, float], dict[float, Any]] = (
        {}
    )
    for t, vals in data.items():
      # Check type.
      if not isinstance(vals, dict):
        raise ValueError(
            f'Expected dict for values at time {t}, got {type(vals)}'
        )
      keys = list(vals.keys())
      # Check for overlapping locations. We check every pair of locations.
      for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
          location1 = keys[i]
          location2 = keys[j]
          # Convert points to ranges.
          range1 = (
              location1
              if isinstance(location1, tuple)
              else (location1, location1)
          )
          range2 = (
              location2
              if isinstance(location2, tuple)
              else (location2, location2)
          )
          if max(range1[0], range2[0]) <= min(range1[1], range2[1]):
            raise ValueError(
                f'Overlapping locations {location1} and {location2} at time'
                f' {t}.'
            )
      # Group values by location.
      for key, val in vals.items():
        if key not in grouped_by_location:
          grouped_by_location[key] = {}
        grouped_by_location[key][t] = val

    # Convert the grouped values to the final format.
    list_of_tuples = []
    for key, time_mapping in grouped_by_location.items():
      list_of_tuples.append(
          (key, TimeVaryingArray.model_validate(time_mapping))
      )

    return {'values': list_of_tuples}

  def get_value(
      self,
      t: chex.Numeric,
      grid_type: Literal['cell', 'face', 'face_right'] = 'cell',
  ) -> array_typing.Array:
    """Returns the evaluated profile on the grid at time t."""
    if self.grid is None:
      raise RuntimeError('grid must be set.')

    match grid_type:
      case 'cell':
        grid_pts = self.grid.cell_centers
      case 'face':
        grid_pts = self.grid.face_centers
      case 'face_right':
        grid_pts = self.grid.face_centers[-1:]  # Make it 1D
      case _:
        raise ValueError(f'Unknown grid type: {grid_type}')

    # Populate specified entries, leaving others at zero
    full_profile = jnp.zeros_like(grid_pts)
    for key, val in self.values:
      # Evaluate the TimeVaryingArray at time t
      evaluated_val = val.get_value(t, grid_type=grid_type)

      # Handle point or range
      if isinstance(key, float):  # Point
        idx = jnp.argmin(jnp.abs(grid_pts - key))
        mask = jnp.arange(grid_pts.shape[0]) == idx
      elif isinstance(key, tuple):  # Range
        mask = (grid_pts >= key[0]) & (grid_pts <= key[1])
      else:
        raise ValueError(f'Unknown key type: {type(key)}')

      full_profile = jnp.where(mask, evaluated_val, full_profile)

    return full_profile

  def tree_flatten(self):
    children = tuple(val for _, val in self.values)
    aux_data = (tuple(key for key, _ in self.values), self.grid)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    keys, grid = aux_data
    values = list(zip(keys, children))
    return cls.model_construct(
        values=values,
        grid=grid,
    )


def _is_positive(array: TimeVaryingArray) -> TimeVaryingArray:
  for _, value in array.value.values():
    if not np.all(value > 0):
      raise ValueError('All values must be positive.')
  return array


PositiveTimeVaryingArray = typing_extensions.Annotated[
    TimeVaryingArray, pydantic.AfterValidator(_is_positive)
]


def _load_from_primitives(
    primitive_values: (
        Mapping[float, interpolated_param.InterpolatedVarSingleAxisInput]
        | float
    ),
) -> Mapping[float, tuple[array_typing.Array, array_typing.Array]]:
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


def _load_from_arrays(
    arrays: tuple[array_typing.Array, ...] | xr.DataArray,
) -> Mapping[float, tuple[np.ndarray, np.ndarray]]:
  """Loads the data from numpy arrays.

  Args:
    arrays: A tuple of (times, rho_norm, values) or (rho_norm, values), or an
      xarray.DataArray.

  Returns:
    A mapping from time to (rho_norm, values)
  """

  if isinstance(arrays, xr.DataArray):
    if interpolated_param.RHO_NORM not in arrays.coords:
      raise ValueError(
          f'"{interpolated_param.RHO_NORM}" must be a coordinate in given'
          ' dataset.'
      )
    if 'time' in arrays.coords:
      arrays = (arrays.time.data, arrays.rho_norm.data, arrays.data)
    else:
      arrays = (arrays.rho_norm.data, arrays.data)

  if len(arrays) == 2:
    # Shortcut for initial condition profile.
    rho_norm, values = arrays  # pytype: disable=bad-unpacking
    return {
        0.0: (
            np.asarray(rho_norm, dtype=jax_utils.get_np_dtype()),
            np.asarray(values, dtype=jax_utils.get_np_dtype()),
        )
    }
  if len(arrays) == 3:
    times = np.asarray(arrays[0], dtype=jax_utils.get_np_dtype())
    rho_norm = np.asarray(arrays[1], dtype=jax_utils.get_np_dtype())
    values = np.asarray(arrays[2], dtype=jax_utils.get_np_dtype())

    if values.ndim != 2:
      raise ValueError(
          f'The values array must have ndim=2, but got {values.ndim}.'
      )

    if rho_norm.ndim == 1:
      rho_norm = np.stack([rho_norm] * len(times))

    return {t: (rho_norm[i], values[i]) for i, t in enumerate(times)}
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
    # The update API assumes all submodels are unique objects. Construct
    # a new Grid1D object (without validation) to ensure this. We do reuse
    # the same NumPy arrays.
    new_grid = Grid1D.model_construct(face_centers=grid.face_centers)
    if submodel.grid is None:
      submodel.__dict__['grid'] = new_grid
    else:
      match mode:
        case 'force':
          submodel.__dict__['grid'] = new_grid
        case 'relaxed':
          pass
        case 'strict':
          raise RuntimeError(
              '`grid` is already set and using strict update mode.'
          )

  for submodel in model.submodels:
    if isinstance(submodel, (TimeVaryingArray, SparseTimeVaryingArray)):
      _update_rule(submodel)


def _is_non_negative(
    time_varying_array: TimeVaryingArray,
) -> TimeVaryingArray:
  for _, value in time_varying_array.value.values():
    if not np.all(value >= 0.0):
      raise ValueError('All values must be non-negative.')
  return time_varying_array


@functools.cache
def get_face_centers(nx: int, dx: float | None = None) -> np.ndarray:
  if dx is None:
    dx = 1.0 / nx
  return np.linspace(0, nx * dx, nx + 1)


NonNegativeTimeVaryingArray: TypeAlias = typing_extensions.Annotated[
    TimeVaryingArray, pydantic.AfterValidator(_is_non_negative)
]
