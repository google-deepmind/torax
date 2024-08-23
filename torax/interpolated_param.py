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

import abc
from collections.abc import Mapping
import enum

import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax import jax_utils
import xarray as xr


class InterpolatedParamBase(abc.ABC):
  """Base class for interpolated params.

  An InterpolatedParamBase child class should implement the interface defined
  below where, given an x-value for where to interpolate, this object
  returns a value. The x-value can be either a time or spatial coordinate
  depending on what we are interpolating over.
  """

  @abc.abstractmethod
  def get_value(self, x: chex.Numeric) -> chex.Array:
    """Returns a value for this parameter interpolated at the given input."""


@enum.unique
class InterpolationMode(enum.Enum):
  """Defines how to do the interpolation.

  InterpolatedParams have many values to interpolate between, and this enum
  defines how exactly that interpolation is computed.

  Assuming inputs [x_0, ..., x_n] and [y_0, ..., y_n], for all modes, the
  interpolated param outputs y_0 for any input less than x_0 and y_n for any
  input greater than x_n.

  Options:
    PIECEWISE_LINEAR: Does piecewise-linear interpolation between the values
      provided. See numpy.interp for a longer description of how it works. (This
      uses JAX, but the behavior is the same.)
    STEP: Step-function interpolation. For any input value x in the range [x_k,
      x_k+1), the output will be y_k.
  """

  PIECEWISE_LINEAR = 'piecewise_linear'
  STEP = 'step'


class PiecewiseLinearInterpolatedParam(InterpolatedParamBase):
  """Parameter using piecewise-linear interpolation to compute its value."""

  def __init__(self, xs: chex.Array, ys: chex.Array):
    """Initialises a piecewise-linear interpolated param, xs must be sorted."""
    self._xs = xs
    self._ys = ys
    jax_utils.assert_rank(self.xs, 1)
    if self.xs.shape[0] != self.ys.shape[0]:
      raise ValueError(
          'xs and ys must have the same number of elements in the first '
          f'dimension. Given: {self.xs.shape} and {self.ys.shape}.'
      )
    diff = jnp.sum(jnp.abs(jnp.sort(self.xs) - self.xs))
    jax_utils.error_if(diff, diff > 1e-8, 'xs must be sorted.')
    if self.ys.ndim == 1:
      self._fn = jnp.interp
    elif self.ys.ndim == 2:
      self._fn = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1)))
    else:
      raise ValueError(
          f'ys must be either 1D or 2D. Given: {self.ys.shape}.'
      )

  @property
  def xs(self) -> chex.Array:
    return self._xs

  @property
  def ys(self) -> chex.Array:
    return self._ys

  def get_value(
      self,
      x: chex.Numeric,
  ) -> chex.Array:
    return self._fn(x, self.xs, self.ys)


class StepInterpolatedParam(InterpolatedParamBase):
  """Parameter using step interpolation to compute its value."""

  def __init__(self, xs: chex.Array, ys: chex.Array):
    """Creates a step interpolated param, xs must be sorted."""
    self._xs = xs
    self._ys = ys
    jax_utils.assert_rank(self.xs, 1)
    if len(self.ys.shape) != 1 and len(self.ys.shape) != 2:
      raise ValueError(
          f'ys must be either 1D or 2D. Given: {self.ys.shape}.'
      )
    if self.xs.shape[0] != self.ys.shape[0]:
      raise ValueError(
          'xs and ys must have the same number of elements in the first '
          f'dimension. Given: {self.xs.shape} and {self.ys.shape}.'
      )
    diff = jnp.sum(jnp.abs(jnp.sort(self.xs) - self.xs))
    jax_utils.error_if(diff, diff > 1e-8, 'xs must be sorted.')
    self._padded_xs = jnp.concatenate(
        [jnp.array([-jnp.inf]), self.xs, jnp.array([jnp.inf])]
    )
    self._padded_ys = jnp.concatenate(
        [jnp.array([self.ys[0]]), self.ys, jnp.array([self.ys[-1]])]
    )

  @property
  def xs(self) -> chex.Array:
    return self._xs

  @property
  def ys(self) -> chex.Array:
    return self._ys

  def get_value(
      self,
      x: chex.Numeric,
  ) -> chex.Array:
    idx = jnp.max(jnp.argwhere(self._padded_xs < x).flatten())
    return self._padded_ys[idx]


# Config input types convertible to InterpolatedParam objects.
InterpolatedVarSingleAxisInput = (
    float
    | dict[float, float]
    | bool
    | dict[float, bool]
    | tuple[chex.Array, chex.Array]
    | xr.DataArray
)
InterpolatedVarTimeRhoInput = (
    # Mapping from time to rho, value interpolated in rho
    Mapping[float, InterpolatedVarSingleAxisInput]
    | float
    | xr.DataArray
    | tuple[chex.Array, chex.Array, chex.Array]
    | tuple[chex.Array, chex.Array]
)


def convert_input_to_xs_ys(
    interp_input: InterpolatedVarSingleAxisInput,
) -> tuple[chex.Array, chex.Array]:
  """Converts config inputs into inputs suitable for constructors."""
  # This function does NOT need to be jittable.
  if isinstance(interp_input, xr.DataArray):
    if len(interp_input.coords) != 1:
      raise ValueError(
          f'Loaded values must have 1 coordinate. Given: {interp_input.coords}'
      )
    if len(interp_input.values.shape) != 1:
      raise ValueError(
          f'Loaded values for {interp_input.name} must be 1D. Given:'
          f' {interp_input.values.shape}.'
      )
    index = list(interp_input.coords)[0]
    return (
        interp_input[index].data,
        interp_input.values,
    )
  if isinstance(interp_input, tuple):
    if len(interp_input) != 2:
      raise ValueError(
          'InterpolatedVarSingleAxisInput tuple must be length 2. Given: '
          f'{interp_input}.'
      )
    xs, ys = interp_input
    sort_order = np.argsort(xs)
    xs = xs[sort_order]
    ys = ys[sort_order]
    return np.asarray(xs), np.asarray(ys)
  if isinstance(interp_input, dict):
    if not interp_input:
      raise ValueError('InterpolatedVarSingleAxisInput must include values.')
    sorted_keys = sorted(interp_input.keys())
    values = [interp_input[key] for key in sorted_keys]
    return np.array(sorted_keys), np.array(values)
  else:
    # The input is a single value.
    return np.array([0]), np.array([interp_input])


def _is_bool(interp_input: InterpolatedVarSingleAxisInput) -> bool:
  if isinstance(interp_input, dict):
    if not interp_input:
      raise ValueError('InterpolatedVarSingleAxisInput must include values.')
    value = list(interp_input.values())[0]
    return isinstance(value, bool)
  return isinstance(interp_input, bool)


def convert_value_to_floats(
    interp_input: InterpolatedVarSingleAxisInput,
) -> InterpolatedVarSingleAxisInput:
  if isinstance(interp_input, dict):
    return {key: float(value) for key, value in interp_input.items()}
  return float(interp_input)


class InterpolatedVarSingleAxis(InterpolatedParamBase):
  """Parameter that may vary based on an input coordinate.

  This class is useful for defining time-dependent runtime parameters, but can
  be used to define any parameters that vary across some range.

  This function allows the interpolation of a 1d array xs, against either a 1d
  or 2d array ys. For example, xs can be time, and ys either a 1d array of
  scalars associated to the times in xs, or a 2d array where the index 0 in ys
  associates a radial array in the index 1 with the times in xs. The
  interpolation of the 2d array is then carried out element-wise and accelerated
  with vmap. Intended use of ys being a 2d array is when the radial slices on
  index 1 have already been interpolated onto appropriate TORAX grids, such as
  cell_centers, faces, or the hires grid. NOTE: this means that the 2d array
  should have shape (n, m) where n is the number of elements in the 1d array and
  m is the number of spatial grid size of the InterpolatedVar1d instance

  See `config.runtime_params.RuntimeParams` and associated tests to see how
  this is used.
  """

  def __init__(
      self,
      value: tuple[chex.Array, chex.Array],
      interpolation_mode: InterpolationMode = (
          InterpolationMode.PIECEWISE_LINEAR
      ),
      is_bool_param: bool = False,
  ):
    """Initializes InterpolatedVarSingleAxis.

    Args:
      value: A tuple of `(xs, ys)` where `xs` is assumed to be a 1D array and
        `ys` can either be a 1D or 2D array with ys.shape[0] = len(xs).
        Additionally it is expected that `xs` is sorted and an error will be
        raised at runtime if this is not the case.
      interpolation_mode: Defines how to interpolate between values in `value`.
      is_bool_param: If True, the input value is assumed to be a bool and is
        converted to a float.
    """
    xs, ys = value
    self._is_bool_param = is_bool_param
    match interpolation_mode:
      case InterpolationMode.PIECEWISE_LINEAR:
        self._param = PiecewiseLinearInterpolatedParam(xs=xs, ys=ys)
      case InterpolationMode.STEP:
        self._param = StepInterpolatedParam(xs=xs, ys=ys)
      case _:
        raise ValueError('Unknown interpolation mode.')

  def get_value(
      self,
      x: chex.Numeric,
  ) -> chex.Array:
    """Returns a single value for this range at the given coordinate."""
    value = self._param.get_value(x)
    if self._is_bool_param:
      return jnp.bool_(value > 0.5)
    return value

  @property
  def param(self) -> InterpolatedParamBase:
    """Returns the JAX-friendly interpolated param used under the hood."""
    return self._param

  @property
  def is_bool_param(self) -> bool:
    """Returns whether this param represents a bool."""
    return self._is_bool_param


class InterpolatedVarTimeRho(InterpolatedParamBase):
  """Interpolates on a grid (time, rho).

  This class linearly interpolates along time to provide a value at any
  (time, rho) pair. For time values that are outside the range of `values` the
  closest defined `InterpolatedVarSingleAxis` is used.

  - NOTE: We assume that rho interpolation is fixed per simulation so take this
  at init and take just time at get_value.
  """

  def __init__(
      self,
      values: Mapping[float, tuple[chex.Array, chex.Array]],
      rho_norm: chex.Array,
      rho_interpolation_mode: InterpolationMode = (
          InterpolationMode.PIECEWISE_LINEAR
      ),
  ):
    """Constructs an `InterpolatedVarTimeRho`.

    Args:
      values: Mapping of times to (rho_norm, values) arrays of equal length.
      rho_norm: The grid to interpolate onto.
      rho_interpolation_mode: The mode in which to do rho interpolation.
    """
    self.rho_norm = rho_norm
    self.sorted_indices = np.array(sorted(values.keys()))
    rho_norm_interpolated_values = np.stack(
        [
            InterpolatedVarSingleAxis(
                values[t], rho_interpolation_mode
            ).get_value(self.rho_norm)
            for t in self.sorted_indices
        ],
        axis=0,
    )
    self._time_interpolated_var = InterpolatedVarSingleAxis(
        (self.sorted_indices, rho_norm_interpolated_values)
    )

  def get_value(self, x: chex.Numeric) -> chex.Array:
    """Returns the value of this parameter interpolated at x=time."""
    return self._time_interpolated_var.get_value(x)


# Type-alias for a variable (in rho_norm) to be interpolated in time.
# If a string is provided, it is assumed to be an InterpolationMode else, the
# default piecewise linear interpolation is used.
TimeInterpolated = (
    InterpolatedVarSingleAxisInput
    | tuple[InterpolatedVarSingleAxisInput, str]
)
