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
from typing import Final, Literal, TypeAlias
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import jax_utils
import xarray as xr

RHO_NORM: Final[str] = 'rho_norm'

_interp_fn = jax.jit(jnp.interp)
_interp_fn_vmap = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1)))


@jax.jit
def _step_interpolation(
    xs: array_typing.Array, x: chex.Numeric
) -> array_typing.Array:
  """Find the indices for step interpolation."""
  # For a given x, we want to find k such that self.xs[k] <= x < self.xs[k+1]
  # and return self.ys[k]. Subtracting 1 gives index k. Setting side='left'
  # means that the step occurs whenever x > self.xs. Clipping is strictly
  # necessary for the case where searchsorted returns index 0.
  # TODO(b/454891040): Make the value at the boundary consistent with the
  # prescribed value.
  return jnp.clip(jnp.searchsorted(xs, x, side='left') - 1, 0, xs.shape[0] - 1)


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


InterpolationModeLiteral: TypeAlias = Literal[
    'step', 'STEP', 'piecewise_linear', 'PIECEWISE_LINEAR'
]


_ArrayOrListOfFloats: TypeAlias = array_typing.Array | list[float]

# Config input types convertible to InterpolatedParam objects.
InterpolatedVarSingleAxisInput: TypeAlias = (
    float
    | dict[float, float]
    | bool
    | dict[float, bool]
    | tuple[_ArrayOrListOfFloats, _ArrayOrListOfFloats]
    | xr.DataArray
)
InterpolatedVarTimeRhoInput: TypeAlias = (
    # Mapping from time to rho, value interpolated in rho
    Mapping[float, InterpolatedVarSingleAxisInput]
    | float
    | xr.DataArray
    | tuple[_ArrayOrListOfFloats, _ArrayOrListOfFloats, _ArrayOrListOfFloats]
    | tuple[_ArrayOrListOfFloats, _ArrayOrListOfFloats]
)


# Type-alias for a variable (in rho_norm) to be interpolated in time.
# If a string is provided, it is assumed to be an InterpolationMode else, the
# default piecewise linear interpolation is used.
TimeInterpolatedInput: TypeAlias = (
    InterpolatedVarSingleAxisInput
    | tuple[InterpolatedVarSingleAxisInput, InterpolationModeLiteral]
)
# Type-alias for a variable to be interpolated in time and rho_norm.
TimeRhoInterpolatedInput: TypeAlias = (
    InterpolatedVarTimeRhoInput
    | tuple[
        InterpolatedVarTimeRhoInput,
        Mapping[
            Literal['time_interpolation_mode', 'rho_interpolation_mode'],
            InterpolationModeLiteral,
        ],
    ]
)


class InterpolatedParamBase(abc.ABC):
  """Base class for interpolated params.

  An InterpolatedParamBase child class should implement the interface defined
  below where, given an x-value for where to interpolate, this object
  returns a value. The x-value can be either a time or spatial coordinate
  depending on what we are interpolating over.
  """

  @abc.abstractmethod
  def get_value(self, x: chex.Numeric) -> array_typing.Array:
    """Returns a value for this parameter interpolated at the given input."""


class _PiecewiseLinearInterpolatedParam(InterpolatedParamBase):
  """Parameter using piecewise-linear interpolation to compute its value."""

  def __init__(self, xs: array_typing.Array, ys: array_typing.Array):
    """Initialises a piecewise-linear interpolated param, xs must be sorted."""

    if not np.issubdtype(xs.dtype, np.floating):
      raise ValueError(f'xs must be a float array, but got {xs.dtype}.')
    if not np.issubdtype(ys.dtype, np.floating):
      raise ValueError(f'ys must be a float array, but got {ys.dtype}.')

    self._xs = xs
    self._ys = ys

    jax_utils.assert_rank(self.xs, 1)
    if self.xs.shape[0] != self.ys.shape[0]:
      raise ValueError(
          'xs and ys must have the same number of elements in the first '
          f'dimension. Given: {self.xs.shape} and {self.ys.shape}.'
      )
    if ys.ndim not in (1, 2):
      raise ValueError(f'ys must be either 1D or 2D. Given: {self.ys.shape}.')

  @property
  def xs(self) -> array_typing.Array:
    return self._xs

  @property
  def ys(self) -> array_typing.Array:
    return self._ys

  def get_value(
      self,
      x: chex.Numeric,
  ) -> array_typing.Array:
    x_shape = getattr(x, 'shape', ())
    is_jax = isinstance(x, jax.Array)
    # This function can be used inside a JITted function, where x are
    # tracers. Thus are required to use the JAX versions of functions in this
    # case.
    interp = _interp_fn if is_jax else np.interp
    full = jnp.full if is_jax else np.full

    match self.ys.ndim:
      # This is simply interp, but with fast paths for common special cases.
      case 1:
        # When ys is size 1, no interpolation is needed: all values are just
        # ys.
        if self.ys.size == 1:
          if x_shape == ():  # pylint: disable=g-explicit-bool-comparison
            return self.ys[0]
          else:
            return full(x_shape, self.ys[0], dtype=self.ys.dtype)
        else:
          return interp(x, self.xs, self.ys)
      # The 2D case is mapped across the last dimension.
      case 2:
        # Special case: no interpolation needed.
        if len(self.ys) == 1 and x_shape == ():  # pylint: disable=g-explicit-bool-comparison
          return self.ys[0]
        else:
          return _interp_fn_vmap(x, self.xs, self.ys)
      case _:
        raise ValueError(f'ys must be either 1D or 2D. Given: {self.ys.shape}.')


class _StepInterpolatedParam(InterpolatedParamBase):
  """Parameter using step interpolation to compute its value."""

  def __init__(self, xs: array_typing.Array, ys: array_typing.Array):
    """Creates a step interpolated param, xs must be sorted."""
    self._xs = jnp.asarray(xs)
    self._ys = jnp.asarray(ys)
    jax_utils.assert_rank(self.xs, 1)
    if self.ys.ndim not in (1, 2):
      raise ValueError(f'ys must be either 1D or 2D. Given: {self.ys.shape}.')
    if self.xs.shape[0] != self.ys.shape[0]:
      raise ValueError(
          'xs and ys must have the same number of elements in the first '
          f'dimension. Given: {self.xs.shape} and {self.ys.shape}.'
      )

  @property
  def xs(self) -> array_typing.Array:
    return self._xs

  @property
  def ys(self) -> array_typing.Array:
    return self._ys

  def get_value(self, x: chex.Numeric) -> array_typing.Array:
    """Returns a single value for this range at the given coordinate."""
    indices = _step_interpolation(self.xs, x)
    return self.ys[indices]


def _is_bool(
    interp_input: InterpolatedVarSingleAxisInput,
) -> bool:
  if isinstance(interp_input, dict):
    if not interp_input:
      raise ValueError('InterpolatedVarSingleAxisInput must include values.')
    value = list(interp_input.values())[0]
    return isinstance(value, bool)
  return isinstance(interp_input, bool)


def _convert_value_to_floats(
    interp_input: InterpolatedVarSingleAxisInput,
) -> InterpolatedVarSingleAxisInput:
  if isinstance(interp_input, dict):
    return {key: float(value) for key, value in interp_input.items()}
  return float(interp_input)


def convert_input_to_xs_ys(
    interp_input: TimeInterpolatedInput,
    default_interpolation_mode: InterpolationMode = InterpolationMode.PIECEWISE_LINEAR,
) -> tuple[np.ndarray, np.ndarray, InterpolationMode, bool]:
  """Converts config inputs into inputs suitable for constructors.

  Args:
    interp_input: The input to convert.
    default_interpolation_mode: The default interpolation mode to use if not
      specified in the input.

  Returns:
    A tuple of (xs, ys, interpolation_mode, is_bool_param) where xs and ys are
    the arrays to be used in the constructor, interpolation_mode is the
    interpolation mode to be used, and is_bool_param is True if the input is a
    bool and False otherwise.
  """
  # This function does NOT need to be jittable.
  interpolation_mode = default_interpolation_mode
  # The param is a InterpolatedVarSingleAxisInput, so we need to convert it to
  # an InterpolatedVarSingleAxis first.
  if isinstance(interp_input, tuple):
    if len(interp_input) != 2:
      raise ValueError(
          'Single axis interpolated var tuple length must be 2. The first '
          'element are the values and the second element is the '
          'interpolation mode or both values should be arrays to be directly '
          f'interpolated. Given: {interp_input}.'
      )
    if isinstance(interp_input[1], str):
      interpolation_mode = InterpolationMode[interp_input[1].upper()]
      interp_input = interp_input[0]

  if _is_bool(interp_input):
    interp_input = _convert_value_to_floats(interp_input)
    is_bool_param = True
  else:
    is_bool_param = False

  if isinstance(interp_input, xr.DataArray):
    if not isinstance(interp_input.coords, Mapping):  # pytype: disable=attribute-error
      raise ValueError('The coords in the xr.DataArray must be a mapping.')
    if 'time' not in interp_input.coords:  # pytype: disable=attribute-error
      raise ValueError(
          'The coords in the xr.DataArray must include a "time" coordinate.'
      )
    return (
        np.asarray(interp_input.coords['time'], dtype=jax_utils.get_np_dtype()),  # pytype: disable=attribute-error
        np.asarray(interp_input.values, dtype=jax_utils.get_np_dtype()),  # pytype: disable=attribute-error
        interpolation_mode,
        is_bool_param,
    )
  if isinstance(interp_input, tuple):
    if len(interp_input) != 2:
      raise ValueError(
          'The time interpolated input tuple must be length 2. Given: '
          f'{interp_input}.'
      )
    xs, ys = interp_input
    xs = np.asarray(xs, dtype=jax_utils.get_np_dtype())
    ys = np.asarray(ys, dtype=jax_utils.get_np_dtype())
    return xs, ys, interpolation_mode, is_bool_param
  if isinstance(interp_input, dict):
    if not interp_input:
      raise ValueError('The time interpolated input dict must be non-empty.')
    return (
        np.array(list(interp_input.keys()), dtype=jax_utils.get_np_dtype()),  # pytype: disable=attribute-error
        np.array(list(interp_input.values()), dtype=jax_utils.get_np_dtype()),  # pytype: disable=attribute-error
        interpolation_mode,
        is_bool_param,
    )
  else:
    # The input is a single value.
    return (
        np.array([0.0], dtype=jax_utils.get_np_dtype()),
        np.array([interp_input], dtype=jax_utils.get_np_dtype()),
        interpolation_mode,
        is_bool_param,
    )


@jax.tree_util.register_pytree_node_class
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
      value: tuple[array_typing.Array, array_typing.Array],
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

    Raises:
      RuntimeError: If the input xs is not sorted.
    """
    self._value = value
    xs, ys = value
    jax_utils.error_if(xs, jnp.any(jnp.diff(xs) < 0), 'xs must be sorted.')

    if not np.issubdtype(xs.dtype, np.floating):
      raise ValueError(f'xs must be a float array, but got {xs.dtype}.')
    if not np.issubdtype(ys.dtype, np.floating):
      raise ValueError(f'ys must be a float array, but got {ys.dtype}.')

    self._is_bool_param = is_bool_param
    self._interpolation_mode = interpolation_mode
    match interpolation_mode:
      case InterpolationMode.PIECEWISE_LINEAR:
        self._param = _PiecewiseLinearInterpolatedParam(xs=xs, ys=ys)
      case InterpolationMode.STEP:
        self._param = _StepInterpolatedParam(xs=xs, ys=ys)
      case _:
        raise ValueError('Unknown interpolation mode.')

  def tree_flatten(self):
    static_params = {
        'interpolation_mode': self.interpolation_mode,
        'is_bool_param': self.is_bool_param,
    }
    return (self._value, static_params)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(children, **aux_data)

  @property
  def is_bool_param(self) -> bool:
    """Returns whether this param represents a bool."""
    return self._is_bool_param

  @property
  def interpolation_mode(self) -> InterpolationMode:
    """Returns the interpolation mode used by this param."""
    return self._interpolation_mode

  def get_value(
      self,
      x: chex.Numeric,
  ) -> array_typing.Array:
    """Returns a single value for this range at the given coordinate."""
    value = self._param.get_value(x)
    if self._is_bool_param:
      return jnp.bool_(value > 0.5)
    return value

  @property
  def param(self) -> InterpolatedParamBase:
    """Returns the JAX-friendly interpolated param used under the hood."""
    return self._param

  def __eq__(self, other: 'InterpolatedVarSingleAxis') -> bool:
    try:
      chex.assert_trees_all_equal(self, other)
    except AssertionError:
      return False
    return True


@jax.tree_util.register_pytree_node_class
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
      values: Mapping[float, tuple[array_typing.Array, array_typing.Array]],
      rho_norm: array_typing.Array,
      time_interpolation_mode: InterpolationMode = (
          InterpolationMode.PIECEWISE_LINEAR
      ),
      rho_interpolation_mode: InterpolationMode = (
          InterpolationMode.PIECEWISE_LINEAR
      ),
  ):
    """Constructs an `InterpolatedVarTimeRho`.

    Args:
      values: Mapping of times to (rho_norm, values) arrays of equal length.
      rho_norm: The grid to interpolate onto.
      time_interpolation_mode: The mode in which to do time interpolation.
      rho_interpolation_mode: The mode in which to do rho interpolation.
    """
    self._rho_interpolation_mode = rho_interpolation_mode
    self._time_interpolation_mode = time_interpolation_mode

    sorted_indices = np.array(sorted(values.keys()))
    rho_norm_interpolated_values = np.stack(
        [
            InterpolatedVarSingleAxis(
                values[t], rho_interpolation_mode
            ).get_value(rho_norm)
            for t in sorted_indices
        ],
        axis=0,
    )
    self._time_interpolated_var = InterpolatedVarSingleAxis(
        value=(sorted_indices, rho_norm_interpolated_values),
        interpolation_mode=time_interpolation_mode,
    )

  def tree_flatten(self):
    children = (self._time_interpolated_var,)
    aux_data = (self._rho_interpolation_mode, self._time_interpolation_mode)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    # Here we construct the object without calling the constructor.
    # This is because the constructor is not jittable.
    obj = object.__new__(InterpolatedVarTimeRho)
    obj._time_interpolated_var = children[0]
    obj._rho_interpolation_mode = aux_data[0]
    obj._time_interpolation_mode = aux_data[1]
    return obj

  @property
  def time_interpolation_mode(self) -> InterpolationMode:
    """Returns the time interpolation mode used by this param."""
    return self._time_interpolation_mode

  @property
  def rho_interpolation_mode(self) -> InterpolationMode:
    """Returns the rho interpolation mode used by this param."""
    return self._rho_interpolation_mode

  def get_value(self, x: chex.Numeric) -> array_typing.Array:
    """Returns the value of this parameter interpolated at x=time."""
    return self._time_interpolated_var.get_value(x)
