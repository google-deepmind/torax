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


class InterpolatedParamBase(abc.ABC):
  """Base class for interpolated params.

  An InterpolatedParamBase child class should implement the interface defined
  below where, given an x-coordinate for where to interpolate, this object
  returns a value.
  """

  @abc.abstractmethod
  def get_value(
      self,
      x: chex.Numeric,
  ) -> jax.Array:
    """Returns a single value for this parameter at the given coordinate."""


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


@chex.dataclass(frozen=True)
class JaxFriendlyInterpolatedParam(InterpolatedParamBase):
  """Base class for JAX-friendly interpolated params.

  Any InterpolatedVar1d implementation that is used within a jitted function or
  used as an argument to a jitted function should inherit from this class.
  """


@chex.dataclass(frozen=True)
class PiecewiseLinearInterpolatedParam(JaxFriendlyInterpolatedParam):
  """Parameter using piecewise-linear interpolation to compute its value."""

  xs: jax.Array  # must be sorted.
  ys: jax.Array

  def __post_init__(self):
    jax_utils.assert_rank(self.xs, 1)
    assert self.xs.shape == self.ys.shape
    diff = jnp.sum(jnp.abs(jnp.sort(self.xs) - self.xs))
    jax_utils.error_if(diff, diff > 1e-8, 'xs must be sorted.')

  def get_value(
      self,
      x: chex.Numeric,
  ) -> jax.Array:
    return jnp.interp(x, self.xs, self.ys)


@chex.dataclass(frozen=True)
class StepInterpolatedParam(JaxFriendlyInterpolatedParam):
  """Parameter using step interpolation to compute its value."""

  xs: jax.Array  # must be sorted.
  ys: jax.Array

  def __post_init__(self):
    jax_utils.assert_rank(self.xs, 1)
    assert self.xs.shape == self.ys.shape
    diff = jnp.sum(jnp.abs(jnp.sort(self.xs) - self.xs))
    jax_utils.error_if(diff, diff > 1e-8, 'xs must be sorted.')
    # Precompute some arrays useful for computing values.
    # Must use object.__setattr__ here because this is frozen dataclass.
    object.__setattr__(
        self,
        '_padded_xs',
        jnp.concatenate([jnp.array([-jnp.inf]), self.xs, jnp.array([jnp.inf])]),
    )
    object.__setattr__(
        self,
        '_padded_ys',
        jnp.concatenate(
            [jnp.array([self.ys[0]]), self.ys, jnp.array([self.ys[-1]])]
        ),
    )

  def get_value(
      self,
      x: chex.Numeric,
  ) -> jax.Array:
    # pytype: disable=attribute-error
    idx = jnp.max(jnp.argwhere(self._padded_xs < x).flatten())
    return self._padded_ys[idx]
    # pytype: enable=attribute-error


# Config input types convertible to InterpolatedParam objects.
InterpolatedVar1dInput = (
    float
    | dict[float, float]
    | bool
    | dict[float, bool]
    | tuple[chex.Array, chex.Array]
)
InterpolatedVar2dInput = (
    # Mapping from time to rho, value interpolated in rho
    Mapping[float, InterpolatedVar1dInput]
    | float
)


def _convert_input_to_xs_ys(
    interp_input: InterpolatedVar1dInput,
) -> tuple[chex.Array, chex.Array]:
  """Converts config inputs into inputs suitable for constructors."""
  # This function does NOT need to be jittable.
  if isinstance(interp_input, tuple):
    if len(interp_input) != 2:
      raise ValueError(
          'InterpolatedVar1dInput tuple must be length 2. Given: '
          f'{interp_input}.'
      )
    xs, ys = interp_input
    sort_order = np.argsort(xs)
    xs = xs[sort_order]
    ys = ys[sort_order]
    return np.asarray(xs), np.asarray(ys)
  if isinstance(interp_input, dict):
    if not interp_input:
      raise ValueError('InterpolatedVar1dInput must include values.')
    sorted_keys = sorted(interp_input.keys())
    values = [interp_input[key] for key in sorted_keys]
    return jnp.array(sorted_keys), jnp.array(values)
  else:
    # The input is a single value.
    return jnp.array([0]), jnp.array([interp_input])


def _is_bool(interp_input: InterpolatedVar1dInput) -> bool:
  if isinstance(interp_input, dict):
    if not interp_input:
      raise ValueError('InterpolatedVar1dInput must include values.')
    value = list(interp_input.values())[0]
    return isinstance(value, bool)
  return isinstance(interp_input, bool)


def _convert_value_to_floats(
    interp_input: InterpolatedVar1dInput,
) -> InterpolatedVar1dInput:
  if isinstance(interp_input, dict):
    return {key: float(value) for key, value in interp_input.items()}
  return float(interp_input)


class InterpolatedVar1d(InterpolatedParamBase):
  """Parameter that may vary based on an input coordinate.

  This class is useful for defining time-dependent runtime parameters, but can
  be used to define any parameters that vary across some range. This class is
  the main "user-facing" class defined in this module.

  See `config.runtime_params.RuntimeParams` and associated tests to see how this
  is used.
  """

  def __init__(
      self,
      value: InterpolatedVar1dInput,
      interpolation_mode: InterpolationMode = (
          InterpolationMode.PIECEWISE_LINEAR
      ),
  ):
    """Initializes InterpolatedVar1d.

    Args:
      value: A single float or a dictionary mapping input coordinates (e.g.
        time) to desired values at those coordinates. If only a single value is
        given, then the output value will be constant. If no values are given in
        the dict, then an error is raised.
      interpolation_mode: Defines how to interpolate between values in `value`.
    """
    self._is_bool_param = _is_bool(value)
    if self._is_bool_param:
      value = _convert_value_to_floats(value)
    xs, ys = _convert_input_to_xs_ys(value)
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
  ) -> jax.Array:
    """Returns a single value for this range at the given coordinate."""
    value = self._param.get_value(x)
    if self._is_bool_param:
      return jnp.bool_(value > 0.5)
    return value

  @property
  def param(self) -> JaxFriendlyInterpolatedParam:
    """Returns the JAX-friendly interpolated param used under the hood."""
    return self._param

  @property
  def is_bool_param(self) -> bool:
    """Returns whether this param represents a bool."""
    return self._is_bool_param


class InterpolatedVar2d:
  """Interpolates on a grid (time, rho).

  - Given `values` that map from time-values to `InterpolatedVar1d`s that tell
  you how to interpolate along rho for different time values this class linearly
  interpolates along time to provide a value at any (time, rho) pair.
  - For time values that are outside the range of `values` the closest defined
  `InterpolatedVar1d` is used.
  """

  def __init__(
      self,
      values: InterpolatedVar2dInput,
      rho_interpolation_mode: InterpolationMode = (
          InterpolationMode.PIECEWISE_LINEAR
      ),
  ):
    # If a float is passed in, will describe constant initial condition profile.
    if isinstance(values, float):

      values = {0.0: {0.0: values}}
    # If a non-nested dict is passed in, it will describe the radial profile for
    # the initial condition."
    if isinstance(values, Mapping) and all(
        isinstance(v, float) for v in values.values()
    ):
      values = {0.0: values}
    self.values = values
    if len(set(values.keys())) != len(values):
      raise ValueError('Indicies in values mapping must be unique.')
    if not values:
      raise ValueError('Values mapping must not be empty.')
    self.times_values = {
        v: InterpolatedVar1d(values[v], rho_interpolation_mode)
        for v in values.keys()
    }
    self.sorted_indices = jnp.array(sorted(values.keys()))

  def get_value(
      self,
      time: chex.Numeric,
      rho: chex.Numeric,
  ) -> jax.Array:
    """Returns the value of this parameter interpolated at the given (time,rho).

    This method is not jittable as it is.

    Args:
      time: The time-coordinate to interpolate at.
      rho: The rho-coordinate to interpolate at.
    Returns:
      The value of the interpolated at the given (time,rho).
    """
    # Find the index that is left of value which time is closest to.
    left = jnp.searchsorted(self.sorted_indices, time, side='left')

    # If time is either smaller or larger, than smallest and largest values
    # we know how to interpolate for, use the boundary interpolater.
    if left == 0:
      return self.times_values[float(self.sorted_indices[0])].get_value(rho)
    if left == len(self.sorted_indices):
      return self.times_values[float(self.sorted_indices[-1])].get_value(rho)

    # Interpolate between the two closest defined interpolaters.
    left_time = float(self.sorted_indices[left - 1])
    right_time = float(self.sorted_indices[left])
    return self.times_values[left_time].get_value(rho) * (right_time - time) / (
        right_time - left_time
    ) + self.times_values[right_time].get_value(rho) * (time - left_time) / (
        right_time - left_time
    )


# In runtime_params, users should be able to either specify the
# InterpolatedVar1d/InterpolatedVar2d object directly or the values that go in
# the constructor. This helps with brevity since a lot of these params are fixed
# floats.
# Type-alias for a scalar variable (in rho_norm) to be interpolated in time.
# If a string is provided, it is assumed to be an InterpolationMode else, the
# default piecewise linear interpolation is used.
TimeInterpolatedScalar = (
    InterpolatedVar1d
    | InterpolatedVar1dInput
    | tuple[InterpolatedVar1dInput, str]
)

# Type-alias for a 1D variable (in rho_norm) to be interpolated in time.
TimeInterpolatedArray = InterpolatedVar2d | InterpolatedVar2dInput
