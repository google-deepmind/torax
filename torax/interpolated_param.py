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
import enum
import chex
import jax.numpy as jnp
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
  ) -> jnp.ndarray:
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

  Any InterpolatedParam implementation that is used within a jitted function or
  used as an argument to a jitted function should inherit from this class.
  """


@chex.dataclass(frozen=True)
class PiecewiseLinearInterpolatedParam(JaxFriendlyInterpolatedParam):
  """Parameter using piecewise-linear interpolation to compute its value."""

  xs: jnp.ndarray  # must be sorted.
  ys: jnp.ndarray

  def __post_init__(self):
    jax_utils.assert_rank(self.xs, 1)
    assert self.xs.shape == self.ys.shape
    diff = jnp.sum(jnp.abs(jnp.sort(self.xs) - self.xs))
    jax_utils.error_if(diff, diff > 1e-8, 'xs must be sorted.')

  def get_value(
      self,
      x: chex.Numeric,
  ) -> jnp.ndarray:
    return jnp.interp(x, self.xs, self.ys)


@chex.dataclass(frozen=True)
class StepInterpolatedParam(JaxFriendlyInterpolatedParam):
  """Parameter using step interpolation to compute its value."""

  xs: jnp.ndarray  # must be sorted.
  ys: jnp.ndarray

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
  ) -> jnp.ndarray:
    # pytype: disable=attribute-error
    idx = jnp.max(jnp.argwhere(self._padded_xs < x).flatten())
    return self._padded_ys[idx]
    # pytype: enable=attribute-error


# Config input types convertible to InterpolatedParam objects.
InterpolatedParamInput = float | dict[float, float] | bool | dict[float, bool]


def _convert_input_to_xs_ys(
    interp_input: InterpolatedParamInput,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Converts config inputs into inputs suitable for constructors."""
  # This function does NOT need to be jittable.
  if isinstance(interp_input, dict):
    if not interp_input:
      raise ValueError('InterpolatedParamInput must include values.')
    sorted_keys = sorted(interp_input.keys())
    values = [interp_input[key] for key in sorted_keys]
    return jnp.array(sorted_keys), jnp.array(values)
  else:
    # The input is a single value.
    return jnp.array([0]), jnp.array([interp_input])


def _is_bool(interp_input: InterpolatedParamInput) -> bool:
  if isinstance(interp_input, dict):
    if not interp_input:
      raise ValueError('InterpolatedParamInput must include values.')
    value = list(interp_input.values())[0]
    return isinstance(value, bool)
  return isinstance(interp_input, bool)


def _convert_value_to_floats(
    interp_input: InterpolatedParamInput,
) -> InterpolatedParamInput:
  if isinstance(interp_input, dict):
    return {key: float(value) for key, value in interp_input.items()}
  return float(interp_input)


class InterpolatedParam(InterpolatedParamBase):
  """Parameter that may vary based on an input coordinate.

  This class is useful for defining time-dependent config parameters, but can
  be used to define any parameters that vary across some range. This class is
  the main "user-facing" class defined in this module.

  See `config.Config` and associated tests to see how this is used.
  """

  def __init__(
      self,
      value: InterpolatedParamInput,
      interpolation_mode: InterpolationMode = (
          InterpolationMode.PIECEWISE_LINEAR
      ),
  ):
    """Initializes InterpolatedParam.

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
  ) -> jnp.ndarray:
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


# In Config, users should be able to either specify the InterpolatedParam object
# directly or the values that go in the constructor. This helps with brevity
# since a lot of these params are fixed floats.
InterpParamOrInterpParamInput = InterpolatedParam | InterpolatedParamInput
