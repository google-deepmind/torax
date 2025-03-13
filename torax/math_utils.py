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

"""Math operations.

Math operations that are needed for Torax, but are not specific to plasma
physics or differential equation solvers.
"""

import enum
import functools

import chex
import jax
from jax import numpy as jnp
import jaxtyping as jt
from torax import array_typing
from torax import jax_utils
from torax.geometry import geometry


@enum.unique
class IntegralPreservationQuantity(enum.Enum):
  """The quantity to preserve the integral of when converting to face values."""
  # Indicate that the volume integral should be preserved.
  VOLUME = 'volume'
  # Indicate that the surface integral should be preserved.
  SURFACE = 'surface'
  # Indicate that the value integral should be preserved.
  VALUE = 'value'


@array_typing.typed
@functools.partial(
    jax_utils.jit,
    static_argnames=['preserved_quantity',],
)
def cell_to_face(
    cell_values: jt.Float[chex.Array, 'rhon'],
    geo: geometry.Geometry,
    preserved_quantity: IntegralPreservationQuantity = IntegralPreservationQuantity.VALUE,
) -> jt.Float[chex.Array, 'rhon+1']:
  """Convert cell values to face values.

  We make four assumptions:
  1) Inner face values are the average of neighbouring cells.
  2) The left most face value is linearly extrapolated from the left most cell
  values.
  3) The transformation from cell to face is integration preserving.
  4) The cell spacing is constant.

  Args:
    cell_values: Values defined on the TORAX cell grid.
    geo: A geometry object.
    preserved_quantity: The quantity to preserve the integral of when converting
      to face values.

  Returns:
    Values defined on the TORAX face grid.
  """
  if len(cell_values) < 2:
    raise ValueError(
        'Cell values must have at least two values to convert to face values.'
    )
  inner_face_values = (cell_values[:-1] + cell_values[1:]) / 2.0
  # Linearly extrapolate to get left value.
  left = cell_values[0] - (inner_face_values[0] - cell_values[0])
  face_values_without_right = jnp.concatenate([left[None], inner_face_values])
  # Preserve integral.
  match preserved_quantity:
    case IntegralPreservationQuantity.VOLUME:
      diff = jnp.sum(
          cell_values * geo.vpr
      ) * geo.drho_norm - jax.scipy.integrate.trapezoid(
          face_values_without_right * geo.vpr_face[:-1], geo.rho_face_norm[:-1]
      )
      right = (
          2 * diff / geo.drho_norm
          - face_values_without_right[-1] * geo.vpr_face[-2]
      ) / geo.vpr_face[-1]
    case IntegralPreservationQuantity.SURFACE:
      diff = jnp.sum(
          cell_values * geo.spr
      ) * geo.drho_norm - jax.scipy.integrate.trapezoid(
          face_values_without_right * geo.spr_face[:-1], geo.rho_face_norm[:-1]
      )
      right = (
          2 * diff / geo.drho_norm
          - face_values_without_right[-1] * geo.spr_face[-2]
      ) / geo.spr_face[-1]
    case IntegralPreservationQuantity.VALUE:
      diff = jnp.sum(
          cell_values
      ) * geo.drho_norm - jax.scipy.integrate.trapezoid(
          face_values_without_right, geo.rho_face_norm[:-1]
      )
      right = 2 * diff / geo.drho_norm - face_values_without_right[-1]

  face_values = jnp.concatenate([face_values_without_right, right[None]])
  return face_values


def tridiag(
    diag: jax.Array, above: jax.Array, below: jax.Array
) -> jax.Array:
  """Builds a tridiagonal matrix.

  Args:
    diag: The main diagonal.
    above: The +1 diagonal.
    below: The -1 diagonal.

  Returns:
    The tridiagonal matrix.
  """

  return jnp.diag(diag) + jnp.diag(above, 1) + jnp.diag(below, -1)


def cumulative_trapezoid(
    y: jax.Array,
    x: jax.Array | None = None,
    dx: float = 1.0,
    axis: int = -1,
    initial: float | None = None,
) -> jax.Array:
  """Cumulatively integrate y = f(x) using the trapezoid rule.

  JAX equivalent of scipy.integrate.cumulative_trapezoid.

  Args:
    y: array of data to integrate.
    x: optional array of sample points corresponding to the `y` values. If not
      provided, `x` defaults to equally spaced with spacing given by `dx`.
    dx: the spacing between sample points when `x` is None (default: 1.0).
    axis: the axis along which to integrate (default: -1)
    initial: a scalar value to prepend to the result. Either None (default) or
      0.0. If `initial=0`, the result is an array with the same shape as `y`. If
      ``initial=None``, the resulting array has one fewer elements than `y`
      along the `axis` dimension.

  Returns:
    The cumulative definite integral approximated by the trapezoidal rule.
  """

  if x is None:
    dx = jnp.asarray(dx, dtype=y.dtype)
  else:
    if x.ndim == 1:
      if y.shape[axis] != len(x):
        raise ValueError(
            f'The length of x is {len(x)}, but expected {y.shape[axis]}.'
        )
    else:
      if x.shape != y.shape:
        raise ValueError(
            'If x is not 1 dimensional, it must have the same shape as y.'
        )

    if x.ndim == 1:
      dx = jnp.diff(x)
      new_shape = [1] * y.ndim
      new_shape[axis] = len(dx)
      dx = jnp.reshape(dx, new_shape)
    else:
      dx = jnp.diff(x, axis=axis)

  y_sliced = functools.partial(jax.lax.slice_in_dim, y, axis=axis)

  out = jnp.cumsum(dx * (y_sliced(1, None) + y_sliced(0, -1)), axis=axis) / 2.0

  if initial is not None:
    if initial != 0.0:
      raise ValueError(
          '`initial` must be 0 or None. Non-zero values have been deprecated'
          ' since SciPy version 1.12.0.'
      )
    initial_array = jnp.asarray(initial, dtype=out.dtype)
    initial_shape = list(out.shape)
    initial_shape[axis] = 1
    initial_array = jnp.broadcast_to(initial_array, initial_shape)
    out = jnp.concatenate((initial_array, out), axis=axis)
  return out


@array_typing.typed
@jax_utils.jit
def cell_integration(
    x: array_typing.ArrayFloat, geo: geometry.Geometry
) -> array_typing.ScalarFloat:
  r"""Integrate a value `x` over the rhon grid.

  Cell variables in TORAX are defined as the average of the face values. This
  method integrates that face value over the rhon grid implicitly using the
  trapezium rule to sum the averaged face values by the face grid spacing.

  Args:
    x: The cell averaged value to integrate.
    geo: The geometry instance.

  Returns:
    Integration over the rhon grid: :math:`\int_0^1 x_{face} d\hat{rho}`
  """
  if x.shape != geo.rho_norm.shape:
    raise ValueError(
        f"For cell_integration, input 'x' must have same shape as the cell grid"
        f"Got x.shape={x.shape}, expected {geo.rho_norm.shape}."
    )
  return jnp.sum(x * geo.drho_norm)


def area_integration(
    value: array_typing.ArrayFloat,
    geo: geometry.Geometry,
) -> array_typing.ScalarFloat:
  """Calculates integral of value using an area metric."""
  return cell_integration(value * geo.spr, geo)


def volume_integration(
    value: array_typing.ArrayFloat,
    geo: geometry.Geometry,
) -> array_typing.ScalarFloat:
  """Calculates integral of value using a volume metric."""
  return cell_integration(value * geo.vpr, geo)


def line_average(
    value: array_typing.ArrayFloat,
    geo: geometry.Geometry,
) -> array_typing.ScalarFloat:
  """Calculates line-averaged value from input profile."""
  return cell_integration(value, geo)


def volume_average(
    value: array_typing.ArrayFloat,
    geo: geometry.Geometry,
) -> array_typing.ScalarFloat:
  """Calculates volume-averaged value from input profile."""
  return cell_integration(value * geo.vpr, geo)/geo.volume_face[-1]
