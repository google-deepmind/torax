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
from torax._src import array_typing
from torax._src import constants
from torax._src.geometry import geometry


@enum.unique
class IntegralPreservationQuantity(enum.Enum):
  """The quantity to preserve the integral of when converting to face values."""

  # Indicate that the volume integral should be preserved.
  VOLUME = 'volume'
  # Indicate that the surface integral should be preserved.
  SURFACE = 'surface'
  # Indicate that the value integral should be preserved.
  VALUE = 'value'


def inner_face_values_from_cell_values(
    *,
    cell_values: chex.Array,
    face_centers: chex.Array,
    cell_centers: chex.Array,
) -> chex.Array:
  """Interpolate inner face values from cell values."""
  face_pts = face_centers[1:-1]
  left_cells = cell_centers[:-1]
  right_cells = cell_centers[1:]

  # Linearly interpolate within cell centers as faces aren't uniformly spaced.
  weights = (face_pts - left_cells) / (right_cells - left_cells)
  inner = (1.0 - weights) * cell_values[:-1] + weights * cell_values[1:]
  return inner


@array_typing.jaxtyped
def cell_to_face(
    cell_values: array_typing.FloatVectorCell,
    geo: geometry.Geometry,
    preserved_quantity: IntegralPreservationQuantity = IntegralPreservationQuantity.VALUE,
) -> array_typing.FloatVectorFace:
  """Convert cell values to face values.

  We make three assumptions:
  1) Inner face values are the interpolation of neighboring cell values.
  2) The left most face value is linearly extrapolated from the left most cell
    values.
  3) The transformation from cell to face is integration preserving and the
    quantity to preserve the integral of is specified by `preserved_quantity`.

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
  inner_face_values = inner_face_values_from_cell_values(
      cell_values=cell_values,
      face_centers=geo.rho_face_norm,
      cell_centers=geo.rho_norm,
  )
  # Linearly extrapolate to get left value.
  left = cell_values[0] - (inner_face_values[0] - cell_values[0])
  face_values_without_right = jnp.concatenate([left[None], inner_face_values])
  # Use the last cell width for the rightmost face calculation
  last_drho = geo.drho_norm[-1]
  # Preserve integral.
  match preserved_quantity:
    case IntegralPreservationQuantity.VOLUME:
      diff = jnp.sum(
          cell_values * geo.vpr * geo.drho_norm
      ) - jax.scipy.integrate.trapezoid(
          face_values_without_right * geo.vpr_face[:-1], geo.rho_face_norm[:-1]
      )
      right = (
          2 * diff / last_drho
          - face_values_without_right[-1] * geo.vpr_face[-2]
      ) / geo.vpr_face[-1]
    case IntegralPreservationQuantity.SURFACE:
      diff = jnp.sum(
          cell_values * geo.spr * geo.drho_norm
      ) - jax.scipy.integrate.trapezoid(
          face_values_without_right * geo.spr_face[:-1], geo.rho_face_norm[:-1]
      )
      right = (
          2 * diff / last_drho
          - face_values_without_right[-1] * geo.spr_face[-2]
      ) / geo.spr_face[-1]
    case IntegralPreservationQuantity.VALUE:
      diff = jnp.sum(
          cell_values * geo.drho_norm
      ) - jax.scipy.integrate.trapezoid(
          face_values_without_right, geo.rho_face_norm[:-1]
      )
      right = 2 * diff / last_drho - face_values_without_right[-1]

  face_values = jnp.concatenate([face_values_without_right, right[None]])
  return face_values


@array_typing.jaxtyped
def tridiag(
    diag: jt.Shaped[array_typing.Array, 'size'],
    above: jt.Shaped[array_typing.Array, 'size-1'],
    below: jt.Shaped[array_typing.Array, 'size-1'],
) -> jt.Shaped[array_typing.Array, 'size size']:
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


@array_typing.jaxtyped
def cell_integration(
    x: array_typing.FloatVectorCell, geo: geometry.Geometry
) -> array_typing.FloatScalar:
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
        'For cell_integration, input "x" must have same shape as the cell grid'
        f'Got x.shape={x.shape}, expected {geo.rho_norm.shape}.'
    )
  return jnp.sum(x * geo.drho_norm)


@array_typing.jaxtyped
def area_integration(
    value: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates integral of value using an area metric."""
  return cell_integration(value * geo.spr, geo)


@array_typing.jaxtyped
def volume_integration(
    value: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates integral of value using a volume metric."""
  return cell_integration(value * geo.vpr, geo)


@array_typing.jaxtyped
def line_average(
    value: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates line-averaged value from input profile."""
  return cell_integration(value, geo)


@array_typing.jaxtyped
def volume_average(
    value: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates volume-averaged value from input profile."""
  return cell_integration(value * geo.vpr, geo) / geo.volume_face[-1]


@array_typing.jaxtyped
def cumulative_cell_integration(
    x: array_typing.FloatVectorCell, geo: geometry.Geometry
) -> array_typing.FloatVectorCell:
  r"""Cumulative integration of a value `x` over the rhon grid.

  Args:
    x: The cell averaged value to integrate.
    geo: The geometry instance.

  Returns:
    Cumulative integration array same size as x.
  """
  if x.shape != geo.rho_norm.shape:
    raise ValueError(
        'For cumulative_cell_integration, input "x" must have same shape as '
        f'the cell grid. Got x.shape={x.shape}, '
        f'expected {geo.rho_norm.shape}.'
    )
  # Uses cumsum to accumulate x * drho_norm.
  # The first element will be x[0] * drho_norm[0].
  return jnp.cumsum(x * geo.drho_norm)


@array_typing.jaxtyped
def cumulative_area_integration(
    value: array_typing.FloatVectorCell,
    geo: geometry.Geometry,
) -> array_typing.FloatVectorCell:
  """Calculates cumulative integral of value using an area metric."""
  return cumulative_cell_integration(value * geo.spr, geo)


@array_typing.jaxtyped
def cumulative_volume_integration(
    value: array_typing.FloatVectorCell,
    geo: geometry.Geometry,
) -> array_typing.FloatVectorCell:
  """Calculates cumulative integral of value using a volume metric."""
  return cumulative_cell_integration(value * geo.vpr, geo)


def safe_divide(y: chex.Array, x: chex.Array) -> chex.Array:
  return y / (x + constants.CONSTANTS.eps)


def inverse_softplus(x: jax.Array) -> jax.Array:
  """Inverse of softplus function."""
  # Enforce minimum value to avoid log(0) or log(negative).
  # We want a function that maps x back to y such that softplus(y) = x.
  # y = log(exp(x) - 1).
  # If x -> 0, y -> -inf.
  # For avoiding overflow/underflow issues with float32:
  # exp(x) overflows if x > 88.
  # But for x > 30, softplus(x) ~ x.
  # For x < 1e-32, exp(x) = 1 and we get log(0). Avoid by clipping.
  return jnp.where(x > 30.0, x, jnp.log(jnp.expm1(jnp.maximum(x, 1e-20))))


def smooth_sqrt(
    x: jax.Array, epsilon: float = constants.CONSTANTS.eps
) -> jax.Array:
  """Smoothed sqrt that linearly extrapolates for x < epsilon.

  This function avoids vanishing gradients for x <= 0, which can happen with
  simple clipping. This function returns a linear extrapolation for x < epsilon,
  connecting smoothly (C1 continuity) to sqrt(x) at x = epsilon.

  For x >= epsilon: sqrt(x)
  For x < epsilon: 2 * eps^1.5 / (3 * eps - x)
  This matches value and derivative at x = epsilon.

  Decays as 1/|x| for x -> -infinity, avoiding vanishing gradients.
  Derivative always positive, avoiding zero-slope trap.

  Args:
    x: Input array.
    epsilon: Threshold below which linear extrapolation is used.

  Returns:
    Approximation of sqrt(x) that is linear for x < epsilon.
  """

  sqrt_eps = jnp.sqrt(epsilon)
  # Guard against negative x in sqrt, even when not selected, to avoid NaN
  # gradients. We clamp to epsilon/2 to ensuring the argument is strictly
  # positive (avoiding sqrt(0) singularity) and that at x=epsilon, the
  # derivative is continuous.
  safe_sqrt_x = jnp.sqrt(jnp.maximum(x, epsilon / 2.0))

  # Rational tail for x < epsilon.
  # Note: 3*eps - x is always positive for x < epsilon (since eps > 0).
  rational_approx = 2 * epsilon * sqrt_eps / (3 * epsilon - x)

  return jnp.where(x >= epsilon, safe_sqrt_x, rational_approx)


def smoothstep_transition(
    x: jax.Array,
    smoothing_start: float,
    smoothing_end: float,
    y_left: jax.Array,
    y_right: jax.Array,
    log_scale: bool = True,
) -> jax.Array:
  """Sigmoid-like transition between two models using smoothstep.

  For x < smoothing_start, returns y_left.
  For x > smoothing_end, returns y_right.
  For smoothing_start < x < smoothing_end, returns a sigmoid transition between
  y_left and y_right.

  The smoothstep function is a sigmoid-like function that exactly satisfies f(0)
  = 0, f(1) = 1, and has zero gradient at the endpoints. The standard logistic
  sigmoid has non-zero gradients at the endpoints, and only approaches 0 and 1
  asymptotically.

  Args:
    x: The independent variable.
    smoothing_start: The start of the sigmoid transition.
    smoothing_end: The end of the sigmoid transition.
    y_left: The value of the left model.
    y_right: The value of the right model.
    log_scale: Whether to use a logarithmic scale for the sigmoid transition. If
      True, the sigmoid transition is applied in log(x) space.

  Returns:
    A weighted average of the left and right models, with the weight given by a
    sigmoid (smoothstep) function.
  """
  if log_scale:
    x, smoothing_start, smoothing_end = jax.tree.map(
        jnp.log, (x, smoothing_start, smoothing_end)
    )
  t = (x - smoothing_start) / (smoothing_end - smoothing_start)
  # Clip t to [0, 1] to avoid extrapolation
  t = jnp.clip(t, 0.0, 1.0)
  # Smoothstep function (https://en.wikipedia.org/wiki/Smoothstep)
  w = 3.0 * t**2 - 2.0 * t**3
  return w * y_right + (1 - w) * y_left
