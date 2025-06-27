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
"""Helper for finding safety factor outputs."""
import functools

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils


@chex.dataclass(frozen=True)
class SafetyFactorFit:
  """Collection of outputs calculated after each simulation step.

  Attributes:
    rho_q_min: rho_norm at the minimum q.
    q_min: Minimum q value.
    rho_q_3_2_first: First outermost rho_norm value that intercepts the
      q=3/2 plane.
    rho_q_2_1_first: First outermost rho_norm value that intercepts the q=2/1
      plane.
    rho_q_3_1_first: First outermost rho_norm value that intercepts the q=3/1
      plane.
    rho_q_3_2_second: Second outermost rho_norm value that intercepts the
      q=3/2 plane.
    rho_q_2_1_second: Second outermost rho_norm value that intercepts the q=2/1
      plane.
    rho_q_3_1_second: Second outermost rho_norm value that intercepts the q=3/1
      plane.
  """

  rho_q_min: array_typing.ScalarFloat
  q_min: array_typing.ScalarFloat
  rho_q_3_2_first: array_typing.ScalarFloat
  rho_q_2_1_first: array_typing.ScalarFloat
  rho_q_3_1_first: array_typing.ScalarFloat
  rho_q_3_2_second: array_typing.ScalarFloat
  rho_q_2_1_second: array_typing.ScalarFloat
  rho_q_3_1_second: array_typing.ScalarFloat


def _sliding_window_of_three(flat_array: jax.Array) -> jax.Array:
  """Sliding window equivalent to numpy.lib.stride_tricks.sliding_window."""
  window_size = 3
  starts = jnp.arange(len(flat_array) - window_size + 1)
  return jax.vmap(
      lambda start: jax.lax.dynamic_slice(flat_array, (start,), (window_size,))
  )(starts)


def _fit_polynomial_to_intervals_of_three(
    rho_norm: jax.Array, q_face: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Fit polynomial to every set of three points in given arrays."""
  q_face_intervals = _sliding_window_of_three(
      q_face,
  )
  rho_norm_intervals = _sliding_window_of_three(
      rho_norm,
  )

  @jax.vmap
  def batch_polyfit(
      q_face_interval: jax.Array, rho_norm_interval: jax.Array
  ) -> jax.Array:
    """Solve Ax=b to get coefficients for a quadratic polynomial."""
    chex.assert_shape(q_face_interval, (3,))
    chex.assert_shape(rho_norm_interval, (3,))
    rho_norm_squared = rho_norm_interval**2
    A = jnp.array([  # pylint: disable=invalid-name
        [rho_norm_squared[0], rho_norm_interval[0], 1],
        [rho_norm_squared[1], rho_norm_interval[1], 1],
        [rho_norm_squared[2], rho_norm_interval[2], 1],
    ])
    b = jnp.array([q_face_interval[0], q_face_interval[1], q_face_interval[2]])
    coeffs = jnp.linalg.solve(A, b)
    return coeffs

  return (
      batch_polyfit(q_face_intervals, rho_norm_intervals),
      rho_norm_intervals,
      q_face_intervals,
  )


@jax.vmap
def _minimum_location_value_in_interval(
    coeffs: jax.Array, rho_norm_interval: jax.Array, q_interval: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Returns the minimum value and location of the fit quadratic in the interval."""
  min_interval, max_interval = rho_norm_interval[0], rho_norm_interval[1]
  q_min_interval, q_max_interval = (
      q_interval[0],
      q_interval[1],
  )
  a, b = coeffs[0], coeffs[1]
  extremum_location = -b / (2 * a)
  extremum_in_interval = jnp.greater(
      extremum_location, min_interval
  ) & jnp.less(extremum_location, max_interval)
  extremum_value = jax.lax.cond(
      extremum_in_interval,
      lambda x: jnp.polyval(coeffs, x),
      lambda x: jnp.inf,
      extremum_location,
  )

  interval_minimum_location, interval_minimum_value = jax.lax.cond(
      jnp.less(q_min_interval, q_max_interval),
      lambda: (min_interval, q_min_interval),
      lambda: (max_interval, q_max_interval),
  )
  overall_minimum_location, overall_minimum_value = jax.lax.cond(
      jnp.less(interval_minimum_value, extremum_value),
      lambda: (interval_minimum_location, interval_minimum_value),
      lambda: (extremum_location, extremum_value),
  )
  return overall_minimum_location, overall_minimum_value


def _find_roots_quadratic(coeffs: jax.Array) -> jax.Array:
  """Finds the roots of a quadratic equation."""
  a, b, c = coeffs[0], coeffs[1], coeffs[2]
  determinant = b**2 - 4.0 * a * c
  roots_exist = jnp.greater(determinant, 0)
  plus_root = jax.lax.cond(
      roots_exist,
      lambda: (-b + jnp.sqrt(determinant)) / (2.0 * a),
      lambda: -jnp.inf,
  )
  minus_root = jax.lax.cond(
      roots_exist,
      lambda: (-b - jnp.sqrt(determinant)) / (2.0 * a),
      lambda: -jnp.inf,
  )
  return jnp.array([plus_root, minus_root])


@functools.partial(jax.vmap, in_axes=(0, 0, None))
def _root_in_interval(
    coeffs: jax.Array, interval: jax.Array, q_surface: float
) -> jax.Array:
  """Finds roots of quadratic(coeffs)=q_surface in the interval."""
  intercept_coeffs = coeffs - jnp.array([0.0, 0.0, q_surface])
  min_interval, max_interval = interval[0], interval[1]
  root_values = _find_roots_quadratic(intercept_coeffs)
  in_interval = jnp.greater(root_values, min_interval) & jnp.less(
      root_values, max_interval
  )
  return jnp.where(in_interval, root_values, -jnp.inf)


@jax_utils.jit
def find_min_q_and_q_surface_intercepts(
    rho_norm: jax.Array, q_face: jax.Array
) -> SafetyFactorFit:
  """Finds the minimum q and the q surface intercepts.

  This method divides the input arrays into groups of three points, fits a
  quadratic to each interval containing three points. It then uses the fit
  quadratics to find the minimum q value as well as any intercepts of the q=3/2,
  q=2/1, and q=3/1 planes.

  As the quadratic fits could have overlapping definitions where the groups
  overlap, we choose the quadratic for the domain [rho_norm[i-1], rho_norm[i]]
  defined using the points (i, i-1, i-2).
  (There is one exception which is the first group for which the domain for
  points in [rho_norm[0], rho_norm[2]] are defined using points (0, 1, 2))

  Args:
    rho_norm: Array of rho_norm values on face grid.
    q_face: Array of q values on face grid.

  Returns:
    rho_q_min: rho_norm at the minimum q.
    q_min: minimum q value.
    outermost_rho_q_3_2: Array of two outermost rho_norm values that intercept
    the q=3/2 plane. If fewer than two intercepts are found, entries will be set
    to -inf.
    outermost_rho_q_2_1: Array of 2 outermost rho_norm values that intercept the
    q=2/1 plane. If fewer than two intercepts are found, entries will be set to
    -inf.
    outermost_rho_q_3_1: Array of 2 outermost rho_norm values that intercept the
    q=3/1 plane. If fewer than two intercepts are found, entries will be set to
    -inf.
  """
  if len(q_face) != len(rho_norm):
    raise ValueError(
        f'Input arrays must have the same length. {len(q_face)} !='
        f' {len(rho_norm)}'
    )
  if len(q_face) < 4:
    raise ValueError('Input arrays must have at least four points.')
  # Sort in case input is not sorted.
  sorted_indices = jnp.argsort(rho_norm)
  rho_norm = rho_norm[sorted_indices]
  q_face = q_face[sorted_indices]
  # Fit polynomial to every set of three points in given arrays.
  poly_coeffs, rho_norm_3, q_face_3 = _fit_polynomial_to_intervals_of_three(
      rho_norm, q_face
  )
  # To bridge across the entire span of rho_norm, and avoid overlapping
  # interval domains, we define the first group from points (0, 2), and all
  # subsequent groups from (i+1, i+2), from i=1 until i=len(rho_norm)-1.
  first_rho_norm = jnp.expand_dims(
      jnp.array([rho_norm[0], rho_norm[2]]), axis=0
  )
  first_q_face = jnp.expand_dims(jnp.array([q_face[0], q_face[2]]), axis=0)
  rho_norms = jnp.concat([first_rho_norm, rho_norm_3[1:, 1:]], axis=0)
  q_faces = jnp.concat([first_q_face, q_face_3[1:, 1:]], axis=0)

  # Find the minimum q value and its location for each interval.
  rho_q_min_intervals, q_min_intervals = _minimum_location_value_in_interval(
      poly_coeffs, rho_norms, q_faces
  )
  # Find the minimum q value and its location out of all intervals.
  arg_q_min = jnp.argmin(q_min_intervals)
  rho_q_min = rho_q_min_intervals[arg_q_min]
  q_min = q_min_intervals[arg_q_min]

  # Find the outermost rho_norm values that intercept the q=3/2, q=2/1, and
  # q=3/1 planes. If none are found, fill from the left with -jnp.inf.
  rho_q_3_2 = _root_in_interval(poly_coeffs, rho_norms, 1.5).flatten()
  outermost_rho_q_3_2 = rho_q_3_2[jnp.argsort(rho_q_3_2)[-2:]]
  rho_q_2_1 = _root_in_interval(poly_coeffs, rho_norms, 2.0).flatten()
  outermost_rho_q_2_1 = rho_q_2_1[jnp.argsort(rho_q_2_1)[-2:]]
  rho_q_3_1 = _root_in_interval(poly_coeffs, rho_norms, 3.0).flatten()
  outermost_rho_q_3_1 = rho_q_3_1[jnp.argsort(rho_q_3_1)[-2:]]

  return SafetyFactorFit(
      rho_q_min=rho_q_min,
      q_min=q_min,
      rho_q_3_2_first=outermost_rho_q_3_2[0],
      rho_q_2_1_first=outermost_rho_q_2_1[0],
      rho_q_3_1_first=outermost_rho_q_3_1[0],
      rho_q_3_2_second=outermost_rho_q_3_2[1],
      rho_q_2_1_second=outermost_rho_q_2_1[1],
      rho_q_3_1_second=outermost_rho_q_3_1[1],
  )
