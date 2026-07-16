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
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SafetyFactorFit:
  """Collection of statistics of safety factor calculated each simulation step.

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

  rho_q_min: array_typing.FloatScalar
  q_min: array_typing.FloatScalar
  rho_q_3_2_first: array_typing.FloatScalar
  rho_q_2_1_first: array_typing.FloatScalar
  rho_q_3_1_first: array_typing.FloatScalar
  rho_q_3_2_second: array_typing.FloatScalar
  rho_q_2_1_second: array_typing.FloatScalar
  rho_q_3_1_second: array_typing.FloatScalar


def _linear_intercepts(
    rho_norm: jax.Array, q_face: jax.Array, q_target: float
) -> jax.Array:
  """Finds intercepts of q_face with q_target using linear interpolation.

  Args:
    rho_norm: Array of rho_norm values on face grid.
    q_face: Array of q values on face grid.
    q_target: q value to find intercepts for.

  Returns:
    Array of rho_norm values of (potential) intercepts in each interval. If no
    intercept is found in an interval, -jnp.inf is placed in the array.
  """
  q_diff = q_face[1:] - q_face[:-1]
  safe_q_diff = jnp.where(q_diff == 0.0, 1.0, q_diff)
  t = (q_target - q_face[:-1]) / safe_q_diff

  # If q_diff is 0 and q_face[:-1] == q_target, we force t = 1.0 (intercept at
  # rho_norm[i+1])
  t = jnp.where((q_diff == 0.0) & (q_face[:-1] == q_target), 1.0, t)

  is_intercept = (t > 0.0) & (t <= 1.0)
  # Mask out cases where q_diff == 0 and q_face[:-1] != q_target
  is_intercept = is_intercept & ~((q_diff == 0.0) & (q_face[:-1] != q_target))

  intercepts = rho_norm[:-1] + t * (rho_norm[1:] - rho_norm[:-1])

  all_intercepts = jnp.where(is_intercept, intercepts, -jnp.inf)

  first_point_intercept = jnp.where(
      q_face[0] == q_target, rho_norm[0], -jnp.inf
  )

  return jnp.concat([jnp.array([first_point_intercept]), all_intercepts])


@jax.jit
def find_min_q_and_q_surface_intercepts(
    rho_norm: jax.Array, q_face: jax.Array
) -> SafetyFactorFit:
  """Finds the minimum q and the q surface intercepts.

  This method uses the input arrays to find the minimum q value and its
  corresponding rho_norm. It then uses linear interpolation between adjacent
  points to find any intercepts of the q=3/2, q=2/1, and q=3/1 planes.

  Args:
    rho_norm: Array of rho_norm values on face grid.
    q_face: Array of q values on face grid.

  Returns:
    Safety factor statistics.
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

  # Find the minimum q value and its location.
  idx_min = jnp.argmin(q_face)
  q_min = q_face[idx_min]
  rho_q_min = rho_norm[idx_min]

  # Find the outermost rho_norm values that intercept the q=3/2, q=2/1, and
  # q=3/1 planes. If none are found, fill from the left with -jnp.inf.
  rho_q_3_2 = _linear_intercepts(rho_norm, q_face, 1.5)
  outermost_rho_q_3_2 = rho_q_3_2[jnp.argsort(rho_q_3_2)[-2:]]

  rho_q_2_1 = _linear_intercepts(rho_norm, q_face, 2.0)
  outermost_rho_q_2_1 = rho_q_2_1[jnp.argsort(rho_q_2_1)[-2:]]

  rho_q_3_1 = _linear_intercepts(rho_norm, q_face, 3.0)
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
