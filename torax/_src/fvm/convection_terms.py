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

"""The `make_convection_terms` function.

Builds the convection terms of the discrete matrix equation.
"""

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import tridiagonal
from torax._src.fvm import cell_variable


# Below this |Péclet number|, `_peclet_to_alpha` uses its Taylor expansion
# instead of the exponential form, avoiding cancellation error near p = 0.
_SMALL_PECLET = 0.1
# |Péclet number| at which `expm1` is clipped to avoid overflow; beyond this the
# weights equal the upwind limits to machine precision.
_LARGE_PECLET = 40.0


def _peclet_to_alpha(p):
  """Face interpolation weight for the exponential convection-diffusion scheme.

  Computes `alpha(p) = ((p - 1) * exp(p) + 1) / (p * (exp(p) - 1))`
  (Scharfetter & Gummel 1969; Patankar 1980 Section 5), which reproduces the
  exact local steady solution with constant coefficients and zero source. See
  also the FiPy discussion of convection schemes (`ExponentialConvectionTerm`):
  https://www.ctcms.nist.gov/fipy/documentation/numerical/scheme.html

  Uses the degree-7 Bernoulli Taylor expansion for `|p| < 0.1` (full accuracy
  in both float32 and float64) and clips `|p|` at 40 in `expm1` so large `|p|`
  matches the upwind limits `1 - 1 / p` (`p >> 1`) and `-1 / p` (`p << -1`)
  without overflow or NaN gradients.

  Args:
    p: Face Péclet number array.

  Returns:
    Face interpolation weight `alpha` in `[0, 1]`.
  """
  is_small = jnp.abs(p) < _SMALL_PECLET
  # Exponential branch for |p| >= _SMALL_PECLET. Masked entries get the
  # placeholder _SMALL_PECLET so that neither 1 / p nor expm1 is evaluated at 0.
  p_safe = jnp.where(is_small, _SMALL_PECLET, p)
  p_clipped = jnp.clip(p_safe, -_LARGE_PECLET, _LARGE_PECLET)
  alpha_exp = 1.0 - 1.0 / p_safe + 1.0 / jnp.expm1(p_clipped)
  # Taylor expansion for |p| < _SMALL_PECLET. Masked entries use 0 so that the
  # polynomial cannot overflow (or produce NaN gradients) at large |p|.
  p_small = jnp.where(is_small, p, 0.0)
  p2 = p_small * p_small
  alpha_series = 0.5 + p_small * (
      1.0 / 12.0 - p2 * (1.0 / 720.0 - p2 * (1.0 / 30240.0 - p2 / 1209600.0))
  )
  return jnp.where(is_small, alpha_series, alpha_exp)


# TODO(b/469726859): Once non-uniform grid is supported add in testing.
def make_convection_terms(
    v_face: array_typing.FloatVectorFace,
    d_face: array_typing.FloatVectorFace,
    var: cell_variable.CellVariable,
) -> tuple[tridiagonal.TriDiagonal, jax.Array]:
  """Makes the terms of the matrix equation derived from the convection term.

  The convection term of the differential equation is of the form
  - (partial / partial r) v u, discretized with ghost-cell boundary conditions
  consistent with `make_diffusion_terms`.

  Args:
    v_face: Convection coefficient on faces.
    d_face: Diffusion coefficient on faces. The relative strength of convection
      to diffusion is used to weight the contribution of neighboring cells when
      calculating face values of u.
    var: CellVariable to define mesh and boundary conditions.

  Returns:
    mat: Tridiagonal matrix of coefficients on u
    c: Vector of terms not dependent on u
  """

  # Avoid divide by zero
  eps = 1e-20
  is_neg = d_face < 0.0
  nonzero_sign = jnp.ones_like(is_neg) - 2 * is_neg
  d_face = nonzero_sign * jnp.maximum(eps, jnp.abs(d_face))

  # FiPy uses half mesh width at the boundaries
  half = jnp.array([0.5], dtype=jax_utils.get_dtype())
  ones = jnp.ones_like(v_face[1:-1])
  scale = jnp.concatenate((half, ones, half))

  distance_to_left_ghost_cell_center = var.cell_widths[0]
  distance_to_right_ghost_cell_center = var.cell_widths[-1]
  cell_spacings = jnp.concat([
      jnp.array([distance_to_left_ghost_cell_center]),
      var.cell_spacings,
      jnp.array([distance_to_right_ghost_cell_center]),
  ])
  ratio = scale * cell_spacings * v_face / d_face

  # left_peclet[i] gives the Péclet number of cell i's left face
  left_peclet = -ratio[:-1]
  right_peclet = ratio[1:]

  left_alpha = _peclet_to_alpha(left_peclet)
  right_alpha = _peclet_to_alpha(right_peclet)

  left_v = v_face[:-1]
  right_v = v_face[1:]

  diag = (left_alpha * left_v - right_alpha * right_v) / var.cell_widths
  vec = jnp.zeros_like(diag)

  if vec.shape[0] < 2:
    raise NotImplementedError(
        'We do not support the case where a single cell'
        ' is affected by both boundary conditions.'
    )

  # Boundary rows need to be special-cased.
  #
  # Check that the boundary conditions are well-posed.
  # These checks are redundant with CellVariable.__post_init__, but including
  # them here for readability because they're in important part of the logic
  # of this function.
  chex.assert_exactly_one_is_none(
      var.left_face_grad_constraint, var.left_face_constraint
  )

  chex.assert_exactly_one_is_none(
      var.right_face_grad_constraint, var.right_face_constraint
  )

  if var.left_face_constraint is not None:
    # Dirichlet condition at leftmost face
    diag_left_face = (
        v_face[0] * (2.0 * left_alpha[0] - 1.0) - v_face[1] * right_alpha[0]
    ) / cell_spacings[0]
    vec_left_face = (
        2.0 * v_face[0] * (1.0 - left_alpha[0]) * var.left_face_constraint
    ) / cell_spacings[0]
  else:
    # Gradient boundary condition at leftmost face
    diag_left_face = (v_face[0] - right_alpha[0] * v_face[1]) / cell_spacings[0]
    vec_left_face = (
        -v_face[0] * (1.0 - left_alpha[0]) * var.left_face_grad_constraint
    )

  diag = diag.at[0].set(diag_left_face)
  vec = vec.at[0].set(vec_left_face)

  if var.right_face_constraint is not None:
    # Dirichlet condition at rightmost face
    diag_right_face = (
        v_face[-2] * left_alpha[-1]
        + v_face[-1] * (1.0 - 2.0 * right_alpha[-1])
    ) / cell_spacings[-1]
    vec_right_face = (
        -2.0
        * v_face[-1]
        * (1.0 - right_alpha[-1])
        * var.right_face_constraint
    ) / cell_spacings[-1]
  else:
    # Gradient boundary condition at rightmost face
    diag_right_face = (
        -(v_face[-1] - v_face[-2] * left_alpha[-1]) / cell_spacings[-1]
    )
    vec_right_face = (
        -v_face[-1] * (1.0 - right_alpha[-1]) * var.right_face_grad_constraint
    )

  diag = diag.at[-1].set(diag_right_face)
  above = -(1.0 - right_alpha) * right_v / var.cell_widths
  below = (1.0 - left_alpha) * left_v / var.cell_widths
  mat = tridiagonal.TriDiagonal(
      diagonal=diag, above=above[:-1], below=below[1:]
  )

  return mat, vec.at[-1].set(vec_right_face)
