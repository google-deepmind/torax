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
from torax import jax_utils
from torax import math_utils
from torax.fvm import cell_variable


def make_convection_terms(
    v_face: jax.Array,
    d_face: jax.Array,
    var: cell_variable.CellVariable,
    dirichlet_mode: str = 'ghost',
    neumann_mode: str = 'ghost',
) -> tuple[jax.Array, jax.Array]:
  """Makes the terms of the matrix equation derived from the convection term.

  The convection term of the differential equation is of the form
  - (partial / partial r) v u

  Args:
    v_face: Convection coefficient on faces.
    d_face: Diffusion coefficient on faces. The relative strength of convection
      to diffusion is used to weight the contribution of neighboring cells when
      calculating face values of u.
    var: CellVariable to define mesh and boundary conditions.
    dirichlet_mode: The strategy to use to handle Dirichlet boundary conditions.
      The default is 'ghost', which has superior stability. 'ghost' -> Boundary
      face values are inferred by constructing a ghost cell then alpha weighting
      cells 'direct' -> Boundary face values are read directly from constraints
      'semi-implicit' -> Matches FiPy. Boundary face values are alpha weighted
      with the constraint value specifying the value of the "other" cell:
      x_{boundary_face} = alpha x_{last_cell} + (1 - alpha) BC
    neumann_mode: Which strategy to use to handle Neumann boundary conditions.
      The default is `ghost`, which has superior stability. 'ghost' -> Boundary
      face values are inferred by constructing a ghost cell then alpha weighting
      cells. 'semi-implicit' -> Matches FiPy. Boundary face values are alpha
      weighted, with the (1 - alpha) weight applied to the external face value
      rather than to a ghost cell.

  Returns:
    mat: Tridiagonal matrix of coefficients on u
    c: Vector of terms not dependent on u
  """

  # Alpha weighting calculated using power law scheme described in
  # https://www.ctcms.nist.gov/fipy/documentation/numerical/scheme.html

  # Avoid divide by zero
  eps = 1e-20
  is_neg = d_face < 0.0
  nonzero_sign = jnp.ones_like(is_neg) - 2 * is_neg
  d_face = nonzero_sign * jnp.maximum(eps, jnp.abs(d_face))

  # FiPy uses half mesh width at the boundaries
  half = jnp.array([0.5], dtype=jax_utils.get_dtype())
  ones = jnp.ones_like(v_face[1:-1])
  scale = jnp.concatenate((half, ones, half))

  ratio = scale * var.dr * v_face / d_face

  # left_peclet[i] gives the Péclet number of cell i's left face
  left_peclet = -ratio[:-1]
  right_peclet = ratio[1:]

  def peclet_to_alpha(p):
    eps = 1e-3
    p = jnp.where(jnp.abs(p) < eps, eps, p)

    alpha_pg10 = (p - 1) / p
    alpha_p0to10 = ((p - 1) + (1 - p / 10) ** 5) / p
    # FiPy doc has a typo on the next line, where we use a + the doc has a
    # -, which is clearly a mistake since it makes the function
    # discontinuous and negative
    alpha_pneg10to0 = ((1 + p / 10) ** 5 - 1) / p
    alpha_plneg10 = -1 / p

    alpha = 0.5 * jnp.ones_like(p)
    alpha = jnp.where(p > 10.0, alpha_pg10, alpha)
    alpha = jnp.where(jnp.logical_and(10.0 >= p, p > eps), alpha_p0to10, alpha)
    alpha = jnp.where(
        jnp.logical_and(-eps > p, p >= -10), alpha_pneg10to0, alpha
    )
    alpha = jnp.where(p < -10.0, alpha_plneg10, alpha)

    return alpha

  left_alpha = peclet_to_alpha(left_peclet)
  right_alpha = peclet_to_alpha(right_peclet)

  left_v = v_face[:-1]
  right_v = v_face[1:]

  diag = (left_alpha * left_v - right_alpha * right_v) / var.dr
  above = -(1.0 - right_alpha) * right_v / var.dr
  above = above[:-1]
  below = (1.0 - left_alpha) * left_v / var.dr
  below = below[1:]
  mat = math_utils.tridiag(diag, above, below)

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
    if dirichlet_mode == 'ghost':
      mat_value = (
          v_face[0] * (2.0 * left_alpha[0] - 1.0) - v_face[1] * right_alpha[0]
      ) / var.dr
      vec_value = (
          2.0 * v_face[0] * (1.0 - left_alpha[0]) * var.left_face_constraint
      ) / var.dr
    elif dirichlet_mode == 'direct':
      vec_value = v_face[0] * var.left_face_constraint / var.dr
      mat_value = -v_face[1] * right_alpha[0]
    elif dirichlet_mode == 'semi-implicit':
      vec_value = (
          v_face[0] * (1.0 - left_alpha[0]) * var.left_face_constraint
      ) / var.dr
      mat_value = mat[0, 0]
      print('left vec_value: ', vec_value)
    else:
      raise ValueError(dirichlet_mode)
  else:
    # Gradient boundary condition at leftmost face
    mat_value = (v_face[0] - right_alpha[0] * v_face[1]) / var.dr
    vec_value = (
        -v_face[0] * (1.0 - left_alpha[0]) * var.left_face_grad_constraint
    )
    if neumann_mode == 'ghost':
      pass  # no adjustment needed
    elif neumann_mode == 'semi-implicit':
      vec_value /= 2.0
    else:
      raise ValueError(neumann_mode)

  mat = mat.at[0, 0].set(mat_value)
  vec = vec.at[0].set(vec_value)

  if var.right_face_constraint is not None:
    # Dirichlet condition at rightmost face
    if dirichlet_mode == 'ghost':
      mat_value = (
          v_face[-2] * left_alpha[-1]
          + v_face[-1] * (1.0 - 2.0 * right_alpha[-1])
      ) / var.dr
      vec_value = (
          -2.0
          * v_face[-1]
          * (1.0 - right_alpha[-1])
          * var.right_face_constraint
      ) / var.dr
    elif dirichlet_mode == 'direct':
      mat_value = v_face[-2] * left_alpha[-1] / var.dr
      vec_value = -v_face[-1] * var.right_face_constraint / var.dr
    elif dirichlet_mode == 'semi-implicit':
      mat_value = mat[-1, -1]
      vec_value = (
          -(v_face[-1] * (1.0 - right_alpha[-1]) * var.right_face_constraint)
          / var.dr
      )
    else:
      raise ValueError(dirichlet_mode)
  else:
    # Gradient boundary condition at rightmost face
    mat_value = -(v_face[-1] - v_face[-2] * left_alpha[-1]) / var.dr
    vec_value = (
        -v_face[-1] * (1.0 - right_alpha[-1]) * var.right_face_grad_constraint
    )
    if neumann_mode == 'ghost':
      pass  # no adjustment needed
    elif neumann_mode == 'semi-implicit':
      vec_value /= 2.0
    else:
      raise ValueError(neumann_mode)

  mat = mat.at[-1, -1].set(mat_value)
  vec = vec.at[-1].set(vec_value)

  return mat, vec
