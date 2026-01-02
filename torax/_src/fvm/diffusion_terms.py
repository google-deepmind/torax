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

"""The `make_diffusion_terms` function.

Builds the diffusion terms of the discrete matrix equation.
"""

import chex
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src.fvm import cell_variable


def make_diffusion_terms(
    d_face: array_typing.FloatVectorFace, var: cell_variable.CellVariable
) -> tuple[array_typing.FloatMatrixCell, array_typing.FloatVectorCell]:
  """Makes the terms of the matrix equation derived from the diffusion term.

  The diffusion term is of the form
  (partial / partial x) D partial x / partial x

  Args:
    d_face: Diffusivity coefficient on faces.
    var: CellVariable (to define geometry and boundary conditions)

  Returns:
    mat: Tridiagonal matrix of coefficients on u
    c: Vector of terms not dependent on u
  """

  # Start by using the formula for the interior rows everywhere
  dx = var.cell_widths
  distance_to_left_ghost_cell_center = dx[0]
  distance_to_right_ghost_cell_center = dx[-1]
  cell_spacings = jnp.concat([
      jnp.array([distance_to_left_ghost_cell_center]),
      var.cell_spacings,
      jnp.array([distance_to_right_ghost_cell_center]),
  ])
  # Fill in the inner diagonal.
  face_flux_right = d_face[1:] / cell_spacings[1:]
  face_flux_left = d_face[:-1] / cell_spacings[:-1]
  diag = (- face_flux_right - face_flux_left) / dx

  off = d_face[1:-1] / var.cell_spacings
  # Divide by different cell widths for the upper and lower diagonals.
  upper_off = off / dx[:-1]
  lower_off = off / dx[1:]

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
    # Left face Dirichlet condition.
    denom_left = cell_spacings[0] * dx[0]
    denom_right = cell_spacings[1] * dx[0]
    diag = diag.at[0].set(-2 * d_face[0] / denom_left - d_face[1] / denom_right)
    vec = vec.at[0].set(2 * d_face[0] * var.left_face_constraint / denom_left)
  else:
    # Left face gradient condition.
    denom_right = cell_spacings[1] * dx[0]
    diag = diag.at[0].set(-d_face[1] / denom_right)
    vec = vec.at[0].set(
        -d_face[0] * var.left_face_grad_constraint / dx[0]
    )
  if var.right_face_constraint is not None:
    # Right face Dirichlet condition.
    denom_left = cell_spacings[-2] * dx[-1]
    denom_right = cell_spacings[-1] * dx[-1]
    diag = diag.at[-1].set(
        -2 * d_face[-1] / denom_right - d_face[-2] / denom_left
    )
    vec = vec.at[-1].set(
        2 * d_face[-1] * var.right_face_constraint / denom_right
    )
  else:
    # Right face gradient condition.
    denom_left = cell_spacings[-2] * dx[-1]
    diag = diag.at[-1].set(-d_face[-2] / denom_left)
    vec = vec.at[-1].set(
        d_face[-1] * var.right_face_grad_constraint / dx[-1]
    )

  # Build the matrix
  mat = math_utils.tridiag(diag, lower_off, upper_off)
  return mat, vec
