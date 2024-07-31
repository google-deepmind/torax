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
"""Functionality for building discrete linear systems.

This file is expected to be used mostly internally by `fvm` itself.

The functionality here is for constructing a description of one discrete
time step of a PDE in terms of a linear equation. In practice, the
actual expressive power of the resulting Jax expression may still be
nonlinear because the coefficients of this linear equation are Jax
expressions, not just numeric values, so nonlinear solvers like
newton_raphson_solve_block can capture nonlinear dynamics even when
each step is expressed using a matrix multiply.
"""

from __future__ import annotations

from typing import TypeAlias

import jax
from jax import numpy as jnp
from torax.fvm import block_1d_coeffs
from torax.fvm import cell_variable
from torax.fvm import convection_terms
from torax.fvm import diffusion_terms


AuxiliaryOutput: TypeAlias = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffs: TypeAlias = block_1d_coeffs.Block1DCoeffs
Block1DCoeffsCallback: TypeAlias = block_1d_coeffs.Block1DCoeffsCallback


def calc_c(
    x: tuple[cell_variable.CellVariable, ...],
    coeffs: Block1DCoeffs,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[jax.Array, jax.Array]:
  """Calculate C and c such that F = C x + c.

  See docstrings for `Block1DCoeff` and `implicit_solve_block` for
  more detail.

  Args:
    x: Tuple containing CellVariables for each channel. This function uses only
      their shape and their boundary conditions, not their values.
    coeffs: Coefficients defining the differential equation.
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    c_mat: matrix C, such that F = C x + c
    c: the vector c
  """

  d_face = coeffs.d_face
  v_face = coeffs.v_face
  source_mat_cell = coeffs.source_mat_cell
  source_cell = coeffs.source_cell

  num_cells = x[0].value.shape[0]
  num_channels = len(x)
  for _, x_i in enumerate(x):
    if x_i.value.shape != (num_cells,):
      raise ValueError(
          f'Expected each x channel to have shape ({num_cells},) '
          f'but got {x_i.value.shape}.'
      )

  zero_block = jnp.zeros((num_cells, num_cells))
  zero_row_of_blocks = [zero_block] * num_channels
  zero_vec = jnp.zeros((num_cells))
  zero_block_vec = [zero_vec] * num_channels

  # Make a matrix C and vector c that will accumulate contributions from
  # diffusion, convection, and source terms.
  # C and c are both block structured, with one block per channel.
  c_mat = [zero_row_of_blocks.copy() for _ in range(num_channels)]
  c = zero_block_vec.copy()

  # Add diffusion terms
  if d_face is not None:
    for i in range(num_channels):
      (
          diffusion_mat,
          diffusion_vec,
      ) = diffusion_terms.make_diffusion_terms(
          d_face[i],
          x[i],
      )

      c_mat[i][i] += diffusion_mat
      c[i] += diffusion_vec

  # Add convection terms
  if v_face is not None:
    for i in range(num_channels):
      if v_face[i] is not None:
        # Resolve diffusion to zeros if it is not specified
        d_face_i = d_face[i] if d_face is not None else None
        d_face_i = jnp.zeros_like(v_face[i]) if d_face_i is None else d_face_i

        (
            conv_mat,
            conv_vec,
        ) = convection_terms.make_convection_terms(
            v_face[i],
            d_face_i,
            x[i],
            dirichlet_mode=convection_dirichlet_mode,
            neumann_mode=convection_neumann_mode,
        )

        c_mat[i][i] += conv_mat
        c[i] += conv_vec

  # Add implicit source terms
  if source_mat_cell is not None:
    for i in range(num_channels):
      for j in range(num_channels):
        source = source_mat_cell[i][j]
        if source is not None:
          c_mat[i][j] += jnp.diag(source)

  # Add explicit source terms
  def add(left: jax.Array, right: jax.Array | None):
    """Addition with adding None treated as no-op."""
    if right is not None:
      return left + right
    return left

  if source_cell is not None:
    c = [add(c_i, source_i) for c_i, source_i in zip(c, source_cell)]

  # Form block structure
  c_mat = jnp.block(c_mat)
  c = jnp.block(c)

  return c_mat, c
