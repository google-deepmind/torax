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
from typing import TypeAlias

import jax
from jax import numpy as jnp
from torax._src import tridiagonal
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import convection_terms
from torax._src.fvm import diffusion_terms

AuxiliaryOutput: TypeAlias = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffs: TypeAlias = block_1d_coeffs.Block1DCoeffs


def calc_c(
    x: tuple[cell_variable.CellVariable, ...],
    coeffs: Block1DCoeffs,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[tridiagonal.BlockTriDiagonal, jax.Array]:
  """Calculate banded blocks and vector c such that F = C x + c.

  Returns the block-tridiagonal representation of C. Each cell-cell coupling
  is a C x C block (where C = num_channels). The matrix structure comes from
  the 1D FVM stencil: each cell couples to itself and its two neighbors.

  Args:
    x: Tuple containing CellVariables for each channel. This function uses only
      their shape and their boundary conditions, not their values.
    coeffs: Coefficients defining the differential equation.
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    A tuple of (c_matrix, c_vec) where:
      c_matrix: BlockTriDiagonal with sub/main/super-diagonal blocks.
      c_vec: Vector c, shape (N, C).
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

  # Add diffusion terms
  if d_face is None:
    c_matrix = tridiagonal.BlockTriDiagonal.zeros(num_cells, num_channels)
    c = jnp.zeros((num_cells, num_channels))
  else:
    d_terms = [
        diffusion_terms.make_diffusion_terms(d_face_i, x_i)
        for d_face_i, x_i in zip(d_face, x)
    ]
    d_mats, c = jax.tree.map(lambda *args: jnp.stack(args, axis=1), *d_terms)
    c_matrix = tridiagonal.BlockTriDiagonal.from_stacked_tridiagonal(d_mats)

  # Add convection terms
  if v_face is not None:
    conv_terms = []
    for i in range(num_channels):
      # Resolve diffusion to zeros if it is not specified
      d_face_i = d_face[i] if d_face is not None else None
      d_face_i = jnp.zeros_like(v_face[i]) if d_face_i is None else d_face_i
      conv_mat, conv_vec = convection_terms.make_convection_terms(
          v_face[i],
          d_face_i,
          x[i],
          dirichlet_mode=convection_dirichlet_mode,
          neumann_mode=convection_neumann_mode,
      )
      conv_terms.append((conv_mat, conv_vec))
    conv_mats, conv_vecs = jax.tree.map(
        lambda *args: jnp.stack(args, axis=1),
        *conv_terms,
    )
    c_matrix += tridiagonal.BlockTriDiagonal.from_stacked_tridiagonal(conv_mats)
    c += conv_vecs

  # Add implicit source terms
  if source_mat_cell is not None:
    diag = c_matrix.diagonal
    for i in range(num_channels):
      for j in range(num_channels):
        source = source_mat_cell[i][j]
        if source is not None:
          diag = diag.at[:, i, j].add(source)
    c_matrix = tridiagonal.BlockTriDiagonal(
        lower=c_matrix.lower,
        diagonal=diag,
        upper=c_matrix.upper,
    )

  if source_cell is not None:
    for i in range(num_channels):
      if source_cell[i] is not None:
        c = c.at[:, i].add(source_cell[i])

  return c_matrix, c
