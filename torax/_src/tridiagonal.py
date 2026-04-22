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

"""Tridiagonal matrix representations and operations."""

import dataclasses

import jax
from jax import numpy as jnp
import jax.scipy.linalg
import jaxtyping as jt
from torax._src import array_typing
from torax._src import jax_utils
import typing_extensions


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TriDiagonal:
  """A tridiagonal matrix representation.

  Attributes:
    diagonal: The main diagonal.
    above: The super-diagonal.
    below: The sub-diagonal.
  """

  diagonal: jt.Float[array_typing.Array, 'size']
  above: jt.Float[array_typing.Array, 'size-1']
  below: jt.Float[array_typing.Array, 'size-1']

  def to_dense(self) -> jt.Float[array_typing.Array, 'size size']:
    return (
        jnp.diag(self.diagonal)
        + jnp.diag(self.above, 1)
        + jnp.diag(self.below, -1)
    )

  def __add__(self, other: typing_extensions.Self) -> typing_extensions.Self:
    return TriDiagonal(
        diagonal=self.diagonal + other.diagonal,
        above=self.above + other.above,
        below=self.below + other.below,
    )

  def matvec(
      self,
      x: jt.Float[array_typing.Array, 'size'],
  ) -> jt.Float[array_typing.Array, 'size']:
    return (
        self.diagonal * x
        + jnp.pad(self.above * x[1:], (0, 1))
        + jnp.pad(self.below * x[:-1], (1, 0))
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BlockTriDiagonal:
  """A block-tridiagonal matrix stored as its three diagonals.

  Attributes:
    lower: Sub-diagonal blocks, shape (num_blocks-1, block_size, block_size).
    diagonal: Main diagonal blocks, shape (num_blocks, block_size, block_size).
    upper: Super-diagonal blocks, shape (num_blocks-1, block_size, block_size).
  """

  lower: jt.Float[array_typing.Array, 'num_blocks-1 block_size block_size']
  diagonal: jt.Float[array_typing.Array, 'num_blocks block_size block_size']
  upper: jt.Float[array_typing.Array, 'num_blocks-1 block_size block_size']

  @property
  def num_blocks(self) -> int:
    """Number of blocks in the main diagonal."""
    return self.diagonal.shape[0]

  @property
  def block_size(self) -> int:
    """Size of each block."""
    return self.diagonal.shape[1]

  def __add__(self, other: typing_extensions.Self) -> typing_extensions.Self:
    return BlockTriDiagonal(
        lower=self.lower + other.lower,
        diagonal=self.diagonal + other.diagonal,
        upper=self.upper + other.upper,
    )

  @classmethod
  def from_block_diagonal(
      cls,
      vals: jt.Float[array_typing.Array, 'num_blocks block_size block_size'],
  ) -> 'BlockTriDiagonal':
    """Creates a block-tridiagonal matrix from diagonal blocks."""
    num_blocks, block_size, _ = vals.shape
    off_diag = jnp.zeros(
        (num_blocks - 1, block_size, block_size), dtype=vals.dtype
    )
    return cls(
        lower=off_diag,
        diagonal=vals,
        upper=off_diag,
    )

  @classmethod
  def zeros(
      cls,
      num_blocks: int,
      block_size: int,
      dtype: jnp.dtype | None = None,
  ) -> 'BlockTriDiagonal':
    """Creates a zero block-tridiagonal matrix."""
    dtype = dtype if dtype is not None else jax_utils.get_dtype()
    return cls.from_block_diagonal(
        vals=jnp.zeros((num_blocks, block_size, block_size), dtype=dtype),
    )

  @classmethod
  def from_diagonal(
      cls,
      vals: jt.Float[array_typing.Array, 'num_blocks block_size'],
  ) -> 'BlockTriDiagonal':
    """Creates a block-tridiagonal matrix from diagonal blocks."""
    return cls.from_block_diagonal(
        vals=vals[..., None, :] * jnp.eye(vals.shape[-1], dtype=vals.dtype),
    )

  @classmethod
  def from_tridiagonals(
      cls,
      tridiagonals: typing_extensions.Iterable[TriDiagonal],
  ) -> 'BlockTriDiagonal':
    """Creates a BlockTriDiagonal from an iterable of per-channel TriDiagonals.

    Each channel contributes a scalar tridiagonal system placed along the (i, i)
    block diagonal.

    Args:
      tridiagonals: Iterable of TriDiagonal objects, each of size N.

    Returns:
      BlockTriDiagonal with block size C, where each (C, C) block is diagonal.
    """
    tridiagonals_seq = tuple(tridiagonals)
    stacked = jax.tree.map(
        lambda *args: jnp.stack(args, axis=1), *tridiagonals_seq
        )
    return cls(
        lower=stacked.below[..., None, :]
        * jnp.eye(stacked.below.shape[-1], dtype=stacked.below.dtype),
        diagonal=stacked.diagonal[..., None, :]
        * jnp.eye(stacked.diagonal.shape[-1], dtype=stacked.diagonal.dtype),
        upper=stacked.above[..., None, :]
        * jnp.eye(stacked.above.shape[-1], dtype=stacked.above.dtype),
    )

  def to_dense(self) -> jt.Float[array_typing.Array, 'total total']:
    """Constructs the dense matrix representation.

    Returns:
      Dense matrix of shape (num_blocks * block_size, num_blocks *
      block_size).
    """
    block_size = self.block_size
    mat = jax.scipy.linalg.block_diag(*self.diagonal)
    if self.num_blocks == 1:
      return mat
    lower_mat = jnp.pad(
        jax.scipy.linalg.block_diag(*self.lower),
        ((block_size, 0), (0, block_size)),
    )
    upper_mat = jnp.pad(
        jax.scipy.linalg.block_diag(*self.upper),
        ((0, block_size), (block_size, 0)),
    )
    return mat + lower_mat + upper_mat

  def solve(
      self, rhs: jt.Float[array_typing.Array, 'num_blocks block_size']
  ) -> jt.Float[array_typing.Array, 'num_blocks block_size']:
    """Solves A @ x = rhs.

    Args:
      rhs: Right-hand side, shape (num_blocks, block_size).

    Returns:
      Solution x, shape (num_blocks, block_size).
    """
    return dense_solve(self, rhs)

  def matvec(
      self, x: jt.Float[array_typing.Array, 'num_blocks block_size']
  ) -> jt.Float[array_typing.Array, 'num_blocks block_size']:
    """Block-tridiagonal matrix-vector multiply: y = A @ x.

    Args:
      x: Input vector, shape (num_blocks, block_size).

    Returns:
      Result y, shape (num_blocks, block_size).
    """
    y_upper = jnp.pad(
        jnp.einsum('nij,nj->ni', self.upper, x[1:]), ((0, 1), (0, 0))
    )
    y_lower = jnp.pad(
        jnp.einsum('nij,nj->ni', self.lower, x[:-1]), ((1, 0), (0, 0))
    )
    return jnp.einsum('nij,nj->ni', self.diagonal, x) + y_upper + y_lower


def dense_solve(
    block_tridiag: BlockTriDiagonal,
    rhs: jt.Float[array_typing.Array, 'num_blocks block_size'],
) -> jt.Float[array_typing.Array, 'num_blocks block_size']:
  """Solves A @ x = rhs using a dense matrix inversion.

  Args:
    block_tridiag: Block tridiagonal matrix.
    rhs: Right-hand side, shape (num_blocks, block_size).

  Returns:
    Solution x, shape (num_blocks, block_size).
  """
  x_flat = jax.scipy.linalg.solve(block_tridiag.to_dense(), rhs.flatten())
  return x_flat.reshape((block_tridiag.num_blocks, block_tridiag.block_size))
