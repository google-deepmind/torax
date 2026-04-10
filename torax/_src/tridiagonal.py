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
import jaxtyping as jt
from torax._src import array_typing


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

  def __add__(self, other: 'TriDiagonal') -> 'TriDiagonal':
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
