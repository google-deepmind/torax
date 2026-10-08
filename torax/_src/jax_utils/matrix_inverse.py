# Copyright 2026 DeepMind Technologies Limited
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

"""Fast matrix inversion implementations for small matrices (dim <= 4)."""

from typing import Literal, TypeAlias
import jax
from jax import numpy as jnp
import jaxtyping as jt
import numpy as np

type Implementations = Literal['solve', 'cramer']

Array: TypeAlias = jax.Array | np.ndarray


def _inv_1x1(
    m: jt.Float[Array, '... 1 1'],
) -> jt.Float[Array, '... 1 1']:
  """Computes analytical inverse of 1x1 matrix."""
  return 1.0 / m


def _inv_2x2(
    m: jt.Float[Array, '... 2 2'],
) -> jt.Float[Array, '... 2 2']:
  """Computes analytical inverse of 2x2 matrix."""
  a = m[..., 0, 0]
  b = m[..., 0, 1]
  c = m[..., 1, 0]
  d = m[..., 1, 1]
  det = a * d - b * c
  inv_det = 1.0 / det
  adj = jnp.stack(
      [
          jnp.stack([d, -b], axis=-1),
          jnp.stack([-c, a], axis=-1),
      ],
      axis=-2,
  )
  return adj * inv_det[..., None, None]


def _inv_3x3(
    m: jt.Float[Array, '... 3 3'],
) -> jt.Float[Array, '... 3 3']:
  """Computes analytical inverse of 3x3 matrix via adjugate cofactors."""
  a00 = m[..., 0, 0]
  a01 = m[..., 0, 1]
  a02 = m[..., 0, 2]
  a10 = m[..., 1, 0]
  a11 = m[..., 1, 1]
  a12 = m[..., 1, 2]
  a20 = m[..., 2, 0]
  a21 = m[..., 2, 1]
  a22 = m[..., 2, 2]

  b00 = a11 * a22 - a12 * a21
  b01 = a02 * a21 - a01 * a22
  b02 = a01 * a12 - a02 * a11

  b10 = a12 * a20 - a10 * a22
  b11 = a00 * a22 - a02 * a20
  b12 = a02 * a10 - a00 * a12

  b20 = a10 * a21 - a11 * a20
  b21 = a01 * a20 - a00 * a21
  b22 = a00 * a11 - a01 * a10

  det = a00 * b00 + a01 * b10 + a02 * b20
  inv_det = 1.0 / det

  adj = jnp.stack(
      [
          jnp.stack([b00, b01, b02], axis=-1),
          jnp.stack([b10, b11, b12], axis=-1),
          jnp.stack([b20, b21, b22], axis=-1),
      ],
      axis=-2,
  )
  return adj * inv_det[..., None, None]


def _inv_4x4(
    m: jt.Float[Array, '... 4 4'],
) -> jt.Float[Array, '... 4 4']:
  """Computes analytical inverse of 4x4 matrix using 2x2 minors."""
  a00 = m[..., 0, 0]
  a01 = m[..., 0, 1]
  a02 = m[..., 0, 2]
  a03 = m[..., 0, 3]

  a10 = m[..., 1, 0]
  a11 = m[..., 1, 1]
  a12 = m[..., 1, 2]
  a13 = m[..., 1, 3]

  a20 = m[..., 2, 0]
  a21 = m[..., 2, 1]
  a22 = m[..., 2, 2]
  a23 = m[..., 2, 3]

  a30 = m[..., 3, 0]
  a31 = m[..., 3, 1]
  a32 = m[..., 3, 2]
  a33 = m[..., 3, 3]

  # 2x2 determinants from rows 0 and 1
  s0 = a00 * a11 - a01 * a10
  s1 = a00 * a12 - a02 * a10
  s2 = a00 * a13 - a03 * a10
  s3 = a01 * a12 - a02 * a11
  s4 = a01 * a13 - a03 * a11
  s5 = a02 * a13 - a03 * a12

  # 2x2 determinants from rows 2 and 3
  c5 = a22 * a33 - a23 * a32
  c4 = a21 * a33 - a23 * a31
  c3 = a21 * a32 - a22 * a31
  c2 = a20 * a33 - a23 * a30
  c1 = a20 * a32 - a22 * a30
  c0 = a20 * a31 - a21 * a30

  det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
  inv_det = 1.0 / det

  b00 = a11 * c5 - a12 * c4 + a13 * c3
  b01 = -a01 * c5 + a02 * c4 - a03 * c3
  b02 = a31 * s5 - a32 * s4 + a33 * s3
  b03 = -a21 * s5 + a22 * s4 - a23 * s3

  b10 = -a10 * c5 + a12 * c2 - a13 * c1
  b11 = a00 * c5 - a02 * c2 + a03 * c1
  b12 = -a30 * s5 + a32 * s2 - a33 * s1
  b13 = a20 * s5 - a22 * s2 + a23 * s1

  b20 = a10 * c4 - a11 * c2 + a13 * c0
  b21 = -a00 * c4 + a01 * c2 - a03 * c0
  b22 = a30 * s4 - a31 * s2 + a33 * s0
  b23 = -a20 * s4 + a21 * s2 - a23 * s0

  b30 = -a10 * c3 + a11 * c1 - a12 * c0
  b31 = a00 * c3 - a01 * c1 + a02 * c0
  b32 = -a30 * s3 + a31 * s1 - a32 * s0
  b33 = a20 * s3 - a21 * s1 + a22 * s0

  adj = jnp.stack(
      [
          jnp.stack([b00, b01, b02, b03], axis=-1),
          jnp.stack([b10, b11, b12, b13], axis=-1),
          jnp.stack([b20, b21, b22, b23], axis=-1),
          jnp.stack([b30, b31, b32, b33], axis=-1),
      ],
      axis=-2,
  )
  return adj * inv_det[..., None, None]


@jax.custom_jvp
def _cramer_inverse(
    a: jt.Float[Array, '... dim dim'],
) -> jt.Float[Array, '... dim dim']:
  """Computes matrix inverse using Cramer's rule."""
  dim = a.shape[-1]
  if dim == 1:
    return _inv_1x1(a)
  elif dim == 2:
    return _inv_2x2(a)
  elif dim == 3:
    return _inv_3x3(a)
  elif dim == 4:
    return _inv_4x4(a)
  # TODO(b/566996136): Support higher dimensions.
  else:
    raise ValueError(f'Cramer inverse not implemented for dimension {dim}.')


@_cramer_inverse.defjvp
def _cramer_inverse_jvp(primals, tangents):
  # Giles (2008) Section 2.2.3 for forward mode autodiff of matrix inverse:
  # d(A^{-1}) = -A^{-1} @ dA @ A^{-1}
  (a,) = primals
  (a_dot,) = tangents
  a_inv = _cramer_inverse(a)
  a_inv_dot = -(a_inv @ a_dot @ a_inv)
  return a_inv, a_inv_dot


def inv(
    a: jt.Float[Array, '... dim dim'], implementation: Implementations = 'solve'
) -> jt.Float[Array, '... dim dim']:
  """Computes matrix inverse."""
  match implementation:
    case 'solve':
      return jnp.linalg.solve(a, jnp.eye(a.shape[-1]))
    case 'cramer':
      return _cramer_inverse(a)
    case _:
      raise ValueError(f'Unknown implementation: {implementation}.')


def fast_matrix_inverse(
    m: jt.Float[Array, '... dim dim'],
) -> jt.Float[Array, '... dim dim']:
  """Computes matrix inverse using closed-form formulas if possible.

  Falls back to jnp.linalg.inv for larger matrices.

  Args:
    m: Square matrix or batch of matrices with trailing dimensions (dim, dim).

  Returns:
    Inverse matrix of the same shape and dtype as m.
  """

  if jax.default_backend() != 'cpu':
    return inv(m, implementation='solve')

  try:
    return inv(m, implementation='cramer')
  except ValueError:
    return inv(m, implementation='solve')
