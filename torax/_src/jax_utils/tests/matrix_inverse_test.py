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

from typing import cast
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.jax_utils import matrix_inverse


class MatrixInverseTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 4, 5)
  def test_invert_matches_linalg_inv(self, dim: int):
    """fast_matrix_inverse matches jnp.linalg.inv for dimensions 1 to 5."""
    key = jax.random.key(42 + dim)
    m = jax.random.normal(key, (dim, dim), dtype=jnp.float64)
    # Ensure well-conditioned
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    actual = matrix_inverse.fast_matrix_inverse(m)
    expected = jnp.linalg.inv(m)

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    eye = jnp.eye(dim, dtype=jnp.float64)
    np.testing.assert_allclose(actual @ m, eye, atol=1e-12)
    np.testing.assert_allclose(m @ actual, eye, atol=1e-12)

  @parameterized.parameters(1, 2, 3, 4)
  def test_batched_invert_matches_linalg_inv(self, dim: int):
    """matrix_inverse handles batched matrices correctly."""
    batch_size = 7
    key = jax.random.key(100 + dim)
    m = jax.random.normal(key, (batch_size, dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    actual = matrix_inverse.fast_matrix_inverse(m)
    expected = jnp.linalg.inv(m)

    np.testing.assert_allclose(actual, expected, atol=1e-12)

  @parameterized.parameters(1, 2, 3, 4)
  def test_invert_identity(self, dim: int):
    """Inverse of identity is identity."""
    eye = jnp.eye(dim, dtype=jnp.float64)
    actual = matrix_inverse.fast_matrix_inverse(eye)
    np.testing.assert_allclose(actual, eye, atol=1e-14)

  @parameterized.parameters(2, 3, 4)
  def test_zero_diagonal_matrix(self, dim: int):
    """Adjugate correctly inverts matrices with zero diagonal entries."""
    # Cyclic permutation matrix has zero diagonal entries everywhere
    p = jnp.roll(jnp.eye(dim, dtype=jnp.float64), shift=1, axis=0)
    actual = matrix_inverse.fast_matrix_inverse(p)
    eye = jnp.eye(dim, dtype=jnp.float64)
    np.testing.assert_allclose(actual @ p, eye, atol=1e-14)
    np.testing.assert_allclose(p @ actual, eye, atol=1e-14)

  @parameterized.parameters(1, 2, 3, 4)
  def test_jit_compiled(self, dim: int):
    """matrix_inverse works under jax.jit."""
    key = jax.random.key(400 + dim)
    m = jax.random.normal(key, (dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    jit_inv = jax.jit(matrix_inverse.fast_matrix_inverse)
    actual = jit_inv(m)
    expected = jnp.linalg.inv(m)

    np.testing.assert_allclose(actual, expected, atol=1e-12)

  @parameterized.parameters(1, 2, 3, 4)
  def test_vmap(self, dim: int):
    """matrix_inverse works under jax.vmap."""
    key = jax.random.key(500 + dim)
    m = jax.random.normal(key, (5, dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    vmap_inv = jax.vmap(matrix_inverse.fast_matrix_inverse)
    actual = vmap_inv(m)
    expected = jnp.linalg.inv(m)

    np.testing.assert_allclose(actual, expected, atol=1e-12)

  def test_multidimensional_batch(self):
    """matrix_inverse handles multidimensional leading batch axes."""
    dim = 4
    key = jax.random.key(200)
    m = jax.random.normal(key, (2, 3, dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    actual = matrix_inverse.fast_matrix_inverse(m)
    expected = jnp.linalg.inv(m)

    np.testing.assert_allclose(actual, expected, atol=1e-12)

  @parameterized.parameters(5, 6)
  def test_fallback_for_large_dim(self, dim: int):
    """Falls back to jnp.linalg.inv for dimensions > 4."""
    key = jax.random.key(300 + dim)
    m = jax.random.normal(key, (dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    actual = matrix_inverse.fast_matrix_inverse(m)
    expected = jnp.linalg.inv(m)

    np.testing.assert_allclose(actual, expected, atol=1e-12)

  @parameterized.parameters(1, 2, 3, 4)
  def test_cramer_inverse_grad_matches_linalg_inv(self, dim: int):
    """Gradients of _cramer_inverse match jnp.linalg.inv."""
    key = jax.random.key(600 + dim)
    m = jax.random.normal(key, (dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    # Linear loss
    grad_cramer_lin = jax.grad(
        lambda x: jnp.sum(matrix_inverse._cramer_inverse(x))
    )(m)
    grad_linalg_lin = jax.grad(lambda x: jnp.sum(jnp.linalg.inv(x)))(m)
    np.testing.assert_allclose(
        grad_cramer_lin, grad_linalg_lin, atol=1e-11, rtol=1e-11
    )

    # Non-linear quadratic loss
    grad_cramer_sq = jax.grad(
        lambda x: jnp.sum(matrix_inverse._cramer_inverse(x) ** 2)
    )(m)
    grad_linalg_sq = jax.grad(lambda x: jnp.sum(jnp.linalg.inv(x) ** 2))(m)
    np.testing.assert_allclose(
        grad_cramer_sq, grad_linalg_sq, atol=1e-11, rtol=1e-11
    )

  @parameterized.parameters(1, 2, 3, 4)
  def test_cramer_inverse_jvp_matches_linalg_inv(self, dim: int):
    """JVP of _cramer_inverse matches jnp.linalg.inv."""
    key1 = jax.random.key(700 + dim)
    key2 = jax.random.key(800 + dim)
    m = jax.random.normal(key1, (dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)
    m_dot = jax.random.normal(key2, (dim, dim), dtype=jnp.float64)

    primal_cramer, tangent_cramer = jax.jvp(
        matrix_inverse._cramer_inverse, (m,), (m_dot,)
    )
    primal_linalg, tangent_linalg = jax.jvp(jnp.linalg.inv, (m,), (m_dot,))

    np.testing.assert_allclose(primal_cramer, primal_linalg, atol=1e-12)
    np.testing.assert_allclose(
        tangent_cramer, tangent_linalg, atol=1e-11, rtol=1e-11
    )

  @parameterized.parameters(1, 2, 3, 4)
  def test_cramer_inverse_vjp_matches_linalg_inv(self, dim: int):
    """VJP of _cramer_inverse matches jnp.linalg.inv."""
    key1 = jax.random.key(900 + dim)
    key2 = jax.random.key(1000 + dim)
    m = jax.random.normal(key1, (dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)
    cotangent = jax.random.normal(key2, (dim, dim), dtype=jnp.float64)

    primal_cramer, vjp_cramer = jax.vjp(matrix_inverse._cramer_inverse, m)
    primal_linalg, vjp_linalg = jax.vjp(jnp.linalg.inv, m)

    (grad_cramer,) = vjp_cramer(cotangent)
    (grad_linalg,) = vjp_linalg(cotangent)

    np.testing.assert_allclose(primal_cramer, primal_linalg, atol=1e-12)
    np.testing.assert_allclose(grad_cramer, grad_linalg, atol=1e-11, rtol=1e-11)

  @parameterized.parameters(1, 2, 3, 4)
  def test_cramer_inverse_batched_grad(self, dim: int):
    """Batched gradients of _cramer_inverse match jnp.linalg.inv."""
    batch_size = 5
    key = jax.random.key(1100 + dim)
    m = jax.random.normal(key, (batch_size, dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    grad_cramer = jax.grad(
        lambda x: jnp.sum(matrix_inverse._cramer_inverse(x) ** 2)
    )(m)
    grad_linalg = jax.grad(lambda x: jnp.sum(jnp.linalg.inv(x) ** 2))(m)

    np.testing.assert_allclose(grad_cramer, grad_linalg, atol=1e-11, rtol=1e-11)

  @parameterized.parameters(1, 2, 3, 4)
  def test_cramer_inverse_grad_jit_and_vmap(self, dim: int):
    """Gradients of _cramer_inverse work under jax.jit and jax.vmap."""
    key = jax.random.key(1200 + dim)
    m = jax.random.normal(key, (3, dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    grad_fn_cramer = jax.jit(
        jax.vmap(
            jax.grad(lambda x: jnp.sum(matrix_inverse._cramer_inverse(x) ** 2))
        )
    )
    grad_fn_linalg = jax.jit(
        jax.vmap(jax.grad(lambda x: jnp.sum(jnp.linalg.inv(x) ** 2)))
    )

    actual = grad_fn_cramer(m)
    expected = grad_fn_linalg(m)

    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)

  @parameterized.parameters(1, 2, 3, 4)
  def test_fast_matrix_inverse_grad(self, dim: int):
    """Gradients of fast_matrix_inverse match jnp.linalg.inv."""
    key = jax.random.key(1300 + dim)
    m = jax.random.normal(key, (dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    grad_fast = jax.grad(
        lambda x: jnp.sum(matrix_inverse.fast_matrix_inverse(x) ** 2)
    )(m)
    grad_linalg = jax.grad(lambda x: jnp.sum(jnp.linalg.inv(x) ** 2))(m)

    np.testing.assert_allclose(grad_fast, grad_linalg, atol=1e-11, rtol=1e-11)

  @parameterized.parameters(1, 2, 3, 4)
  def test_inv_implementations(self, dim: int):
    """inv with 'solve' and 'cramer' matches jnp.linalg.inv."""
    key = jax.random.key(1400 + dim)
    m = jax.random.normal(key, (dim, dim), dtype=jnp.float64)
    m = m + 5.0 * jnp.eye(dim, dtype=jnp.float64)

    expected = jnp.linalg.inv(m)
    actual_solve = matrix_inverse.inv(m, implementation='solve')
    actual_cramer = matrix_inverse.inv(m, implementation='cramer')

    np.testing.assert_allclose(actual_solve, expected, atol=1e-12)
    np.testing.assert_allclose(actual_cramer, expected, atol=1e-12)

  def test_inv_invalid_implementation(self):
    """inv raises ValueError on invalid implementation."""
    m = jnp.eye(2, dtype=jnp.float64)
    with self.assertRaises(ValueError):
      matrix_inverse.inv(
          m, implementation=cast(matrix_inverse.Implementations, 'unsupported')
      )

  def test_cramer_inverse_invalid_dimension(self):
    """_cramer_inverse raises ValueError for dimension > 4."""
    m = jnp.eye(5, dtype=jnp.float64)
    with self.assertRaises(ValueError):
      matrix_inverse._cramer_inverse(m)


if __name__ == '__main__':
  absltest.main()
