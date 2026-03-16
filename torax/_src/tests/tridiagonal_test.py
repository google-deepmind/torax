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

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax._src import tridiagonal


class TridiagonalTest(absltest.TestCase):

  def test_tridiag_to_dense(self):
    diag = jnp.array([1.0, 2.0, 3.0])
    above = jnp.array([4.0, 5.0])
    below = jnp.array([6.0, 7.0])

    tri_mat = tridiagonal.TriDiagonal(diag, above, below)
    dense_mat = tri_mat.to_dense()

    expected_dense = jnp.array([
        [1.0, 4.0, 0.0],
        [6.0, 2.0, 5.0],
        [0.0, 7.0, 3.0],
    ])

    np.testing.assert_array_equal(dense_mat, expected_dense)

  def test_tridiag_add(self):
    tri_mat_1 = tridiagonal.TriDiagonal(
        diagonal=jnp.array([1.0, 2.0, 3.0]),
        above=jnp.array([4.0, 5.0]),
        below=jnp.array([6.0, 7.0]),
    )
    tri_mat_2 = tridiagonal.TriDiagonal(
        diagonal=jnp.array([10.0, 20.0, 30.0]),
        above=jnp.array([40.0, 50.0]),
        below=jnp.array([60.0, 70.0]),
    )

    sum_tri_mat = tri_mat_1 + tri_mat_2

    expected_diag = jnp.array([11.0, 22.0, 33.0])
    expected_above = jnp.array([44.0, 55.0])
    expected_below = jnp.array([66.0, 77.0])

    np.testing.assert_array_equal(sum_tri_mat.diagonal, expected_diag)
    np.testing.assert_array_equal(sum_tri_mat.above, expected_above)
    np.testing.assert_array_equal(sum_tri_mat.below, expected_below)

  def test_tridiag_matvec(self):
    diag = jnp.array([1.0, 2.0, 3.0])
    above = jnp.array([4.0, 5.0])
    below = jnp.array([6.0, 7.0])
    tri_mat = tridiagonal.TriDiagonal(diag, above, below)
    x = jnp.array([1.0, 2.0, 3.0])

    result = tri_mat.matvec(x)
    expected = tri_mat.to_dense() @ x

    np.testing.assert_allclose(result, expected)


class BlockTriDiagonalTest(absltest.TestCase):

  def _make_block_tridiag(
      self, num_blocks: int, block_size: int
  ) -> tridiagonal.BlockTriDiagonal:
    """Helper to create a BlockTriDiagonal with deterministic values."""
    rng = np.random.RandomState(42)
    lower = jnp.array(
        rng.randn(num_blocks - 1, block_size, block_size), dtype=jnp.float64
    )
    diag_blocks = jnp.array(
        rng.randn(num_blocks, block_size, block_size), dtype=jnp.float64
    )
    upper = jnp.array(
        rng.randn(num_blocks - 1, block_size, block_size), dtype=jnp.float64
    )
    return tridiagonal.BlockTriDiagonal(
        lower=lower, diagonal=diag_blocks, upper=upper
    )

  def _make_nonsingular_block_tridiag(
      self, num_blocks: int, block_size: int
  ) -> tridiagonal.BlockTriDiagonal:
    """Helper to create a diagonally-dominant BlockTriDiagonal for solve."""
    rng = np.random.RandomState(0)
    lower = jnp.array(
        rng.randn(num_blocks - 1, block_size, block_size), dtype=jnp.float64
    )
    upper = jnp.array(
        rng.randn(num_blocks - 1, block_size, block_size), dtype=jnp.float64
    )
    # Make diagonal blocks diagonally dominant to ensure non-singularity.
    diag_blocks = jnp.array(
        rng.randn(num_blocks, block_size, block_size), dtype=jnp.float64
    )
    diag_blocks = diag_blocks + 10.0 * jnp.eye(block_size, dtype=jnp.float64)
    return tridiagonal.BlockTriDiagonal(
        lower=lower, diagonal=diag_blocks, upper=upper
    )

  def test_num_blocks_and_block_size(self):
    bt = self._make_block_tridiag(num_blocks=4, block_size=3)
    self.assertEqual(bt.num_blocks, 4)
    self.assertEqual(bt.block_size, 3)

  def test_zeros(self):
    bt = tridiagonal.BlockTriDiagonal.zeros(
        num_blocks=3, block_size=2, dtype=jnp.float64
    )
    np.testing.assert_array_equal(bt.lower, jnp.zeros((2, 2, 2)))
    np.testing.assert_array_equal(bt.diagonal, jnp.zeros((3, 2, 2)))
    np.testing.assert_array_equal(bt.upper, jnp.zeros((2, 2, 2)))

  def test_from_block_diagonal(self):
    vals = jnp.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]],
    ])
    bt = tridiagonal.BlockTriDiagonal.from_block_diagonal(vals)

    np.testing.assert_array_equal(bt.diagonal, vals)
    np.testing.assert_array_equal(bt.lower, jnp.zeros((2, 2, 2)))
    np.testing.assert_array_equal(bt.upper, jnp.zeros((2, 2, 2)))

  def test_from_diagonal(self):
    vals = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bt = tridiagonal.BlockTriDiagonal.from_diagonal(vals)

    # Each block should be a diagonal matrix.
    for i in range(3):
      expected_block = jnp.diag(vals[i])
      np.testing.assert_array_equal(bt.diagonal[i], expected_block)

    # Off-diagonals should be zero.
    np.testing.assert_array_equal(bt.lower, jnp.zeros((2, 2, 2)))
    np.testing.assert_array_equal(bt.upper, jnp.zeros((2, 2, 2)))

  def test_from_stacked_tridiagonal(self):
    # 3 blocks, 2 channels.
    stacked = tridiagonal.TriDiagonal(
        diagonal=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        above=jnp.array([[7.0, 8.0], [9.0, 10.0]]),
        below=jnp.array([[11.0, 12.0], [13.0, 14.0]]),
    )
    bt = tridiagonal.BlockTriDiagonal.from_stacked_tridiagonal(stacked)

    self.assertEqual(bt.num_blocks, 3)
    self.assertEqual(bt.block_size, 2)

    # Diagonal blocks should be diagonal matrices.
    np.testing.assert_array_equal(
        bt.diagonal[0], jnp.diag(jnp.array([1.0, 2.0]))
    )
    np.testing.assert_array_equal(
        bt.diagonal[1], jnp.diag(jnp.array([3.0, 4.0]))
    )
    np.testing.assert_array_equal(
        bt.diagonal[2], jnp.diag(jnp.array([5.0, 6.0]))
    )

    # Off-diagonal blocks should also be diagonal matrices.
    np.testing.assert_array_equal(bt.upper[0], jnp.diag(jnp.array([7.0, 8.0])))
    np.testing.assert_array_equal(bt.upper[1], jnp.diag(jnp.array([9.0, 10.0])))
    np.testing.assert_array_equal(
        bt.lower[0], jnp.diag(jnp.array([11.0, 12.0]))
    )
    np.testing.assert_array_equal(
        bt.lower[1], jnp.diag(jnp.array([13.0, 14.0]))
    )

  def test_to_dense_single_block(self):
    diag = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    bt = tridiagonal.BlockTriDiagonal(
        lower=jnp.zeros((0, 2, 2)),
        diagonal=diag,
        upper=jnp.zeros((0, 2, 2)),
    )
    dense = bt.to_dense()
    np.testing.assert_array_equal(dense, diag[0])

  def test_to_dense(self):
    bt = self._make_block_tridiag(num_blocks=3, block_size=2)
    dense = bt.to_dense()

    # Verify shape.
    self.assertEqual(dense.shape, (6, 6))

    # Verify diagonal blocks.
    for i in range(3):
      np.testing.assert_array_equal(
          dense[2 * i : 2 * i + 2, 2 * i : 2 * i + 2], bt.diagonal[i]
      )

    # Verify upper blocks.
    for i in range(2):
      np.testing.assert_array_equal(
          dense[2 * i : 2 * i + 2, 2 * (i + 1) : 2 * (i + 1) + 2],
          bt.upper[i],
      )

    # Verify lower blocks.
    for i in range(2):
      np.testing.assert_array_equal(
          dense[2 * (i + 1) : 2 * (i + 1) + 2, 2 * i : 2 * i + 2],
          bt.lower[i],
      )

  def test_add(self):
    bt1 = self._make_block_tridiag(num_blocks=3, block_size=2)
    rng = np.random.RandomState(99)
    bt2 = tridiagonal.BlockTriDiagonal(
        lower=jnp.array(rng.randn(2, 2, 2), dtype=jnp.float64),
        diagonal=jnp.array(rng.randn(3, 2, 2), dtype=jnp.float64),
        upper=jnp.array(rng.randn(2, 2, 2), dtype=jnp.float64),
    )
    result = bt1 + bt2

    np.testing.assert_allclose(result.lower, bt1.lower + bt2.lower)
    np.testing.assert_allclose(result.diagonal, bt1.diagonal + bt2.diagonal)
    np.testing.assert_allclose(result.upper, bt1.upper + bt2.upper)

  def test_add_matches_dense(self):
    bt1 = self._make_block_tridiag(num_blocks=3, block_size=2)
    rng = np.random.RandomState(99)
    bt2 = tridiagonal.BlockTriDiagonal(
        lower=jnp.array(rng.randn(2, 2, 2), dtype=jnp.float64),
        diagonal=jnp.array(rng.randn(3, 2, 2), dtype=jnp.float64),
        upper=jnp.array(rng.randn(2, 2, 2), dtype=jnp.float64),
    )
    result = bt1 + bt2

    np.testing.assert_allclose(
        result.to_dense(), bt1.to_dense() + bt2.to_dense()
    )

  def test_matvec(self):
    bt = self._make_block_tridiag(num_blocks=4, block_size=3)
    rng = np.random.RandomState(7)
    x = jnp.array(rng.randn(4, 3), dtype=jnp.float64)

    result = bt.matvec(x)
    expected = (bt.to_dense() @ x.flatten()).reshape(4, 3)

    np.testing.assert_allclose(result, expected, atol=1e-12)

  def test_matvec_single_block(self):
    diag = jnp.array([[[2.0, 1.0], [0.5, 3.0]]])
    bt = tridiagonal.BlockTriDiagonal(
        lower=jnp.zeros((0, 2, 2)),
        diagonal=diag,
        upper=jnp.zeros((0, 2, 2)),
    )
    x = jnp.array([[1.0, 2.0]])

    result = bt.matvec(x)
    expected = jnp.array([[4.0, 6.5]])

    np.testing.assert_allclose(result, expected)

  def test_solve(self):
    bt = self._make_nonsingular_block_tridiag(num_blocks=4, block_size=3)
    rng = np.random.RandomState(123)
    x_true = jnp.array(rng.randn(4, 3), dtype=jnp.float64)
    rhs = bt.matvec(x_true)

    x_solved = bt.solve(rhs)

    np.testing.assert_allclose(x_solved, x_true, atol=1e-10)

  def test_solve_recovers_rhs(self):
    """Verify A @ solve(A, b) = b."""
    bt = self._make_nonsingular_block_tridiag(num_blocks=3, block_size=2)
    rng = np.random.RandomState(55)
    rhs = jnp.array(rng.randn(3, 2), dtype=jnp.float64)

    x = bt.solve(rhs)
    reconstructed_rhs = bt.matvec(x)

    np.testing.assert_allclose(reconstructed_rhs, rhs, atol=1e-10)

  def test_from_diagonal_to_dense_matches_manual(self):
    """from_diagonal should produce the same dense matrix as manual diag."""
    vals = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bt = tridiagonal.BlockTriDiagonal.from_diagonal(vals)
    dense = bt.to_dense()

    expected = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(dense, expected)

  def test_from_stacked_tridiagonal_to_dense_matches_per_channel(self):
    """Each channel should form an independent scalar tridiagonal system."""
    stacked = tridiagonal.TriDiagonal(
        diagonal=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        above=jnp.array([[7.0, 8.0], [9.0, 10.0]]),
        below=jnp.array([[11.0, 12.0], [13.0, 14.0]]),
    )
    bt = tridiagonal.BlockTriDiagonal.from_stacked_tridiagonal(stacked)
    dense = bt.to_dense()

    # Build per-channel scalar tridiag and interleave.
    ch0 = tridiagonal.TriDiagonal(
        diagonal=jnp.array([1.0, 3.0, 5.0]),
        above=jnp.array([7.0, 9.0]),
        below=jnp.array([11.0, 13.0]),
    )
    ch1 = tridiagonal.TriDiagonal(
        diagonal=jnp.array([2.0, 4.0, 6.0]),
        above=jnp.array([8.0, 10.0]),
        below=jnp.array([12.0, 14.0]),
    )
    d0 = ch0.to_dense()
    d1 = ch1.to_dense()
    # Channel 0 occupies rows/cols 0, 2, 4 and channel 1 occupies 1, 3, 5.
    # But block ordering interleaves them as (ch0, ch1) per block.
    expected_full = jnp.zeros((6, 6))
    for r in range(3):
      for c in range(3):
        expected_full = expected_full.at[2 * r, 2 * c].set(d0[r, c])
        expected_full = expected_full.at[2 * r + 1, 2 * c + 1].set(d1[r, c])

    np.testing.assert_allclose(dense, expected_full)


if __name__ == '__main__':
  absltest.main()
