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


if __name__ == '__main__':
  absltest.main()
