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
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize
from torax._src.solver import jax_root_finding


# Adapted from the example in:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
def function_to_find_root(x, a, b):
  array_construct = jnp.array if isinstance(x, jax.Array) else np.array
  return array_construct(
      [
          x[0] + 0.5 * (x[0] - b * x[1]) ** 3.0 - 1.0,
          a * (x[1] - b * x[0]) ** 3.0 + x[1],
      ],
      dtype=x.dtype,
  )


class NewtonRaphsonSolveBlockTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)

  @parameterized.named_parameters(
      # All search directions are positive.
      ('positive_search', 0.5, 0.1, (0.0, 0.0)),
      # Assume find the root even with a negative search direction.
      ('negative_search', 1.0, 1.0, (2.0, 1.0)),
  )
  def test_root_newton_raphson_basic(
      self, a: float, b: float, x0: tuple[float, float]):
    dtype = np.float64
    tol = 1e-9
    f_closed = functools.partial(function_to_find_root, a=a, b=b)

    x_init = np.array(x0, dtype=dtype)
    sol_np = optimize.root(f_closed, x0, tol=tol)

    @jax.jit(static_argnames=['tol', 'maxiter'])
    def root_jax(x, tol, maxiter):
      return jax_root_finding.root_newton_raphson(
          f_closed, x, tol=tol, maxiter=maxiter
      )

    sol_jax, metadata = root_jax(x_init, tol=tol, maxiter=100)

    with self.subTest('solver_correctness_against_scipy'):
      chex.assert_trees_all_close(sol_np.x, sol_jax, atol=tol)
      self.assertFalse(bool(metadata.error))

    with self.subTest('auxiliary_data'):
      self.assertGreater(int(metadata.iterations), 0)
      self.assertEqual(int(metadata.error), 0)

    with self.subTest('maxiter'):
      _, metadata = root_jax(x_init, tol=tol, maxiter=1)
      self.assertEqual(int(metadata.iterations), 1)
      self.assertEqual(int(metadata.error), 1)
      self.assertTrue(jnp.isdtype(metadata.iterations, 'integral'))

    def loss(x, a, b):
      root = jax_root_finding.root_newton_raphson(
          functools.partial(function_to_find_root, a=a, b=b), x, tol=tol
      )[0]
      return jnp.sum(root**2)

    eps = 1e-4
    a_grad_diff = (loss(x_init, a + eps, b) - loss(sol_np.x, a - eps, b)) / (
        2 * eps
    )
    b_grad_diff = (loss(x_init, a, b + eps) - loss(sol_np.x, a, b - eps)) / (
        2 * eps
    )
    x_grad, a_grad, b_grad = jax.grad(loss, argnums=(0, 1, 2))(x_init, a, b)

    with self.subTest('gradient_correctness'):
      chex.assert_trees_all_equal(x_grad, jnp.array([0.0, 0.0], dtype=dtype))
      chex.assert_trees_all_close(a_grad, a_grad_diff, atol=1e-4)
      chex.assert_trees_all_close(b_grad, b_grad_diff, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
