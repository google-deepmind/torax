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

import re
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize
from torax._src.solver import jax_fixed_point


def _func_np(x, c1, c2):
  return np.sqrt(c1 / (x + c2))


def _func_jnp(x, c1, c2):
  return jnp.sqrt(c1 / (x + c2))


class FixedPointTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)

  @parameterized.product(
      method=['del2', 'iteration'], maxiter=[1, 2, 500], xtol=[1e-08, 1e-3, 1.0]
  )
  def test_fixed_point_basic(self, method, maxiter, xtol):

    c1 = np.array([10, 12.0])
    c2 = np.array([3, 5.0])
    x = np.array([1.2, 1.3])

    # If there is no convergence, the SciPy implementation will raise a
    # RuntimeError of the form
    # 'Failed to converge after 1 iterations, value is [1.49240205 1.37228787]'
    # Extract the value from the message.
    try:
      out_np = optimize.fixed_point(
          _func_np, x, args=(c1, c2), method=method, maxiter=maxiter, xtol=xtol
      )
    except RuntimeError as e:
      out_np = np.array(
          [float(f) for f in re.split(r'\[|]|\s+', e.args[0])[-3:-1]]
      )

    @jax.jit
    def fixed_point(x):
      return jax_fixed_point.fixed_point(
          _func_jnp, x, args=(c1, c2), method=method, maxiter=maxiter, xtol=xtol
      )

    out_jnp = fixed_point(x)
    chex.assert_trees_all_close(out_np, out_jnp, atol=1e-8)

  def test_fixed_point_none(self):
    c1 = np.array([10, 12.0])
    c2 = np.array([3, 5.0])
    x = np.array([1.2, 1.3])
    maxiter = 100

    out_expected = x
    for _ in range(maxiter):
      out_expected = _func_jnp(out_expected, c1, c2)

    out_jnp = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        method='iteration',
        maxiter=maxiter,
        xtol=None,
    )
    chex.assert_trees_all_close(out_expected, out_jnp, atol=1e-8)

  def test_fixed_point_residual_norm(self):
    c1 = np.array([10, 12.0])
    c2 = np.array([3, 5.0])
    x = np.array([1.2, 1.3])

    # Test with atol
    out_jnp_atol = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        method='iteration',
        maxiter=500,
        atol=1e-5,
        xtol=None,
    )

    # Verify it gives close result to standard fixed point
    out_expected = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        method='iteration',
        maxiter=500,
        xtol=1e-5,
    )
    chex.assert_trees_all_close(out_expected, out_jnp_atol, atol=1e-5)

    # Test with rtol
    out_jnp_rtol = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        method='iteration',
        maxiter=500,
        rtol=1e-5,
        xtol=None,
    )
    chex.assert_trees_all_close(out_expected, out_jnp_rtol, atol=1e-5)

  def test_fixed_point_backtracking(self):
    c1 = np.array([10, 12.0])
    c2 = np.array([3, 5.0])
    x = np.array([1.2, 1.3])

    out_with_backtracking = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        method='iteration',
        maxiter=500,
        use_backtracking=True,
        delta_reduction_factor=0.5,
        max_backtrack_steps=5,
        atol=1e-5,
        xtol=None,
    )

    out_without_backtracking = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        method='iteration',
        maxiter=500,
        use_backtracking=False,
        atol=1e-5,
        xtol=None,
    )
    chex.assert_trees_all_close(
        out_with_backtracking, out_without_backtracking, atol=1e-5
    )


if __name__ == '__main__':
  absltest.main()
