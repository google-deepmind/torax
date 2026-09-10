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

  @parameterized.named_parameters(
      dict(
          testcase_name='maxiter',
          maxiter=2,
          atol=0.0,
          rtol=0.0,
      ),
      dict(
          testcase_name='atol',
          maxiter=500,
          atol=1e-8,
          rtol=0.0,
      ),
      dict(
          testcase_name='rtol',
          maxiter=500,
          atol=0.0,
          rtol=1e-5,
      ),
  )
  def test_fixed_point_convergence(self, maxiter, atol, rtol):
    c1 = np.array([10, 12.0])
    c2 = np.array([3, 5.0])
    x = np.array([1.2, 1.3])

    out_jnp = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
    )

    # Scipy's fixed_point raises a RuntimeError if the maximum number of
    # iterations is reached. As a fallback, we can extract the expected output
    # from the error message.
    try:
      out_expected = optimize.fixed_point(
          _func_np,
          x,
          args=(c1, c2),
          maxiter=maxiter,
          method='iteration',
      )
    except RuntimeError as e:
      out_expected = np.array(
          [float(f) for f in re.split(r'\[|]|\s+', e.args[0])[-3:-1]]
      )

    chex.assert_trees_all_close(out_expected, out_jnp)


if __name__ == '__main__':
  absltest.main()
