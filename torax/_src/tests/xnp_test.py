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
import os

import chex
import jax
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from torax._src import jax_utils
from torax._src import xnp


class XNPTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.original_flag = jax_utils.env_bool('EXPERIMENTAL_COMPILE', False)
    os.environ['EXPERIMENTAL_COMPILE'] = 'True'
    if 'jax_enable_x64' in jax.config.values:
      self.original_x64 = jax.config.read('jax_enable_x64')
    else:
      self.original_x64 = None
    jax.config.update('jax_enable_x64', True)

  def tearDown(self):
    super().tearDown()
    os.environ['EXPERIMENTAL_COMPILE'] = str(self.original_flag)
    if self.original_x64 is not None:
      jax.config.update('jax_enable_x64', self.original_x64)

  def test_xnp(self):
    def f(x: chex.Array):
      x = x + 1
      x = xnp.square(x)
      return xnp.sin(x)

    jit_f = xnp.jit(f)
    x = np.random.rand(10)
    jit_output = jit_f(x)
    non_jit_output = f(x)
    self.assertIsInstance(jit_output, jax.Array)
    self.assertIsInstance(non_jit_output, np.ndarray)
    chex.assert_trees_all_equal_shapes_and_dtypes(jit_output, non_jit_output)
    np.testing.assert_allclose(jit_output, non_jit_output)

  def test_xnp_with_experimental_compile_off_and_on(self):
    x = np.random.rand(10)
    # Disable experimental compile.
    os.environ['EXPERIMENTAL_COMPILE'] = 'False'

    def f(x: chex.Array):
      return xnp.sin(x)

    # With compile turned off, we should run with numpy.
    self.assertIsInstance(xnp.jit(f)(x), np.ndarray)

    # Re-enable experimental compile.
    os.environ['EXPERIMENTAL_COMPILE'] = 'True'
    self.assertIsInstance(xnp.jit(f)(x), jax.Array)


if __name__ == '__main__':
  absltest.main()
