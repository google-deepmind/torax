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

"""Unit tests for torax.jax_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
from torax import jax_utils


class JaxUtilsTest(parameterized.TestCase):
  """Unit tests for the `torax.jax_utils` module."""

  def _should_error(self):
    """Assert that errors are on."""
    x = jnp.array(0)
    cond = x == 0

    with self.assertRaises(jax.lib.xla_extension.XlaRuntimeError):
      x = jax_utils.error_if(x, cond, msg="")

  def _should_not_error(self):
    """Call error_if, expecting it to be disabled.

    Because we don't catch any exceptions, the test will fail if it
    is not actually disabled.
    """
    x = jnp.array(0)
    cond = x == 0

    x = jax_utils.error_if(x, cond, msg="")

  def test_enable_errors(self):
    """Test that jax_utils.enable_errors enables / disables errors."""

    # Errors should be on by default
    self._should_error()

    # Test that we can turn them off
    with jax_utils.enable_errors(False):
      self._should_not_error()

      # Test that we can turn them back on explicitly
      with jax_utils.enable_errors(True):
        self._should_error()

      # Now test that the stack unwinds correctly

      self._should_not_error()

    self._should_error()

  def test_jit(self):
    """Test for jax_utils.jit."""

    x1 = jnp.array(3.3)

    def f(x, y):
      if y:
        return jnp.sin(x)
      else:
        return jnp.cos(x)

    out_ref = f(x1, y=True)

    f_jit = jax_utils.jit(f, static_argnames=["y"], assert_max_traces=1)
    self.assertTrue(hasattr(f_jit, "lower"))
    chex.assert_trees_all_close(f_jit(x1, y=True), out_ref)

    with self.assertRaises(AssertionError):
      f_jit(jnp.array([3.3]), y=False)


if __name__ == "__main__":
  absltest.main()
