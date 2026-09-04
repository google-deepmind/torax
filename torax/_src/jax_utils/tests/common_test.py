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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
from torax._src import jax_utils


class JaxUtilsTest(parameterized.TestCase):

  def setUp(self):
    """Clear the get_dtype and get_int_dtype caches before each test."""
    super().setUp()
    jax_utils.get_dtype.cache_clear()
    jax_utils.get_int_dtype.cache_clear()

  def tearDown(self):
    """Clear the get_dtype and get_int_dtype caches after each test."""
    super().tearDown()
    jax_utils.get_dtype.cache_clear()
    jax_utils.get_int_dtype.cache_clear()

  def _should_error(self):
    """Assert that errors are on."""
    x = jnp.array(0)
    cond = x == 0

    with self.assertRaises(RuntimeError):
      jax_utils.error_if(x, cond, msg='')

  def _should_not_error(self):
    """Call error_if, expecting it to be disabled.

    Because we don't catch any exceptions, the test will fail if it
    is not actually disabled.
    """
    x = jnp.array(0)
    cond = x == 0

    jax_utils.error_if(x, cond, msg='')

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

  def test_error_if_raises_under_jit(self):
    """Test that error_if raises RuntimeError under jax.jit."""

    @jax.jit
    def f(x):
      return jax_utils.error_if(x, x < 0, 'x must be non-negative')

    with self.assertRaises(RuntimeError):
      f(jnp.array(-1.0))

  def test_error_if_passes_under_jit(self):
    """Test that error_if passes without error under jax.jit."""

    @jax.jit
    def f(x):
      return jax_utils.error_if(x, x < 0, 'x must be non-negative')

    result = f(jnp.array(1.0))
    chex.assert_trees_all_equal(result, jnp.array(1.0))

  def test_error_if_compatible_with_grad(self):
    """Test that error_if is compatible with jax.grad under jit."""

    @jax.jit
    def f(x):
      x = jax_utils.error_if(x, x < 0, 'x must be non-negative')
      return x**2

    chex.assert_trees_all_close(jax.grad(f)(jnp.array(3.0)), jnp.array(6.0))

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_default_dtype(self):
    """Test that the default dtype is float64 when JAX_PRECISION is not set."""
    self.assertEqual(jax_utils.get_dtype(), jnp.float64)

  @mock.patch.dict(os.environ, {'JAX_PRECISION': 'f64'})
  def test_f64_dtype(self):
    """Test that the dtype is float64 when JAX_PRECISION is set to 'f64'."""
    self.assertEqual(jax_utils.get_dtype(), jnp.float64)

  @mock.patch.dict(os.environ, {'JAX_PRECISION': 'f32'})
  def test_f32_dtype(self):
    """Test that the dtype is float32 when JAX_PRECISION is set to 'f32'."""
    self.assertEqual(jax_utils.get_dtype(), jnp.float32)

  @mock.patch.dict(os.environ, {'JAX_PRECISION': 'f16'})
  def test_empty_dtype(self):
    """Test an assertion error is raised for an invalid value."""
    with self.assertRaisesRegex(
        AssertionError, r'Unknown JAX precision environment variable'
    ):
      jax_utils.get_dtype()

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_default_int_dtype(self):
    """Test that the default dtype is int64 when JAX_PRECISION is not set."""
    self.assertEqual(jax_utils.get_int_dtype(), jnp.int64)

  @mock.patch.dict(os.environ, {'JAX_PRECISION': 'f64'})
  def test_f64_int_dtype(self):
    """Test that the dtype is int64 when JAX_PRECISION is set to 'f64'."""
    self.assertEqual(jax_utils.get_int_dtype(), jnp.int64)

  @mock.patch.dict(os.environ, {'JAX_PRECISION': 'f32'})
  def test_f32_int_dtype(self):
    """Test that the dtype is int32 when JAX_PRECISION is set to 'f32'."""
    self.assertEqual(jax_utils.get_int_dtype(), jnp.int32)

  def test_get_number_of_compiles(self):
    """Check assumptions on JAX internals are valid."""

    def f(x: jax.Array):
      return x

    jit_f = jax.jit(f)
    self.assertTrue(hasattr(jit_f, '_cache_size'))
    # Should be 0 before any calls.
    self.assertEqual(jax_utils.get_number_of_compiles(jit_f), 0)

    # Should be 1 after one call.
    jit_f(jnp.array(0))
    self.assertEqual(jax_utils.get_number_of_compiles(jit_f), 1)
    # Should be 1 after another call with same shape.
    jit_f(jnp.array(1))
    self.assertEqual(jax_utils.get_number_of_compiles(jit_f), 1)

    # Should be 2 after another call with different shape.
    jit_f(jnp.array([1]))
    self.assertEqual(jax_utils.get_number_of_compiles(jit_f), 2)

  @parameterized.parameters(['map', 'vectorize'])
  def test_batched_cond(self, implementation):
    pred = jnp.array([True, False])
    x = jnp.array([[2, 3.0, 4.0], [5.0, 6.0, 7.0]])
    out = jax_utils.batched_cond(
        pred=pred,
        true_fun=lambda x, y: x * y,
        false_fun=lambda x, y: x * y**2,
        operands=(x, x),
        implementation=implementation,
    )
    out_gt = jnp.array(
        [[4.0, 9.0, 16.0], [125.0, 216.0, 343.0]], dtype=jnp.float32
    )
    chex.assert_trees_all_equal(out, out_gt)

  @parameterized.parameters(['map', 'vectorize'])
  def test_batched_cond_concrete_special(self, implementation):
    pred = jnp.array([True])
    x = jnp.array([[2, 3.0, 4.0]])

    @jax.jit
    def f(x):
      return jax_utils.batched_cond(
          pred=pred,
          true_fun=lambda x, y: x * y,
          false_fun=lambda x, y: x * y**2,
          operands=(x, x),
          implementation=implementation,
      )

    out = f(x)
    out_gt = jnp.array([[4.0, 9.0, 16.0]], dtype=jnp.float32)
    chex.assert_trees_all_equal(out, out_gt)


if __name__ == '__main__':
  absltest.main()
