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
import numpy as np
from torax._src import jax_utils


class JaxUtilsTest(parameterized.TestCase):

  def setUp(self):
    """Clear the get_dtype and get_int_dtype caches before each test."""
    super().setUp()
    jax_utils.get_dtype.cache_clear()
    jax_utils.get_int_dtype.cache_clear()
    jax_utils.get_np_dtype.cache_clear()

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

  @parameterized.parameters(['while_loop', 'pure_callback'])
  def test_non_inlined_function(self, implementation):

    @jax.jit(static_argnames='z')
    def f(x, z, y=2.0):
      if z == 'left':
        return x['temp1'] * y + jnp.sin(x['temp2'])
      else:
        return x['temp1'] + jnp.cos(x['temp2'])

    # Verify that this is JITable.
    f_non_inlined = jax.jit(
        jax_utils.non_inlined_function(f=f, implementation=implementation),
        static_argnames=['z'],
    )

    x = {'temp1': jnp.array(1.3), 'temp2': jnp.array(2.6)}
    chex.assert_trees_all_close(f_non_inlined(x, z='left'), f(x, z='left'))

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

  def test_max_steps_while_loop(self):
    terminating_step = 4

    def cond_fun(state):
      i, _ = state
      return i < terminating_step

    def body_fun(state):
      i, value = state
      next_i = i + 1
      next_value = jnp.sin(value)
      return next_i, next_value

    init_state = (0, 0.5)
    max_steps = 10

    with self.subTest('forward_agrees_with_while_loop'):
      output_state = jax_utils.while_loop_bounded(
          cond_fun, body_fun, init_state, max_steps
      )
      chex.assert_trees_all_close(
          output_state, jax.lax.while_loop(cond_fun, body_fun, init_state)
      )

    def f_while(x, max_steps=max_steps):
      init_state = (0, x)
      return jax_utils.while_loop_bounded(
          cond_fun, body_fun, init_state, max_steps=max_steps
      )[1]

    def f(x, n_times=terminating_step):
      result = x
      for _ in range(n_times):
        result = jnp.sin(result)
      return result

    with self.subTest('forward_agrees_with_explicit'):
      chex.assert_trees_all_close(f_while(0.5), f(0.5))
    with self.subTest('grad_agrees_with_explicit'):
      chex.assert_trees_all_close(jax.grad(f_while)(0.5), jax.grad(f)(0.5))

    with self.subTest('max_steps_is_respected'):
      final_i, final_value = jax_utils.while_loop_bounded(
          cond_fun, body_fun, init_state, max_steps=2
      )
      self.assertEqual(final_i, 2)
      chex.assert_trees_all_close(final_value, f(0.5, n_times=2))
      chex.assert_trees_all_close(
          jax.grad(f_while)(0.5, max_steps=2), jax.grad(f)(0.5, n_times=2)
      )
      chex.assert_trees_all_close(f_while(0.5, max_steps=3), f(0.5, n_times=3))

  # --- Tests for get_np_dtype ---

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_default_np_dtype(self):
    """Test that the default np dtype is float64."""
    self.assertEqual(jax_utils.get_np_dtype(), np.float64)

  @mock.patch.dict(os.environ, {'JAX_PRECISION': 'f64'})
  def test_f64_np_dtype(self):
    """Test np dtype is float64 when JAX_PRECISION is 'f64'."""
    self.assertEqual(jax_utils.get_np_dtype(), np.float64)

  @mock.patch.dict(os.environ, {'JAX_PRECISION': 'f32'})
  def test_f32_np_dtype(self):
    """Test np dtype is float32 when JAX_PRECISION is 'f32'."""
    self.assertEqual(jax_utils.get_np_dtype(), np.float32)

  @mock.patch.dict(os.environ, {'JAX_PRECISION': 'f16'})
  def test_invalid_np_dtype(self):
    """Test an assertion error is raised for an invalid precision."""
    with self.assertRaisesRegex(
        AssertionError, r'Unknown JAX precision environment variable'
    ):
      jax_utils.get_np_dtype()

  # --- Tests for env_bool ---

  @mock.patch.dict(os.environ, {'MY_FLAG': 'True'})
  def test_env_bool_true_string(self):
    """Test env_bool returns True for 'True'."""
    self.assertTrue(jax_utils.env_bool('MY_FLAG', False))

  @mock.patch.dict(os.environ, {'MY_FLAG': 'true'})
  def test_env_bool_lowercase_true(self):
    """Test env_bool returns True for 'true'."""
    self.assertTrue(jax_utils.env_bool('MY_FLAG', False))

  @mock.patch.dict(os.environ, {'MY_FLAG': '1'})
  def test_env_bool_one_string(self):
    """Test env_bool returns True for '1'."""
    self.assertTrue(jax_utils.env_bool('MY_FLAG', False))

  @mock.patch.dict(os.environ, {'MY_FLAG': 'False'})
  def test_env_bool_false_string(self):
    """Test env_bool returns False for 'False'."""
    self.assertFalse(jax_utils.env_bool('MY_FLAG', True))

  @mock.patch.dict(os.environ, {'MY_FLAG': 'false'})
  def test_env_bool_lowercase_false(self):
    """Test env_bool returns False for 'false'."""
    self.assertFalse(jax_utils.env_bool('MY_FLAG', True))

  @mock.patch.dict(os.environ, {'MY_FLAG': '0'})
  def test_env_bool_zero_string(self):
    """Test env_bool returns False for '0'."""
    self.assertFalse(jax_utils.env_bool('MY_FLAG', True))

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_env_bool_missing_uses_default(self):
    """Test env_bool returns the default when the variable is not set."""
    self.assertTrue(jax_utils.env_bool('NONEXISTENT', True))
    self.assertFalse(jax_utils.env_bool('NONEXISTENT', False))

  @mock.patch.dict(os.environ, {'MY_FLAG': 'maybe'})
  def test_env_bool_invalid_value_raises(self):
    """Test env_bool raises ValueError for unrecognized strings."""
    with self.assertRaisesRegex(ValueError, r'Unrecognized boolean string'):
      jax_utils.env_bool('MY_FLAG', False)

  # --- Tests for assert_rank ---

  def test_assert_rank_correct_rank(self):
    """Test assert_rank passes for a correctly-ranked array."""
    x = jnp.zeros((3, 4))
    jax_utils.assert_rank(x, 2)  # Should not raise.

  def test_assert_rank_wrong_rank(self):
    """Test assert_rank raises for an incorrectly-ranked array."""
    x = jnp.zeros((3, 4))
    with self.assertRaises(AssertionError):
      jax_utils.assert_rank(x, 1)

  def test_assert_rank_scalar(self):
    """Test assert_rank passes for a scalar with rank 0."""
    x = jnp.array(1.0)
    jax_utils.assert_rank(x, 0)  # Should not raise.

  # --- Tests for error_if ---

  def test_error_if_passthrough_when_disabled(self):
    """Test error_if returns the input unchanged when errors are disabled."""
    x = jnp.array(42.0)
    with jax_utils.enable_errors(False):
      result = jax_utils.error_if(x, jnp.array(True), msg='should not fire')
    chex.assert_trees_all_equal(result, x)

  def test_error_if_passthrough_identity(self):
    """Test that error_if returns the exact same array object when disabled."""
    x = jnp.array(7.0)
    with jax_utils.enable_errors(False):
      result = jax_utils.error_if(x, jnp.array(True), msg='noop')
    # When errors are disabled, the function is a no-op passthrough.
    self.assertIs(result, x)


if __name__ == '__main__':
  absltest.main()
