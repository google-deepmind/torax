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
from jax import numpy as jnp
from torax import jax_utils


class JaxUtilsTest(parameterized.TestCase):

  def setUp(self):
    """Clear the get_dtype and get_int_dtype caches before each test."""
    super().setUp()
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


if __name__ == '__main__':
  absltest.main()
