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
"""Polymorphic JAX/NumPy module."""

import contextlib
import functools
import threading
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils

# Export all symbols from jax.numpy API for type checkers (including editors).
if TYPE_CHECKING:
  # pylint: disable=wildcard-import
  # pylint: disable=g-bad-import-order
  # pylint: disable=g-import-not-at-top
  from jax.numpy import *
  # pylint: enable=wildcard-import
  # pylint: enable=g-bad-import-order
  # pylint: enable=g-import-not-at-top


# Create thread-local storage.
thread_context = threading.local()


@contextlib.contextmanager
def _jit_context():
  """A context manager that sets a thread-local flag to indicate JAX context."""
  original_is_jax = getattr(thread_context, 'is_jax', None)
  thread_context.is_jax = True
  try:
    yield
  finally:
    if original_is_jax is None:
      del thread_context.is_jax
    else:
      thread_context.is_jax = original_is_jax


def jit(*args, **kwargs):
  """JAX jit wrapper that sets the thread-local flag to indicate JAX context."""
  func = args[0]

  @functools.wraps(func)
  def wrapped_func(*func_args, **func_kwargs):
    with _jit_context():
      return func(*func_args, **func_kwargs)

  # if EXPERIMENTAL_COMPILE is not set use default `False` and do not JIT.
  if jax_utils.env_bool('EXPERIMENTAL_COMPILE', False):
    return jax.jit(wrapped_func, *args[1:], **kwargs)
  else:
    return func


def _get_current_lib():
  """Determines whether to use JAX or NumPy."""
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    return jnp
  else:
    return np


def __getattr__(name):  # pylint: disable=invalid-name
  """Returns the corresponding function from the current library (jnp or np)."""
  current_lib = _get_current_lib()
  try:
    return getattr(current_lib, name)
  except AttributeError as exc:
    raise AttributeError(f"Module 'xnp' has no attribute '{name}'") from exc


def __dir__():  # pylint: disable=invalid-name
  """Provides a list of potential attributes for code completion."""
  common_attributes = set(dir(jnp)) | set(dir(np))

  return sorted(list(common_attributes))
