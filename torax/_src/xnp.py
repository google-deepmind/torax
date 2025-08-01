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
from typing import Any, Callable, TYPE_CHECKING, TypeVar

from absl import logging as native_logging
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils

# Export all symbols from jax.numpy API for type checkers (including editors).
# pylint: disable=wildcard-import
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
if TYPE_CHECKING:
  from jax.numpy import *
# pylint: enable=wildcard-import
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top


T = TypeVar('T')
BooleanNumeric = Any  # A bool, or a Boolean array.

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


def py_while(
    cond_fun: Callable[list[T], BooleanNumeric],
    body_fun: Callable[list[T], T],
    init_val: T,
) -> T:
  """Pure Python implementation of jax.lax.while_loop.

  This gives us a way to write code that could easily be changed to be
  Jax-compatible in the future (if we want to compute its gradient or
  compile it, etc.) without having to pay the high compile time cost
  of jax.lax.while_loop.

  Args:
    cond_fun: function of type ``a -> Bool``.
    body_fun: function of type ``a -> a``.
    init_val: value of type ``a``, a type that can be a scalar, array, or any
      pytree (nested Python tuple/list/dict) thereof, representing the initial
      loop carry value.

  Returns:
    The output from the final iteration of body_fun, of type ``a``.

  .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
  """

  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val


def while_loop(
    cond_fun: Callable[list[T], BooleanNumeric],
    body_fun: Callable[list[T], T],
    init_val: T,
):
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    return jax.lax.while_loop(cond_fun, body_fun, init_val)
  else:
    return py_while(cond_fun, body_fun, init_val)


# pylint: disable=g-bare-generic
def py_cond(
    cond_val: bool,
    true_fun: Callable,
    false_fun: Callable,
    *operands,
) -> Any:
  """Pure Python implementation of jax.lax.cond.

  This gives us a way to write code that could easily be changed to be
  Jax-compatible in the future, if we want to expand the scope of the jit
  compilation.

  Args:
    cond_val: The condition.
    true_fun: Function to be called if cond==True.
    false_fun: Function to be called if cond==False.
    *operands: The operands to be passed to the functions.

  Returns:
    The output from either true_fun or false_fun.
  """
  if cond_val:
    return true_fun(*operands)
  else:
    return false_fun(*operands)


def cond(
    cond_val: bool,
    true_fun: Callable[..., Any],  # pytype: disable=invalid-annotation
    false_fun: Callable[..., Any],  # pytype: disable=invalid-annotation
    *operands,
) -> Any:
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    return jax.lax.cond(cond_val, true_fun, false_fun, *operands)
  else:
    return py_cond(cond_val, true_fun, false_fun, *operands)


def py_fori_loop(
    lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T  # pytype: disable=invalid-annotation
) -> T:
  """Pure Python implementation of jax.lax.fori_loop.

  This gives us a way to write code that could easily be changed to be
  Jax-compatible in the future, if we want to expand the scope of the jit
  compilation.

  Args:
    lower: lower integer of loop
    upper: upper integer of loop. upper<=lower will produce no iterations.
    body_fun: function of type ``a -> a``.
    init_val: value of type ``a``, a type that can be a scalar, array, or any
      pytree (nested Python tuple/list/dict) thereof, representing the initial
      loop carry value.

  Returns:
    The output from the final iteration of body_fun, of type ``a``.

  .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
  """
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val


def fori_loop(
    lower: int,
    upper: int,
    body_fun: Callable[..., Any],  # pytype: disable=invalid-annotation
    init_val: Any,
):
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    return jax.lax.fori_loop(lower, upper, body_fun, init_val)
  else:
    return py_fori_loop(lower, upper, body_fun, init_val)


def logging(msg: Any, *args: Any, **kwargs: Any):
  """Logging wrapper that works under xnp.jit.

  This function makes use of jax.debug.print when running under xnp.jit,
  otherwise it uses native logging.

  Note, jax.debug.print makes use of jax.debug.callback behind the scenes which:
  - does not guarantee execution.
  https://docs.jax.dev/en/latest/external-callbacks.html#flavors-of-callback
  - will cause cache misses when using the persistent cache.
  https://docs.jax.dev/en/latest/persistent_compilation_cache.html#pitfalls

  Therefore advise using for debug logging only if running under xnp.jit.

  Args:
    msg: message to log.
    *args: arguments to be passed to the logging function.
    **kwargs: keyword arguments to be passed to the logging function.
  """
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    jax.debug.print(msg, *args, **kwargs)
  else:
    native_logging.info(msg, *args, **kwargs)


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
