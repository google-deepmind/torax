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

"""Commonly repeated jax expressions."""

import contextlib
import dataclasses
import functools
import os
from typing import Any, Callable, Optional, TypeVar, Union
import chex
import equinox as eqx
import jax
from jax import numpy as jnp
import numpy as np


T = TypeVar('T')
BooleanNumeric = Any  # A bool, or a Boolean array.


@functools.cache
def get_dtype() -> type(jnp.float32):
  # Default TORAX JAX precision is f64
  precision = os.getenv('JAX_PRECISION', 'f64')
  assert precision == 'f64' or precision == 'f32', (
      'Unknown JAX precision environment variable: %s' % precision
  )
  return jnp.float64 if precision == 'f64' else jnp.float32


@functools.cache
def get_np_dtype() -> type(np.float32):
  # Default TORAX JAX precision is f64
  precision = os.getenv('JAX_PRECISION', 'f64')
  assert precision == 'f64' or precision == 'f32', (
      'Unknown JAX precision environment variable: %s' % precision
  )
  return np.float64 if precision == 'f64' else np.float32


@functools.cache
def get_int_dtype() -> type(jnp.int32):
  # Default TORAX JAX precision is f64
  precision = os.getenv('JAX_PRECISION', 'f64')
  assert precision == 'f64' or precision == 'f32', (
      'Unknown JAX precision environment variable: %s' % precision
  )
  return jnp.int64 if precision == 'f64' else jnp.int32


def env_bool(name: str, default: bool) -> bool:
  """Get a bool from an environment variable.

  Args:
    name: The name of the environment variable.
    default: The default value of the bool.

  Returns:
    value: The value of the bool.
  """
  if name not in os.environ:
    return default
  str_value = os.environ[name]
  if str_value in ['1', 'True', 'true']:
    return True
  if str_value in ['0', 'False', 'false']:
    return False
  raise ValueError(f'Unrecognized boolean string {str_value}.')


# If True, `error_if` functions will raise errors.  Otherwise they are
# pass throughs.
# Default to False, because host_callbacks are incompatible with the
# persistent compilation cache.
_ERRORS_ENABLED: bool = env_bool('TORAX_ERRORS_ENABLED', False)


@contextlib.contextmanager
def enable_errors(value: bool):
  """Enables / disables `error_if` inside a code block.

  Example:

  with enable_errors(False):
    my_sim.run() # NaNs etc will be ignored

  Args:
    value: Sets `errors_enabled` to this value

  Yields:
    Cleanup function restoring previous value
  """
  global _ERRORS_ENABLED
  previous_value = _ERRORS_ENABLED
  _ERRORS_ENABLED = value
  yield
  if previous_value is not None:
    _ERRORS_ENABLED = previous_value


def error_if(
    var: jax.Array,
    cond: jax.Array,
    msg: str,
) -> jax.Array:
  """Raises error if cond is true, and `errors_enabled` is True.

  This is just a wrapper around `equinox.error_if`, gated by `errors_enabled`.

  Args:
    var: The variable to pass through.
    cond: Boolean array, error if cond is true.
    msg: Message to print on error.

  Returns:
    var: Identity wrapper that must be used for the check to be included.
  """
  if not _ERRORS_ENABLED:
    return var
  return eqx.error_if(var, cond, msg)


def error_if_negative(
    var: jax.Array, name: str, to_wrap: Optional[jax.Array] = None
) -> jax.Array:
  """Check that a variable is non-negative.

  Similar to error_if_not_positive, but 0 is allowed in this function.

  Args:
    var: The variable to check.
    name: Name of the variable.
    to_wrap: If `var` won't be used in your jax function, specify another
      variable that will be.

  Returns:
    var: Identity wrapper that must be used for the check to be included.
  """
  msg = f'{name} must be >= 0.'
  min_var = jnp.min(var)
  if to_wrap is None:
    to_wrap = var
  return error_if(to_wrap, min_var < 0, msg)


def jax_default(value: chex.Numeric) -> ...:
  """Define a dataclass field with a jax-type default value.

  Args:
    value: The default value of the field.

  Returns:
    field: The dataclass field.
  """
  jax_value = lambda: jnp.array(value)
  return dataclasses.field(default_factory=jax_value)


def compat_linspace(
    start: Union[chex.Numeric, jax.Array], stop: jax.Array, num: jax.Array
) -> jax.Array:
  """See np.linspace.

  This implementation of a subset of the linspace API reproduces the
  output of numpy better (at least when run in float64 mode) than
  jnp.linspace does.

  Args:
    start: first value
    stop: last value
    num: Number of points in the series

  Returns:
    linspace: array of shape (num) increasing linearly from `start` to `stop`
  """
  return jnp.arange(num) * ((stop - start) / (num - 1)) + start


def assert_rank(
    inputs: chex.Numeric | jax.stages.ArgInfo,
    rank: int,
) -> None:
  """Wrapper around chex.assert_rank that supports jax.stages.ArgInfo."""
  if isinstance(inputs, jax.stages.ArgInfo):
    chex.assert_rank(inputs.shape, rank)
  else:
    chex.assert_rank(inputs, rank)


def select(
    cond: jax.Array | bool,
    true_val: jax.Array,
    false_val: jax.Array,
) -> jax.Array:
  """Wrapper around jnp.where for readability."""
  return jnp.where(cond, true_val, false_val)


def is_tracer(var: jax.Array) -> bool:
  """Checks whether `var` is a jax tracer.

  Args:
    var: The jax variable to inspect.

  Returns:
    output: True `var` is a tracer, False if concrete.
  """

  try:
    if var.sum() > 0:
      return False
    return False
  except jax.errors.TracerBoolConversionError:
    return True
  assert False  # Should be unreachable


def jit(*args, **kwargs) -> Callable[..., Any]:
  """Calls jax.jit if TORAX_COMPILATION_ENABLED is True, otherwise no-op."""
  if env_bool('TORAX_COMPILATION_ENABLED', True):
    return jax.jit(*args, **kwargs)
  return args[0]


def py_while(
    cond_fun: Callable[[T], BooleanNumeric],
    body_fun: Callable[[T], T],
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


def py_fori_loop(
    lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T
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


# pylint: disable=g-bare-generic
def py_cond(
    cond: bool,
    true_fun: Callable,
    false_fun: Callable,
) -> Any:
  """Pure Python implementation of jax.lax.cond.

  This gives us a way to write code that could easily be changed to be
  Jax-compatible in the future, if we want to expand the scope of the jit
  compilation.

  Args:
    cond: The condition
    true_fun: Function to be called if cond==True.
    false_fun: Function to be called if cond==False.

  Returns:
    The output from either true_fun or false_fun.
  """
  if cond:
    return true_fun()
  else:
    return false_fun()


# pylint: enable=g-bare-generic
