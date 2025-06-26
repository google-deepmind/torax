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
import sys
import typing
from typing import Any, Callable, Literal, TypeVar

import chex
import equinox as eqx
import jax
from jax import numpy as jnp
import numpy as np
import typing_extensions


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


def assert_rank(
    inputs: chex.Numeric | jax.stages.ArgInfo,
    rank: int,
) -> None:
  """Wrapper around chex.assert_rank that supports jax.stages.ArgInfo."""
  if isinstance(inputs, jax.stages.ArgInfo):
    chex.assert_rank(inputs.shape, rank)
  else:
    chex.assert_rank(inputs, rank)


def jit(*args, **kwargs) -> Callable[..., Any]:
  """Calls jax.jit if TORAX_COMPILATION_ENABLED is True, otherwise no-op."""
  if env_bool('TORAX_COMPILATION_ENABLED', True):
    return jax.jit(*args, **kwargs)
  return args[0]


def get_number_of_compiles(
    jitted_function: Callable[..., Any],
) -> int:
  """Helper function for debugging JAX compilation.

  This counts the number of times the function has been JIT compiled. This does
  not include any uses of the AOT compile workflow.

  Args:
    jitted_function: A function that has been wrapped with `jax.jit`.

  Returns:
    The number of times the function has been compiled.
  Raises:
    RuntimeError: If the function does not have a _cache_size attribute.
  """
  # pylint: disable=protected-access
  if not hasattr(jitted_function, '_cache_size'):
    raise RuntimeError(
        'The function does not have a _cache_size attribute. Possibly because'
        ' the function was not jitted.'
    )
  return jitted_function._cache_size()
  # pylint: enable=protected-access


# pylint: enable=g-bare-generic


# TODO(b/424382924)
def non_inlined_function(
    f: Callable[..., Any],
    implementation: Literal['while_loop', 'pure_callback'],
) -> Callable[..., Any]:
  """A decorator that prevents XLA from inlining a function.

  XLA inlines all functions in a computational graph. As XLA does global
  optimization, the compile times increase super-linearly. This decorator
  allows preventing inlining of functions using `jax.lax.while_loop` or
  `jax.pure_callback`. In the case of `jax.pure_callback`, what is called from
  the Python callback is a black box to XLA, and cannot be inlined. In the case
  of `jax.lax.while_loop`, the body function is not inlined with the rest of the
  computation.

  Args:
    f: The function to be called.
    implementation: If 'while_loop', use `jax.lax.while_loop` with a single
      iteration. If 'pure_callback', use `jax.pure_callback`. This comes at the
      cost of a roughly 0.7ms constant overhead per call. It is recommended that
      `f` is a JITted function in this case, as it will be called directly from
      Python.

  Returns:
    The function.
  """

  match implementation:
    case 'while_loop':
      return _non_inlined_function_while_loop(f)
    case 'pure_callback':
      return _non_inlined_function_pure_callback(f)
    case _:
      raise ValueError(f'Unknown implementation: {implementation}')


def _non_inlined_function_pure_callback(f):
  """A decorator that prevents XLA from inlining a function."""

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    result_shape_dtypes = jax.eval_shape(f, *args, **kwargs)
    return jax.pure_callback(f, result_shape_dtypes, *args, **kwargs)

  return wrapper


def _non_inlined_function_while_loop(f):
  """A decorator that prevents XLA from inlining a function."""

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    continue_loop = True
    empty_out = _init_pytree(jax.eval_shape(f, *args, **kwargs))

    def body(val):
      return False, val[1], val[2], f(*val[1], **val[2])

    def cond(val):
      return val[0]

    out = jax.lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=(continue_loop, args, kwargs, empty_out),
    )
    return out[-1]

  return wrapper


def _init_pytree(t):

  def init_array(x):
    if isinstance(x, jax.ShapeDtypeStruct):
      return jnp.empty(shape=x.shape, dtype=x.dtype)
    else:
      return x

  return jax.tree_util.tree_map(init_array, t)


def _dataclass_transform():
  if sys.version_info >= (3, 11):
    return typing.dataclass_transform()
  else:
    # Remove once support for 3.10 can be dropped.
    return typing_extensions.dataclass_transform()


@_dataclass_transform()
def jax_dataclass(
    cls=None,
    *,
    init=True,
    repr=True,  # pylint: disable=redefined-builtin
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    kw_only=False,
):
  """Decorator for dataclasses that are registered with JAX.

  This can be used just like a normal dataclass with the additional feature of
  being able to mark certain fields of the dataclass as static, as shown below.

  @jax_dataclass
  class JaxDataclass:
    a: bool = dataclasses.field(metadata=dict(static=True))
    b: chex.Array

  Args:
    cls: The dataclass to decorate.
    init: See :py:func:`dataclasses.dataclass`.
    repr: See :py:func:`dataclasses.dataclass`.
    eq: See :py:func:`dataclasses.dataclass`.
    order: See :py:func:`dataclasses.dataclass`.
    unsafe_hash: See :py:func:`dataclasses.dataclass`.
    frozen: See :py:func:`dataclasses.dataclass`.
    kw_only: See :py:func:`dataclasses.dataclass`.
  """
  if cls is None:
    return functools.partial(
        jax_dataclass,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
        kw_only=kw_only,
        init=init,
    )
  dataclass_cls = dataclasses.dataclass(
      repr=repr,
      eq=eq,
      order=order,
      unsafe_hash=unsafe_hash,
      frozen=frozen,
      kw_only=kw_only,  # pytype: disable=not-supported-yet
      init=init,  # pytype: disable=not-supported-yet
  )(cls)
  jax.tree_util.register_dataclass(dataclass_cls)

  return dataclass_cls
