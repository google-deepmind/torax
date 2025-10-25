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
import functools
import inspect
import os
from typing import Any, Callable, Literal, ParamSpec, TypeAlias, TypeVar

import chex
import equinox as eqx
import jax
from jax import numpy as jnp
import numpy as np

T = TypeVar('T')
BooleanNumeric: TypeAlias = Any  # A bool, or a Boolean array.
_State = ParamSpec('_State')
PyTree: TypeAlias = Any


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

  # Skip error checking during JAX cache analysis to avoid pytree leaf issues
  if os.getenv('JAX_EXPLAIN_CACHE_MISSES') == '1':
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
    f: A JITted function.
    implementation: If 'while_loop', use `jax.lax.while_loop` with a single
      iteration. If 'pure_callback', use `jax.pure_callback`. This comes at the
      cost of a roughly 0.7ms constant overhead per call. It is recommended that
      `f` is a JITted function in this case, as it will be called directly from
      Python.

  Returns:
    The function.
  """

  if not hasattr(f, 'lower'):
    raise ValueError('Must be a JITted function.')

  if not hasattr(f, '_jit_info'):
    raise ValueError('The function must have a _jit_info attribute.')

  static_argnames = f._jit_info.static_argnames  # pylint: disable=protected-access

  match implementation:
    case 'while_loop':
      return _non_inlined_function_while_loop(
          f, static_argnames=static_argnames
      )
    case 'pure_callback':
      return _non_inlined_function_pure_callback(
          f, static_argnames=static_argnames
      )
    case _:
      raise ValueError(f'Unknown implementation: {implementation}')


def _non_inlined_function_pure_callback(f, static_argnames):
  """A decorator that prevents XLA from inlining a function."""

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    nonlocal f, static_argnames
    bound = inspect.signature(f).bind(*args, **kwargs)
    bound.apply_defaults()
    kwargs = bound.arguments
    if 'self' in kwargs:
      kwargs.pop('self')

    if static_argnames:
      static_args = {k: bound.arguments[k] for k in static_argnames}

      kwargs = {k: v for k, v in kwargs.items() if k not in static_argnames}
      f = functools.partial(f, **static_args)

    result_shape_dtypes = jax.eval_shape(f, **kwargs)
    return jax.pure_callback(f, result_shape_dtypes, **kwargs)

  return wrapper


def _non_inlined_function_while_loop(f, static_argnames):
  """A decorator that prevents XLA from inlining a function."""

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    nonlocal f, static_argnames
    bound = inspect.signature(f).bind(*args, **kwargs)
    bound.apply_defaults()
    kwargs = bound.arguments
    if 'self' in kwargs:
      kwargs.pop('self')

    if static_argnames:
      static_args = {k: bound.arguments[k] for k in static_argnames}
      kwargs = {k: v for k, v in kwargs.items() if k not in static_argnames}
      f = functools.partial(f, **static_args)

    continue_loop = True
    empty_out = _init_pytree(jax.eval_shape(f, **kwargs))

    def body(val):
      return False, val[1], f(**val[1])

    def cond(val):
      return val[0]

    out = jax.lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=(continue_loop, kwargs, empty_out),
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


def batched_cond(
    pred: jax.Array,
    true_fun: Callable[..., PyTree],
    false_fun: Callable[..., PyTree],
    operands: tuple[PyTree, ...],
    implementation: Literal['vectorize', 'map'] = 'vectorize',
):
  """A batched version of `jax.lax.cond`.

  JAX provides two approaches for implementing a batched version of
  `jax.lax.cond`, neither of which is always faster:
  `implementation='vectorize'` is equivalent to `jnp.select`, which evaluates
  both braches for every batch element. This is fully vectorized, allowing for
  parallel execution on CPU/GPU, but requiring twice the number of function
  evaluations. `implementation='map'` will sequentially evaluate `jax.lax.cond`,
  preventing vectorized execution, but only requiring a single function
  evaluation per batch element.

  This function also handles the special case where `pred` is a concrete list of
  length-1, in which case we can avoid tracing both branches like `jax.lax.cond`
  does by doing the control-flow in Python.

  Args:
    pred: Boolean 1D array `[batch_size]`, indicating which branch function to
      apply.
    true_fun: Function (A -> B), to be applied if `pred` is True.
    false_fun: Function (A -> B), to be applied if `pred` is False.
    operands: A tuple of arguments to pass to the functions. Each `jax.Array`
      (every PyTree leaf) must have a leading batch dimension of size
      `batch_size`.
    implementation: The implementation to use. 'vectorize' compiles to a
      `jax.lax.select`, where both branches are evaluated. 'map' uses
      `jax.lax.map`.

  Returns:
    The result of applying the appropriate function to each element of the
    batch.
  """

  if not isinstance(operands, tuple):
    raise ValueError('The args must be a tuple.')

  if pred.ndim != 1 or pred.dtype != jnp.bool:
    raise ValueError('pred must be a 1D array of bools.')

  # For the special case where `pred` is a concrete list of length 1, we can
  # avoid tracing both branches by doing the control flow in Python.
  if len(pred) == 1 and not isinstance(pred, jax.core.Tracer):
    f = true_fun if bool(pred) else false_fun
    operands = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), operands)
    out = f(*operands)
    return jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), out)

  f = lambda args: jax.lax.cond(args[0], true_fun, false_fun, *args[1])
  match implementation:
    case 'vectorize':
      # This is compiled to a jax.lax.select, where both branches are evaluated.
      return jax.vmap(f)((pred, operands))
    case 'map':
      return jax.lax.map(f, (pred, operands))
    case _:
      raise ValueError(f'Unknown implementation: {implementation}')


@functools.partial(
    jax.jit,
    static_argnames=['cond_fun', 'body_fun', 'max_steps', 'scan_unroll'],
)
def while_loop_bounded(
    cond_fun: Callable[[_State], BooleanNumeric],
    body_fun: Callable[[_State], _State],
    init_val: _State,
    max_steps: int,
    scan_unroll: int = 1,
) -> _State:
  """A reverse-mode differentiable while_loop.

  This makes use of jax.lax.scan and `max_steps` to define a fixed size
  computational graph. The body_fun is called the same number of times it would
  be under a jax.lax.while_loop i.e. until `cond_fun` returns False (unless the
  `max_steps` is reached).

  Args:
    cond_fun: As in jax.lax.while_loop.
    body_fun: As in jax.lax.while_loop.
    init_val: As in jax.lax.while_loop.
    max_steps: An integer, the maximum number of iterations the loop can
      perform. This is crucial for defining a fixed computational graph for
      scan.
    scan_unroll: The number of iterations to unroll the internal scan by.

  Returns:
    The final state after `cond_fun` returns `False` or `max_steps` are reached.
  """
  # Initial carry for the scan: (current_state, while_loop_condition_met)
  initial_scan_carry = (init_val, jnp.array(True, dtype=jnp.bool_))

  def scan_body(carry, _):
    current_state, cond_prev = carry
    # Only execute cond if the previous cond was True.
    should_execute_body = jax.lax.cond(
        cond_prev, cond_fun, lambda _: False, current_state
    )
    # If the `while_loop` would have terminated, we no-op.
    next_state = jax.lax.cond(
        should_execute_body, body_fun, lambda s: s, current_state
    )

    return (next_state, should_execute_body), None

  (final_state, _), _ = jax.lax.scan(
      scan_body, initial_scan_carry, length=max_steps, unroll=scan_unroll
  )

  return final_state
