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

"""JAX fixed point functions."""

from typing import Any, Callable, Literal, TypeAlias
import jax
import jax.numpy as jnp
from torax._src import jax_utils

PyTree: TypeAlias = Any


def fixed_point(
    func: Callable[..., jax.Array],
    x0: PyTree,
    args: tuple[PyTree, ...] = (),
    xtol: float | None = 1e-08,
    maxiter: int = 500,
    method: Literal['del2', 'iteration'] = 'del2',
) -> PyTree:
  """A JAX version of `scipy.optimize.fixed_point`.

  Unlike `scipy.optimize.fixed_point`, this function will not raise a
  `RuntimeError` if convergence is not reached.

  Args:
    func: The function to solve, of the form `f(x, *args)` returning a
      `jax.Array` of the same shape as `x`.
    x0: The initial guess.
    args: Additional arguments to pass to the function.
    xtol: The tolerance on the absolute value of the function value. If None, no
      tolerance is used and `maxiter` iterations will be performed.
    maxiter: The maximum number of iterations to perform.
    method: The method to use. 'del2' (the default) uses Steffensen’s Method
      with Aitken’s Del^2 convergence acceleration, taken from Burden, Faires,
      “Numerical Analysis”, 5th edition, pg. 80. 'iteration' just iterates the
      function until the tolerance is reached.

  Returns:
    The fixed point `jax.Array`.
  """
  if method not in ['del2', 'iteration']:
    raise ValueError(f'Invalid method: {method}')
  if maxiter <= 0:
    raise ValueError(f'Invalid maxiter: {maxiter} must be positive.')

  def body(x):
    x, count, _ = x
    out1 = func(x, *args)
    if method == 'del2':
      out2 = func(out1, *args)

      def _del2(p0, p1, p2):
        d = p2 - 2.0 * p1 + x
        out3 = x - (p1 - p0) ** 2 / d
        return jax.lax.select(d != 0, out3, p2)

      out = jax.tree.map(_del2, x, out1, out2)
    else:
      out = out1

    if xtol:

      def _relative_error(actual, expected):
        relative_error = (actual - expected) / expected
        relerr = jax.lax.select(x != 0, relative_error, actual)
        return jnp.all(jnp.abs(relerr) < xtol)

      stop = jax.tree.map(_relative_error, out, x)
      stop = jnp.all(jnp.hstack(jax.tree.leaves(stop)))
    else:
      stop = jnp.array(False, dtype=jnp.bool_)
    count += 1
    return out, count, stop

  def cond(x):
    _, count, stop = x
    return jnp.logical_not(stop) & (count < maxiter)

  count = jnp.array(0, dtype=jax_utils.get_int_dtype())
  stop = jnp.array(False, dtype=jnp.bool_)
  x_init = (x0, count, stop)

  if xtol is None:
    return jax.lax.fori_loop(0, maxiter, lambda i, val: body(val), x_init)[0]
  else:
    return jax.lax.while_loop(cond, body, x_init)[0]


def _fixed_point_del2(p0, p1, d):
  return (p0 + p1 - d) / 2.0
