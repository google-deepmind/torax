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
from torax._src.solver import linesearch

PyTree: TypeAlias = Any


def fixed_point(
    func: Callable[..., jax.Array],
    x0: PyTree,
    args: tuple[PyTree, ...] = (),
    xtol: float | None = 1e-08,
    maxiter: int = 500,
    method: Literal['del2', 'iteration'] = 'del2',
    atol: float | None = None,
    rtol: float | None = None,
    use_backtracking: bool = True,
    delta_reduction_factor: float = 0.5,
    max_backtrack_steps: int = 10,
) -> PyTree:
  """A JAX version of `scipy.optimize.fixed_point` with backtracking linesearch.

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
    atol: Absolute tolerance on the residual norm.
    rtol: Relative tolerance on the residual norm.
    use_backtracking: If true, use backtracking linesearch in 'iteration'
      method.
    delta_reduction_factor: Factor by which step_size is reduced each step.
    max_backtrack_steps: Maximum number of backtracking steps.

  Returns:
    The fixed point `jax.Array`.
  """
  if method not in ['del2', 'iteration']:
    raise ValueError(f'Invalid method: {method}')
  if maxiter <= 0:
    raise ValueError(f'Invalid maxiter: {maxiter} must be positive.')

  def residual_fn(x):
    return jax.tree.map(lambda a, b: a - b, func(x, *args), x)

  def norm_fn(res):
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(res)))

  def residual_norm(x):
    return norm_fn(residual_fn(x))

  if rtol is not None:
    initial_residual_norm = residual_norm(x0)
  else:
    initial_residual_norm = jnp.array(0.0)

  def body(x_state):
    x, count, _ = x_state
    out1 = func(x, *args)
    if method == 'del2':
      out2 = func(out1, *args)

      def _del2(p0, p1, p2):
        d = p2 - 2.0 * p1 + x
        out3 = x - (p1 - p0) ** 2 / d
        return jax.lax.select(d != 0, out3, p2)

      out = jax.tree.map(_del2, x, out1, out2)
    else:
      if use_backtracking:
        direction = jax.tree.map(lambda a, b: a - b, out1, x)

        init_res = direction
        init_norm = norm_fn(init_res)

        decrease = 1e-4
        current_norm_sq = init_norm**2

        def accept_fn(step_size, trial_norm):
          target = (1.0 - 2.0 * decrease * step_size) * current_norm_sq
          return (trial_norm**2) <= target

        ls_state = linesearch.backtracking_linesearch(
            residual_fn=residual_fn,
            x_init=x,
            direction=direction,
            accept_fn=accept_fn,
            norm_fn=norm_fn,
            initial_residual=init_res,
            initial_residual_norm=init_norm,
            delta_reduction_factor=delta_reduction_factor,
            max_steps=max_backtrack_steps,
        )
        out = ls_state.x
      else:
        out = out1

    if atol is not None or rtol is not None:
      # Terminate based on residual norm.
      res_norm = residual_norm(out)
      combined_tol = atol + rtol * initial_residual_norm
      stop = res_norm <= combined_tol
    elif xtol:
      # Terminate based on relative error.
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

  if xtol is None and atol is None and rtol is None:
    return jax.lax.fori_loop(0, maxiter, lambda i, val: body(val), x_init)[0]
  else:
    return jax.lax.while_loop(cond, body, x_init)[0]


def _fixed_point_del2(p0, p1, d):
  return (p0 + p1 - d) / 2.0
