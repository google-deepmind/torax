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

from typing import Any, Callable, TypeAlias
import jax
import jax.numpy as jnp
from torax._src import jax_utils

PyTree: TypeAlias = Any


def fixed_point(
    func: Callable[..., PyTree],
    x0: PyTree,
    args: tuple[PyTree, ...] = (),
    maxiter: int = 500,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    termination_criterion: str = 'tolerance',
) -> PyTree:
  """Solves `func(x, *args) = x` for `x` using Picard fixed-point iteration.

  Iterates x_new = func(x_old, *args) until either the requested tolerance is
  satisfied or the maximum number of iterations is reached.

  Args:
    func: The function to solve, of the form `f(x, *args)` returning a `PyTree`
      of the same structure as `x`.
    x0: The initial guess.
    args: Additional arguments to pass to the function.
    maxiter: The maximum number of iterations to perform.
    atol: Absolute tolerance on the residual norm.
    rtol: Relative tolerance on the residual norm.
    termination_criterion: The criterion to use for terminating the iteration.
      If 'max_iterations', the iteration will terminate after `maxiter`
      iterations. If 'tolerance', the iteration will terminate when the residual
      norm is below the tolerance specified by `atol` and `rtol`.

  Returns:
    The fixed point `PyTree`.
  """
  if maxiter <= 0:
    raise ValueError(f'Invalid maxiter: {maxiter} must be positive.')
  if termination_criterion not in ['max_iterations', 'tolerance']:
    raise ValueError(
        f'Invalid termination criterion: {termination_criterion} must be'
        ' "max_iterations" or "tolerance".'
    )

  def norm_fn(x: PyTree) -> jnp.ndarray:
    """Computes the L2 norm of a PyTree."""
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(x)))

  def body(carry):
    x, _, count = carry
    f_x = func(x, *args)
    residual = jax.tree.map(lambda a, b: a - b, f_x, x)
    residual_norm = norm_fn(residual)
    return f_x, residual_norm, count + 1

  # Take the initial step x1 = func(x0, *args).
  x1 = func(x0, *args)
  initial_residual = jax.tree.map(lambda a, b: a - b, x1, x0)
  initial_residual_norm = norm_fn(initial_residual)
  count = jnp.array(1, dtype=jax_utils.get_int_dtype())
  initial_carry = (x1, initial_residual_norm, count)

  # TODO(b/515250945): Ensure that automatic differentiation is supported.
  # Currently, the branch using fori_loop supports autodiff, but differentiates
  # through the entire loop. The branch using while_loop does not allow for
  # automatic differentiation. Consider switching to whilei_loop.
  if termination_criterion == 'max_iterations':
    x_final, _, _ = jax.lax.fori_loop(
        1, maxiter, lambda i, val: body(val), initial_carry
    )
    return x_final
  else:
    tol = atol + rtol * initial_residual_norm

    def cond(carry):
      _, residual_norm, count = carry
      is_converged = residual_norm <= tol
      return (count < maxiter) & jnp.logical_not(is_converged)

    x_final, _, _ = jax.lax.while_loop(cond, body, initial_carry)
    return x_final
