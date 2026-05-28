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
  """Solves `func(x, *args) = x` for `x`.

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

  def residual_sq_norm(x, f_x):
    """Computes the squared norm of the residual `x - f(x)`."""
    leaf_sq_norms = jax.tree.map(lambda a, b: jnp.sum((a - b) ** 2), f_x, x)
    return jax.tree_util.tree_reduce(jnp.add, leaf_sq_norms)

  def body(carry):
    x, _, count = carry
    f_x = func(x, *args)
    sq_norm = residual_sq_norm(x, f_x)
    count += 1
    return f_x, sq_norm, count

  # TODO(b/515250945): Ensure that automatic differentiation is supported.
  # Currently, the branch using fori_loop supports autodiff, but differentiates
  # through the entire loop. The branch using while_loop does not allow for
  # automatic differentiation. Consider switching to whilei_loop.
  if termination_criterion == 'max_iterations':
    count = jnp.array(0, dtype=jax_utils.get_int_dtype())
    initial_sq_norm = jnp.array(jnp.inf, dtype=jax_utils.get_dtype())
    initial_carry = (x0, initial_sq_norm, count)
    x_final, _, _ = jax.lax.fori_loop(
        0, maxiter, lambda i, val: body(val), initial_carry
    )
    return x_final

  else:
    # Precompute the tolerance for convergence.
    # TODO(b/515255142): pass in the initial residual as an argument, and use it
    # as the basis for the tolerance instead of calculating it here.
    f_x0 = func(x0, *args)
    initial_sq_norm = residual_sq_norm(x0, f_x0)
    initial_residual_norm = jnp.sqrt(initial_sq_norm)
    tol = atol + rtol * initial_residual_norm
    sq_tol = tol**2

    def cond(carry):
      _, sq_norm, count = carry
      is_converged = sq_norm <= sq_tol
      return (count < maxiter) & jnp.logical_not(is_converged)

    # Initial count starts at 1 since we do one evaluation of `func` in the
    # initialization above.
    initial_count = jnp.array(1, dtype=jax_utils.get_int_dtype())
    initial_carry = (f_x0, initial_sq_norm, initial_count)
    x_final, _, _ = jax.lax.while_loop(cond, body, initial_carry)
    return x_final
