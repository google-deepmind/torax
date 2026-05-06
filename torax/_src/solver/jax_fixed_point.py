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
    atol: float = 0.0,
    rtol: float = 0.0,
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

  Returns:
    The fixed point `PyTree`.
  """
  if maxiter <= 0:
    raise ValueError(f'Invalid maxiter: {maxiter} must be positive.')

  def residual_sq_norm(x, f_x):
    """Computes the squared norm of the residual `x - f(x)`."""
    leaf_sq_norms = jax.tree.map(lambda a, b: jnp.sum((a - b) ** 2), f_x, x)
    return jax.tree_util.tree_reduce(jnp.add, leaf_sq_norms)

  # Precompute the tolerance for convergence.
  f_x0 = func(x0, *args)
  initial_sq_norm = residual_sq_norm(x0, f_x0)
  initial_residual_norm = jnp.sqrt(initial_sq_norm)
  tol = atol + rtol * initial_residual_norm
  sq_tol = tol**2

  def body(carry):
    x, _, count = carry
    f_x = func(x, *args)
    sq_norm = residual_sq_norm(x, f_x)
    count += 1
    return f_x, sq_norm, count

  def cond(carry):
    _, sq_norm, count = carry
    is_converged = sq_norm <= sq_tol
    return (count < maxiter) & jnp.logical_not(is_converged)

  # Initialize the iteration counter to 1, since we start counting from the
  # first iteration in the `body` function.
  count = jnp.array(1, dtype=jax_utils.get_int_dtype())
  init_carry = (f_x0, initial_sq_norm, count)
  return jax.lax.while_loop(cond, body, init_carry)[0]
