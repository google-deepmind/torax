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
from torax._src.solver import linesearch

PyTree: TypeAlias = Any


def fixed_point(
    func: Callable[..., PyTree],
    x0: PyTree,
    args: tuple[PyTree, ...] = (),
    maxiter: int = 500,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    termination_criterion: str = 'tolerance',
    use_backtracking: bool = False,
    sufficient_decrease: float = 1e-4,
    step_size_reduction_factor: float = 0.5,
    max_backtrack_steps: int = 50,
) -> PyTree:
  """Solves `func(x, *args) = x` for `x` with backtracking linesearch.

  Iterates x_new = func(x_old, *args) until either the requested tolerance is
  satisfied or the maximum number of iterations is reached.
  If `use_backtracking` is True, the iteration is of the form
  x_new = x_old + alpha * (f(x_old) - x_old), where alpha is chosen via
  backtracking linesearch.

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
    use_backtracking: If true, use backtracking linesearch.
    sufficient_decrease: Control parameter for Armijo condition in backtracking
      linesearch. Residual norm must decrease by at least this factor for the
      step to be accepted.
    step_size_reduction_factor: Factor by which step_size is reduced during
      backtracking linesearch.
    max_backtrack_steps: Maximum number of backtracking steps.

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

  def residual_fn(x):
    """Computes the residual R(x) = f(x) - x."""
    f_x = func(x, *args)
    return jax.tree.map(lambda a, b: a - b, f_x, x)

  def norm_fn(x):
    """Computes the L2 norm of a PyTree."""
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(x)))

  def body(carry):
    x, residual, residual_norm, count = carry

    if use_backtracking:

      def armijo_condition(step_size, trial_norm):
        return (
            trial_norm <= (1 - sufficient_decrease * step_size) * residual_norm
        )

      # Damped Picard update is x_k+1 = x_k + alpha * (f(x_k) - x_k), where
      # alpha is the step size. We do linesearch to find an acceptable alpha.
      # Note: this will *fail* if f'(x_k) > 1.
      ls_state = linesearch.backtracking_linesearch(
          residual_fn=residual_fn,
          x_init=x,
          direction=residual,
          accept_fn=armijo_condition,
          norm_fn=norm_fn,
          initial_residual=residual,
          initial_residual_norm=residual_norm,
          delta_reduction_factor=step_size_reduction_factor,
          max_steps=max_backtrack_steps,
      )
      x_next = ls_state.x
      residual_next = ls_state.residual
    else:
      # Standard Picard: x_{k+1} = f(x_k) = x_k + R(x_k)
      x_next = jax.tree.map(lambda a, b: a + b, x, residual)
      residual_next = residual_fn(x_next)

    count += 1
    residual_norm_next = norm_fn(residual_next)
    return x_next, residual_next, residual_norm_next, count

  # Take a single full implicit step (undamped).
  x1 = func(x0, *args)
  residual = residual_fn(x1)
  residual_norm = norm_fn(residual)
  count = jnp.array(1, dtype=jax_utils.get_int_dtype())
  carry = (x1, residual, residual_norm, count)

  # TODO(b/515250945): Ensure that automatic differentiation is supported.
  # Currently, the branch using fori_loop supports autodiff, but differentiates
  # through the entire loop. The branch using while_loop does not allow for
  # automatic differentiation. Consider switching to whilei_loop.
  if termination_criterion == 'max_iterations':
    x_final, _, _, _ = jax.lax.fori_loop(
        1, maxiter, lambda i, val: body(val), carry
    )
    return x_final
  else:
    # Precompute the tolerance for convergence.
    tol = atol + rtol * residual_norm

    def cond(carry):
      _, _, residual_norm, count = carry
      is_converged = residual_norm <= tol
      return (count < maxiter) & jnp.logical_not(is_converged)

    x_final, _, _, _ = jax.lax.while_loop(cond, body, carry)
    return x_final
