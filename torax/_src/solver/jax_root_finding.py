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

"""JAX root finding functions."""

import dataclasses
import functools
from typing import Callable, Final

import jax
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.solver import linesearch

# Delta is a vector. If no entry of delta is above this magnitude, we terminate
# the delta loop. This is to avoid getting stuck in an infinite loop in edge
# cases with bad numerics.
MIN_DELTA: Final[float] = 1e-7


def _residual_scalar(x):
  return jnp.mean(jnp.abs(x))


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RootMetadata:
  iterations: jax.Array
  residual: jax.Array
  last_tau: jax.Array
  error: jax.Array


def root_newton_raphson(
    fun: Callable[[jax.Array], jax.Array],
    x0: jax.Array | np.ndarray,
    *,
    maxiter: int = 30,
    tol: float = 1e-5,
    coarse_tol: float = 1e-2,
    delta_reduction_factor: float = 0.5,
    tau_min: float = 0.01,
    sufficient_decrease: float = 1e-4,
    log_iterations: bool = False,
    use_jax_custom_root: bool = True,
    custom_jac: Callable[[jax.Array], jax.Array] | None = None,
    norm_fn: Callable[[jax.Array], jax.Array] = _residual_scalar,
) -> tuple[jax.Array, RootMetadata]:
  """A differentiable Newton-Raphson root finder.

  A similar API to scipy.optimize.root.

  Args:
    fun: The function to find the root of.
    x0: The initial guess of the location of the root.
    maxiter: Quit iterating after this many iterations reached.
    tol: Quit iterating after norm_fn(residual) <= tol.
    coarse_tol: Coarser allowed tolerance for cases when solver develops small
      steps in the vicinity of the solution.
    delta_reduction_factor: Multiply by delta_reduction_factor after each failed
      line search step.
    tau_min: Minimum delta/delta_original allowed before the newton raphson
      routine resets at a lower timestep.
    sufficient_decrease: Acceptance threshold for sufficient decrease in the
      line search.
    log_iterations: If true, output diagnostic information from within iteration
      loop.
    use_jax_custom_root: If true, use jax.lax.custom_root to allow for
      differentiable solving. This can increase compile times even when no
      derivatives are requested.
    custom_jac: If provided, use this function to compute the Jacobian of `fun`
      instead of jax.jacfwd.
    norm_fn: Scalar norm function applied to residual vectors for line search
      acceptance and convergence checks. Defaults to L1 norm.

  Returns:
    A tuple `(x_root, RootMetadata(...))`.
  """

  def _newton_raphson(
      f: Callable[[jax.Array], jax.Array],
      x: jax.Array | np.ndarray,
      jacobian_fun: Callable[[jax.Array], jax.Array] | None = None,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    init_x_new_vec = jnp.asarray(x)
    residual_fun = f

    if jacobian_fun is None:
      jacobian_fun = jax.jit(jax.jacfwd(f), inline=jax.Inline.XLA_LATE)

    # initialize state dict being passed around Newton-Raphson iterations
    residual_vec_init_x_new = residual_fun(init_x_new_vec)
    initial_state = {
        'x': init_x_new_vec,
        # jax.lax.custom_root is broken with aux outputs of integer type. Use
        # float for the iterations https://github.com/jax-ml/jax/issues/24295.
        'iterations': jnp.array(0, dtype=jax_utils.get_dtype()),
        'residual': residual_vec_init_x_new,
        'last_tau': jnp.array(1.0, dtype=jax_utils.get_dtype()),
        'residual_norm': norm_fn(residual_vec_init_x_new),
    }

    # carry out iterations.
    cond_fun = functools.partial(
        _cond, tol=tol, tau_min=tau_min, maxiter=maxiter
    )
    body_fun = functools.partial(
        _body,
        jacobian_fun=jacobian_fun,
        residual_fun=residual_fun,
        log_iterations=log_iterations,
        delta_reduction_factor=delta_reduction_factor,
        sufficient_decrease=sufficient_decrease,
        norm_fn=norm_fn,
    )
    output_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    x_out = output_state.pop('x')
    return x_out, output_state

  # jax.lax.custom_root allows for differentiating through the solver,
  # efficiently. As the solver has a jax.lax.while_loop, it cannot be
  # reverse-mode differentiated. But even if we could, this would be highly
  # inefficient. This uses the implicit function theorem to differentiate
  # through the solver with only needing the result of the solver,
  # rather than the entire solver computational graph.
  # See also this discussion:
  # https://docs.jax.dev/en/latest/advanced-autodiff.html#example-implicit-function-differentiation-of-iterative-implementations

  def back(
      g: Callable[[jax.Array], jax.Array],
      y: jax.Array,
  ) -> jax.Array:
    return jnp.linalg.solve(jax.jacfwd(g)(y), y)

  if use_jax_custom_root:
    if custom_jac is not None:
      raise ValueError('custom_jac is not compatible with use_jax_custom_root.')
    x_out, metadata = jax.lax.custom_root(
        f=fun,
        initial_guess=x0,
        solve=_newton_raphson,
        tangent_solve=back,
        has_aux=True,
    )
  else:
    x_out, metadata = _newton_raphson(fun, x0, jacobian_fun=custom_jac)

  # Error flag tells the caller whether x_new successfully reduces the residual
  # below the tolerance.
  error = _get_error_flag(
      norm=metadata.pop('residual_norm'),
      coarse_tol=coarse_tol,
      tol=tol,
  )
  # Workaround for https://github.com/google/jax/issues/24295: cast iterations
  # to the correct int dtype.
  metadata['iterations'] = metadata['iterations'].astype(
      jax_utils.get_int_dtype()
  )
  return x_out, RootMetadata(**metadata, error=error)


def _get_error_flag(
    norm: jax.Array,
    coarse_tol: float,
    tol: float,
) -> jax.Array:
  """Computes the flag indicating whether a solve step converged.

  Args:
    norm: Scalar norm of the final residual vector.
    coarse_tol: Coarser allowed tolerance for cases when solver exits early due
      to small steps in the solution vicinity.
    tol: Fine convergence tolerance.

  Returns:
    An integer scalar flag indicating the convergence status:
      - 0: Residual converged within fine tolerance (`norm < tol`).
      - 1: Not converged (`norm >= coarse_tol`).
      - 2: Residual within reasonable tolerance (`norm < coarse_tol`).
  """
  return jax.lax.cond(
      norm < tol,
      lambda: 0,
      lambda: jax.lax.cond(
          norm < coarse_tol,
          lambda: 2,  # tol < norm < coarse_tol
          lambda: 1,  # norm > coarse_tol
      ),
  )


def _cond(
    state: dict[str, jax.Array],
    tau_min: float,
    maxiter: int,
    tol: float,
) -> bool:
  """Check if exit condition reached for Newton-Raphson iterations."""
  iteration = state['iterations'][...]
  return jnp.bool_(
      jnp.logical_and(
          jnp.logical_and(state['residual_norm'] > tol, iteration < maxiter),
          state['last_tau'] > tau_min,
      )
  )


def _body(
    input_state: dict[str, jax.Array],
    jacobian_fun: Callable[[jax.Array], jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
    log_iterations: bool,
    delta_reduction_factor: float,
    sufficient_decrease: float,
    norm_fn: Callable[[jax.Array], jax.Array],
) -> dict[str, jax.Array]:
  """Calculates next guess in Newton-Raphson iteration."""
  rhs = -input_state['residual']
  init_norm = input_state['residual_norm']

  a_mat = jacobian_fun(input_state['x'])

  direction = jnp.linalg.solve(a_mat, rhs)

  def accept_fn(step_size, trial_norm):
    return (
        trial_norm <= (1.0 - sufficient_decrease * step_size) * init_norm
    ) & (~jnp.isnan(trial_norm))

  ls_state = linesearch.backtracking_linesearch(
      residual_fn=residual_fun,
      x_init=input_state['x'],
      direction=direction,
      accept_fn=accept_fn,
      norm_fn=norm_fn,
      initial_residual=input_state['residual'],
      initial_residual_norm=init_norm,
      delta_reduction_factor=delta_reduction_factor,
      max_steps=100,
      min_step_norm=MIN_DELTA,
  )

  output_state = {
      'x': ls_state.x,
      'residual': ls_state.residual,
      'iterations': input_state['iterations'] + 1,
      'last_tau': ls_state.step_size,
      'residual_norm': ls_state.residual_norm,
  }

  if log_iterations:
    jax.debug.print(
        'Iteration: {iteration:d}. Residual: {residual:.16f}. tau = {tau:.6f}',
        iteration=output_state['iterations'].astype(jax_utils.get_int_dtype()),
        residual=output_state['residual_norm'],
        tau=ls_state.step_size,
    )

  return output_state
