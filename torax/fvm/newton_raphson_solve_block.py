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

"""The `newton_raphson_solve_block` function.

See function docstring for details.
"""

import functools
from typing import Callable, Final

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from torax import jax_utils
from torax import state as state_module
from torax.config import runtime_params_slice
from torax.fvm import block_1d_coeffs
from torax.fvm import calc_coeffs
from torax.fvm import cell_variable
from torax.fvm import enums
from torax.fvm import fvm_conversions
from torax.fvm import residual_and_loss
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
from torax.stepper import predictor_corrector_method
from torax.transport_model import transport_model as transport_model_lib


# Delta is a vector. If no entry of delta is above this magnitude, we terminate
# the delta loop. This is to avoid getting stuck in an infinite loop in edge
# cases with bad numerics.
MIN_DELTA: Final[float] = 1e-7


def _log_iterations(
    residual: jax.Array,
    iterations: jax.Array,
    delta_reduction: jax.Array | None = None,
    dt: jax.Array | None = None,
) -> None:
  """Logs info on internal Newton-Raphson iterations.

  Args:
    residual: Scalar residual.
    iterations: Number of iterations taken so far in the solve block.
    delta_reduction: Current tau used in this iteration.
    dt: Current dt used in this iteration.
  """
  if dt is not None:
    logging.info(
        'Iteration: %d. Residual: %.16f. dt = %.6f',
        iterations,
        residual,
        dt,
    )

  elif delta_reduction is not None:
    logging.info(
        'Iteration: %d. Residual: %.16f. tau = %.6f',
        iterations,
        residual,
        delta_reduction,
    )
  else:
    logging.info('Iteration: %d. Residual: %.16f', iterations, residual)


def newton_raphson_solve_block(
    dt: jax.Array,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t: geometry.Geometry,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state_module.CoreProfiles,
    core_profiles_t_plus_dt: state_module.CoreProfiles,
    transport_model: transport_model_lib.TransportModel,
    explicit_source_profiles: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    pedestal_model: pedestal_model_lib.PedestalModel,
    coeffs_callback: calc_coeffs.CoeffsCallback,
    evolving_names: tuple[str, ...],
    initial_guess_mode: enums.InitialGuessMode,
    maxiter: int,
    tol: float,
    coarse_tol: float,
    delta_reduction_factor: float,
    tau_min: float,
    log_iterations: bool = False,
) -> tuple[
    tuple[cell_variable.CellVariable, ...],
    state_module.SolverNumericOutputs,
    block_1d_coeffs.AuxiliaryOutput,
]:
  # pyformat: disable  # pyformat removes line breaks needed for reability
  """Runs one time step of a Newton-Raphson based root-finding on the equation defined by `coeffs`.

  This solver is relatively generic in that it models diffusion, convection,
  etc. abstractly. The caller must do the problem-specific physics calculations
  to obtain the coefficients for a particular problem.

  This solver uses iterative root finding on the linearized residual
  between two sides of the equation describing a theta method update.

  The linearized residual for a trial x_new is:

  R(x_old) + jacobian(R(x_old))*(x_new - x_old)

  Setting delta = x_new - x_old, we solve the linear system:

  A*x_new = b, with A = jacobian(R(x_old)), b = A*x_old - R(x_old)

  Each successive iteration sets x_new = x_old - delta, until the residual
  or delta is under a tolerance (tol).
  If either the delta step leads to an unphysical state, represented by NaNs in
  the residual, or if the residual doesn't shrink following the delta step,
  then delta is successively reduced by a delta_reduction_factor.
  If tau = delta_now / delta_original is below a tolerance, then the iterations
  stop. If residual > tol then the function exits with an error flag, producing
  either a warning or recalculation with a lower dt.

  Args:
    dt: Discrete time step.
    static_runtime_params_slice: Static runtime parameters. Changes to these
      runtime params will trigger recompilation.
    dynamic_runtime_params_slice_t: Runtime parameters for time t (the start
      time of the step). These config params can change from step to step
      without triggering a recompilation.
    dynamic_runtime_params_slice_t_plus_dt: Runtime parameters for time t + dt.
    geo_t: Geometry at time t.
    geo_t_plus_dt: Geometry at time t + dt.
    x_old: Tuple containing CellVariables for each channel with their values at
      the start of the time step.
    core_profiles_t: Core plasma profiles which contain all available prescribed
      quantities at the start of the time step. This includes evolving boundary
      conditions and prescribed time-dependent profiles that are not being
      evolved by the PDE system.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    transport_model: Turbulent transport model callable.
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    source_models: Collection of source callables to generate source PDE
      coefficients.
    pedestal_model: Model of the pedestal's behavior.
    coeffs_callback: Calculates diffusion, convection etc. coefficients given a
      core_profiles. Repeatedly called by the iterative optimizer.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    initial_guess_mode: chooses the initial_guess for the iterative method,
      either x_old or linear step. When taking the linear step, it is also
      recommended to use Pereverzev-Corrigan terms if the transport coefficients
      are stiff, e.g. from QLKNN. This can be set by setting use_pereverzev =
      True in the solver config.
    maxiter: Quit iterating after this many iterations reached.
    tol: Quit iterating after the average absolute value of the residual is <=
      tol.
    coarse_tol: Coarser allowed tolerance for cases when solver develops small
      steps in the vicinity of the solution.
    delta_reduction_factor: Multiply by delta_reduction_factor after each failed
      line search step.
    tau_min: Minimum delta/delta_original allowed before the newton raphson
      routine resets at a lower timestep.
    log_iterations: If true, output diagnostic information from within iteration
      loop.

  Returns:
    x_new: Tuple, with x_new[i] giving channel i of x at the next time step
    solver_numeric_outputs: state_module.SolverNumericOutputs. Iteration and
      error info. For the error, 0 signifies residual < tol at exit, 1 signifies
      residual > tol, steps became small.
    aux_output: Extra auxiliary output from calc_coeffs.
  """
  # pyformat: enable

  coeffs_old = coeffs_callback(
      dynamic_runtime_params_slice_t,
      geo_t,
      core_profiles_t,
      x_old,
      explicit_call=True,
  )

  match initial_guess_mode:
    # LINEAR initial guess will provide the initial guess using the predictor-
    # corrector method if predictor_corrector=True in the solver config
    case enums.InitialGuessMode.LINEAR:
      # returns transport coefficients with additional pereverzev terms
      # if set by runtime_params, needed if stiff transport models (e.g. qlknn)
      # are used.
      coeffs_exp_linear = coeffs_callback(
          dynamic_runtime_params_slice_t,
          geo_t,
          core_profiles_t,
          x_old,
          allow_pereverzev=True,
          explicit_call=True,
      )

      # See linear_theta_method.py for comments on the predictor_corrector API
      x_new_guess = tuple(
          [core_profiles_t_plus_dt[name] for name in evolving_names]
      )
      init_x_new, _ = predictor_corrector_method.predictor_corrector_method(
          dt=dt,
          static_runtime_params_slice=static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
          geo_t_plus_dt=geo_t_plus_dt,
          x_old=x_old,
          x_new_guess=x_new_guess,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          coeffs_exp=coeffs_exp_linear,
          coeffs_callback=coeffs_callback,
      )
      init_x_new_vec = fvm_conversions.cell_variable_tuple_to_vec(init_x_new)
    case enums.InitialGuessMode.X_OLD:
      init_x_new_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)
    case _:
      raise ValueError(
          f'Unknown option for first guess in iterations: {initial_guess_mode}'
      )

  # Create a residual() function with only one argument: x_new.
  # The other arguments (dt, x_old, etc.) are fixed.
  # Note that core_profiles_t_plus_dt only contains the known quantities at
  # t_plus_dt, e.g. boundary conditions and prescribed profiles.
  residual_fun = functools.partial(
      residual_and_loss.theta_method_block_residual,
      dt=dt,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      transport_model=transport_model,
      explicit_source_profiles=explicit_source_profiles,
      source_models=source_models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      pedestal_model=pedestal_model,
  )
  jacobian_fun = functools.partial(
      residual_and_loss.theta_method_block_jacobian,
      dt=dt,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      evolving_names=evolving_names,
      transport_model=transport_model,
      pedestal_model=pedestal_model,
      explicit_source_profiles=explicit_source_profiles,
      source_models=source_models,
      coeffs_old=coeffs_old,
  )

  cond_fun = functools.partial(cond, tol=tol, tau_min=tau_min, maxiter=maxiter)
  delta_cond_fun = functools.partial(
      delta_cond,
      residual_fun=residual_fun,
  )
  body_fun = functools.partial(
      body,
      jacobian_fun=jacobian_fun,
      delta_cond_fun=delta_cond_fun,
      delta_reduction_factor=delta_reduction_factor,
      log_iterations=log_iterations,
  )

  # initialize state dict being passed around Newton-Raphson iterations
  residual_vec_init_x_new, aux_output_init_x_new = residual_fun(init_x_new_vec)
  initial_state = {
      'x': init_x_new_vec,
      'iterations': jnp.array(0, dtype=jax_utils.get_int_dtype()),
      'residual': residual_vec_init_x_new,
      'last_tau': jnp.array(1.0, dtype=jax_utils.get_dtype()),
      'aux_output': aux_output_init_x_new,
  }

  # log initial state if requested
  if log_iterations:
    _log_iterations(
        residual=residual_scalar(initial_state['residual']),
        iterations=initial_state['iterations'],
        dt=dt,
    )

  # carry out iterations. jax.lax.while needed for JAX-compliance
  output_state = jax_utils.py_while(cond_fun, body_fun, initial_state)

  # Create updated CellVariable instances based on state_plus_dt which has
  # updated boundary conditions and prescribed profiles.
  x_new = fvm_conversions.vec_to_cell_variable_tuple(
      output_state['x'], core_profiles_t_plus_dt, evolving_names
  )

  # Tell the caller whether or not x_new successfully reduces the residual below
  # the tolerance by providing an extra output, error.
  # error = 0: residual converged within fine tolerance (tol)
  # error = 1: not converged. Possibly backtrack to smaller dt and retry
  # error = 2: residual not strictly converged but is still within reasonable
  # tolerance (coarse_tol). Can occur when solver exits early due to small steps
  # in solution vicinity. Proceed but provide a warning to user.
  error = jax_utils.py_cond(
      residual_scalar(output_state['residual']) < tol,
      lambda: 0,  # Called when True
      lambda: jax_utils.py_cond(  # Called when False
          residual_scalar(output_state['residual']) < coarse_tol,
          lambda: 2,  # Called when True
          lambda: 1,  # Called when False
      ),
  )
  solver_numeric_outputs = state_module.SolverNumericOutputs(
      inner_solver_iterations=int(output_state['iterations']),
      solver_error_state=error,
      outer_solver_iterations=1,
  )

  coeffs_final = coeffs_callback(
      dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt,
      core_profiles_t_plus_dt,
      x_new,
      allow_pereverzev=True,
  )

  return x_new, solver_numeric_outputs, coeffs_final.auxiliary_outputs


def residual_scalar(x):
  return np.mean(np.abs(x))


def cond(
    state: dict[str, jax.Array],
    tau_min: float,
    maxiter: int,
    tol: float,
) -> bool:
  """Check if exit condition reached for Newton-Raphson iterations."""
  iteration = state['iterations'][...]
  return jnp.bool_(
      jnp.logical_and(
          jnp.logical_and(
              residual_scalar(state['residual']) > tol, iteration < maxiter
          ),
          state['last_tau'] > tau_min,
      )
  )


def body(
    input_state: dict[str, jax.Array],
    jacobian_fun,
    delta_cond_fun,
    delta_reduction_factor,
    log_iterations,
) -> dict[str, jax.Array]:
  """Calculates next guess in Newton-Raphson iteration."""

  delta_body_fun = functools.partial(
      delta_body,
      delta_reduction_factor=delta_reduction_factor,
  )

  a_mat, _ = jacobian_fun(input_state['x'])  # Ignore the aux output here.
  rhs = -input_state['residual']
  # delta = x_new - x_old
  # tau = delta/delta0, where delta0 is the delta that sets the linearized
  # residual to zero. tau < 1 when needed such that x_new meets
  # conditions of reduced residual and valid state quantities.
  # If tau < taumin while residual > tol, then the routine exits with an
  # error flag, leading to either a warning or recalculation at lower dt
  initial_delta_state = {
      'x': input_state['x'],
      'delta': jnp.linalg.solve(a_mat, rhs),
      'residual_old': input_state['residual'],
      'residual_new': input_state['residual'],
      'aux_output_new': input_state['aux_output'],
      'tau': jnp.array(1.0, dtype=jax_utils.get_dtype()),
  }
  output_delta_state = jax_utils.py_while(
      delta_cond_fun, delta_body_fun, initial_delta_state
  )

  output_state = {
      'x': input_state['x'] + output_delta_state['delta'],
      'residual': output_delta_state['residual_new'],
      'iterations': (
          jnp.array(
              input_state['iterations'][...], dtype=jax_utils.get_int_dtype()
          )
          + 1
      ),
      'last_tau': output_delta_state['tau'],
      'aux_output': output_delta_state['aux_output_new'],
  }
  if log_iterations:
    _log_iterations(
        residual=residual_scalar(output_state['residual']),
        iterations=output_state['iterations'],
        delta_reduction=output_delta_state['tau'],
    )

  return output_state


def delta_cond(
    delta_state: dict[str, jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
) -> bool:
  """Check if delta obtained from Newton step is valid.

  Args:
    delta_state: see `delta_body`.
    residual_fun: Residual function.

  Returns:
    True if the new value of `x` causes any NaNs or has increased the residual
    relative to the old value of `x`.
  """
  x_old = delta_state['x']
  x_new = x_old + delta_state['delta']
  residual_vec_x_old = delta_state['residual_old']
  residual_scalar_x_old = residual_scalar(residual_vec_x_old)
  # Avoid sanity checking inside residual, since we directly
  # afterwards check sanity on the output (NaN checking)
  # TODO(b/312453092) consider instead sanity-checking x_new
  with jax_utils.enable_errors(False):
    residual_vec_x_new, aux_output_x_new = residual_fun(x_new)
    residual_scalar_x_new = residual_scalar(residual_vec_x_new)
    delta_state['residual_new'] = residual_vec_x_new
    delta_state['aux_output_new'] = aux_output_x_new
  return jnp.bool_(
      jnp.logical_and(
          jnp.max(delta_state['delta']) > MIN_DELTA,
          jnp.logical_or(
              residual_scalar_x_old < residual_scalar_x_new,
              jnp.isnan(residual_scalar_x_new),
          ),
      ),
  )


def delta_body(
    input_delta_state: dict[str, jax.Array], delta_reduction_factor: float
) -> dict[str, jax.Array]:
  """Reduces step size for this Newton iteration."""

  return input_delta_state | dict(
      delta=input_delta_state['delta'] * delta_reduction_factor,
      tau=jnp.array(input_delta_state['tau'][...], dtype=jax_utils.get_dtype())
      * delta_reduction_factor,
  )
