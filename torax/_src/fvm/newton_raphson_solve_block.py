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
from typing import Final

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state as state_module
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.fvm import fvm_conversions
from torax._src.fvm import jax_root_finding
from torax._src.fvm import residual_and_loss
from torax._src.geometry import geometry
from torax._src.solver import predictor_corrector_method
from torax._src.sources import source_profiles


# Delta is a vector. If no entry of delta is above this magnitude, we terminate
# the delta loop. This is to avoid getting stuck in an infinite loop in edge
# cases with bad numerics.
MIN_DELTA: Final[float] = 1e-7


@functools.partial(
    jax.jit,
    static_argnames=[
        'evolving_names',
        'coeffs_callback',
        'initial_guess_mode',
        'log_iterations',
    ],
)
def newton_raphson_solve_block(
    dt: array_typing.FloatScalar,
    runtime_params_t: runtime_params_slice.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t: geometry.Geometry,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state_module.CoreProfiles,
    core_profiles_t_plus_dt: state_module.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
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
    runtime_params_t: Runtime parameters for time t.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
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
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    physics_models: Physics models used for the calculations.
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
  """
  # pyformat: enable

  coeffs_old = coeffs_callback(
      runtime_params_t,
      geo_t,
      core_profiles_t,
      x_old,
      explicit_source_profiles=explicit_source_profiles,
      explicit_call=True,
  )

  match initial_guess_mode:
    # LINEAR initial guess will provide the initial guess using the predictor-
    # corrector method if predictor_corrector=True in the solver config
    case enums.InitialGuessMode.LINEAR:
      # returns transport coefficients with additional pereverzev terms
      # if set by runtime_params, needed if stiff transport models
      # (e.g. qlknn) are used.
      coeffs_exp_linear = coeffs_callback(
          runtime_params_t,
          geo_t,
          core_profiles_t,
          x_old,
          explicit_source_profiles=explicit_source_profiles,
          allow_pereverzev=True,
          explicit_call=True,
      )

      # See linear_theta_method.py for comments on the predictor_corrector API
      x_new_guess = convertors.core_profiles_to_solver_x_tuple(
          core_profiles_t_plus_dt, evolving_names
      )
      init_x_new = predictor_corrector_method.predictor_corrector_method(
          dt=dt,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo_t_plus_dt=geo_t_plus_dt,
          x_old=x_old,
          x_new_guess=x_new_guess,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          coeffs_exp=coeffs_exp_linear,
          coeffs_callback=coeffs_callback,
          explicit_source_profiles=explicit_source_profiles,
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
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      physics_models=physics_models,
      explicit_source_profiles=explicit_source_profiles,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
  )

  # TODO(b/438673871): Set use_jax_custom_root=True once the compile time
  # regression is resolved.
  x_root, metadata = jax_root_finding.root_newton_raphson(
      fun=residual_fun,
      x0=init_x_new_vec,
      maxiter=maxiter,
      tol=tol,
      coarse_tol=coarse_tol,
      delta_reduction_factor=delta_reduction_factor,
      tau_min=tau_min,
      log_iterations=log_iterations,
      use_jax_custom_root=False,
  )

  # Create updated CellVariable instances based on state_plus_dt which has
  # updated boundary conditions and prescribed profiles.
  x_new = fvm_conversions.vec_to_cell_variable_tuple(
      x_root, core_profiles_t_plus_dt, evolving_names
  )
  solver_numeric_outputs = state_module.SolverNumericOutputs(
      inner_solver_iterations=metadata.iterations,
      solver_error_state=metadata.error,
      outer_solver_iterations=jnp.array(1, jax_utils.get_int_dtype()),
      sawtooth_crash=False,
  )

  return x_new, solver_numeric_outputs
