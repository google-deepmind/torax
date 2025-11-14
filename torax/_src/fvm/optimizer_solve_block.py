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
"""The `optimizer_solve_block` function.

See function docstring for details.
"""
import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss
from torax._src.geometry import geometry
from torax._src.solver import predictor_corrector_method
from torax._src.sources import source_profiles

AuxiliaryOutput: TypeAlias = block_1d_coeffs.AuxiliaryOutput


@functools.partial(
    jax.jit,
    static_argnames=[
        'physics_models',
        'coeffs_callback',
        'evolving_names',
        'initial_guess_mode',
    ],
)
def optimizer_solve_block(
    dt: jax.Array,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo_t: geometry.Geometry,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    coeffs_callback: calc_coeffs.CoeffsCallback,
    evolving_names: tuple[str, ...],
    initial_guess_mode: enums.InitialGuessMode,
    maxiter: int,
    tol: float,
) -> tuple[
    tuple[cell_variable.CellVariable, ...],
    state.SolverNumericOutputs,
]:
  # pyformat: disable  # pyformat removes line breaks needed for readability
  """Runs one time step of an optimization-based solver on the equation defined by `coeffs`.

  This solver is relatively generic in that it models diffusion, convection,
  etc. abstractly. The caller must do the problem-specific physics calculations
  to obtain the coefficients for a particular problem.

  This solver uses iterative optimization to minimize the norm of the residual
  between two sides of the equation describing a theta method update.

  Args:
    dt: Discrete time step.
    runtime_params_t: Runtime params for time t (the start time of the step).
      These runtime params can change from step to step without triggering a
      recompilation.
    runtime_params_t_plus_dt: Runtime params for time t + dt.
    geo_t: Geometry object used to initialize auxiliary outputs at time t.
    geo_t_plus_dt: Geometry object used to initialize auxiliary outputs at time
      t + dt.
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
    initial_guess_mode: Chooses the initial_guess for the iterative method,
      either x_old or linear step. When taking the linear step, it is also
      recommended to use Pereverzev-Corrigan terms if the transport use
      pereverzev terms for linear solver. Is only applied in the nonlinear
      solver for the optional initial guess from the linear solver.
    maxiter: See docstring of `jaxopt.LBFGS`.
    tol: See docstring of `jaxopt.LBFGS`.

  Returns:
    x_new: Tuple, with x_new[i] giving channel i of x at the next time step
    solver_numeric_outputs: SolverNumericOutputs. Info about iterations and
      errors
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
    # corrector method if use_predictor_corrector=True in the solver runtime
    # params
    case enums.InitialGuessMode.LINEAR:
      # returns transport coefficients with additional pereverzev terms
      # if set by runtime_params, needed if stiff transport models (e.g. qlknn)
      # are used.
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

  solver_numeric_outputs = state.SolverNumericOutputs(
      inner_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
      outer_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
      solver_error_state=jnp.array(0, jax_utils.get_int_dtype()),
      sawtooth_crash=False,
  )

  # Advance jaxopt_solver by one timestep
  (
      x_new_vec,
      final_loss,
      solver_numeric_outputs.inner_solver_iterations,
  ) = residual_and_loss.jaxopt_solver(
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      init_x_new_vec=init_x_new_vec,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      physics_models=physics_models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      maxiter=maxiter,
      tol=tol,
  )

  # Create updated CellVariable instances based on core_profiles_t_plus_dt which
  # has updated boundary conditions and prescribed profiles.
  x_new = fvm_conversions.vec_to_cell_variable_tuple(
      x_new_vec, core_profiles_t_plus_dt, evolving_names
  )

  # Tell the caller whether or not x_new successfully reduces the loss below
  # the tolerance by providing an extra output, error.
  solver_numeric_outputs.solver_error_state = jax.lax.cond(
      final_loss > tol,
      lambda: 1,  # Called when True
      lambda: 0,  # Called when False
  )

  return x_new, solver_numeric_outputs
