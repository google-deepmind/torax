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

from typing import TypeAlias

import jax
from torax import state
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


AuxiliaryOutput: TypeAlias = block_1d_coeffs.AuxiliaryOutput


def optimizer_solve_block(
    dt: jax.Array,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t: geometry.Geometry,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    transport_model: transport_model_lib.TransportModel,
    explicit_source_profiles: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    pedestal_model: pedestal_model_lib.PedestalModel,
    coeffs_callback: calc_coeffs.CoeffsCallback,
    evolving_names: tuple[str, ...],
    initial_guess_mode: enums.InitialGuessMode,
    maxiter: int,
    tol: float,
) -> tuple[
    tuple[cell_variable.CellVariable, ...],
    state.SolverNumericOutputs,
    block_1d_coeffs.AuxiliaryOutput,
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
    static_runtime_params_slice: Static runtime parameters. Changes to these
      runtime params will trigger recompilation. A key parameter in this params
      slice is theta_imp, a coefficient in [0, 1] determining which solution
      method to use. We solve transient_coeff (x_new - x_old) / dt = theta_imp
      F(t_new) + (1 - theta_imp) F(t_old). Three values of theta_imp correspond
      to named solution methods: theta_imp = 1: Backward Euler implicit method
      (default). theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Forward Euler
      explicit method.
    dynamic_runtime_params_slice_t: Runtime params for time t (the start time of
      the step). These runtime params can change from step to step without
      triggering a recompilation.
    dynamic_runtime_params_slice_t_plus_dt: Runtime params for time t + dt.
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
    aux_output: Extra auxiliary output from the calc_coeffs.
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
    # corrector method if predictor_corrector=True in the stepper runtime params
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

  solver_numeric_outputs = state.SolverNumericOutputs()

  # Advance jaxopt_solver by one timestep
  (
      x_new_vec,
      final_loss,
      _,
      solver_numeric_outputs.inner_solver_iterations,
  ) = residual_and_loss.jaxopt_solver(
      dt=dt,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      init_x_new_vec=init_x_new_vec,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      transport_model=transport_model,
      explicit_source_profiles=explicit_source_profiles,
      source_models=source_models,
      pedestal_model=pedestal_model,
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

  coeffs_final = coeffs_callback(
      dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt,
      core_profiles_t_plus_dt,
      x_new,
      allow_pereverzev=True,
  )

  return x_new, solver_numeric_outputs, coeffs_final.auxiliary_outputs
