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

import jax
from torax import fvm
from torax import geometry
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import block_1d_coeffs
from torax.fvm import cell_variable
from torax.fvm import fvm_conversions
from torax.fvm import residual_and_loss
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
from torax.stepper import predictor_corrector_method
from torax.transport_model import transport_model as transport_model_lib


AuxiliaryOutput = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffsCallback = block_1d_coeffs.Block1DCoeffsCallback
InitialGuessMode = fvm.InitialGuessMode

# Default values, so that other modules that pass through arguments to
# these functions can use the same defaults.
# TODO(b/330172917) allow these variables to be set in config
INITIAL_GUESS_MODE = InitialGuessMode.LINEAR
MAXITER = 100
TOL = 1e-12


def optimizer_solve_block(
    dt: jax.Array,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    transport_model: transport_model_lib.TransportModel,
    explicit_source_profiles: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    coeffs_callback: Block1DCoeffsCallback,
    evolving_names: tuple[str, ...],
    initial_guess_mode: InitialGuessMode = INITIAL_GUESS_MODE,
    maxiter=MAXITER,
    tol=TOL,
) -> tuple[tuple[cell_variable.CellVariable, ...], int, AuxiliaryOutput]:
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
    geo: Geometry object used to initialize auxiliary outputs.
    x_old: Tuple containing CellVariables for each channel with their values at
      the start of the time step.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    transport_model: Turbulent transport model callable.
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    source_models: Collection of source callables to generate source PDE
      coefficients.
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
    error: int. 0 signifies loss < tol at exit, 1 signifies loss > tol
    aux_output: Extra auxiliary output from the calc_coeffs.
  """
  # pyformat: enable

  coeffs_old = coeffs_callback(
      dynamic_runtime_params_slice_t, x_old, explicit_call=True
  )

  match initial_guess_mode:
    # LINEAR initial guess will provide the initial guess using the predictor-
    # corrector method if predictor_corrector=True in the stepper runtime params
    case InitialGuessMode.LINEAR:
      # returns transport coefficients with additional pereverzev terms
      # if set by runtime_params, needed if stiff transport models (e.g. qlknn)
      # are used.
      coeffs_exp_linear = coeffs_callback(
          dynamic_runtime_params_slice_t,
          x_old,
          allow_pereverzev=True,
          explicit_call=True,
      )
      # See linear_theta_method.py for comments on the predictor_corrector API
      x_new_init = tuple(
          [core_profiles_t_plus_dt[name] for name in evolving_names]
      )
      init_val = (
          x_new_init,
          # Initialized here with correct shapes to help with tracing in case
          # this is jitted.
          (
              source_models_lib.build_all_zero_profiles(
                  geo,
                  source_models,
              ),
              state.CoreTransport.zeros(geo),
          ),
      )
      init_x_new, _ = predictor_corrector_method.predictor_corrector_method(
          dt=dt,
          static_runtime_params_slice=static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
          init_val=init_val,
          x_old=x_old,
          coeffs_exp=coeffs_exp_linear,
          coeffs_callback=coeffs_callback,
      )
      init_x_new_vec = fvm_conversions.cell_variable_tuple_to_vec(init_x_new)
    case InitialGuessMode.X_OLD:
      init_x_new_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)
    case _:
      raise ValueError(
          f'Unknown option for first guess in iterations: {initial_guess_mode}'
      )

  # Advance jaxopt_solver by one timestep
  x_new_vec, final_loss, aux_output = residual_and_loss.jaxopt_solver(
      dt=dt,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
      geo=geo,
      x_old=x_old,
      init_x_new_vec=init_x_new_vec,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      transport_model=transport_model,
      explicit_source_profiles=explicit_source_profiles,
      source_models=source_models,
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
  error = jax.lax.cond(
      final_loss > tol,
      lambda: 1,  # Called when True
      lambda: 0,  # Called when False
  )

  return x_new, error, aux_output
