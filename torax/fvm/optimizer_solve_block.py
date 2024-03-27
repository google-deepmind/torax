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
from torax import config_slice
from torax import fvm
from torax import geometry
from torax import state
from torax.fvm import block_1d_coeffs
from torax.fvm import cell_variable
from torax.fvm import fvm_conversions
from torax.fvm import residual_and_loss
from torax.sources import source_profiles
from torax.stepper import predictor_corrector_method
from torax.transport_model import transport_model as transport_model_lib


AuxiliaryOutput = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffsCallback = block_1d_coeffs.Block1DCoeffsCallback
InitialGuessMode = fvm.InitialGuessMode

# Default values, so that other modules that pass through arguments to
# these functions can use the same defaults.
# TODO( b/330172917)
INITIAL_GUESS_MODE = InitialGuessMode.LINEAR
MAXITER = 100
TOL = 1e-12


def optimizer_solve_block(
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    evolving_names: tuple[str, ...],
    dt: jax.Array,
    coeffs_callback: Block1DCoeffsCallback,
    dynamic_config_slice_t: config_slice.DynamicConfigSlice,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    geo: geometry.Geometry,
    transport_model: transport_model_lib.TransportModel,
    sources: source_profiles.Sources,
    explicit_source_profiles: source_profiles.SourceProfiles,
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
    x_old: Tuple containing CellVariables for each channel with their values at
      the start of the time step.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    dt: Discrete time step.
    coeffs_callback: Calculates diffusion, convection etc. coefficients given a
      core_profiles. Repeatedly called by the iterative optimizer.
    dynamic_config_slice_t: Runtime configuration for time t (the start time of
      the step). These config params can change from step to step without
      triggering a recompilation.
    dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt.
    static_config_slice: Static runtime configuration. Changes to these config
      params will trigger recompilation. A key parameter in static_config slice
      is theta_imp, a coefficient in [0, 1] determining which solution method to
      use. We solve transient_coeff (x_new - x_old) / dt = theta_imp F(t_new) +
      (1 - theta_imp) F(t_old). Three values of theta_imp correspond to named
      solution methods: theta_imp = 1: Backward Euler implicit method (default).
      theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Forward Euler explicit
      method.
    geo: Geometry object used to initialize auxiliary outputs.
    transport_model: Turbulent transport model callable.
    sources: Collection of source callables to generate source PDE coefficients
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
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
      x_old, dynamic_config_slice_t, explicit_call=True
  )

  match initial_guess_mode:
    # LINEAR initial guess will provide the initial guess using the predictor-
    # corrector method if predictor_corrector=True in the solver config
    case InitialGuessMode.LINEAR:
      # returns transport coefficients with additional pereverzev terms
      # if set by config, needed if stiff transport models (e.g. qlknn)
      # are used.
      coeffs_exp_linear = coeffs_callback(
          x_old,
          dynamic_config_slice_t,
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
              state.CoreTransport.zeros(geo),
              state.AuxOutput.zeros(geo),
          ),
      )
      init_x_new, _ = predictor_corrector_method.predictor_corrector_method(
          init_val=init_val,
          x_old=x_old,
          dt=dt,
          coeffs_exp=coeffs_exp_linear,
          coeffs_callback=coeffs_callback,
          dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
          static_config_slice=static_config_slice,
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
      init_x_new_vec=init_x_new_vec,
      x_old=x_old,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      geo=geo,
      dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
      static_config_slice=static_config_slice,
      dt=dt,
      evolving_names=evolving_names,
      coeffs_old=coeffs_old,
      transport_model=transport_model,
      sources=sources,
      explicit_source_profiles=explicit_source_profiles,
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
