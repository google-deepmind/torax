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
import dataclasses
import functools

import jax
from jax import numpy as jnp
import jaxopt
from torax import calc_coeffs
from torax import config_slice
from torax import fvm
from torax import geometry
from torax.fvm import block_1d_coeffs
from torax.fvm import cell_variable
from torax.fvm import residual_and_loss
from torax.stepper import predictor_corrector_method


AuxiliaryOutput = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffsCallback = block_1d_coeffs.Block1DCoeffsCallback
InitialGuessMode = fvm.InitialGuessMode

# Default values, so that other modules that pass through arguments to
# these functions can use the same defaults.
INITIAL_GUESS_MODE = InitialGuessMode.LINEAR
MAXITER = 100
TOL = 1e-12


def optimizer_solve_block(
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_update_fns: tuple[cell_variable.CellVariableUpdateFn, ...],
    dt: jax.Array,
    coeffs_callback: Block1DCoeffsCallback,
    dynamic_config_slice_t: config_slice.DynamicConfigSlice,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    geo: geometry.Geometry,
    theta_imp: jax.Array | float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
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
    x_new_update_fns: Tuple containing callables that update the CellVariables
      in x_new to the correct boundary conditions at time t + dt.
    dt: Discrete time step.
    coeffs_callback: Calculates diffusion, convection etc. coefficients given a
      state. Repeatedly called by the iterative optimizer.
    dynamic_config_slice_t: Runtime configuration for time t (the start time of
      the step). These config params can change from step to step without
      triggering a recompilation.
    dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt.
    static_config_slice: Static runtime configuration. Changes to these config
      parrams will trigger recompilation
    geo: geometry object used to initialize auxiliary outputs
    theta_imp: Coefficient in [0, 1] determining which solution method to use.
      We solve transient_coeff (x_new - x_old) / dt = theta_imp F(t_new) + (1 -
      theta_imp) F(t_old). Three values of theta_imp correspond to named
      solution methods: theta_imp = 1: Backward Euler implicit method (default).
      theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Produces results
      equivalent to explicit method, but should not be used because this
      function will needless call the linear algebra solver. Use
      explicit_stepper` instead.
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.
    initial_guess_mode: chooses the initial_guess for the iterative method,
      either x_old or linear step. When taking the linear step, it is also
      recommended to use Pereverzev-Corrigan terms if the transport use
      pereverzev terms for linear solver. Is only applied in the nonlinear
      solver for the optional initial guess from the linear solver
    maxiter: See docstring of `jaxopt.LBFGS`
    tol: See docstring of `jaxopt.LBFGS`

  Returns:
    x_new: Tuple, with x_new[i] giving channel i of x at the next time step
    error: int. 0 signifies loss < tol at exit, 1 signifies loss > tol
    aux_output: Extra auxiliary output from the coeffs_callback.
  """
  # pyformat: enable

  coeffs_old = coeffs_callback(x_old, dynamic_config_slice_t)

  match initial_guess_mode:
    # LINEAR initial guess will provide the initial guess using the predictor-
    # corrector method if predictor_corrector=True in the solver config
    case InitialGuessMode.LINEAR:
      # returns transport coefficients with additional pereverzev terms
      # if set by config, needed if stiff transport models (e.g. qlknn)
      # are used.
      coeffs_exp_linear = coeffs_callback(
          x_old, dynamic_config_slice_t, allow_pereverzev=True
      )
      # See linear_theta_method.py for comments on the predictor_corrector API
      init_val = (
          x_old,
          calc_coeffs.AuxOutput.build_from_geo(geo),
      )
      init_x_new, _ = predictor_corrector_method.predictor_corrector_method(
          init_val=init_val,
          x_new_update_fns=x_new_update_fns,
          dt=dt,
          coeffs_exp=coeffs_exp_linear,
          coeffs_callback=coeffs_callback,
          dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
          static_config_slice=static_config_slice,
      )
      init_x_new_vec = jnp.concatenate([var.value for var in init_x_new])
    case InitialGuessMode.X_OLD:
      init_x_new_vec = jnp.concatenate([var.value for var in x_old])
    case _:
      raise ValueError(
          f'Unknown option for first guess in iterations: {initial_guess_mode}'
      )

  num_channels = len(x_old)

  # Create a loss() function with only one argument: x_new.
  # The other arguments (dt, x_old, etc.) are fixed.
  loss = functools.partial(
      residual_and_loss.theta_method_block_loss,
      dt=dt,
      x_old=x_old,
      x_new_update_fns=x_new_update_fns,
      coeffs_callback=coeffs_callback,
      coeffs_old=coeffs_old,
      dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
      theta_imp=theta_imp,
      convection_dirichlet_mode=convection_dirichlet_mode,
      convection_neumann_mode=convection_neumann_mode,
  )

  solver = jaxopt.LBFGS(fun=loss, maxiter=maxiter, tol=tol, has_aux=True)
  solver_output = solver.run(init_x_new_vec)
  x_new_vec = solver_output.params
  aux_output = solver_output.state.aux

  x_new_values = jnp.split(x_new_vec, num_channels)

  # Make new CellVariable instances with same constraints as originals
  x_new = [
      dataclasses.replace(var, value=value)
      for var, value in zip(x_old, x_new_values)
  ]
  x_new = tuple(x_new)

  # Tell the caller whether or not x_new successfully reduces the loss below
  # the tolerance by providing an extra output, error.
  loss_scalar, _ = loss(x_new_vec)
  error = jax.lax.cond(
      loss_scalar > tol,
      lambda: 1,  # Called when True
      lambda: 0,  # Called when False
  )

  return x_new, error, aux_output
