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

"""The `implicit_solve_block` function.

See function docstring for details.
"""

import dataclasses

import jax
from jax import numpy as jnp
from torax import config_slice
from torax.fvm import block_1d_coeffs
from torax.fvm import cell_variable
from torax.fvm import residual_and_loss


def implicit_solve_block(
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_vec_guess: jax.Array,
    x_new_update_fns: tuple[cell_variable.CellVariableUpdateFn, ...],
    dt: jax.Array,
    coeffs_old: block_1d_coeffs.Block1DCoeffs,
    coeffs_callback: block_1d_coeffs.Block1DCoeffsCallback,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    theta_imp: jax.Array | float = 1.0,
    allow_pereverzev: bool = True,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[
    tuple[cell_variable.CellVariable, ...],
    block_1d_coeffs.AuxiliaryOutput,
]:
  # pyformat: disable  # pyformat removes line breaks needed for readability
  """Runs one time step of an implicit solver on the equation defined by `coeffs`.

  This solver is relatively generic in that it models diffusion, convection,
  etc. abstractly. The caller must do the problem-specific physics calculations
  to obtain the coefficients for a particular problem.

  Args:
    x_old: Tuple containing CellVariables for each channel with their values at
      the start of the time step.
    x_new_vec_guess: Initial guess for x_new. Used to compute new coefficients
      at time t + dt.
    x_new_update_fns: Tuple containing callables that update the CellVariables
      in x_new to the correct boundary conditions at time t + dt.
    dt: Discrete time step.
    coeffs_old: Coefficients defining the equation, computed for time t.
      Coefficients can depend on time-varying parameters, which is why both the
      coefficients at time t and t+dt are required for the explicit and implicit
      components, respectively.
    coeffs_callback: Calculates diffusion, convection etc. coefficients given a
      state. In practice, this is typically a sim.CoeffsCallback.
    dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt. Used
      with the coeffs_callback to compute the coefficients used with x_new.
    theta_imp: Coefficient in [0, 1] determining which solution method to use.
      We solve transient_coeff (x_new - x_old) / dt = theta_imp F(t_new) + (1 -
      theta_imp) F(t_old). Three values of theta_imp correspond to named
      solution methods: theta_imp = 1: Backward Euler implicit method (default).
      theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Produces results
      equivalent to explicit method, but should not be used because this
      function will needless call the linear algebra solver. Use
      `explicit_stepper` instead.
    allow_pereverzev: Passed to the coeffs_callback to determine whether the
      callback can use pereverzev-corrigan terms. Note that, if True, the actual
      use of the terms depends on the static_config_slice.solver.use_pereverzev
      (depending on the implementation of the coeffs_callback).
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    x_new: Tuple, with x_new[i] giving channel i of x at the next time step
    Auxiliary output from computing the matrices used to solve for x_new.
  """
  # pyformat: enable

  # In the linear case, we can use the same matrix formulation from the
  # nonlinear case but instead use linalg.solve to directly solve
  # residual, where the implicit coefficients are calculated with
  # an approximation of x_new, e.g. x_old for a single-step linear solve,
  # or from Picard iterations with predictor-corrector.
  # See residual_and_loss.theta_method_matrix_equation for a complete
  # description of how the equation is set up.

  num_channels = len(x_old)
  x_old_vec = jnp.concatenate([var.value for var in x_old])

  lhs_mat, lhs_vec, rhs_mat, rhs_vec, aux_output = (
      residual_and_loss.theta_method_matrix_equation(
          x_new_vec=x_new_vec_guess,
          x_old=x_old,
          x_new_update_fns=x_new_update_fns,
          dt=dt,
          coeffs_old=coeffs_old,
          coeffs_callback=coeffs_callback,
          dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
          theta_imp=theta_imp,
          convection_dirichlet_mode=convection_dirichlet_mode,
          convection_neumann_mode=convection_neumann_mode,
          allow_pereverzev=allow_pereverzev,
      )
  )

  rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec - lhs_vec
  x_new = jnp.linalg.solve(lhs_mat, rhs)

  x_new = jnp.split(x_new, num_channels)

  # Make new CellVariable instances with updated constraints as well.
  out = [
      update_fn(dataclasses.replace(var, value=value))
      for var, value, update_fn in zip(x_old, x_new, x_new_update_fns)
  ]
  out = tuple(out)

  return out, aux_output
