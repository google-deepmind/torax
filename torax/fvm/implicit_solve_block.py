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
import functools
import jax
from jax import numpy as jnp
from torax import jax_utils
from torax.fvm import block_1d_coeffs
from torax.fvm import cell_variable
from torax.fvm import fvm_conversions
from torax.fvm import residual_and_loss


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'convection_dirichlet_mode',
        'convection_neumann_mode',
        'theta_implicit',
    ],
)
def implicit_solve_block(
    dt: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    coeffs_old: block_1d_coeffs.Block1DCoeffs,
    coeffs_new: block_1d_coeffs.Block1DCoeffs,
    theta_implicit: float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[cell_variable.CellVariable, ...]:
  # pyformat: disable  # pyformat removes line breaks needed for readability
  """Runs one time step of an implicit solver on the equation defined by `coeffs`.

  This solver is relatively generic in that it models diffusion, convection,
  etc. abstractly. The caller must do the problem-specific physics calculations
  to obtain the coefficients for a particular problem.

  Args:
    dt: Discrete time step.
    x_old: Tuple containing CellVariables for each channel with their values at
    x_new_guess: Tuple containing initial guess for x_new.
    coeffs_old: Coefficients defining the equation, computed for time t.
    coeffs_new: Coefficients defining the equation, computed for time t+dt.
    theta_implicit: Coefficient in [0, 1] determining which solution method to
      use. We solve transient_coeff (x_new - x_old) / dt = theta_implicit
      F(t_new) + (1 - theta_implicit) F(t_old). Three values of theta_implicit
      correspond to named solution methods: theta_implicit = 1: Backward Euler
      implicit method (default). theta_implicit = 0.5: Crank-Nicolson.
      theta_implicit = 0: Forward Euler explicit method
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    x_new: Tuple, with x_new[i] giving channel i of x at the next time step
  """
  # pyformat: enable

  # In the linear case, we can use the same matrix formulation from the
  # nonlinear case but instead use linalg.solve to directly solve
  # residual, where the implicit coefficients are calculated with
  # an approximation of x_new, e.g. x_old for a single-step linear solve,
  # or from Picard iterations with predictor-corrector.
  # See residual_and_loss.theta_method_matrix_equation for a complete
  # description of how the equation is set up.

  x_old_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)

  lhs_mat, lhs_vec, rhs_mat, rhs_vec = (
      residual_and_loss.theta_method_matrix_equation(
          dt=dt,
          x_old=x_old,
          x_new_guess=x_new_guess,
          coeffs_old=coeffs_old,
          coeffs_new=coeffs_new,
          theta_implicit=theta_implicit,
          convection_dirichlet_mode=convection_dirichlet_mode,
          convection_neumann_mode=convection_neumann_mode,
      )
  )

  rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec - lhs_vec
  x_new = jnp.linalg.solve(lhs_mat, rhs)

  # Create updated CellVariable instances based on state_plus_dt which has
  # updated boundary conditions and prescribed profiles.
  x_new = jnp.split(x_new, len(x_old))
  out = [
      dataclasses.replace(var, value=value)
      for var, value in zip(x_new_guess, x_new)
  ]
  out = tuple(out)

  return out
