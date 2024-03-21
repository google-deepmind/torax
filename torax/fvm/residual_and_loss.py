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
"""Residual functions and loss functions.

Residual functions define a full differential equation and give a vector
measuring (left hand side) - (right hand side). Loss functions collapse
these to scalar functions, for example using mean squared error.
Residual functions are for use with e.g. the Newton-Raphson method
while loss functions can be minimized using any optimization method.
"""

import jax
from jax import numpy as jnp
from torax import config_slice
from torax import jax_utils
from torax import state as state_module
from torax.fvm import block_1d_coeffs
from torax.fvm import cell_variable
from torax.fvm import discrete_system
from torax.fvm import fvm_conversions

AuxiliaryOutput = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffs = block_1d_coeffs.Block1DCoeffs
Block1DCoeffsCallback = block_1d_coeffs.Block1DCoeffsCallback


def theta_method_matrix_equation(
    x_new_vec: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    state_t_plus_dt: state_module.State,
    evolving_names: tuple[str, ...],
    dt: jax.Array,
    coeffs_old: Block1DCoeffs,
    coeffs_callback: Block1DCoeffsCallback,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    theta_imp: jax.Array | float = 1.0,
    allow_pereverzev: bool = False,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, AuxiliaryOutput]:
  """Returns the left-hand and right-hand sides of the theta method equation.

  The theta method solves a differential equation
  ```
  tc_out partial (tc_in x) / partial t = F
  ```
  where `tc` is the transient coefficient, with `tc_out`
  being outside the partial derivative and `tc_in` inside it.

  We rearrange this to
  ```
  partial tc_in x / partial t = F / tc_out
  ```

  The theta method calculates one discrete time step by solving:
  ```
  (tc_in_new x_new - tc_in_old x_old) / dt =
  theta_imp F_new / tc_out_new + theta_exp F_old / tc_out_old
  ```

  The equation is on the cell grid where `tc` is never zero. Therefore
  it's safe to multiply equation by `dt/tc_in_new` and scale the residual to
  `x`, which has O(1) values and thus the residual is scaled appropriately.

  We thus rearrange to:
  ```
  x_new - tc_in_old/tc_in_new x_old =
  dt theta_imp F_new / (tc_out_new tc_in_new) +
  dt theta_exp F_old / (tc_out_old tc_in_new)
  ```

  Rearranging we obtain
  ```
  x_new - dt theta_imp F_new / (tc_out_new tc_in_new) =
  tc_in_old/tc_in_new x_old + dt theta_exp F_old / (tc_out_old tc_in_new)
  ```

  We now substitute in `F = Cu + c`:

  ```
  (I - dt theta_imp diag(1/(tc_out_new tc_in_new)) C_new) x_new
  - dt theta_imp diag(1/(tc_out_new tc_in_new)) c_new
  =
  (diag(tc_in_old/tc_in_new)
  + dt theta_exp diag(1/(tc_out_old tc_in_new)) C_old) x_old
  + dt theta_exp diag(1/(tc_out_old tc_in_new)) c_old
  ```

  Args:
    x_new_vec: A single vector containing all values of x_new concatenated
      together.
    x_old: The starting x defined as a tuple of CellVariables.
    state_t_plus_dt: Sim state which contains all available prescribed
      quantities at the end of the time step. This includes evolving boundary
      conditions and prescribed time-dependent profiles that are not being
      evolved by the PDE system.
    evolving_names: The names of variables within the state that should evolve.
    dt: Time step duration.
    coeffs_old: The coefficients calculated at x_old. We have this passed in
      separately rather than using `coeffs_callback` to reduce the size of the
      computation graph passed to the `minimize` function and thus reduces
      compilation time.
    coeffs_callback: Calculates diffusion, convection etc. coefficients given a
      state. Repeatedly called by the nonlinear solver as x_new changes. This
      callback is allowed to inject "fake" coefficients, e.g. such as keeping
      some coefficients locked to coeffs_old, introducing stop_gradient, etc. In
      particular a callback that injects coeffs_old with stop_gradient on every
      timestep should make the nonlinear solver solution equivalent to the
      `implicit_solve_block` solution. This function assumes that the callback
      has already been called once with first=True to create `coeffs_old`.
    dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt. Used
      with the coeffs_callback to compute the coefficients used with x_new.
    theta_imp: Coefficient on implicit term of theta method.
    allow_pereverzev: Passed to the coeffs_callback to determine whether the
      callback can use pereverzev-corrigan terms. These are typically not
      desired with nonlinear solvers. Note that, if True, the actual use of the
      terms depends on the static_config_slice.solver.use_pereverzev (depending
      on the implementation of the coeffs_callback).
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    For the equation A x_new + a_vec = B x_old + b_vec. This function returns
     - left-hand side matrix, A
     - left-hand side vector, a
     - right-hand side matrix B
     - right-hand side vector, b
     - auxiliary output from calculating new coefficients for x_new.
  """

  # Convert input flattened evolving state vector to tuple of CellVariables
  x_new = fvm_conversions.vec_to_cell_variable_tuple(
      x_new_vec, state_t_plus_dt, evolving_names
  )

  coeffs_new = coeffs_callback(
      x_new, dynamic_config_slice_t_plus_dt, allow_pereverzev=allow_pereverzev
  )
  aux_output = coeffs_new.auxiliary_outputs

  tc_out_old = jnp.concatenate(coeffs_old.transient_out_cell)
  tc_in_old = jnp.concatenate(coeffs_old.transient_in_cell)
  tc_out_new = jnp.concatenate(coeffs_new.transient_out_cell)
  tc_in_new = jnp.concatenate(coeffs_new.transient_in_cell)

  eps = 1e-7
  # adding sanity checks for values in denominators
  tc_in_new = jax_utils.error_if(
      tc_in_new, jnp.any(tc_in_new < eps), msg='tc_in_new unexpectedly < eps'
  )
  tc_in_new = jax_utils.error_if(
      tc_in_new,
      jnp.any(tc_out_new * tc_in_new < eps),
      msg='tc_out_new*tc_in_new unexpectedly < eps',
  )
  tc_in_new = jax_utils.error_if(
      tc_in_new,
      jnp.any(tc_out_old * tc_in_new < eps),
      msg='tc_out_old*tc_in_new unexpectedly < eps',
  )

  left_transient = jnp.identity(len(x_new_vec))
  right_transient = jnp.diag(jnp.squeeze(tc_in_old / tc_in_new))

  c_mat_old, c_old = discrete_system.calc_c(
      coeffs_old,
      x_old,
      convection_dirichlet_mode,
      convection_neumann_mode,
  )
  c_mat_new, c_new = discrete_system.calc_c(
      coeffs_new,
      x_new,
      convection_dirichlet_mode,
      convection_neumann_mode,
  )

  theta_exp = 1.0 - theta_imp

  lhs_mat = left_transient - dt * theta_imp * jnp.dot(
      jnp.diag(1 / (tc_out_new * tc_in_new)), c_mat_new
  )
  lhs_vec = -theta_imp * dt * (1 / (tc_out_new * tc_in_new)) * c_new
  rhs_mat = right_transient + dt * theta_exp * jnp.dot(
      jnp.diag(1 / (tc_out_old * tc_in_new)), c_mat_old
  )
  rhs_vec = dt * theta_exp * (1 / (tc_out_old * tc_in_new)) * c_old

  return lhs_mat, lhs_vec, rhs_mat, rhs_vec, aux_output


def theta_method_block_residual(
    x_new_vec: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    state_t_plus_dt: state_module.State,
    evolving_names: tuple[str, ...],
    dt: jax.Array,
    coeffs_old: Block1DCoeffs,
    coeffs_callback: Block1DCoeffsCallback,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    theta_imp: jax.Array | float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[jax.Array, AuxiliaryOutput]:
  """Residual of theta-method equation for state at next time-step.

  Args:
    x_new_vec: A single vector containing all values of x_new concatenated
      together.
    x_old: The starting x defined as a tuple of CellVariables.
    state_t_plus_dt: Sim state which contains all available prescribed
      quantities at the end of the time step. This includes evolving boundary
      conditions and prescribed time-dependent profiles that are not being
      evolved by the PDE system.
    evolving_names: The names of variables within the state that should evolve.
    dt: Time step duration.
    coeffs_old: The coefficients calculated at x_old. We have this passed in
      separately rather than using `coeffs_callback` to reduce the size of the
      computation graph passed to the `minimize` function and thus reduces
      compilation time.
    coeffs_callback: Calculates diffusion, convection etc. coefficients given a
      state. Repeatedly called by the nonlinear solver as x_new changes. This
      callback is allowed to inject "fake" coefficients, e.g. such as keeping
      some coefficients locked to coeffs_old, introducing stop_gradient, etc. In
      particular a callback that injects coeffs_old with stop_gradient on every
      timestep should make the nonlinear solver solution equivalent to the
      `implicit_solve_block` solution. This function assumes that the callback
      has already been called once with first=True to create `coeffs_old`.
    dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt. Used
      with the coeffs_callback to compute the coefficients used with x_new.
    theta_imp: Coefficient on implicit term of theta method.
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    residual: Vector residual between LHS and RHS of the theta method equation.
    aux_output: Auxiliary outputs coming from the coeffs_callback()
  """
  x_old_vec = jnp.concatenate([var.value for var in x_old])
  lhs_mat, lhs_vec, rhs_mat, rhs_vec, aux_output = theta_method_matrix_equation(
      x_new_vec=x_new_vec,
      x_old=x_old,
      state_t_plus_dt=state_t_plus_dt,
      evolving_names=evolving_names,
      dt=dt,
      coeffs_old=coeffs_old,
      coeffs_callback=coeffs_callback,
      dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
      theta_imp=theta_imp,
      convection_dirichlet_mode=convection_dirichlet_mode,
      convection_neumann_mode=convection_neumann_mode,
  )

  lhs = jnp.dot(lhs_mat, x_new_vec) + lhs_vec
  rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec

  residual = lhs - rhs
  return residual, aux_output


def theta_method_block_loss(
    x_new_vec: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    state_t_plus_dt: state_module.State,
    evolving_names: tuple[str, ...],
    dt: jax.Array,
    coeffs_old: Block1DCoeffs,
    coeffs_callback: Block1DCoeffsCallback,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    theta_imp: jax.Array | float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[jax.Array, AuxiliaryOutput]:
  """Loss for the optimizer method of nonlinear solution.

  Args:
    x_new_vec: A single vector containing all values of x_new concatenated
      together.
    x_old: The starting x defined as a tuple of CellVariables.
    state_t_plus_dt: Sim state which contains all available prescribed
      quantities at the end of the time step. This includes evolving boundary
      conditions and prescribed time-dependent profiles that are not being
      evolved by the PDE system.
    evolving_names: The names of variables within the state that should evolve.
    dt: Time step duration.
    coeffs_old: The coefficients calculated at x_old. We have this passed in
      separately rather than using `coeffs_callback` to reduce the size of the
      computation graph passed to the `minimize` function and thus reduces
      compilation time.
    coeffs_callback: Calculates diffusion, convection etc. coefficients given a
      state. Repeatedly called by the nonlinear solver as x_new changes. This
      callback is allowed to inject "fake" coefficients, e.g. such as keeping
      some coefficients locked to coeffs_old, introducing stop_gradient, etc. In
      particular a callback that injects coeffs_old with stop_gradient on every
      timestep should make the nonlinear solver solution equivalent to the
      `implicit_solve_block` solution. This function assumes that the callback
      has already been called once with first=True to create `coeffs_old`.
    dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt. Used
      with the coeffs_callback to compute the coefficients used with x_new.
    theta_imp: Coefficient on implicit term of theta method.
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    loss: mean squared loss of theta method residual.
    aux_output: Auxiliary output coming from the coeffs_callback.
  """

  residual, aux_output = theta_method_block_residual(
      x_new_vec=x_new_vec,
      x_old=x_old,
      state_t_plus_dt=state_t_plus_dt,
      evolving_names=evolving_names,
      dt=dt,
      coeffs_old=coeffs_old,
      coeffs_callback=coeffs_callback,
      dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
      theta_imp=theta_imp,
      convection_dirichlet_mode=convection_dirichlet_mode,
      convection_neumann_mode=convection_neumann_mode,
  )
  loss = jnp.mean(jnp.square(residual))
  return loss, aux_output
