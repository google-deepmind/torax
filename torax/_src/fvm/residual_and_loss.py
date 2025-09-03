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

import functools
from typing import TypeAlias

import chex
import jax
from jax import numpy as jnp
import jaxopt
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import updaters
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import discrete_system
from torax._src.fvm import fvm_conversions
from torax._src.geometry import geometry
from torax._src.sources import source_profiles

Block1DCoeffs: TypeAlias = block_1d_coeffs.Block1DCoeffs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'convection_dirichlet_mode',
        'convection_neumann_mode',
        'theta_implicit',
    ],
)
def theta_method_matrix_equation(
    dt: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    coeffs_old: Block1DCoeffs,
    coeffs_new: Block1DCoeffs,
    theta_implicit: float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Returns the left-hand and right-hand sides of the theta method equation.

  The theta method solves a differential equation

    tc_out partial (tc_in x) / partial t = F

  where `tc` is the transient coefficient, with `tc_out`
  being outside the partial derivative and `tc_in` inside it.

  We rearrange this to

    partial tc_in x / partial t = F / tc_out

  The theta method calculates one discrete time step by solving:

    | (tc_in_new x_new - tc_in_old x_old) / dt =
    | theta_implicit F_new / tc_out_new + theta_exp F_old / tc_out_old

  The equation is on the cell grid where `tc` is never zero. Therefore
  it's safe to multiply equation by `dt/tc_in_new` and scale the residual to
  `x`, which has O(1) values and thus the residual is scaled appropriately.

  We thus rearrange to:

    | x_new - tc_in_old/tc_in_new x_old =
    | dt theta_implicit F_new / (tc_out_new tc_in_new) +
    | dt theta_exp F_old / (tc_out_old tc_in_new)

  Rearranging we obtain

    | x_new - dt theta_implicit F_new / (tc_out_new tc_in_new) =
    | tc_in_old/tc_in_new x_old + dt theta_exp F_old / (tc_out_old tc_in_new)

  We now substitute in `F = Cu + c`:

    | (I - dt theta_implicit diag(1/(tc_out_new tc_in_new)) C_new) x_new
    | - dt theta_implicit diag(1/(tc_out_new tc_in_new)) c_new
    | =
    | (diag(tc_in_old/tc_in_new)
    | + dt theta_exp diag(1/(tc_out_old tc_in_new)) C_old) x_old
    | + dt theta_exp diag(1/(tc_out_old tc_in_new)) c_old

  Args:
    dt: Time step duration.
    x_old: The starting x defined as a tuple of CellVariables.
    x_new_guess: Current guess of x_new defined as a tuple of CellVariables.
    coeffs_old: The coefficients calculated at x_old.
    coeffs_new: The coefficients calculated at x_new.
    theta_implicit: Coefficient on implicit term of theta method.
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
  """

  x_new_guess_vec = fvm_conversions.cell_variable_tuple_to_vec(x_new_guess)

  theta_exp = 1.0 - theta_implicit

  tc_in_old = jnp.concatenate(coeffs_old.transient_in_cell)
  tc_out_new = jnp.concatenate(coeffs_new.transient_out_cell)
  tc_in_new = jnp.concatenate(coeffs_new.transient_in_cell)
  chex.assert_rank(tc_in_old, 1)
  chex.assert_rank(tc_out_new, 1)
  chex.assert_rank(tc_in_new, 1)

  eps = 1e-7
  # adding sanity checks for values in denominators
  # TODO(b/326577625) remove abs in checks once x_new range is restricted
  tc_in_new = jax_utils.error_if(
      tc_in_new,
      jnp.any(jnp.abs(tc_in_new) < eps),
      msg='|tc_in_new| unexpectedly < eps',
  )
  tc_in_new = jax_utils.error_if(
      tc_in_new,
      jnp.any(jnp.abs(tc_out_new * tc_in_new) < eps),
      msg='|tc_out_new*tc_in_new| unexpectedly < eps',
  )

  left_transient = jnp.identity(len(x_new_guess_vec))
  right_transient = jnp.diag(jnp.squeeze(tc_in_old / tc_in_new))

  c_mat_new, c_new = discrete_system.calc_c(
      x_new_guess,
      coeffs_new,
      convection_dirichlet_mode,
      convection_neumann_mode,
  )

  broadcasted = jnp.expand_dims(1 / (tc_out_new * tc_in_new), 1)

  lhs_mat = left_transient - dt * theta_implicit * broadcasted * c_mat_new
  lhs_vec = -theta_implicit * dt * (1 / (tc_out_new * tc_in_new)) * c_new

  if theta_exp > 0.0:
    tc_out_old = jnp.concatenate(coeffs_old.transient_out_cell)
    tc_in_new = jax_utils.error_if(
        tc_in_new,
        jnp.any(jnp.abs(tc_out_old * tc_in_new) < eps),
        msg='|tc_out_old*tc_in_new| unexpectedly < eps',
    )
    c_mat_old, c_old = discrete_system.calc_c(
        x_old,
        coeffs_old,
        convection_dirichlet_mode,
        convection_neumann_mode,
    )
    broadcasted = jnp.expand_dims(1 / (tc_out_old * tc_in_new), 1)
    rhs_mat = right_transient + dt * theta_exp * broadcasted * c_mat_old
    rhs_vec = dt * theta_exp * (1 / (tc_out_old * tc_in_new)) * c_old
  else:
    rhs_mat = right_transient
    rhs_vec = jnp.zeros_like(x_new_guess_vec)

  return lhs_mat, lhs_vec, rhs_mat, rhs_vec


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def theta_method_block_residual(
    x_new_guess_vec: jax.Array,
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
) -> jax.Array:
  """Residual of theta-method equation for core profiles at next time-step.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      core profiles.
    dt: Time step duration.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: The geometry at time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    physics_models: Physics models used for the calculations.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.

  Returns:
    residual: Vector residual between LHS and RHS of the theta method equation.
  """
  x_old_vec = jnp.concatenate([var.value for var in x_old])
  # Prepare core_profiles_t_plus_dt for calc_coeffs. Explanation:
  # 1. The original (before iterative solving) core_profiles_t_plus_dt contained
  #    updated boundary conditions and prescribed profiles.
  # 2. Before calling calc_coeffs, we need to update the evolving subset of the
  #    core_profiles_t_plus_dt CellVariables with the current x_new_guess.
  # 3. Ion and impurity density and charge states are also updated here, since
  #    they are state dependent (on n_e and T_e).
  x_new_guess = fvm_conversions.vec_to_cell_variable_tuple(
      x_new_guess_vec, core_profiles_t_plus_dt, evolving_names
  )
  core_profiles_t_plus_dt = updaters.update_core_profiles_during_step(
      x_new_guess,
      runtime_params_t_plus_dt,
      geo_t_plus_dt,
      core_profiles_t_plus_dt,
      evolving_names,
  )
  coeffs_new = calc_coeffs.calc_coeffs(
      runtime_params=runtime_params_t_plus_dt,
      geo=geo_t_plus_dt,
      core_profiles=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      physics_models=physics_models,
      evolving_names=evolving_names,
      use_pereverzev=False,
  )

  solver_params = runtime_params_t_plus_dt.solver
  lhs_mat, lhs_vec, rhs_mat, rhs_vec = theta_method_matrix_equation(
      dt=dt,
      x_old=x_old,
      x_new_guess=x_new_guess,
      coeffs_old=coeffs_old,
      coeffs_new=coeffs_new,
      theta_implicit=solver_params.theta_implicit,
      convection_dirichlet_mode=solver_params.convection_dirichlet_mode,
      convection_neumann_mode=solver_params.convection_neumann_mode,
  )

  lhs = jnp.dot(lhs_mat, x_new_guess_vec) + lhs_vec
  rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec

  residual = lhs - rhs
  return residual


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def theta_method_block_loss(
    x_new_guess_vec: jax.Array,
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
) -> jax.Array:
  """Loss for the optimizer method of nonlinear solution.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      core profiles.
    dt: Time step duration.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: geometry object at time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE
    physics_models: Physics models used for the calculations.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.

  Returns:
    loss: mean squared loss of theta method residual.
  """

  residual = theta_method_block_residual(
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      x_new_guess_vec=x_new_guess_vec,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      physics_models=physics_models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
  )
  loss = jnp.mean(jnp.square(residual))
  return loss


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def jaxopt_solver(
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    init_x_new_vec: jax.Array,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
    maxiter: int,
    tol: float,
) -> tuple[jax.Array, float, int]:
  """Advances jaxopt solver by one timestep.

  Args:
    dt: Time step duration.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: geometry object for time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    init_x_new_vec: Flattened array of initial guess of x_new for all evolving
      core profiles.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE.
    physics_models: Physics models used for the calculations.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    maxiter: maximum number of iterations of jaxopt solver.
    tol: tolerance for jaxopt solver convergence.

  Returns:
    x_new_vec: Flattened evolving profile array after jaxopt evolution.
    final_loss: loss after jaxopt evolution
    num_iterations: number of iterations ran in jaxopt
  """

  loss = functools.partial(
      theta_method_block_loss,
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      physics_models=physics_models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
  )
  solver = jaxopt.LBFGS(fun=loss, maxiter=maxiter, tol=tol)
  solver_output = solver.run(init_x_new_vec)
  x_new_vec = solver_output.params
  final_loss = loss(x_new_vec)
  num_iterations = solver_output.state.iter_num

  return x_new_vec, final_loss, num_iterations
