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
from torax._src import models as models_lib
from torax._src import state
from torax._src import tridiagonal
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import updaters
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import discrete_system
from torax._src.fvm import fvm_conversions
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.sources import source_profiles

Block1DCoeffs: TypeAlias = block_1d_coeffs.Block1DCoeffs


@jax.jit(
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
) -> tuple[
    tridiagonal.BlockTriDiagonal,
    jax.Array,
    tridiagonal.BlockTriDiagonal,
    jax.Array,
]:
  """Returns the banded left-hand and right-hand sides of the theta method.

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
    A tuple of (lhs, lhs_vec, rhs, rhs_vec) where:
      lhs_matrix: BlockTriDiagonal for the LHS matrix A.
      lhs_vec: LHS vector a.
      rhs_matrix: BlockTriDiagonal for the RHS matrix B.
      rhs_vec: RHS vector b.
  """

  theta_exp = 1.0 - theta_implicit

  tc_in_old = jnp.stack(coeffs_old.transient_in_cell, axis=-1)
  tc_out_new = jnp.stack(coeffs_new.transient_out_cell, axis=-1)
  tc_in_new = jnp.stack(coeffs_new.transient_in_cell, axis=-1)
  chex.assert_rank(tc_in_old, 2)
  chex.assert_rank(tc_out_new, 2)
  chex.assert_rank(tc_in_new, 2)

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

  scale_new = dt * theta_implicit / (tc_out_new * tc_in_new)

  c_new_matrix, c_new_forcing = discrete_system.calc_c(
      x_new_guess,
      coeffs_new,
      convection_dirichlet_mode,
      convection_neumann_mode,
  )

  # Compute LHS = I - scale_new * C_new directly, avoiding intermediate
  # BlockTriDiagonal objects. The transient part (I) only contributes to the
  # diagonal, so off-diagonal blocks are just -scale * C_new.
  ch_idx = jnp.arange(len(x_old))
  lhs_diag = -scale_new[:, :, None] * c_new_matrix.diagonal
  lhs_diag = lhs_diag.at[:, ch_idx, ch_idx].add(1.0)
  lhs_matrix = tridiagonal.BlockTriDiagonal(
      lower=-scale_new[1:, :, None] * c_new_matrix.lower,
      diagonal=lhs_diag,
      upper=-scale_new[:-1, :, None] * c_new_matrix.upper,
  )
  lhs_vec = -scale_new * c_new_forcing

  if theta_exp > 0.0:
    tc_out_old = jnp.stack(coeffs_old.transient_out_cell, axis=-1)
    tc_in_new = jax_utils.error_if(
        tc_in_new,
        jnp.any(jnp.abs(tc_out_old * tc_in_new) < eps),
        msg='|tc_out_old*tc_in_new| unexpectedly < eps',
    )
    c_old_matrix, c_old_forcing = discrete_system.calc_c(
        x_old,
        coeffs_old,
        convection_dirichlet_mode,
        convection_neumann_mode,
    )

    scale_old = dt * theta_exp / (tc_out_old * tc_in_new)

    # Compute RHS = diag(tc_in_old/tc_in_new) + scale_old * C_old directly.
    # The transient part only contributes to the diagonal.
    rhs_diag = scale_old[:, :, None] * c_old_matrix.diagonal
    rhs_diag = rhs_diag.at[:, ch_idx, ch_idx].add((tc_in_old / tc_in_new))
    rhs_matrix = tridiagonal.BlockTriDiagonal(
        lower=scale_old[1:, :, None] * c_old_matrix.lower,
        diagonal=rhs_diag,
        upper=scale_old[:-1, :, None] * c_old_matrix.upper,
    )
    rhs_vec = scale_old * c_old_forcing
  else:
    rhs_matrix = tridiagonal.BlockTriDiagonal.from_diagonal(
        tc_in_old / tc_in_new
        )
    rhs_vec = jnp.zeros(
        (rhs_matrix.num_blocks, rhs_matrix.block_size), dtype=tc_in_new.dtype
        )

  return lhs_matrix, lhs_vec, rhs_matrix, rhs_vec


@jax.jit(
    static_argnames=[
        'evolving_names',
        'models',
    ],
)
def theta_method_block_residual(
    x_new_guess_vec: jax.Array,
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    models: models_lib.Models,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
    pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
) -> jax.Array:
  """Residual of theta-method equation for core profiles at next time-step.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      core profiles.
    dt: Time step duration.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: The geometry at time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    core_profiles_t: Core plasma profiles which contain all available prescribed
      quantities at the start of the time step.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    models: Models used for the calculations.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    pedestal_transition_state: State of the pedestal transition model if using
      the formation model with adaptive source.

  Returns:
    residual: Vector residual between LHS and RHS of the theta method equation.
  """
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
      prev_core_profiles=core_profiles_t,
      dt=dt,
      evolving_names=evolving_names,
  )
  coeffs_new = calc_coeffs.calc_coeffs(
      runtime_params=runtime_params_t_plus_dt,
      geo=geo_t_plus_dt,
      core_profiles=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      models=models,
      evolving_names=evolving_names,
      use_pereverzev=False,
      pedestal_transition_state=pedestal_transition_state,
  )

  solver_params = runtime_params_t_plus_dt.solver
  lhs, lhs_vec, rhs, rhs_vec = theta_method_matrix_equation(
      dt=dt,
      x_old=x_old,
      x_new_guess=x_new_guess,
      coeffs_old=coeffs_old,
      coeffs_new=coeffs_new,
      theta_implicit=solver_params.theta_implicit,
      convection_dirichlet_mode=solver_params.convection_dirichlet_mode,
      convection_neumann_mode=solver_params.convection_neumann_mode,
  )

  # TODO(b/505253351) Remove the reshape and transpose.
  x_old_array = fvm_conversions.cell_variable_tuple_to_array(x_old, axis=1)
  # Reshape x_new_guess_vec to a 2D array with shape (num_channels, num_cells)
  # then transpose it to (num_cells, num_channels) to allow for block
  # tridiagonal matvec multiplication with lhs and rhs.
  num_cells, num_channels = x_old_array.shape
  x_new_array = x_new_guess_vec.reshape(num_channels, num_cells).T

  lhs_result = lhs.matvec(x_new_array) + lhs_vec
  rhs_result = rhs.matvec(x_old_array) + rhs_vec

  return (lhs_result - rhs_result).T.reshape(-1)


@jax.jit(
    static_argnames=[
        'models',
        'evolving_names',
    ],
)
def theta_method_block_loss(
    x_new_guess_vec: jax.Array,
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    models: models_lib.Models,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
    pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
) -> jax.Array:
  """Loss for the optimizer method of nonlinear solution.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      core profiles.
    dt: Time step duration.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: geometry object at time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    core_profiles_t: Core profiles from the previous time step.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE
    models: Models used for the calculations.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    pedestal_transition_state: State of the pedestal transition model if using
      the formation model with adaptive source.

  Returns:
    loss: mean squared loss of theta method residual.
  """

  residual = theta_method_block_residual(
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      x_new_guess_vec=x_new_guess_vec,
      core_profiles_t=core_profiles_t,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      models=models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      pedestal_transition_state=pedestal_transition_state,
  )
  loss = jnp.mean(jnp.square(residual))
  return loss


@jax.jit(
    static_argnames=[
        'models',
        'evolving_names',
    ],
)
def jaxopt_solver(
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    init_x_new_vec: jax.Array,
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    models: models_lib.Models,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
    pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
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
    core_profiles_t: Core profiles from the previous time step.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE.
    models: Models used for the calculations.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    pedestal_transition_state: State of the pedestal transition model if using
      the formation model with adaptive source.
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
      core_profiles_t=core_profiles_t,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      models=models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      pedestal_transition_state=pedestal_transition_state,
  )
  solver = jaxopt.LBFGS(fun=loss, maxiter=maxiter, tol=tol, implicit_diff=True)
  solver_output = solver.run(init_x_new_vec)
  x_new_vec = solver_output.params
  final_loss = loss(x_new_vec)
  num_iterations = solver_output.state.iter_num

  return x_new_vec, final_loss, num_iterations
