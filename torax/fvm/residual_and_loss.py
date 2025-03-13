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
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.core_profiles import updaters
from torax.fvm import block_1d_coeffs
from torax.fvm import calc_coeffs
from torax.fvm import cell_variable
from torax.fvm import discrete_system
from torax.fvm import fvm_conversions
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
from torax.transport_model import transport_model as transport_model_lib


AuxiliaryOutput: TypeAlias = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffs: TypeAlias = block_1d_coeffs.Block1DCoeffs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'convection_dirichlet_mode',
        'convection_neumann_mode',
        'theta_imp',
    ],
)
def theta_method_matrix_equation(
    dt: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    coeffs_old: Block1DCoeffs,
    coeffs_new: Block1DCoeffs,
    theta_imp: float = 1.0,
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
    | theta_imp F_new / tc_out_new + theta_exp F_old / tc_out_old

  The equation is on the cell grid where `tc` is never zero. Therefore
  it's safe to multiply equation by `dt/tc_in_new` and scale the residual to
  `x`, which has O(1) values and thus the residual is scaled appropriately.

  We thus rearrange to:

    | x_new - tc_in_old/tc_in_new x_old =
    | dt theta_imp F_new / (tc_out_new tc_in_new) +
    | dt theta_exp F_old / (tc_out_old tc_in_new)

  Rearranging we obtain

    | x_new - dt theta_imp F_new / (tc_out_new tc_in_new) =
    | tc_in_old/tc_in_new x_old + dt theta_exp F_old / (tc_out_old tc_in_new)

  We now substitute in `F = Cu + c`:

    | (I - dt theta_imp diag(1/(tc_out_new tc_in_new)) C_new) x_new
    | - dt theta_imp diag(1/(tc_out_new tc_in_new)) c_new
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
    theta_imp: Coefficient on implicit term of theta method.
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

  theta_exp = 1.0 - theta_imp

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

  lhs_mat = left_transient - dt * theta_imp * broadcasted * c_mat_new
  lhs_vec = -theta_imp * dt * (1 / (tc_out_new * tc_in_new)) * c_new

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
        'static_runtime_params_slice',
        'transport_model',
        'source_models',
        'evolving_names',
        'pedestal_model',
    ],
)
def theta_method_block_residual(
    x_new_guess_vec: jax.Array,
    dt: jax.Array,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    transport_model: transport_model_lib.TransportModel,
    explicit_source_profiles: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
    pedestal_model: pedestal_model_lib.PedestalModel,
) -> tuple[jax.Array, AuxiliaryOutput]:
  """Residual of theta-method equation for core profiles at next time-step.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      core profiles.
    dt: Time step duration.
    static_runtime_params_slice: Static runtime parameters. Changes to these
      runtime params will trigger recompilation. A key parameter in this params
      slice is theta_imp, a coefficient in [0, 1] determining which solution
      method to use. We solve transient_coeff (x_new - x_old) / dt = theta_imp
      F(t_new) + (1 - theta_imp) F(t_old). Three values of theta_imp correspond
      to named solution methods: theta_imp = 1: Backward Euler implicit method
      (default). theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Forward Euler
      explicit method.
    dynamic_runtime_params_slice_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: The geometry at time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    transport_model: Turbulent transport model callable.
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    source_models: Collection of source callables to generate source PDE
      coefficients.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    pedestal_model: Model of the pedestal's behavior.

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
  #    they are state dependent (on ne and temp_el).
  x_new_guess = fvm_conversions.vec_to_cell_variable_tuple(
      x_new_guess_vec, core_profiles_t_plus_dt, evolving_names
  )
  core_profiles_t_plus_dt = updaters.update_core_profiles_during_step(
      x_new_guess,
      static_runtime_params_slice,
      dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt,
      core_profiles_t_plus_dt,
      evolving_names,
  )
  coeffs_new = calc_coeffs.calc_coeffs(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
      geo=geo_t_plus_dt,
      core_profiles=core_profiles_t_plus_dt,
      transport_model=transport_model,
      explicit_source_profiles=explicit_source_profiles,
      source_models=source_models,
      evolving_names=evolving_names,
      use_pereverzev=False,
      pedestal_model=pedestal_model,
  )

  lhs_mat, lhs_vec, rhs_mat, rhs_vec = theta_method_matrix_equation(
      dt=dt,
      x_old=x_old,
      x_new_guess=x_new_guess,
      coeffs_old=coeffs_old,
      coeffs_new=coeffs_new,
      theta_imp=static_runtime_params_slice.stepper.theta_imp,
      convection_dirichlet_mode=static_runtime_params_slice.stepper.convection_dirichlet_mode,
      convection_neumann_mode=static_runtime_params_slice.stepper.convection_neumann_mode,
  )

  lhs = jnp.dot(lhs_mat, x_new_guess_vec) + lhs_vec
  rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec

  residual = lhs - rhs
  return residual, coeffs_new.auxiliary_outputs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'static_runtime_params_slice',
        'transport_model',
        'source_models',
        'evolving_names',
        'pedestal_model',
    ],
)
def theta_method_block_jacobian(*args, **kwargs):
  return jax.jacfwd(theta_method_block_residual, has_aux=True)(*args, **kwargs)


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'static_runtime_params_slice',
        'transport_model',
        'source_models',
        'evolving_names',
        'pedestal_model',
    ],
)
def theta_method_block_loss(
    x_new_guess_vec: jax.Array,
    dt: jax.Array,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    transport_model: transport_model_lib.TransportModel,
    explicit_source_profiles: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
    pedestal_model: pedestal_model_lib.PedestalModel,
) -> tuple[jax.Array, AuxiliaryOutput]:
  """Loss for the optimizer method of nonlinear solution.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      core profiles.
    dt: Time step duration.
    static_runtime_params_slice: Static runtime parameters. Changes to these
      runtime params will trigger recompilation. A key parameter in this params
      slice is theta_imp, a coefficient in [0, 1] determining which solution
      method to use. We solve transient_coeff (x_new - x_old) / dt = theta_imp
      F(t_new) + (1 - theta_imp) F(t_old). Three values of theta_imp correspond
      to named solution methods: theta_imp = 1: Backward Euler implicit method
      (default). theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Forward Euler
      explicit method.
    dynamic_runtime_params_slice_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: geometry object at time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    transport_model: turbulent transport model callable
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE
    source_models: Collection of source callables to generate source PDE
      coefficients.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    pedestal_model: Model of the pedestal's behavior.

  Returns:
    loss: mean squared loss of theta method residual.
  """

  residual, aux_output = theta_method_block_residual(
      dt=dt,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      x_new_guess_vec=x_new_guess_vec,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      transport_model=transport_model,
      explicit_source_profiles=explicit_source_profiles,
      source_models=source_models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      pedestal_model=pedestal_model,
  )
  loss = jnp.mean(jnp.square(residual))
  return loss, aux_output


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'static_runtime_params_slice',
        'transport_model',
        'source_models',
        'evolving_names',
        'pedestal_model',
    ],
)
def jaxopt_solver(
    dt: jax.Array,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    init_x_new_vec: jax.Array,
    core_profiles_t_plus_dt: state.CoreProfiles,
    transport_model: transport_model_lib.TransportModel,
    pedestal_model: pedestal_model_lib.PedestalModel,
    explicit_source_profiles: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
    maxiter: int,
    tol: float,
) -> tuple[jax.Array, float, AuxiliaryOutput, int]:
  """Advances jaxopt solver by one timestep.

  Args:
    dt: Time step duration.
    static_runtime_params_slice: Static runtime parameters. Changes to these
      runtime params will trigger recompilation. A key parameter in this params
      slice is theta_imp, a coefficient in [0, 1] determining which solution
      method to use. We solve transient_coeff (x_new - x_old) / dt = theta_imp
      F(t_new) + (1 - theta_imp) F(t_old). Three values of theta_imp correspond
      to named solution methods: theta_imp = 1: Backward Euler implicit method
      (default). theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Forward Euler
      explicit method.
    dynamic_runtime_params_slice_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: geometry object for time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    init_x_new_vec: Flattened array of initial guess of x_new for all evolving
      core profiles.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    transport_model: turbulent transport model callable.
    pedestal_model: Model of the pedestal's behavior.
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE.
    source_models: Collection of source callables to generate source PDE
      coefficients.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    maxiter: maximum number of iterations of jaxopt solver.
    tol: tolerance for jaxopt solver convergence.

  Returns:
    x_new_vec: Flattened evolving profile array after jaxopt evolution.
    final_loss: loss after jaxopt evolution
    aux_output: auxiliary outputs from calc_coeffs.
    num_iterations: number of iterations ran in jaxopt
  """

  loss = functools.partial(
      theta_method_block_loss,
      dt=dt,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      transport_model=transport_model,
      explicit_source_profiles=explicit_source_profiles,
      source_models=source_models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      pedestal_model=pedestal_model,
  )
  solver = jaxopt.LBFGS(fun=loss, maxiter=maxiter, tol=tol, has_aux=True)
  solver_output = solver.run(init_x_new_vec)
  x_new_vec = solver_output.params
  aux_output = solver_output.state.aux
  final_loss, _ = loss(x_new_vec)
  num_iterations = solver_output.state.iter_num

  return x_new_vec, final_loss, aux_output, num_iterations
