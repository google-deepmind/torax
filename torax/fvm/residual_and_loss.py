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
import jax
from jax import numpy as jnp
import jaxopt
from torax import calc_coeffs
from torax import config_slice
from torax import geometry
from torax import jax_utils
from torax import state as state_module
from torax import update_state
from torax.fvm import block_1d_coeffs
from torax.fvm import cell_variable
from torax.fvm import discrete_system
from torax.fvm import fvm_conversions
from torax.sources import source_profiles
from torax.transport_model import transport_model as transport_model_lib

AuxiliaryOutput = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffs = block_1d_coeffs.Block1DCoeffs
Block1DCoeffsCallback = block_1d_coeffs.Block1DCoeffsCallback


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'convection_dirichlet_mode',
        'convection_neumann_mode',
        'theta_imp',
    ],
)
def theta_method_matrix_equation(
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    x_old: tuple[cell_variable.CellVariable, ...],
    dt: jax.Array,
    coeffs_old: Block1DCoeffs,
    coeffs_new: Block1DCoeffs,
    theta_imp: float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
    x_new_guess: Current guess of x_new defined as a tuple of CellVariables.
    x_old: The starting x defined as a tuple of CellVariables.
    dt: Time step duration.
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
     - auxiliary output from calculating new coefficients for x_new.
  """

  x_new_guess_vec = fvm_conversions.cell_variable_tuple_to_vec(x_new_guess)

  theta_exp = 1.0 - theta_imp

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

  left_transient = jnp.identity(len(x_new_guess_vec))
  right_transient = jnp.diag(jnp.squeeze(tc_in_old / tc_in_new))

  c_mat_new, c_new = discrete_system.calc_c(
      coeffs_new,
      x_new_guess,
      convection_dirichlet_mode,
      convection_neumann_mode,
  )

  lhs_mat = left_transient - dt * theta_imp * jnp.dot(
      jnp.diag(1 / (tc_out_new * tc_in_new)), c_mat_new
  )
  lhs_vec = -theta_imp * dt * (1 / (tc_out_new * tc_in_new)) * c_new

  if theta_exp > 0.0:
    tc_out_old = jnp.concatenate(coeffs_old.transient_out_cell)
    tc_in_new = jax_utils.error_if(
        tc_in_new,
        jnp.any(tc_out_old * tc_in_new < eps),
        msg='tc_out_old*tc_in_new unexpectedly < eps',
    )
    c_mat_old, c_old = discrete_system.calc_c(
        coeffs_old,
        x_old,
        convection_dirichlet_mode,
        convection_neumann_mode,
    )
    rhs_mat = right_transient + dt * theta_exp * jnp.dot(
        jnp.diag(1 / (tc_out_old * tc_in_new)), c_mat_old
    )
    rhs_vec = dt * theta_exp * (1 / (tc_out_old * tc_in_new)) * c_old
  else:
    rhs_mat = right_transient
    rhs_vec = jnp.zeros_like(x_new_guess_vec)

  return lhs_mat, lhs_vec, rhs_mat, rhs_vec


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'static_config_slice',
        'evolving_names',
        'transport_model',
        'sources',
    ],
)
def theta_method_block_residual(
    x_new_guess_vec: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    state_t_plus_dt: state_module.State,
    evolving_names: tuple[str, ...],
    geo: geometry.Geometry,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    dt: jax.Array,
    coeffs_old: Block1DCoeffs,
    transport_model: transport_model_lib.TransportModel,
    sources: source_profiles.Sources,
    explicit_source_profiles: source_profiles.SourceProfiles,
) -> tuple[jax.Array, AuxiliaryOutput]:
  """Residual of theta-method equation for state at next time-step.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      state profiles.
    x_old: The starting x defined as a tuple of CellVariables.
    state_t_plus_dt: Sim state which contains all available prescribed
      quantities at the end of the time step. This includes evolving boundary
      conditions and prescribed time-dependent profiles that are not being
      evolved by the PDE system.
    evolving_names: The names of variables within the state that should evolve.
    geo: Geometry object.
    dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt.
    static_config_slice: Static runtime configuration. Changes to these config
      params will trigger recompilation. A key parameter in static_config slice
      is theta_imp, a coefficient in [0, 1] determining which solution method
      to use. We solve transient_coeff (x_new - x_old) / dt = theta_imp F(t_new)
      + (1 - theta_imp) F(t_old). Three values of theta_imp correspond to named
      solution methods: theta_imp = 1: Backward Euler implicit method (default).
      theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Forward Euler explicit
      method.
    dt: Time step duration.
    coeffs_old: The coefficients calculated at x_old.
    transport_model: Turbulent transport model callable.
    sources: Collection of source callables to generate source PDE coefficients
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.

  Returns:
    residual: Vector residual between LHS and RHS of the theta method equation.
  """
  x_old_vec = jnp.concatenate([var.value for var in x_old])
  # Create updated CellVariable instances based on state_plus_dt which has
  # updated boundary conditions and prescribed profiles.
  x_new_guess = fvm_conversions.vec_to_cell_variable_tuple(
      x_new_guess_vec, state_t_plus_dt, evolving_names
  )
  state_t_plus_dt = update_state.update_state(
      state_t_plus_dt,
      x_new_guess,
      evolving_names,
      dynamic_config_slice_t_plus_dt,
  )
  coeffs_new = calc_coeffs.calc_coeffs(
      state=state_t_plus_dt,
      evolving_names=evolving_names,
      geo=geo,
      dynamic_config_slice=dynamic_config_slice_t_plus_dt,
      static_config_slice=static_config_slice,
      transport_model=transport_model,
      explicit_source_profiles=explicit_source_profiles,
      sources=sources,
      use_pereverzev=False,
  )

  lhs_mat, lhs_vec, rhs_mat, rhs_vec = theta_method_matrix_equation(
      x_new_guess=x_new_guess,
      x_old=x_old,
      dt=dt,
      coeffs_old=coeffs_old,
      coeffs_new=coeffs_new,
      theta_imp=static_config_slice.solver.theta_imp,
      convection_dirichlet_mode=static_config_slice.solver.convection_dirichlet_mode,
      convection_neumann_mode=static_config_slice.solver.convection_neumann_mode,
  )

  lhs = jnp.dot(lhs_mat, x_new_guess_vec) + lhs_vec
  rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec

  residual = lhs - rhs
  return residual, coeffs_new.auxiliary_outputs


theta_method_block_jacobian = jax.jacfwd(
    theta_method_block_residual, has_aux=True
)
theta_method_block_jacobian = jax_utils.jit(
    theta_method_block_jacobian,
    static_argnames=[
        'static_config_slice',
        'evolving_names',
        'transport_model',
        'sources',
    ],
)


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'static_config_slice',
        'evolving_names',
        'transport_model',
        'sources',
    ],
)
def theta_method_block_loss(
    x_new_guess_vec: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    state_t_plus_dt: state_module.State,
    evolving_names: tuple[str, ...],
    geo: geometry.Geometry,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    dt: jax.Array,
    coeffs_old: Block1DCoeffs,
    transport_model: transport_model_lib.TransportModel,
    sources: source_profiles.Sources,
    explicit_source_profiles: source_profiles.SourceProfiles,
) -> tuple[jax.Array, AuxiliaryOutput]:
  """Loss for the optimizer method of nonlinear solution.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      state profiles.
    x_old: The starting x defined as a tuple of CellVariables.
    state_t_plus_dt: Sim state which contains all available prescribed
      quantities at the end of the time step. This includes evolving boundary
      conditions and prescribed time-dependent profiles that are not being
      evolved by the PDE system.
    evolving_names: The names of variables within the state that should evolve.
    geo: geometry object
    dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt.
    static_config_slice: Static runtime configuration. Changes to these config
      params will trigger recompilation. A key parameter in static_config slice
      is theta_imp, a coefficient in [0, 1] determining which solution method
      to use. We solve transient_coeff (x_new - x_old) / dt = theta_imp F(t_new)
      + (1 - theta_imp) F(t_old). Three values of theta_imp correspond to named
      solution methods: theta_imp = 1: Backward Euler implicit method (default).
      theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Forward Euler explicit
      method.
    dt: Time step duration.
    coeffs_old: The coefficients calculated at x_old.
    transport_model: turbulent transport model callable
    sources: collection of source callables to generate source PDE coefficients
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE

  Returns:
    loss: mean squared loss of theta method residual.
  """

  residual, aux_output = theta_method_block_residual(
      x_new_guess_vec=x_new_guess_vec,
      x_old=x_old,
      state_t_plus_dt=state_t_plus_dt,
      evolving_names=evolving_names,
      geo=geo,
      dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
      static_config_slice=static_config_slice,
      dt=dt,
      coeffs_old=coeffs_old,
      transport_model=transport_model,
      sources=sources,
      explicit_source_profiles=explicit_source_profiles,
  )
  loss = jnp.mean(jnp.square(residual))
  return loss, aux_output


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'static_config_slice',
        'evolving_names',
        'transport_model',
        'sources',
    ],
)
def jaxopt_solver(
    init_x_new_vec: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    state_t_plus_dt: state_module.State,
    evolving_names: tuple[str, ...],
    geo: geometry.Geometry,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    dt: jax.Array,
    coeffs_old: Block1DCoeffs,
    transport_model: transport_model_lib.TransportModel,
    sources: source_profiles.Sources,
    explicit_source_profiles: source_profiles.SourceProfiles,
    maxiter: int,
    tol: float,
) -> tuple[jax.Array, float, AuxiliaryOutput]:
  """Advances jaxopt solver by one timestep.

  Args:
    init_x_new_vec: Flattened array of initial guess of x_new for all evolving
      state profiles.
    x_old: The starting x defined as a tuple of CellVariables.
    state_t_plus_dt: Sim state which contains all available prescribed
      quantities at the end of the time step. This includes evolving boundary
      conditions and prescribed time-dependent profiles that are not being
      evolved by the PDE system.
    evolving_names: The names of variables within the state that should evolve.
    geo: geometry object.
    dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt.
    static_config_slice: Static runtime configuration. Changes to these config
      params will trigger recompilation. A key parameter in static_config slice
      is theta_imp, a coefficient in [0, 1] determining which solution method
      to use. We solve transient_coeff (x_new - x_old) / dt = theta_imp F(t_new)
      + (1 - theta_imp) F(t_old). Three values of theta_imp correspond to named
      solution methods: theta_imp = 1: Backward Euler implicit method (default).
      theta_imp = 0.5: Crank-Nicolson. theta_imp = 0: Forward Euler explicit
      method.
    dt: Time step duration.
    coeffs_old: The coefficients calculated at x_old.
    transport_model: turbulent transport model callable.
    sources: collection of source callables to generate source PDE coefficients
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE.
    maxiter: maximum number of iterations of jaxopt solver.
    tol: tolerance for jaxopt solver convergence.

  Returns:
    x_new_vec: Flattened evolving profile array after jaxopt evolution.
    aux_output: auxilliary outputs from calc_coeffs.
  """

  loss = functools.partial(
      theta_method_block_loss,
      dt=dt,
      x_old=x_old,
      state_t_plus_dt=state_t_plus_dt,
      geo=geo,
      dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
      static_config_slice=static_config_slice,
      evolving_names=evolving_names,
      coeffs_old=coeffs_old,
      transport_model=transport_model,
      sources=sources,
      explicit_source_profiles=explicit_source_profiles,
  )
  solver = jaxopt.LBFGS(fun=loss, maxiter=maxiter, tol=tol, has_aux=True)
  solver_output = solver.run(init_x_new_vec)
  x_new_vec = solver_output.params
  aux_output = solver_output.state.aux
  final_loss, _ = loss(x_new_vec)

  return x_new_vec, final_loss, aux_output
