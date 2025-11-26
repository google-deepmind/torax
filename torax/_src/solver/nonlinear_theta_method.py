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

"""The NonLinearThetaMethod class."""
import abc
import dataclasses

import jax
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.fvm import newton_raphson_solve_block
from torax._src.fvm import optimizer_solve_block
from torax._src.geometry import geometry
from torax._src.pedestal_policy import pedestal_policy
from torax._src.solver import runtime_params as solver_runtime_params_lib
from torax._src.solver import solver
from torax._src.sources import source_profiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OptimizerRuntimeParams(solver_runtime_params_lib.RuntimeParams):
  n_max_iterations: int
  loss_tol: float
  initial_guess_mode: int = dataclasses.field(metadata={'static': True})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NewtonRaphsonRuntimeParams(solver_runtime_params_lib.RuntimeParams):
  maxiter: int
  residual_tol: float
  residual_coarse_tol: float
  delta_reduction_factor: float
  tau_min: float
  initial_guess_mode: int = dataclasses.field(metadata={'static': True})
  log_iterations: bool = dataclasses.field(metadata={'static': True})


class NonlinearThetaMethod(solver.Solver):
  """Time step update using nonlinear solvers and the theta method."""

  def _x_new(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      pedestal_policy_state_t: pedestal_policy.PedestalPolicyState,
      pedestal_policy_state_t_plus_dt: pedestal_policy.PedestalPolicyState,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """See Solver._x_new docstring."""

    coeffs_callback = calc_coeffs.CoeffsCallback(
        physics_models=self.physics_models,
        evolving_names=evolving_names,
    )
    (
        x_new,
        solver_numeric_outputs,
    ) = self._x_new_helper(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
        pedestal_policy_state_t=pedestal_policy_state_t,
        pedestal_policy_state_t_plus_dt=pedestal_policy_state_t_plus_dt,
        coeffs_callback=coeffs_callback,
        evolving_names=evolving_names,
    )

    return (
        x_new,
        solver_numeric_outputs,
    )

  @abc.abstractmethod
  def _x_new_helper(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      pedestal_policy_state_t: pedestal_policy.PedestalPolicyState,
      pedestal_policy_state_t_plus_dt: pedestal_policy.PedestalPolicyState,
      coeffs_callback: calc_coeffs.CoeffsCallback,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Abstract method for subclasses to implement the specific nonlinear solve.

    This helper method is called by `_x_new` after it has constructed the
    `coeffs_callback`. Subclasses should implement this method to call their
    respective nonlinear solver implementation (e.g., Newton-Raphson or an
    optimizer-based approach).

    Args:
      dt: Time step duration.
      runtime_params_t: Runtime parameters for time t (the start time of the
        step).
      runtime_params_t_plus_dt: Runtime parameters for time t + dt, used for
        implicit calculations in the solver.
      geo_t: Magnetic geometry at time t.
      geo_t_plus_dt: Magnetic geometry at time t + dt.
      core_profiles_t: Core plasma profiles at the beginning of the time step.
      core_profiles_t_plus_dt: Core plasma profiles which contain all available
        prescribed quantities at the end of the time step. This includes
        evolving boundary conditions and prescribed time-dependent profiles that
        are not being evolved by the PDE system.
      explicit_source_profiles: Pre-calculated sources implemented as explicit
        sources in the PDE.
      pedestal_policy_state_t: pedestal policy state at time t
      pedestal_policy_state_t_plus_dt: pedestal policy state at time t + dt
      coeffs_callback: Calculates diffusion, convection etc. coefficients given
        a core_profiles, geometry, runtime_params. Repeatedly called by the
        iterative solvers.
      evolving_names: The names of variables within the core profiles that
        should evolve.

    Returns:
      A tuple containing:
        - The new values of the evolving variables at time t + dt.
        - Solver iteration and error info.
    """
    ...


class OptimizerThetaMethod(NonlinearThetaMethod):
  """Minimize the squared norm of the residual of the theta method equation."""

  def _x_new_helper(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      pedestal_policy_state_t: pedestal_policy.PedestalPolicyState,
      pedestal_policy_state_t_plus_dt: pedestal_policy.PedestalPolicyState,
      coeffs_callback: calc_coeffs.CoeffsCallback,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """See abstract method docstring in NonlinearThetaMethod."""
    solver_params = runtime_params_t.solver
    assert isinstance(solver_params, OptimizerRuntimeParams)
    (
        x_new,
        solver_numeric_outputs,
    ) = optimizer_solve_block.optimizer_solve_block(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=convertors.core_profiles_to_solver_x_tuple(
            core_profiles_t, evolving_names
        ),
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        physics_models=self.physics_models,
        explicit_source_profiles=explicit_source_profiles,
        pedestal_policy_state_t=pedestal_policy_state_t,
        pedestal_policy_state_t_plus_dt=pedestal_policy_state_t_plus_dt,
        coeffs_callback=coeffs_callback,
        evolving_names=evolving_names,
        initial_guess_mode=enums.InitialGuessMode(
            solver_params.initial_guess_mode,
        ),
        maxiter=solver_params.n_max_iterations,
        tol=solver_params.loss_tol,
    )
    return (
        x_new,
        solver_numeric_outputs,
    )


class NewtonRaphsonThetaMethod(NonlinearThetaMethod):
  """Nonlinear theta method using Newton Raphson."""

  def _x_new_helper(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      pedestal_policy_state_t: pedestal_policy.PedestalPolicyState,
      pedestal_policy_state_t_plus_dt: pedestal_policy.PedestalPolicyState,
      coeffs_callback: calc_coeffs.CoeffsCallback,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """See abstract method docstring in NonlinearThetaMethod."""
    solver_params = runtime_params_t.solver
    assert isinstance(solver_params, NewtonRaphsonRuntimeParams)

    (
        x_new,
        solver_numeric_outputs,
    ) = newton_raphson_solve_block.newton_raphson_solve_block(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=convertors.core_profiles_to_solver_x_tuple(
            core_profiles_t, evolving_names
        ),
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
        physics_models=self.physics_models,
        coeffs_callback=coeffs_callback,
        evolving_names=evolving_names,
        pedestal_policy_state_t=pedestal_policy_state_t,
        pedestal_policy_state_t_plus_dt=pedestal_policy_state_t_plus_dt,
        log_iterations=solver_params.log_iterations,
        initial_guess_mode=enums.InitialGuessMode(
            solver_params.initial_guess_mode
        ),
        maxiter=solver_params.maxiter,
        tol=solver_params.residual_tol,
        coarse_tol=solver_params.residual_coarse_tol,
        delta_reduction_factor=solver_params.delta_reduction_factor,
        tau_min=solver_params.tau_min,
    )
    return (
        x_new,
        solver_numeric_outputs,
    )
