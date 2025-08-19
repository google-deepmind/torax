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
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.fvm import newton_raphson_solve_block
from torax._src.fvm import optimizer_solve_block
from torax._src.geometry import geometry
from torax._src.solver import runtime_params
from torax._src.solver import solver
from torax._src.sources import source_profiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicOptimizerRuntimeParams(runtime_params.DynamicRuntimeParams):
  n_max_iterations: int
  loss_tol: float
  initial_guess_mode: int = dataclasses.field(metadata={'static': True})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicNewtonRaphsonRuntimeParams(runtime_params.DynamicRuntimeParams):
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
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """See Solver._x_new docstring."""

    coeffs_callback = calc_coeffs.CoeffsCallback(
        static_runtime_params_slice=static_runtime_params_slice,
        physics_models=self.physics_models,
        evolving_names=evolving_names,
    )
    (
        x_new,
        solver_numeric_outputs,
    ) = self._x_new_helper(
        dt=dt,
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
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
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
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
      static_runtime_params_slice: Static runtime parameters. Changes to these
        runtime params will trigger recompilation.
      dynamic_runtime_params_slice_t: Runtime parameters for time t (the start
        time of the step).
      dynamic_runtime_params_slice_t_plus_dt: Runtime parameters for time t +
        dt, used for implicit calculations in the solver.
      geo_t: Magnetic geometry at time t.
      geo_t_plus_dt: Magnetic geometry at time t + dt.
      core_profiles_t: Core plasma profiles at the beginning of the time step.
      core_profiles_t_plus_dt: Core plasma profiles which contain all available
        prescribed quantities at the end of the time step. This includes
        evolving boundary conditions and prescribed time-dependent profiles that
        are not being evolved by the PDE system.
      explicit_source_profiles: Pre-calculated sources implemented as explicit
        sources in the PDE.
      coeffs_callback: Calculates diffusion, convection etc. coefficients given
        a core_profiles, geometry, dynamic_runtime_params. Repeatedly called by
        the iterative solvers.
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
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      coeffs_callback: calc_coeffs.CoeffsCallback,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """See abstract method docstring in NonlinearThetaMethod."""
    solver_params = dynamic_runtime_params_slice_t.solver
    assert isinstance(solver_params, DynamicOptimizerRuntimeParams)
    (
        x_new,
        solver_numeric_outputs,
    ) = optimizer_solve_block.optimizer_solve_block(
        dt=dt,
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=convertors.core_profiles_to_solver_x_tuple(
            core_profiles_t, evolving_names
        ),
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        physics_models=self.physics_models,
        explicit_source_profiles=explicit_source_profiles,
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
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      coeffs_callback: calc_coeffs.CoeffsCallback,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """See abstract method docstring in NonlinearThetaMethod."""
    solver_params = dynamic_runtime_params_slice_t.solver
    assert isinstance(solver_params, DynamicNewtonRaphsonRuntimeParams)

    (
        x_new,
        solver_numeric_outputs,
    ) = newton_raphson_solve_block.newton_raphson_solve_block(
        dt=dt,
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
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
