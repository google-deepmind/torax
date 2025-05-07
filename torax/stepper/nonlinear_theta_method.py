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

import chex
import jax
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import calc_coeffs
from torax.fvm import cell_variable
from torax.fvm import enums
from torax.fvm import newton_raphson_solve_block
from torax.fvm import optimizer_solve_block
from torax.geometry import geometry
from torax.sources import source_profiles
from torax.stepper import runtime_params
from torax.stepper import stepper


@chex.dataclass(frozen=True)
class DynamicOptimizerRuntimeParams(runtime_params.DynamicRuntimeParams):
  initial_guess_mode: int
  maxiter: int
  tol: float


@chex.dataclass(frozen=True)
class DynamicNewtonRaphsonRuntimeParams(runtime_params.DynamicRuntimeParams):
  log_iterations: bool
  initial_guess_mode: int
  maxiter: int
  residual_tol: float
  residual_coarse_tol: float
  delta_reduction_factor: float
  tau_min: float


class NonlinearThetaMethod(stepper.Solver):
  """Time step update using theta method.

  Attributes:
    transport_model: A TransportModel subclass, calculates transport coeffs.
    source_models: All TORAX sources used to compute both the explicit and
      implicit source profiles used for each time step as terms in the state
      evolution equations. Though the explicit profiles are computed outside the
      call to Solver, the same sources should be used to compute those. The
      Sources are exposed here to provide a single source of truth for which
      sources are used during a run.
  """

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
      source_profiles.SourceProfiles,
      state.CoreTransport,
      state.SolverNumericOutputs,
  ]:
    """See Solver._x_new docstring."""

    coeffs_callback = calc_coeffs.CoeffsCallback(
        static_runtime_params_slice=static_runtime_params_slice,
        transport_model=self.transport_model,
        explicit_source_profiles=explicit_source_profiles,
        source_models=self.source_models,
        pedestal_model=self.pedestal_model,
        evolving_names=evolving_names,
    )
    x_new, core_sources, core_transport, solver_numeric_outputs = self._x_new_helper(
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

    return x_new, core_sources, core_transport, solver_numeric_outputs

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
      source_profiles.SourceProfiles,
      state.CoreTransport,
      state.SolverNumericOutputs,
  ]:
    """Final implementation of x_new after callback has been created etc."""
    ...


class OptimizerThetaMethod(NonlinearThetaMethod):
  """Minimize the squared norm of the residual of the theta method equation.

  Attributes:
    transport_model: A TransportModel subclass, calculates transport coeffs.
    callback_class: Which class should be used to calculate the coefficients.
    initial_guess_mode: Passed through to `fvm.optimizer_solve_block`.
    maxiter: Passed through to `jaxopt.LBFGS`.
    tol: Passed through to `jaxopt.LBFGS`.
  """

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
      source_profiles.SourceProfiles,
      state.CoreTransport,
      state.SolverNumericOutputs,
  ]:
    """Final implementation of x_new after callback has been created etc."""
    solver_params = dynamic_runtime_params_slice_t.solver
    assert isinstance(solver_params, DynamicOptimizerRuntimeParams)
    # Unpack the outputs of the optimizer_solve_block.
    x_new, solver_numeric_outputs, (core_sources, core_transport) = (
        optimizer_solve_block.optimizer_solve_block(
            dt=dt,
            static_runtime_params_slice=static_runtime_params_slice,
            dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
            dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
            geo_t=geo_t,
            geo_t_plus_dt=geo_t_plus_dt,
            x_old=tuple([core_profiles_t[name] for name in evolving_names]),
            core_profiles_t=core_profiles_t,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            transport_model=self.transport_model,
            explicit_source_profiles=explicit_source_profiles,
            source_models=self.source_models,
            pedestal_model=self.pedestal_model,
            coeffs_callback=coeffs_callback,
            evolving_names=evolving_names,
            initial_guess_mode=enums.InitialGuessMode(
                solver_params.initial_guess_mode,
            ),
            maxiter=solver_params.maxiter,
            tol=solver_params.tol,
        )
    )
    return x_new, core_sources, core_transport, solver_numeric_outputs


class NewtonRaphsonThetaMethod(NonlinearThetaMethod):
  """Nonlinear theta method using Newton Raphson.

  Attributes:
    transport_model: A TransportModel subclass, calculates transport coeffs.
    callback_class: Which class should be used to calculate the coefficients.
  """

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
      source_profiles.SourceProfiles,
      state.CoreTransport,
      state.SolverNumericOutputs,
  ]:
    """Final implementation of x_new after callback has been created etc."""
    solver_params = dynamic_runtime_params_slice_t.solver
    assert isinstance(solver_params, DynamicNewtonRaphsonRuntimeParams)
    # disable error checking in residual, since Newton-Raphson routine has
    # error checking based on result of each linear step

    # Unpack the outputs of the newton_raphson_solve_block.
    x_new, solver_numeric_outputs, (core_sources, core_transport) = (
        newton_raphson_solve_block.newton_raphson_solve_block(
            dt=dt,
            static_runtime_params_slice=static_runtime_params_slice,
            dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
            dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
            geo_t=geo_t,
            geo_t_plus_dt=geo_t_plus_dt,
            x_old=tuple([core_profiles_t[name] for name in evolving_names]),
            core_profiles_t=core_profiles_t,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            transport_model=self.transport_model,
            pedestal_model=self.pedestal_model,
            explicit_source_profiles=explicit_source_profiles,
            source_models=self.source_models,
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
    )
    return x_new, core_sources, core_transport, solver_numeric_outputs
