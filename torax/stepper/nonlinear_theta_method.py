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

from __future__ import annotations

import abc
from typing import Literal

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
from torax.stepper import linear_theta_method
from torax.stepper import runtime_params
from torax.stepper import stepper


@chex.dataclass(frozen=True)
class DynamicOptimizerRuntimeParams(runtime_params.DynamicRuntimeParams):
  initial_guess_mode: int
  maxiter: int
  tol: float


class Optimizer(linear_theta_method.Linear):
  """A basic version of the pedestal model that uses direct specification.

  Attributes:
    stepper_type: The type of stepper to use, hardcoded to 'optimizer'.
    initial_guess_mode: The initial guess mode for the optimizer.
    maxiter: The maximum number of iterations for the optimizer.
    tol: The tolerance for the optimizer.
  """

  stepper_type: Literal['optimizer'] = 'optimizer'
  initial_guess_mode: enums.InitialGuessMode = enums.InitialGuessMode.LINEAR
  maxiter: int = 100
  tol: float = 1e-12

  def build_dynamic_params(
      self,
  ) -> DynamicOptimizerRuntimeParams:
    return DynamicOptimizerRuntimeParams(
        chi_per=self.chi_per,
        d_per=self.d_per,
        initial_guess_mode=self.initial_guess_mode.value,
        maxiter=self.maxiter,
        tol=self.tol,
        corrector_steps=self.corrector_steps,
    )

  def build_stepper(
      self, transport_model, source_models, pedestal_model
  ) -> OptimizerThetaMethod:
    return OptimizerThetaMethod(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )


@chex.dataclass(frozen=True)
class DynamicNewtonRaphsonRuntimeParams(runtime_params.DynamicRuntimeParams):
  log_iterations: bool
  initial_guess_mode: int
  maxiter: int
  tol: float
  coarse_tol: float
  delta_reduction_factor: float
  tau_min: float


class NewtonRaphson(linear_theta_method.Linear):
  """Model for non linear Newton-Raphson stepper.

  Attributes:
    stepper_type: The type of stepper to use, hardcoded to 'newton_raphson'.
    log_iterations: If True, log internal iterations in Newton-Raphson solver.
    initial_guess_mode: The initial guess mode for the Newton-Raphson solver.
    maxiter: The maximum number of iterations for the Newton-Raphson solver.
    tol: The tolerance for the Newton-Raphson solver.
    coarse_tol: The coarse tolerance for the Newton-Raphson solver.
    delta_reduction_factor: The delta reduction factor for the Newton-Raphson
      solver.
    tau_min: The minimum value of tau for the Newton-Raphson solver.
  """

  stepper_type: Literal['newton_raphson'] = 'newton_raphson'
  log_iterations: bool = False
  initial_guess_mode: enums.InitialGuessMode = enums.InitialGuessMode.LINEAR
  maxiter: int = 30
  tol: float = 1e-5
  coarse_tol: float = 1e-2
  delta_reduction_factor: float = 0.5
  tau_min: float = 0.01

  def build_dynamic_params(self) -> DynamicNewtonRaphsonRuntimeParams:
    return DynamicNewtonRaphsonRuntimeParams(
        chi_per=self.chi_per,
        d_per=self.d_per,
        log_iterations=self.log_iterations,
        initial_guess_mode=self.initial_guess_mode.value,
        maxiter=self.maxiter,
        tol=self.tol,
        coarse_tol=self.coarse_tol,
        corrector_steps=self.corrector_steps,
        delta_reduction_factor=self.delta_reduction_factor,
        tau_min=self.tau_min,
    )

  def build_stepper(
      self, transport_model, source_models, pedestal_model
  ) -> NewtonRaphsonThetaMethod:
    return NewtonRaphsonThetaMethod(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )


class NonlinearThetaMethod(stepper.Stepper):
  """Time step update using theta method.

  Attributes:
    transport_model: A TransportModel subclass, calculates transport coeffs.
    source_models: All TORAX sources used to compute both the explicit and
      implicit source profiles used for each time step as terms in the state
      evolution equations. Though the explicit profiles are computed outside the
      call to Stepper, the same sources should be used to compute those. The
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
      state.StepperNumericOutputs,
  ]:
    """See Stepper._x_new docstring."""

    coeffs_callback = calc_coeffs.CoeffsCallback(
        static_runtime_params_slice=static_runtime_params_slice,
        transport_model=self.transport_model,
        explicit_source_profiles=explicit_source_profiles,
        source_models=self.source_models,
        pedestal_model=self.pedestal_model,
        evolving_names=evolving_names,
    )
    x_new, core_sources, core_transport, stepper_numeric_outputs = self._x_new_helper(
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

    return x_new, core_sources, core_transport, stepper_numeric_outputs

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
      state.StepperNumericOutputs,
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
      state.StepperNumericOutputs,
  ]:
    """Final implementation of x_new after callback has been created etc."""
    stepper_params = dynamic_runtime_params_slice_t.stepper
    assert isinstance(stepper_params, DynamicOptimizerRuntimeParams)
    # Unpack the outputs of the optimizer_solve_block.
    x_new, stepper_numeric_outputs, (core_sources, core_transport) = (
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
                stepper_params.initial_guess_mode,
            ),
            maxiter=stepper_params.maxiter,
            tol=stepper_params.tol,
        )
    )
    return x_new, core_sources, core_transport, stepper_numeric_outputs


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
      state.StepperNumericOutputs,
  ]:
    """Final implementation of x_new after callback has been created etc."""
    stepper_params = dynamic_runtime_params_slice_t.stepper
    assert isinstance(stepper_params, DynamicNewtonRaphsonRuntimeParams)
    # disable error checking in residual, since Newton-Raphson routine has
    # error checking based on result of each linear step

    # Unpack the outputs of the newton_raphson_solve_block.
    x_new, stepper_numeric_outputs, (core_sources, core_transport) = (
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
            log_iterations=stepper_params.log_iterations,
            initial_guess_mode=enums.InitialGuessMode(
                stepper_params.initial_guess_mode
            ),
            maxiter=stepper_params.maxiter,
            tol=stepper_params.tol,
            coarse_tol=stepper_params.coarse_tol,
            delta_reduction_factor=stepper_params.delta_reduction_factor,
            tau_min=stepper_params.tau_min,
        )
    )
    return x_new, core_sources, core_transport, stepper_numeric_outputs
