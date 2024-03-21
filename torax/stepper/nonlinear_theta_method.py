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
from typing import Type

import jax
from torax import calc_coeffs
from torax import config_slice
from torax import fvm
from torax import geometry
from torax import sim
from torax import state as state_module
from torax.fvm import newton_raphson_solve_block
from torax.fvm import optimizer_solve_block
from torax.sources import source_profiles
from torax.stepper import stepper
from torax.transport_model import transport_model as transport_model_lib


class NonlinearThetaMethod(stepper.Stepper):
  """Time step update using theta method.

  Attributes:
    transport_model: A TransportModel subclass, calculates transport coeffs.
    sources: All TORAX sources used to compute both the explicit and implicit
      source profiles used for each time step as terms in the state evolution
      equations. Though the explicit profiles are computed outside the call to
      Stepper, the same sources should be used to compute those. The Sources are
      exposed here to provide a single source of truth for which sources are
      used during a run.
    callback_class: Which class should be used to calculate the PDE coefficients
       for the linear and predictor-corrector initial guess routines.
  """

  def __init__(
      self,
      transport_model: transport_model_lib.TransportModel,
      sources: source_profiles.Sources,
      callback_class: Type[sim.CoeffsCallback] = sim.CoeffsCallback,
  ):
    super().__init__(transport_model, sources)
    self.callback_class = callback_class

  def _x_new(
      self,
      state_t: state_module.State,
      state_t_plus_dt: state_module.State,
      evolving_names: tuple[str, ...],
      geo: geometry.Geometry,
      dynamic_config_slice_t: config_slice.DynamicConfigSlice,
      dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      dt: jax.Array,
      mask: jax.Array,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[tuple[fvm.CellVariable, ...], int, calc_coeffs.AuxOutput]:
    """See Stepper._x_new docstring."""

    coeffs_callback = self.callback_class(
        state_t=state_t,
        evolving_names=evolving_names,
        geo=geo,
        static_config_slice=static_config_slice,
        transport_model=self.transport_model,
        mask=mask,
        explicit_source_profiles=explicit_source_profiles,
        sources=self.sources,
    )
    x_new, error, aux_output = self._x_new_helper(
        dynamic_config_slice_t=dynamic_config_slice_t,
        dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
        static_config_slice=static_config_slice,
        state_t=state_t,
        state_t_plus_dt=state_t_plus_dt,
        evolving_names=evolving_names,
        geo=geo,
        mask=mask,
        explicit_source_profiles=explicit_source_profiles,
        dt=dt,
        coeffs_callback=coeffs_callback,
    )

    return x_new, error, aux_output

  @abc.abstractmethod
  def _x_new_helper(
      self,
      dynamic_config_slice_t: config_slice.DynamicConfigSlice,
      dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      state_t: state_module.State,
      state_t_plus_dt: state_module.State,
      evolving_names: tuple[str, ...],
      geo: geometry.Geometry,
      mask: jax.Array,
      explicit_source_profiles: source_profiles.SourceProfiles,
      dt: jax.Array,
      coeffs_callback: sim.CoeffsCallback,
  ) -> tuple[tuple[fvm.CellVariable, ...], int, calc_coeffs.AuxOutput]:
    """Final implementation of x_new after callback has been created etc."""
    ...

  def _artificially_linear(self) -> bool:
    """If True, the Stepper has been hacked to be linear in practice."""
    if issubclass(self.callback_class, sim.FrozenCoeffsCallback):
      return True
    return False


class OptimizerThetaMethod(NonlinearThetaMethod):
  """Minimize the squared norm of the residual of the theta method equation.

  Attributes:
    transport_model: A TransportModel subclass, calculates transport coeffs.
    callback_class: Which class should be used to calculate the coefficients.
    initial_guess_mode: Passed through to `fvm.optimizer_solve_block`.
    maxiter: Passed through to `jaxopt.LBFGS`
    tol: Passed through to `jaxopt.LBFGS`
  """

  def __init__(
      self,
      transport_model: transport_model_lib.TransportModel,
      sources: source_profiles.Sources,
      callback_class: Type[sim.CoeffsCallback] = sim.CoeffsCallback,
      initial_guess_mode: fvm.InitialGuessMode = optimizer_solve_block.INITIAL_GUESS_MODE,
      maxiter: int = optimizer_solve_block.MAXITER,
      tol: float = optimizer_solve_block.TOL,
  ):
    self.maxiter = maxiter
    self.tol = tol
    self.initial_guess_mode = initial_guess_mode
    super().__init__(transport_model, sources, callback_class)

  def _x_new_helper(
      self,
      dynamic_config_slice_t: config_slice.DynamicConfigSlice,
      dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      state_t: state_module.State,
      state_t_plus_dt: state_module.State,
      evolving_names: tuple[str, ...],
      geo: geometry.Geometry,
      mask: jax.Array,
      explicit_source_profiles: source_profiles.SourceProfiles,
      dt: jax.Array,
      coeffs_callback: sim.CoeffsCallback,
  ) -> tuple[tuple[fvm.CellVariable, ...], int, calc_coeffs.AuxOutput]:
    """Final implementation of x_new after callback has been created etc."""
    return optimizer_solve_block.optimizer_solve_block(
        x_old=tuple([state_t[name] for name in evolving_names]),
        state_t_plus_dt=state_t_plus_dt,
        evolving_names=evolving_names,
        dt=dt,
        coeffs_callback=coeffs_callback,
        dynamic_config_slice_t=dynamic_config_slice_t,
        dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
        static_config_slice=static_config_slice,
        geo=geo,
        transport_model=self.transport_model,
        sources=self.sources,
        mask=mask,
        explicit_source_profiles=explicit_source_profiles,
        # theta_imp is not time-dependent. Not all parameters in the
        # dynamic_config_slice need to be time-dependent. They can simply change
        # from simulation run to simulation run without triggering a recompile.
        theta_imp=dynamic_config_slice_t.solver.theta_imp,
        convection_dirichlet_mode=(
            static_config_slice.solver.convection_dirichlet_mode
        ),
        convection_neumann_mode=(
            static_config_slice.solver.convection_neumann_mode
        ),
        initial_guess_mode=self.initial_guess_mode,
        maxiter=self.maxiter,
        tol=self.tol,
    )

  def _artificially_linear(self) -> bool:
    """If True, the Stepper has been hacked to be linear in practice."""
    if self.maxiter == 0:
      return True
    return super()._artificially_linear()


class NewtonRaphsonThetaMethod(NonlinearThetaMethod):
  """Nonlinear theta method using Newton Raphson.

  Attributes:
    transport_model: A TransportModel subclass, calculates transport coeffs.
    callback_class: Which class should be used to calculate the coefficients.
    initial_guess_mode: Passed through to `stepper.newton_raphson_solve_block`.
    maxiter: Passed through to `stepper.newton_raphson_solve_block`
    tol: Passed through to `stepper.newton_raphson_solve_block`
    delta_reduction_factor: Passed through to
      `stepper.newton_raphson_solve_block`
    tau_min: Passed through to `stepper.newton_raphson_solve_block`
  """

  def __init__(
      self,
      transport_model: transport_model_lib.TransportModel,
      sources: source_profiles.Sources,
      callback_class: Type[sim.CoeffsCallback] = sim.CoeffsCallback,
      initial_guess_mode: fvm.InitialGuessMode = newton_raphson_solve_block.INITIAL_GUESS_MODE,
      maxiter: int = newton_raphson_solve_block.MAXITER,
      tol: float = newton_raphson_solve_block.TOL,
      coarse_tol: float = newton_raphson_solve_block.COARSE_TOL,
      delta_reduction_factor: float = newton_raphson_solve_block.DELTA_REDUCTION_FACTOR,
      tau_min: float = newton_raphson_solve_block.TAU_MIN,
  ):
    self.initial_guess_mode = initial_guess_mode
    self.maxiter = maxiter
    self.tol = tol
    self.coarse_tol = coarse_tol
    self.delta_reduction_factor = delta_reduction_factor
    self.tau_min = tau_min
    super().__init__(transport_model, sources, callback_class)

  def _x_new_helper(
      self,
      dynamic_config_slice_t: config_slice.DynamicConfigSlice,
      dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      state_t: state_module.State,
      state_t_plus_dt: state_module.State,
      evolving_names: tuple[str, ...],
      geo: geometry.Geometry,
      mask: jax.Array,
      explicit_source_profiles: source_profiles.SourceProfiles,
      dt: jax.Array,
      coeffs_callback: sim.CoeffsCallback,
  ) -> tuple[tuple[fvm.CellVariable, ...], int, calc_coeffs.AuxOutput]:
    """Final implementation of x_new after callback has been created etc."""
    # disable error checking in residual, since Newton-Raphson routine has
    # error checking based on result of each linear step

    return newton_raphson_solve_block.newton_raphson_solve_block(
        x_old=tuple([state_t[name] for name in evolving_names]),
        state_t_plus_dt=state_t_plus_dt,
        evolving_names=evolving_names,
        dt=dt,
        coeffs_callback=coeffs_callback,
        dynamic_config_slice_t=dynamic_config_slice_t,
        dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
        static_config_slice=static_config_slice,
        geo=geo,
        # theta_imp is not time-dependent. Not all parameters in the
        # dynamic_config_slice need to be time-dependent. They can simply change
        # from simulation run to simulation run without triggering a recompile.
        theta_imp=dynamic_config_slice_t.solver.theta_imp,
        transport_model=self.transport_model,
        sources=self.sources,
        mask=mask,
        explicit_source_profiles=explicit_source_profiles,
        log_iterations=static_config_slice.solver.log_iterations,
        convection_dirichlet_mode=(
            static_config_slice.solver.convection_dirichlet_mode
        ),
        convection_neumann_mode=(
            static_config_slice.solver.convection_neumann_mode
        ),
        initial_guess_mode=self.initial_guess_mode,
        maxiter=self.maxiter,
        tol=self.tol,
        coarse_tol=self.coarse_tol,
        delta_reduction_factor=self.delta_reduction_factor,
    )

  def _artificially_linear(self) -> bool:
    """If True, the Stepper has been hacked to be linear in practice."""
    if self.maxiter == 0:
      return True
    return super()._artificially_linear()
