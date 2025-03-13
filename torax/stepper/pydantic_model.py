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

"""Pydantic config for Stepper."""
import functools
from typing import Any, Literal, Union

import pydantic
from torax.fvm import enums
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.stepper import nonlinear_theta_method
from torax.stepper import runtime_params
from torax.stepper import stepper as stepper_lib
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import transport_model as transport_model_lib
from typing_extensions import Annotated


# pylint: disable=invalid-name
class LinearThetaMethod(torax_pydantic.BaseModelFrozen):
  """Model for the linear stepper.

  This is also the base model for the NewtonRaphsonThetaMethod and
  OptimizerThetaMethod steppers as they share the same parameters due to the
  nonlinear solvers having an option to use the a linear stepper solution as an
  initial guess.

  Attributes:
    stepper_type: The type of stepper to use, hardcoded to 'linear'.
    theta_imp: The theta value in the theta method 0 = explicit, 1 = fully
      implicit, 0.5 = Crank-Nicolson.
    predictor_corrector: Enables predictor_corrector iterations with the linear
      solver. If False, compilation is faster.
    corrector_steps: The number of corrector steps for the predictor-corrector
      linear solver. 0 means a pure linear solve with no corrector steps.
    convection_dirichlet_mode: See `fvm.convection_terms` docstring,
      `dirichlet_mode` argument.
    convection_neumann_mode: See `fvm.convection_terms` docstring,
      `neumann_mode` argument.
    use_pereverzev: Use pereverzev terms for linear solver. Is only applied in
      the nonlinear solver for the optional initial guess from the linear solver
    chi_per: (deliberately) large heat conductivity for Pereverzev rule.
    d_per: (deliberately) large particle diffusion for Pereverzev rule.
  """

  stepper_type: Literal['linear'] = 'linear'
  theta_imp: torax_pydantic.UnitInterval = 1.0
  predictor_corrector: bool = True
  corrector_steps: pydantic.PositiveInt = 1
  convection_dirichlet_mode: Literal['ghost', 'direct', 'semi-implicit'] = (
      'ghost'
  )
  convection_neumann_mode: Literal['ghost', 'semi-implicit'] = 'ghost'
  use_pereverzev: bool = False
  chi_per: pydantic.PositiveFloat = 20.0
  d_per: pydantic.NonNegativeFloat = 10.0

  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    return runtime_params.DynamicRuntimeParams(
        chi_per=self.chi_per,
        d_per=self.d_per,
        corrector_steps=self.corrector_steps,
    )

  def build_stepper(
      self,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
      pedestal_model: pedestal_model_lib.PedestalModel,
  ) -> stepper_lib.Stepper:
    return linear_theta_method.LinearThetaMethod(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )

  @property
  def linear_solver(self) -> bool:
    return True


class NewtonRaphsonThetaMethod(LinearThetaMethod):
  """Model for nonlinear Newton-Raphson stepper.

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
  maxiter: pydantic.NonNegativeInt = 30
  tol: float = 1e-5
  coarse_tol: float = 1e-2
  delta_reduction_factor: float = 0.5
  tau_min: float = 0.01

  @property
  def linear_solver(self) -> bool:
    return self.initial_guess_mode == enums.InitialGuessMode.LINEAR

  def build_dynamic_params(
      self,
  ) -> nonlinear_theta_method.DynamicNewtonRaphsonRuntimeParams:
    return nonlinear_theta_method.DynamicNewtonRaphsonRuntimeParams(
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
      self,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
      pedestal_model: pedestal_model_lib.PedestalModel,
  ) -> nonlinear_theta_method.NewtonRaphsonThetaMethod:
    return nonlinear_theta_method.NewtonRaphsonThetaMethod(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )


class OptimizerThetaMethod(LinearThetaMethod):
  """Model for nonlinear OptimizerThetaMethod stepper.

  Attributes:
    stepper_type: The type of stepper to use, hardcoded to 'optimizer'.
    initial_guess_mode: The initial guess mode for the optimizer.
    maxiter: The maximum number of iterations for the optimizer.
    tol: The tolerance for the optimizer.
  """

  stepper_type: Literal['optimizer'] = 'optimizer'
  initial_guess_mode: enums.InitialGuessMode = enums.InitialGuessMode.LINEAR
  maxiter: pydantic.NonNegativeInt = 100
  tol: float = 1e-12

  @property
  def linear_solver(self) -> bool:
    return self.initial_guess_mode == enums.InitialGuessMode.LINEAR

  def build_dynamic_params(
      self,
  ) -> nonlinear_theta_method.DynamicOptimizerRuntimeParams:
    return nonlinear_theta_method.DynamicOptimizerRuntimeParams(
        chi_per=self.chi_per,
        d_per=self.d_per,
        initial_guess_mode=self.initial_guess_mode.value,
        maxiter=self.maxiter,
        tol=self.tol,
        corrector_steps=self.corrector_steps,
    )

  def build_stepper(
      self,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
      pedestal_model: pedestal_model_lib.PedestalModel,
  ) -> nonlinear_theta_method.OptimizerThetaMethod:
    return nonlinear_theta_method.OptimizerThetaMethod(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )


StepperConfig = Union[
    Annotated[LinearThetaMethod, pydantic.Tag('linear')],
    Annotated[NewtonRaphsonThetaMethod, pydantic.Tag('newton_raphson')],
    Annotated[OptimizerThetaMethod, pydantic.Tag('optimizer')],
]


def get_discriminator_value(model: dict[str, Any]) -> str:
  return model.get('stepper_type', 'linear')


class Stepper(torax_pydantic.BaseModelFrozen):
  """Config for a stepper.

  The `from_dict` method of constructing this class supports the config
  described in: https://torax.readthedocs.io/en/latest/configuration.html
  """
  stepper_config: Annotated[
      StepperConfig, pydantic.Discriminator(get_discriminator_value)
  ]

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    # If we are running with the standard class constructor we don't need to do
    # any custom validation.
    if 'stepper_config' in data:
      return data

    return {'stepper_config': data}

  @functools.cached_property
  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    return self.stepper_config.build_dynamic_params()

  def build_static_params(self) -> runtime_params.StaticRuntimeParams:
    return runtime_params.StaticRuntimeParams(
        theta_imp=self.stepper_config.theta_imp,
        convection_dirichlet_mode=self.stepper_config.convection_dirichlet_mode,
        convection_neumann_mode=self.stepper_config.convection_neumann_mode,
        use_pereverzev=self.stepper_config.use_pereverzev,
        predictor_corrector=self.stepper_config.predictor_corrector,
    )

  def build_stepper_model(
      self,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
      pedestal_model: pedestal_model_lib.PedestalModel,
  ) -> stepper_lib.Stepper:
    return self.stepper_config.build_stepper(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )
