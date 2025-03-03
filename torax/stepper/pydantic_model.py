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
from typing import Any, Literal, Union

import pydantic
from torax.fvm import enums
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class LinearThetaMethod(torax_pydantic.BaseModelMutable):
  """Model for the linear stepper.

  This is also the base model for the Newton-Raphson and Optimizer steppers as
  they share the same parameters.

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
  stepper_type: Literal['linear']
  theta_imp: torax_pydantic.UnitInterval = 1.0
  predictor_corrector: bool = True
  corrector_steps: pydantic.PositiveInt = 1
  convection_dirichlet_mode: str = 'ghost'
  convection_neumann_mode: str = 'ghost'
  use_pereverzev: bool = False
  chi_per: pydantic.PositiveFloat = 20.0
  d_per: pydantic.NonNegativeFloat = 10.0


class NewtonRaphsonThetaMethod(LinearThetaMethod):
  """Model for non linear NewtonRaphsonThetaMethod stepper.

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
  stepper_type: Literal['newton_raphson']
  log_iterations: bool = False
  initial_guess_mode: enums.InitialGuessMode = enums.InitialGuessMode.LINEAR
  maxiter: pydantic.NonNegativeInt = 30
  tol: float = 1e-5
  coarse_tol: float = 1e-2
  delta_reduction_factor: float = 0.5
  tau_min: float = 0.01


class OptimizerThetaMethod(LinearThetaMethod):
  """Model for non linear OptimizerThetaMethod stepper.

  Attributes:
    stepper_type: The type of stepper to use, hardcoded to 'optimizer'.
    initial_guess_mode: The initial guess mode for the optimizer.
    maxiter: The maximum number of iterations for the optimizer.
    tol: The tolerance for the optimizer.
  """

  stepper_type: Literal['optimizer']
  initial_guess_mode: enums.InitialGuessMode = enums.InitialGuessMode.LINEAR
  maxiter: pydantic.NonNegativeInt = 100
  tol: float = 1e-12


StepperConfig = Union[
    LinearThetaMethod, NewtonRaphsonThetaMethod, OptimizerThetaMethod
]


class Stepper(torax_pydantic.BaseModelMutable):
  """Config for a stepper.

  The `from_dict` method of constructing this class supports the config
  described in: https://torax.readthedocs.io/en/latest/configuration.html
  """
  stepper_config: StepperConfig = pydantic.Field(discriminator='stepper_type')

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    # If we are running with the standard class constructor we don't need to do
    # any custom validation.
    if 'stepper_config' in data:
      return data

    return {'stepper_config': data}
