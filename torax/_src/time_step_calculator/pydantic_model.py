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

"""Pydantic config for time step calculators."""

import abc
import enum
from typing import Annotated, Literal, Self

import chex
import pydantic
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import from_previous_time_step_calculator
from torax._src.time_step_calculator import runtime_params
from torax._src.time_step_calculator import time_step_calculator
from torax._src.torax_pydantic import torax_pydantic


@enum.unique
class TimeStepCalculatorType(enum.StrEnum):
  """Types of time step calculators."""

  CHI = 'chi'
  FIXED = 'fixed'
  FROM_PREVIOUS_DT = 'from_previous_dt'


class BaseTimeStepCalculator(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base config for a time step calculator.

  Attributes:
    tolerance: The tolerance within the final time for which the simulation will
      be considered done.
    min_dt: Minimum timestep allowed in simulation.
    max_dt: Maximum timesteps allowed in the simulation. This is only used with
      the `chi_time_step_calculator` and `from_previous_time_step_calculator`
      time_step_calculators.
    adaptive_dt: Iterative reduction of dt if nonlinear step does not converge.
      If nonlinear step does not converge, then the step is redone iteratively
      at successively lower dt until convergence is reached.
    dt_reduction_factor: Factor by which to reduce dt if adaptive_dt is True.
  """

  tolerance: float = 1e-7
  min_dt: torax_pydantic.Second = 1e-8
  max_dt: torax_pydantic.Second = 2.0
  adaptive_dt: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  dt_reduction_factor: pydantic.PositiveFloat = 3.0

  @pydantic.model_validator(mode='after')
  def model_validation(self) -> Self:
    if self.min_dt > self.max_dt:
      raise ValueError(
          'max_dt must be greater than or equal to min_dt. '
          f'max_dt: {self.max_dt}, min_dt: {self.min_dt}'
      )
    return self

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    del t
    return runtime_params.RuntimeParams(
        tolerance=self.tolerance,
        min_dt=self.min_dt,
        max_dt=self.max_dt,
        dt_reduction_factor=self.dt_reduction_factor,
        adaptive_dt=self.adaptive_dt,
    )

  @abc.abstractmethod
  def build_time_step_calculator(
      self,
  ) -> time_step_calculator.TimeStepCalculator:
    """Builds a time step calculator from the config."""


class ChiTimeStepCalculator(BaseTimeStepCalculator):
  """Config for the chi time step calculator.

  Attributes:
    calculator_type: The type of time step calculator to use, hardcoded to
      'chi'.
    chi_timestep_prefactor: Prefactor in front of chi_timestep_calculator base
      timestep dt=dx^2/(2*chi). In most use-cases can be increased further above
      this.
  """

  calculator_type: Annotated[
      Literal[TimeStepCalculatorType.CHI], torax_pydantic.JAX_STATIC
  ] = TimeStepCalculatorType.CHI
  chi_timestep_prefactor: pydantic.PositiveFloat = 50.0

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.ChiRuntimeParams:
    base_params = super().build_runtime_params(t)
    return runtime_params.ChiRuntimeParams(
        **vars(base_params),
        chi_timestep_prefactor=self.chi_timestep_prefactor,
    )

  def build_time_step_calculator(
      self,
  ) -> chi_time_step_calculator.ChiTimeStepCalculator:
    return chi_time_step_calculator.ChiTimeStepCalculator()


class FixedTimeStepCalculator(BaseTimeStepCalculator):
  """Config for the fixed time step calculator.

  Attributes:
    calculator_type: The type of time step calculator to use, hardcoded to
      'fixed'.
    fixed_dt: Timestep used for `fixed_time_step_calculator`.
  """

  calculator_type: Annotated[
      Literal[TimeStepCalculatorType.FIXED], torax_pydantic.JAX_STATIC
  ] = TimeStepCalculatorType.FIXED
  fixed_dt: torax_pydantic.NonNegativeTimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(1e-1)
  )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.FixedRuntimeParams:
    base_params = super().build_runtime_params(t)
    return runtime_params.FixedRuntimeParams(
        **vars(base_params),
        fixed_dt=self.fixed_dt.get_value(t),
    )

  def build_time_step_calculator(
      self,
  ) -> fixed_time_step_calculator.FixedTimeStepCalculator:
    return fixed_time_step_calculator.FixedTimeStepCalculator()


class FromPreviousTimeStepCalculator(BaseTimeStepCalculator):
  """Config for the from_previous_dt time step calculator.

  Attributes:
    calculator_type: The type of time step calculator to use, hardcoded to
      'from_previous_dt'.
    fixed_dt: Initial timestep used for `from_previous_time_step_calculator`.
  """

  calculator_type: Annotated[
      Literal[TimeStepCalculatorType.FROM_PREVIOUS_DT],
      torax_pydantic.JAX_STATIC,
  ] = TimeStepCalculatorType.FROM_PREVIOUS_DT
  fixed_dt: torax_pydantic.NonNegativeTimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(1e-1)
  )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.FromPreviousRuntimeParams:
    base_params = super().build_runtime_params(t)
    return runtime_params.FromPreviousRuntimeParams(
        **vars(base_params),
        fixed_dt=self.fixed_dt.get_value(t),
    )

  def build_time_step_calculator(
      self,
  ) -> from_previous_time_step_calculator.FromPreviousTimeStepCalculator:
    return from_previous_time_step_calculator.FromPreviousTimeStepCalculator()


ChiTimeStepCalculatorConfig = ChiTimeStepCalculator
FixedTimeStepCalculatorConfig = FixedTimeStepCalculator
FromPreviousTimeStepCalculatorConfig = FromPreviousTimeStepCalculator

TimeStepCalculatorConfig = Annotated[
    ChiTimeStepCalculator
    | FixedTimeStepCalculator
    | FromPreviousTimeStepCalculator,
    pydantic.Field(discriminator='calculator_type'),
]
TimeStepCalculator = TimeStepCalculatorConfig
