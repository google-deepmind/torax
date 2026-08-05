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

import enum
from typing import Annotated

import pydantic
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import from_previous_time_step_calculator
from torax._src.time_step_calculator import pellet_aware_time_step_calculator
from torax._src.time_step_calculator import runtime_params
from torax._src.time_step_calculator import time_step_calculator
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Self


@enum.unique
class TimeStepCalculatorType(enum.Enum):
  """Types of time step calculators."""

  CHI = 'chi'
  FIXED = 'fixed'
  FROM_PREVIOUS_DT = 'from_previous_dt'
  PELLET_AWARE = 'pellet_aware'


class TimeStepCalculator(torax_pydantic.BaseModelFrozen):
  """Config for a time step calculator.

  Attributes:
    calculator_type: The type of time step calculator to use.
    tolerance: The tolerance within the final time for which the simulation will
      be considered done.
    base_calculator: For the 'pellet_aware' calculator, the base time
      step calculator used away from pellet events. If None, the
      'chi' calculator is used.
    trigger_tolerance: For the 'pellet_aware' calculator, the time tolerance for
      deciding whether the current time coincides with a pellet trigger or an
      ablation boundary.
    window_after_pellet: For the 'pellet_aware' calculator, the duration of the
      window after a pellet trigger during which dt_after_pellet is used.
    dt_after_pellet: For the 'pellet_aware' calculator, the time step used during
      the window after a pellet trigger. If None, the base calculator's step is
      used.
  """

  calculator_type: Annotated[
      TimeStepCalculatorType, torax_pydantic.JAX_STATIC
  ] = TimeStepCalculatorType.CHI
  tolerance: float = 1e-7
  base_calculator: 'TimeStepCalculator | None' = None
  trigger_tolerance: float = 1e-8
  window_after_pellet: float = 0.0
  dt_after_pellet: float | None = None

  @pydantic.model_validator(mode='after')
  def _validate_base_calculator(self) -> Self:
    if self.base_calculator is not None:
      if self.calculator_type != TimeStepCalculatorType.PELLET_AWARE:
        raise ValueError(
            'base_calculator is only used by the pellet_aware time step '
            'calculator.'
        )
      if (
          self.base_calculator.calculator_type
          == TimeStepCalculatorType.PELLET_AWARE
      ):
        raise ValueError(
            'The base_calculator of a pellet_aware calculator cannot itself be '
            'pellet_aware.'
        )
    return self

  def build_runtime_params(self) -> runtime_params.RuntimeParams:
    return runtime_params.RuntimeParams(tolerance=self.tolerance)

  def build_time_step_calculator(
      self,
  ) -> time_step_calculator.TimeStepCalculator:
    match self.calculator_type:
      case TimeStepCalculatorType.CHI:
        return chi_time_step_calculator.ChiTimeStepCalculator()
      case TimeStepCalculatorType.FIXED:
        return fixed_time_step_calculator.FixedTimeStepCalculator()
      case TimeStepCalculatorType.FROM_PREVIOUS_DT:
        return (
            from_previous_time_step_calculator.FromPreviousTimeStepCalculator()
        )
      case TimeStepCalculatorType.PELLET_AWARE:
        # Default the calculator to 'chi' when none is configured.
        base_calculator_config = self.base_calculator or TimeStepCalculator()
        return pellet_aware_time_step_calculator.PelletAwareTimeStepCalculator(
            base_calculator=base_calculator_config.build_time_step_calculator(),
            trigger_tolerance=self.trigger_tolerance,
            window_after_pellet=self.window_after_pellet,
            dt_after_pellet=self.dt_after_pellet,
        )


# Resolve the self-referential 'base_calculator' annotation.
TimeStepCalculator.model_rebuild()
