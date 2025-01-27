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
from typing import Any
from torax.config import pydantic_base
from torax.time_step_calculator import chi_time_step_calculator
from torax.time_step_calculator import fixed_time_step_calculator
from torax.time_step_calculator import time_step_calculator


@enum.unique
class TimeStepCalculatorType(enum.Enum):
  """Types of time step calculators."""

  CHI = 'chi'
  FIXED = 'fixed'


class TimeStepCalculator(pydantic_base.Base):
  """Config for a time step calculator."""

  calculator_type: TimeStepCalculatorType = TimeStepCalculatorType.CHI
  init_kwargs: dict[str, Any] = {}

  @property
  def time_step_calculator(self) -> time_step_calculator.TimeStepCalculator:
    match self.calculator_type:
      case TimeStepCalculatorType.CHI:
        return chi_time_step_calculator.ChiTimeStepCalculator(
            **self.init_kwargs
        )
      case TimeStepCalculatorType.FIXED:
        return fixed_time_step_calculator.FixedTimeStepCalculator(
            **self.init_kwargs
        )
