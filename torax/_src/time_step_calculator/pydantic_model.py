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

from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import runtime_params
from torax._src.time_step_calculator import time_step_calculator
from torax._src.torax_pydantic import torax_pydantic


@enum.unique
class TimeStepCalculatorType(enum.Enum):
  """Types of time step calculators."""

  CHI = 'chi'
  FIXED = 'fixed'


class TimeStepCalculator(torax_pydantic.BaseModelFrozen):
  """Config for a time step calculator.

  Attributes:
    calculator_type: The type of time step calculator to use.
    tolerance: The tolerance within the final time for which the simulation will
      be considered done.
  """

  calculator_type: Annotated[
      TimeStepCalculatorType, torax_pydantic.JAX_STATIC
  ] = TimeStepCalculatorType.CHI
  tolerance: float = 1e-7

  def build_runtime_params(self) -> runtime_params.RuntimeParams:
    return runtime_params.RuntimeParams(tolerance=self.tolerance)

  @property
  def time_step_calculator(self) -> time_step_calculator.TimeStepCalculator:
    match self.calculator_type:
      case TimeStepCalculatorType.CHI:
        return chi_time_step_calculator.ChiTimeStepCalculator()
      case TimeStepCalculatorType.FIXED:
        return fixed_time_step_calculator.FixedTimeStepCalculator()
