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

"""Defines the runtime (dynamic) configuration of formulas used in sources."""

from __future__ import annotations

import dataclasses

from torax import interpolated_param


# Type-alias for clarity.
TimeDependentField = interpolated_param.InterpParamOrInterpParamInput


@dataclasses.dataclass
class Exponential:
  """Configures an exponential formula.

  See formulas.Exponential for more information on how this config is used.
  """

  # floats to parameterize the different formulas.
  total: TimeDependentField = 1.0
  c1: TimeDependentField = 1.0
  c2: TimeDependentField = 1.0

  # If True, uses r_norm when calculating the source profiles.
  use_normalized_r: bool = False


@dataclasses.dataclass
class Gaussian:
  """Configures a Gaussian formula.

  See formulas.Gaussian for more information on how this config is used.
  """

  # floats to parameterize the different formulas.
  total: TimeDependentField = 1.0
  c1: TimeDependentField = 1.0
  c2: TimeDependentField = 1.0

  # If True, uses r_norm when calculating the source profiles.
  use_normalized_r: bool = False


@dataclasses.dataclass
class FormulaConfig:
  """Contains all formula configs."""

  exponential: Exponential = dataclasses.field(default_factory=Exponential)
  gaussian: Gaussian = dataclasses.field(default_factory=Gaussian)
  custom_params: dict[str, TimeDependentField] = dataclasses.field(
      default_factory=lambda: {},
  )
