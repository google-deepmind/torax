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

import chex
from torax import interpolated_param
from torax.runtime_params import config_slice_args


# Type-alias for clarity.
TimeDependentField = interpolated_param.InterpParamOrInterpParamInput


@dataclasses.dataclass
class FormulaConfig:
  """Configures a formula.

  This config can include time-varying parameters which are interpolated as
  the simulation runs. For new formula implementations, extend this class and
  add the formula-specific parameters required.

  The Gaussian and Exponential config classes, and their implementations in
  formulas.py, are useful, simple examples for how to do this.
  """

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicFormula:
    """Interpolates this config to a dynamic config for time t."""
    del t  # Unused because there are no params in the base class.
    return DynamicFormula()


@chex.dataclass(frozen=True)
class DynamicFormula:
  """Base class for dynamic configs."""


@dataclasses.dataclass
class Exponential(FormulaConfig):
  """Configures an exponential formula.

  See formulas.Exponential for more information on how this config is used.
  """

  # floats to parameterize the different formulas.
  total: TimeDependentField = 1.0
  c1: TimeDependentField = 1.0
  c2: TimeDependentField = 1.0

  # If True, uses r_norm when calculating the source profiles.
  use_normalized_r: bool = False

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicExponential:
    return DynamicExponential(
        **config_slice_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicExponential,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class DynamicExponential(DynamicFormula):

  total: float
  c1: float
  c2: float
  use_normalized_r: bool


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

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicGaussian:
    return DynamicGaussian(
        **config_slice_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicGaussian,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class DynamicGaussian(DynamicFormula):

  total: float
  c1: float
  c2: float
  # If True, uses r_norm when calculating the source profiles.
  use_normalized_r: bool
