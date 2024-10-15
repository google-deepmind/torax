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
from typing import TypeAlias

import chex
from torax import array_typing
from torax import geometry
from torax import interpolated_param
from torax.config import base
from torax.config import config_args


TimeInterpolatedInput: TypeAlias = interpolated_param.TimeInterpolatedInput


@chex.dataclass(frozen=True)
class DynamicFormula:
  """Base class for dynamic configs."""


@dataclasses.dataclass
class Exponential(base.RuntimeParametersConfig['ExponentialProvider']):
  """Configures an exponential formula.

  See formulas.Exponential for more information on how this config is used.
  """

  # floats to parameterize the different formulas.
  total: TimeInterpolatedInput = 1.0
  c1: TimeInterpolatedInput = 1.0
  c2: TimeInterpolatedInput = 1.0

  # If True, uses r_norm when calculating the source profiles.
  use_normalized_r: bool = False

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> ExponentialProvider:
    del torax_mesh  # Unused.
    return ExponentialProvider(
        runtime_params_config=self,
        total=config_args.get_interpolated_var_single_axis(self.total,),
        c1=config_args.get_interpolated_var_single_axis(self.c1,),
        c2=config_args.get_interpolated_var_single_axis(self.c2,),
    )


@chex.dataclass
class ExponentialProvider(base.RuntimeParametersProvider['DynamicExponential']):
  """Runtime parameter provider for a single source/sink term."""

  runtime_params_config: Exponential
  total: interpolated_param.InterpolatedVarSingleAxis
  c1: interpolated_param.InterpolatedVarSingleAxis
  c2: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
    ) -> DynamicExponential:
    return DynamicExponential(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicExponential(DynamicFormula):

  total: array_typing.ScalarFloat
  c1: array_typing.ScalarFloat
  c2: array_typing.ScalarFloat
  use_normalized_r: bool


@dataclasses.dataclass
class Gaussian(base.RuntimeParametersConfig['GaussianProvider']):
  """Configures a Gaussian formula.

  See formulas.Gaussian for more information on how this config is used.
  """

  # floats to parameterize the different formulas.
  total: TimeInterpolatedInput = 1.0
  c1: TimeInterpolatedInput = 1.0
  c2: TimeInterpolatedInput = 1.0

  # If True, uses r_norm when calculating the source profiles.
  use_normalized_r: bool = False

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> GaussianProvider:
    del torax_mesh  # Unused.
    return GaussianProvider(
        runtime_params_config=self,
        total=config_args.get_interpolated_var_single_axis(self.total,),
        c1=config_args.get_interpolated_var_single_axis(self.c1,),
        c2=config_args.get_interpolated_var_single_axis(self.c2,),)


@chex.dataclass
class GaussianProvider(base.RuntimeParametersProvider['DynamicGaussian']):
  """Runtime parameter provider for a single source/sink term."""
  runtime_params_config: Gaussian
  total: interpolated_param.InterpolatedVarSingleAxis
  c1: interpolated_param.InterpolatedVarSingleAxis
  c2: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
    ) -> DynamicGaussian:
    return DynamicGaussian(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicGaussian(DynamicFormula):

  total: array_typing.ScalarFloat
  c1: array_typing.ScalarFloat
  c2: array_typing.ScalarFloat
  # If True, uses r_norm when calculating the source profiles.
  use_normalized_r: bool
