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

"""Plasma composition parameters used throughout TORAX simulations."""

from __future__ import annotations

import chex
from torax import array_typing
from torax import geometry
from torax import interpolated_param
from torax.config import base


# pylint: disable=invalid-name


@chex.dataclass
class PlasmaComposition(
    base.RuntimeParametersConfig['PlasmaCompositionProvider']
):
  """Configuration for the plasma composition."""

  # amu of main ion (if multiple isotope, make average)
  Ai: float = 2.5
  # charge of main ion
  Zi: float = 1.0
  # needed for qlknn and fusion power
  Zeff: interpolated_param.TimeInterpolated = 1.0
  Zimp: interpolated_param.TimeInterpolated = (
      10.0  # impurity charge state assumed for dilution
  )

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> PlasmaCompositionProvider:
    return PlasmaCompositionProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class PlasmaCompositionProvider(
    base.RuntimeParametersProvider['DynamicPlasmaComposition']
):
  """Prepared plasma composition."""

  runtime_params_config: PlasmaComposition
  Zeff: interpolated_param.InterpolatedVarSingleAxis
  Zimp: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> DynamicPlasmaComposition:
    return DynamicPlasmaComposition(**self.get_dynamic_params_kwargs(t))


@chex.dataclass
class DynamicPlasmaComposition:
  Ai: array_typing.ScalarFloat
  Zi: array_typing.ScalarFloat
  Zeff: array_typing.ScalarFloat
  Zimp: array_typing.ScalarFloat
