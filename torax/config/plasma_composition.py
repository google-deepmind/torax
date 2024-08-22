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
from torax import geometry
from torax import interpolated_param
from torax.config import base
from torax.config import config_args


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
    return PlasmaCompositionProvider(
        runtime_params_config=self,
        Zeff=config_args.get_interpolated_var_single_axis(self.Zeff),
        Zimp=config_args.get_interpolated_var_single_axis(self.Zimp),
    )


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
    return DynamicPlasmaComposition(
        Ai=self.runtime_params_config.Ai,
        Zi=self.runtime_params_config.Zi,
        Zeff=float(self.Zeff.get_value(t)),
        Zimp=float(self.Zimp.get_value(t)),
    )


@chex.dataclass
class DynamicPlasmaComposition:
  Ai: float
  Zi: float
  Zeff: float
  Zimp: float
