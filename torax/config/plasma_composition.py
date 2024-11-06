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

import dataclasses
import logging

import chex
from torax import array_typing
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
  Zeff: interpolated_param.InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: 1.0
  )
  Zimp: interpolated_param.TimeInterpolatedInput = (
      10.0  # impurity charge state assumed for dilution
  )
  Aimp: interpolated_param.TimeInterpolatedInput = (
      20.18  # impurity mass in amu
  )

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> PlasmaCompositionProvider:
    if torax_mesh is None:
      raise ValueError(
          'torax_mesh is required to make a PlasmaCompositionProvider'
      )
    return PlasmaCompositionProvider(
        runtime_params_config=self,
        Zeff=config_args.get_interpolated_var_2d(
            self.Zeff,
            torax_mesh.cell_centers,
        ),
        Zeff_face=config_args.get_interpolated_var_2d(
            self.Zeff,
            torax_mesh.face_centers,
        ),
        Zimp=config_args.get_interpolated_var_single_axis(self.Zimp),
        Aimp=config_args.get_interpolated_var_single_axis(self.Aimp),
    )

  def __post_init__(self):
    if not interpolated_param.rhonorm1_defined_in_timerhoinput(self.Zeff):
      logging.info("""
          Config input Zeff not directly defined at rhonorm=1.0.
          Zeff_face at rhonorm=1.0 set from constant values or constant extrapolation.
          """)


@chex.dataclass
class PlasmaCompositionProvider(
    base.RuntimeParametersProvider['DynamicPlasmaComposition']
):
  """Prepared plasma composition."""

  runtime_params_config: PlasmaComposition
  Zeff: interpolated_param.InterpolatedVarTimeRho
  Zeff_face: interpolated_param.InterpolatedVarTimeRho
  Zimp: interpolated_param.InterpolatedVarSingleAxis
  Aimp: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicPlasmaComposition:
    return DynamicPlasmaComposition(**self.get_dynamic_params_kwargs(t))


@chex.dataclass
class DynamicPlasmaComposition:
  Ai: float
  Zi: float
  Zeff: array_typing.ArrayFloat
  Zeff_face: array_typing.ArrayFloat
  Zimp: array_typing.ScalarFloat
  Aimp: array_typing.ScalarFloat
