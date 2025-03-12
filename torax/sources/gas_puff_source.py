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

"""Gas puff source for the ne equation."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal

import chex
import pydantic
from torax import array_typing
from torax import interpolated_param
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class GasPuffSourceConfig(runtime_params_lib.SourceModelBase):
  """Gas puff source for the ne equation.

  Attributes:
    source_name: Name of the source, hardcoded to 'gas_puff_source'
    puff_decay_length: exponential decay length of gas puff ionization
      [normalized radial coord]
    S_puff_tot: total gas puff particles/s
  """
  source_name: Literal['gas_puff_source'] = 'gas_puff_source'
  puff_decay_length: torax_pydantic.UnitInterval = 0.05
  S_puff_tot: pydantic.NonNegativeFloat = 1e22
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED


@dataclasses.dataclass(kw_only=True)
class GasPuffRuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for GasPuffSource."""

  # exponential decay length of gas puff ionization [normalized radial coord]
  puff_decay_length: runtime_params_lib.TimeInterpolatedInput = 0.05
  # total gas puff particles/s
  S_puff_tot: runtime_params_lib.TimeInterpolatedInput = 1e22
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def make_provider(
      self,
      torax_mesh: torax_pydantic.Grid1D | None = None,
  ) -> GasPuffRuntimeParamsProvider:
    if torax_mesh is None:
      raise ValueError(
          'torax_mesh is required for GasPuffRuntimeParams.make_provider.'
      )
    return GasPuffRuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class GasPuffRuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: GasPuffRuntimeParams
  puff_decay_length: interpolated_param.InterpolatedVarSingleAxis
  S_puff_tot: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicGasPuffRuntimeParams:
    return DynamicGasPuffRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicGasPuffRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  puff_decay_length: array_typing.ScalarFloat
  S_puff_tot: array_typing.ScalarFloat


# Default formula: exponential with nref normalization.
def calc_puff_source(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Calculates external source term for n from puffs."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicGasPuffRuntimeParams)
  return (formulas.exponential_profile(
      decay_start=1.0,
      width=dynamic_source_runtime_params.puff_decay_length,
      total=(
          dynamic_source_runtime_params.S_puff_tot
          / dynamic_runtime_params_slice.numerics.nref
      ),
      geo=geo,
  ),)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GasPuffSource(source.Source):
  """Gas puff source for the ne equation."""

  SOURCE_NAME: ClassVar[str] = 'gas_puff_source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'calc_puff_source'
  model_func: source.SourceProfileFunction = calc_puff_source

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.NE,)
