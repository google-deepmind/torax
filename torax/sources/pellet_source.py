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
"""Pellet source for the ne equation."""
from __future__ import annotations

import dataclasses
from typing import ClassVar
from typing import Literal

import chex
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
def calc_pellet_source(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Calculates external source term for n from pellets."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicPelletRuntimeParams)
  return (
      formulas.gaussian_profile(
          center=dynamic_source_runtime_params.pellet_deposition_location,
          width=dynamic_source_runtime_params.pellet_width,
          total=(
              dynamic_source_runtime_params.S_pellet_tot
              / dynamic_runtime_params_slice.numerics.nref
          ),
          geo=geo,
      ),
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PelletSource(source.Source):
  """Pellet source for the ne equation."""

  SOURCE_NAME: ClassVar[str] = 'pellet_source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'calc_pellet_source'
  model_func: source.SourceProfileFunction = calc_pellet_source

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.NE,)


@chex.dataclass(frozen=True)
class DynamicPelletRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  pellet_width: array_typing.ScalarFloat
  pellet_deposition_location: array_typing.ScalarFloat
  S_pellet_tot: array_typing.ScalarFloat


@chex.dataclass
class PelletRuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: PelletRuntimeParams
  pellet_width: interpolated_param.InterpolatedVarSingleAxis
  pellet_deposition_location: interpolated_param.InterpolatedVarSingleAxis
  S_pellet_tot: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicPelletRuntimeParams:
    return DynamicPelletRuntimeParams(**self.get_dynamic_params_kwargs(t))


@dataclasses.dataclass(kw_only=True)
class PelletRuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for PelletSource."""

  # Gaussian width of pellet deposition [normalized radial coord],
  # (continuous pellet model)
  pellet_width: runtime_params_lib.TimeInterpolatedInput = 0.1
  # Pellet source Gaussian central location [normalized radial coord]
  # (continuous pellet model)
  pellet_deposition_location: runtime_params_lib.TimeInterpolatedInput = 0.85
  # total pellet particles/s (continuous pellet model)
  S_pellet_tot: runtime_params_lib.TimeInterpolatedInput = 2e22
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def make_provider(
      self,
      torax_mesh: torax_pydantic.Grid1D | None = None,
  ) -> PelletRuntimeParamsProvider:
    return PelletRuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


class PelletSourceConfig(runtime_params_lib.SourceModelBase):
  """Pellet source for the ne equation.

  Attributes:
    pellet_width: Gaussian width of pellet deposition [normalized radial coord]
    pellet_deposition_location: Gaussian center of pellet deposition [normalized
      radial coord]
    S_pellet_tot: total pellet particles/s
    mode: Defines how the source values are computed (from a model, from a file,
      etc.)
  """

  source_name: Literal['pellet_source'] = 'pellet_source'
  pellet_width: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )
  pellet_deposition_location: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.85)
  )
  S_pellet_tot: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(2e22)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED
