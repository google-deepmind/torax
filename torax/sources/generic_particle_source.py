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
"""Generic particle source for the ne equation."""
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
def calc_generic_particle_source(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Calculates external source term for n from SBI."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(
      dynamic_source_runtime_params,
      DynamicParticleRuntimeParams,
  )
  return (
      formulas.gaussian_profile(
          center=dynamic_source_runtime_params.deposition_location,
          width=dynamic_source_runtime_params.particle_width,
          total=(
              dynamic_source_runtime_params.S_tot
              / dynamic_runtime_params_slice.numerics.nref
          ),
          geo=geo,
      ),
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericParticleSource(source.Source):
  """Neutral-beam injection source for the ne equation."""

  SOURCE_NAME: ClassVar[str] = 'generic_particle_source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'calc_generic_particle_source'
  model_func: source.SourceProfileFunction = calc_generic_particle_source

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.NE,)


@chex.dataclass(frozen=True)
class DynamicParticleRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  particle_width: array_typing.ScalarFloat
  deposition_location: array_typing.ScalarFloat
  S_tot: array_typing.ScalarFloat


@chex.dataclass
class GenericParticleSourceRuntimeParamsProvider(
    runtime_params_lib.RuntimeParamsProvider
):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: GenericParticleSourceRuntimeParams
  particle_width: interpolated_param.InterpolatedVarSingleAxis
  deposition_location: interpolated_param.InterpolatedVarSingleAxis
  S_tot: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicParticleRuntimeParams:
    return DynamicParticleRuntimeParams(
        particle_width=float(self.particle_width.get_value(t)),
        deposition_location=float(self.deposition_location.get_value(t)),
        S_tot=float(self.S_tot.get_value(t)),
        prescribed_values=self.prescribed_values.get_value(t),
    )


@dataclasses.dataclass(kw_only=True)
class GenericParticleSourceRuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for particle source."""

  # particle source Gaussian width in normalized radial coord
  particle_width: runtime_params_lib.TimeInterpolatedInput = 0.25
  # particle source Gaussian central location in normalized radial coord
  deposition_location: runtime_params_lib.TimeInterpolatedInput = 0.0
  # total particle source
  S_tot: runtime_params_lib.TimeInterpolatedInput = 1e22
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> GenericParticleSourceRuntimeParamsProvider:
    return GenericParticleSourceRuntimeParamsProvider(
        **self.get_provider_kwargs(torax_mesh)
    )


class GenericParticleSourceConfig(runtime_params_lib.SourceModelBase):
  """Generic particle source for the ne equation.

  Attributes:
    particle_width: particle source Gaussian width in normalized radial coord
    deposition_location: particle source Gaussian center in normalized radial
      coord
    S_tot: total particle source particles/s
    mode: Defines how the source values are computed (from a model, from a file,
      etc.)
  """
  source_name: Literal['generic_particle_source'] = 'generic_particle_source'
  particle_width: torax_pydantic.UnitInterval = 0.25
  deposition_location: torax_pydantic.UnitInterval = 0.0
  S_tot: pydantic.NonNegativeFloat = 1e22
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED
