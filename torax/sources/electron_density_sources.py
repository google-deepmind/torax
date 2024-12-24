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

"""Sources for the ne equation."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import chex
import jax
from torax import array_typing
from torax import interpolated_param
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_models


# pylint: disable=invalid-name
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
      torax_mesh: geometry.Grid1D | None = None,
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
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles | None = None,
    unused_source_models: source_models.SourceModels | None = None,
) -> jax.Array:
  """Calculates external source term for n from puffs."""
  del (
      unused_source_models,
      static_runtime_params_slice,
  )  # Unused.
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicGasPuffRuntimeParams)
  return formulas.exponential_profile(
      c1=1.0,
      c2=dynamic_source_runtime_params.puff_decay_length,
      total=(
          dynamic_source_runtime_params.S_puff_tot
          / dynamic_runtime_params_slice.numerics.nref
      ),
      geo=geo,
  )


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


@chex.dataclass(frozen=True)
class DynamicParticleRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  particle_width: array_typing.ScalarFloat
  deposition_location: array_typing.ScalarFloat
  S_tot: array_typing.ScalarFloat


def calc_generic_particle_source(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles | None = None,
    unused_source_models: source_models.SourceModels | None = None,
) -> jax.Array:
  """Calculates external source term for n from SBI."""
  del (
      unused_source_models,
      static_runtime_params_slice,
  )  # Unused.
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicParticleRuntimeParams)
  return formulas.gaussian_profile(
      c1=dynamic_source_runtime_params.deposition_location,
      c2=dynamic_source_runtime_params.particle_width,
      total=(
          dynamic_source_runtime_params.S_tot
          / dynamic_runtime_params_slice.numerics.nref
      ),
      geo=geo,
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
      torax_mesh: geometry.Grid1D | None = None,
  ) -> PelletRuntimeParamsProvider:
    return PelletRuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


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


@chex.dataclass(frozen=True)
class DynamicPelletRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  pellet_width: array_typing.ScalarFloat
  pellet_deposition_location: array_typing.ScalarFloat
  S_pellet_tot: array_typing.ScalarFloat


def calc_pellet_source(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles | None = None,
    unused_source_models: source_models.SourceModels | None = None,
) -> jax.Array:
  """Calculates external source term for n from pellets."""
  del (
      unused_source_models,
      static_runtime_params_slice,
  )  # Unused.
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicPelletRuntimeParams)
  return formulas.gaussian_profile(
      c1=dynamic_source_runtime_params.pellet_deposition_location,
      c2=dynamic_source_runtime_params.pellet_width,
      total=(
          dynamic_source_runtime_params.S_pellet_tot
          / dynamic_runtime_params_slice.numerics.nref
      ),
      geo=geo,
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
