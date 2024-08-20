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

import chex
import jax
from torax import geometry
from torax import interpolated_param
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_models


# pylint: disable=invalid-name
@dataclasses.dataclass(kw_only=True)
class GasPuffRuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for GasPuffSource."""
  # exponential decay length of gas puff ionization [normalized radial coord]
  puff_decay_length: runtime_params_lib.TimeInterpolated = 0.05
  # total gas puff particles/s
  S_puff_tot: runtime_params_lib.TimeInterpolated = 1e22

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> GasPuffRuntimeParamsProvider:
    if torax_mesh is None:
      raise ValueError(
          'torax_mesh is required for GasPuffRuntimeParams.make_provider.'
      )
    return GasPuffRuntimeParamsProvider(
        runtime_params_config=self,
        formula=self.formula.make_provider(torax_mesh),
        prescribed_values=config_args.get_interpolated_var_2d(
            self.prescribed_values, torax_mesh.cell_centers
        ),
        puff_decay_length=config_args.get_interpolated_var_single_axis(
            self.puff_decay_length,
        ),
        S_puff_tot=config_args.get_interpolated_var_single_axis(
            self.S_puff_tot,
        ),
    )


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
    return DynamicGasPuffRuntimeParams(
        puff_decay_length=float(self.puff_decay_length.get_value(t)),
        S_puff_tot=float(self.S_puff_tot.get_value(t)),
        mode=self.runtime_params_config.mode.value,
        is_explicit=self.runtime_params_config.is_explicit,
        formula=self.formula.build_dynamic_params(t),
        prescribed_values=self.prescribed_values.get_value(t),
    )


@chex.dataclass(frozen=True)
class DynamicGasPuffRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  puff_decay_length: float
  S_puff_tot: float


# Default formula: exponential with nref normalization.
def _calc_puff_source(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    unused_state: state.CoreProfiles | None = None,
    unused_source_models: source_models.SourceModels | None = None,
) -> jax.Array:
  """Calculates external source term for n from puffs."""
  assert isinstance(dynamic_source_runtime_params, DynamicGasPuffRuntimeParams)
  return formulas.exponential_profile(
      c1=1.0,
      c2=dynamic_source_runtime_params.puff_decay_length,
      total=(
          dynamic_source_runtime_params.S_puff_tot
          / dynamic_runtime_params_slice.numerics.nref
      ),
      use_normalized_r=True,
      geo=geo,
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GasPuffSource(source.SingleProfileNeSource):
  """Gas puff source for the ne equation."""

  formula: source.SourceProfileFunction = _calc_puff_source


@dataclasses.dataclass(kw_only=True)
class NBIParticleRuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for NBI particle source."""

  # NBI particle source Gaussian width in normalized radial coord
  nbi_particle_width: runtime_params_lib.TimeInterpolated = 0.25
  # NBI particle source Gaussian central location in normalized radial coord
  nbi_deposition_location: runtime_params_lib.TimeInterpolated = 0.0
  # NBI total particle source
  S_nbi_tot: runtime_params_lib.TimeInterpolated = 1e22

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> NBIParticleRuntimeParamsProvider:
    if torax_mesh is None:
      raise ValueError(
          'torax_mesh is required for NBIParticleRuntimeParams.make_provider.'
      )
    return NBIParticleRuntimeParamsProvider(
        runtime_params_config=self,
        formula=self.formula.make_provider(torax_mesh),
        prescribed_values=config_args.get_interpolated_var_2d(
            self.prescribed_values, torax_mesh.cell_centers
        ),
        nbi_particle_width=config_args.get_interpolated_var_single_axis(
            self.nbi_particle_width,
        ),
        nbi_deposition_location=config_args.get_interpolated_var_single_axis(
            self.nbi_deposition_location,
        ),
        S_nbi_tot=config_args.get_interpolated_var_single_axis(
            self.S_nbi_tot,
        ),
    )


@chex.dataclass
class NBIParticleRuntimeParamsProvider(
    runtime_params_lib.RuntimeParamsProvider
):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: NBIParticleRuntimeParams
  nbi_particle_width: interpolated_param.InterpolatedVarSingleAxis
  nbi_deposition_location: interpolated_param.InterpolatedVarSingleAxis
  S_nbi_tot: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicNBIParticleRuntimeParams:
    return DynamicNBIParticleRuntimeParams(
        nbi_particle_width=float(self.nbi_particle_width.get_value(t)),
        nbi_deposition_location=float(
            self.nbi_deposition_location.get_value(t)
        ),
        S_nbi_tot=float(self.S_nbi_tot.get_value(t)),
        mode=self.runtime_params_config.mode.value,
        is_explicit=self.runtime_params_config.is_explicit,
        formula=self.formula.build_dynamic_params(t),
        prescribed_values=self.prescribed_values.get_value(t),
    )


@chex.dataclass(frozen=True)
class DynamicNBIParticleRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  nbi_particle_width: float
  nbi_deposition_location: float
  S_nbi_tot: float


def _calc_nbi_source(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    unused_state: state.CoreProfiles | None = None,
    unused_source_models: source_models.SourceModels | None = None,
) -> jax.Array:
  """Calculates external source term for n from SBI."""
  assert isinstance(
      dynamic_source_runtime_params, DynamicNBIParticleRuntimeParams
  )
  return formulas.gaussian_profile(
      c1=dynamic_source_runtime_params.nbi_deposition_location,
      c2=dynamic_source_runtime_params.nbi_particle_width,
      total=(
          dynamic_source_runtime_params.S_nbi_tot
          / dynamic_runtime_params_slice.numerics.nref
      ),
      use_normalized_r=True,
      geo=geo,
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class NBIParticleSource(source.SingleProfileNeSource):
  """Neutral-beam injection source for the ne equation."""

  formula: source.SourceProfileFunction = _calc_nbi_source


@dataclasses.dataclass(kw_only=True)
class PelletRuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for PelletSource."""

  # Gaussian width of pellet deposition [normalized radial coord],
  # (continuous pellet model)
  pellet_width: runtime_params_lib.TimeInterpolated = 0.1
  # Pellet source Gaussian central location [normalized radial coord]
  # (continuous pellet model)
  pellet_deposition_location: runtime_params_lib.TimeInterpolated = 0.85
  # total pellet particles/s (continuous pellet model)
  S_pellet_tot: runtime_params_lib.TimeInterpolated = 2e22

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> PelletRuntimeParamsProvider:
    if torax_mesh is None:
      raise ValueError(
          'torax_mesh is required for PelletRuntimeParams.make_provider.'
      )
    return PelletRuntimeParamsProvider(
        runtime_params_config=self,
        formula=self.formula.make_provider(torax_mesh),
        prescribed_values=config_args.get_interpolated_var_2d(
            self.prescribed_values, torax_mesh.cell_centers
        ),
        pellet_width=config_args.get_interpolated_var_single_axis(
            self.pellet_width,
        ),
        pellet_deposition_location=config_args.get_interpolated_var_single_axis(
            self.pellet_deposition_location,
        ),
        S_pellet_tot=config_args.get_interpolated_var_single_axis(
            self.S_pellet_tot,
        ),
    )


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
    return DynamicPelletRuntimeParams(
        pellet_width=float(self.pellet_width.get_value(t)),
        pellet_deposition_location=float(
            self.pellet_deposition_location.get_value(t)
        ),
        S_pellet_tot=float(self.S_pellet_tot.get_value(t)),
        mode=self.runtime_params_config.mode.value,
        is_explicit=self.runtime_params_config.is_explicit,
        formula=self.formula.build_dynamic_params(t),
        prescribed_values=self.prescribed_values.get_value(t),
    )


@chex.dataclass(frozen=True)
class DynamicPelletRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  pellet_width: float
  pellet_deposition_location: float
  S_pellet_tot: float


def _calc_pellet_source(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    unused_state: state.CoreProfiles | None = None,
    unused_source_models: source_models.SourceModels | None = None,
) -> jax.Array:
  """Calculates external source term for n from pellets."""
  assert isinstance(dynamic_source_runtime_params, DynamicPelletRuntimeParams)
  return formulas.gaussian_profile(
      c1=dynamic_source_runtime_params.pellet_deposition_location,
      c2=dynamic_source_runtime_params.pellet_width,
      total=(
          dynamic_source_runtime_params.S_pellet_tot
          / dynamic_runtime_params_slice.numerics.nref
      ),
      use_normalized_r=True,
      geo=geo,
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PelletSource(source.SingleProfileNeSource):
  """Pellet source for the ne equation."""

  formula: source.SourceProfileFunction = _calc_pellet_source


# pylint: enable=invalid-name

# The sources below don't have any source-specific implementations, so their
# bodies are empty. You can refer to their base class to see the implementation.
# We define new classes here to:
#  a) support any future source-specific implementation.
#  b) better readability and human-friendly error messages when debugging.


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RecombinationDensitySink(source.SingleProfileNeSource):
  """Recombination sink for the electron density equation."""


PelletSourceBuilder = source.make_source_builder(
    PelletSource, runtime_params_type=PelletRuntimeParams
)
GasPuffSourceBuilder = source.make_source_builder(
    GasPuffSource, runtime_params_type=GasPuffRuntimeParams
)
NBIParticleSourceBuilder = source.make_source_builder(
    NBIParticleSource, runtime_params_type=NBIParticleRuntimeParams
)
RecombinationDensitySinkBuilder = source.make_source_builder(
    RecombinationDensitySink
)
