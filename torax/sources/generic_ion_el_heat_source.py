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

"""Generic heat source for both ion and electron heat."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Optional

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
# Many variables throughout this function are capitalized based on physics
# notational conventions rather than on Google Python style
# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for the generic heat source."""

  # external heat source parameters
  # Gaussian width in normalized radial coordinate
  w: runtime_params_lib.TimeInterpolatedInput = 0.25
  # Source Gaussian central location (in normalized r)
  rsource: runtime_params_lib.TimeInterpolatedInput = 0.0
  # total heating
  Ptot: runtime_params_lib.TimeInterpolatedInput = 120e6
  # electron heating fraction
  el_heat_fraction: runtime_params_lib.TimeInterpolatedInput = 0.66666
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> 'RuntimeParamsProvider':
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: RuntimeParams
  w: interpolated_param.InterpolatedVarSingleAxis
  rsource: interpolated_param.InterpolatedVarSingleAxis
  Ptot: interpolated_param.InterpolatedVarSingleAxis
  el_heat_fraction: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> 'DynamicRuntimeParams':
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  w: array_typing.ScalarFloat
  rsource: array_typing.ScalarFloat
  Ptot: array_typing.ScalarFloat
  el_heat_fraction: array_typing.ScalarFloat


def calc_generic_heat_source(
    geo: geometry.Geometry,
    rsource: float,
    w: float,
    Ptot: float,
    el_heat_fraction: float,
) -> tuple[chex.Array, chex.Array]:
  """Computes ion/electron heat source terms.

  Flexible prescribed heat source term.

  Args:
    geo: Geometry describing the torus.
    rsource: Source Gaussian central location
    w: Gaussian width
    Ptot: total heating
    el_heat_fraction: fraction of heating deposited on electrons

  Returns:
    source_ion: source term for ions.
    source_el: source term for electrons.
  """
  # Calculate heat profile.
  profile = formulas.gaussian_profile(geo, c1=rsource, c2=w, total=Ptot)
  source_ion = profile * (1 - el_heat_fraction)
  source_el = profile * el_heat_fraction

  return source_ion, source_el


# pytype: disable=name-error
def default_formula(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_source_models: Optional['source_models.SourceModels'],
) -> tuple[chex.Array, ...]:
  """Returns the default formula-based ion/electron heat source profile."""
  # pytype: enable=name-error
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  ion, el = calc_generic_heat_source(
      geo,
      dynamic_source_runtime_params.rsource,
      dynamic_source_runtime_params.w,
      dynamic_source_runtime_params.Ptot,
      dynamic_source_runtime_params.el_heat_fraction,
  )
  return (ion, el)


# pylint: enable=invalid-name


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericIonElectronHeatSource(source.Source):
  """Generic heat source for both ion and electron heat."""

  SOURCE_NAME: ClassVar[str] = 'generic_ion_el_heat_source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'default_formula'
  model_func: source.SourceProfileFunction = default_formula

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (
        source.AffectedCoreProfile.TEMP_ION,
        source.AffectedCoreProfile.TEMP_EL,
    )
