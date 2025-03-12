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
from typing import ClassVar, Literal

import chex
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import base
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class GenericIonElHeatSourceConfig(base.SourceModelBase):
  """Configuration for the GenericIonElHeatSource.

  Attributes:
    w: Gaussian width in normalized radial coordinate
    rsource: Source Gaussian central location (in normalized r)
    Ptot: Total heating: high default based on total ITER power including alphas
    el_heat_fraction: Electron heating fraction
  """
  source_name: Literal['generic_ion_el_heat_source'] = (
      'generic_ion_el_heat_source'
  )
  w: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(0.25)
  rsource: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.0
  )
  Ptot: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      120e6
  )
  el_heat_fraction: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.66666)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return default_formula

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        prescribed_values=self.prescribed_values.get_value(t),
        w=self.w.get_value(t),
        rsource=self.rsource.get_value(t),
        Ptot=self.Ptot.get_value(t),
        el_heat_fraction=self.el_heat_fraction.get_value(t),
    )

  def build_source(self) -> GenericIonElectronHeatSource:
    return GenericIonElectronHeatSource(model_func=self.model_func)


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
  profile = formulas.gaussian_profile(geo, center=rsource, width=w, total=Ptot)
  source_ion = profile * (1 - el_heat_fraction)
  source_el = profile * el_heat_fraction

  return source_ion, source_el


def default_formula(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Returns the default formula-based ion/electron heat source profile."""
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
