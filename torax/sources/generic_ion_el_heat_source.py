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
import dataclasses
from typing import ClassVar, Literal

import chex
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.neoclassical.conductivity import base as conductivity_base
from torax.sources import base
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles
from torax.torax_pydantic import torax_pydantic


# Default value for the model function to be used for the electron cyclotron
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'gaussian'


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  gaussian_width: array_typing.ScalarFloat
  gaussian_location: array_typing.ScalarFloat
  P_total: array_typing.ScalarFloat
  electron_heat_fraction: array_typing.ScalarFloat
  absorption_fraction: array_typing.ScalarFloat


def calc_generic_heat_source(
    geo: geometry.Geometry,
    gaussian_location: float,
    gaussian_width: float,
    P_total: float,
    electron_heat_fraction: float,
    absorption_fraction: float,
) -> tuple[chex.Array, chex.Array]:
  """Computes ion/electron heat source terms.

  Flexible prescribed heat source term.

  Args:
    geo: Geometry describing the torus.
    gaussian_location: Source Gaussian central location
    gaussian_width: Gaussian width
    P_total: total heating
    electron_heat_fraction: fraction of heating deposited on electrons
    absorption_fraction: fraction of absorbed power

  Returns:
    source_ion: source term for ions.
    source_el: source term for electrons.
  """
  # Calculate heat profile.
  absorbed_power = P_total * absorption_fraction
  profile = formulas.gaussian_profile(
      geo, center=gaussian_location, width=gaussian_width, total=absorbed_power
  )
  source_ion = profile * (1 - electron_heat_fraction)
  source_el = profile * electron_heat_fraction

  return source_ion, source_el


def default_formula(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[chex.Array, ...]:
  """Returns the default formula-based ion/electron heat source profile."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  ion, el = calc_generic_heat_source(
      geo,
      dynamic_source_runtime_params.gaussian_location,
      dynamic_source_runtime_params.gaussian_width,
      dynamic_source_runtime_params.P_total,
      dynamic_source_runtime_params.electron_heat_fraction,
      dynamic_source_runtime_params.absorption_fraction,
  )
  return (ion, el)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericIonElectronHeatSource(source.Source):
  """Generic heat source for both ion and electron heat."""

  SOURCE_NAME: ClassVar[str] = 'generic_heat'
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


class GenericIonElHeatSourceConfig(base.SourceModelBase):
  """Configuration for the GenericIonElHeatSource.

  Attributes:
    gaussian_width: Gaussian width in normalized radial coordinate
    gaussian_location: Source Gaussian central location (in normalized r)
    P_total: Total heating: high default based on total ITER power including
      alphas
    electron_heat_fraction: Electron heating fraction
  """

  model_name: Literal['gaussian'] = 'gaussian'
  gaussian_width: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.25)
  )
  gaussian_location: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  P_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      120e6
  )
  electron_heat_fraction: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.66666)
  )
  absorption_fraction: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
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
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        gaussian_width=self.gaussian_width.get_value(t),
        gaussian_location=self.gaussian_location.get_value(t),
        P_total=self.P_total.get_value(t),
        electron_heat_fraction=self.electron_heat_fraction.get_value(t),
        absorption_fraction=self.absorption_fraction.get_value(t),
    )

  def build_source(self) -> GenericIonElectronHeatSource:
    return GenericIonElectronHeatSource(model_func=self.model_func)
