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
from typing import Annotated, ClassVar, Literal

import chex
import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base
from torax._src.sources import formulas
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  gaussian_width: array_typing.FloatScalar
  gaussian_location: array_typing.FloatScalar
  P_total: array_typing.FloatScalar
  electron_heat_fraction: array_typing.FloatScalar
  absorption_fraction: array_typing.FloatScalar


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GenericIonElectronHeatSource(source.Source):
  """Generic heat source for both ion and electron heat."""

  SOURCE_ID: ClassVar[str] = 'generic_heat'
  AFFECTED_CORE_PROFILES: ClassVar[tuple[source.AffectedCoreProfile, ...]] = (
      source.AffectedCoreProfile.TEMP_ION,
      source.AffectedCoreProfile.TEMP_EL,
  )

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None = None,
      conductivity: conductivity_base.Conductivity | None = None,
  ) -> tuple[source.SourceProfileElement, ...]:
    """Returns the default formula-based ion/electron heat source profile."""
    del core_profiles, calculated_source_profiles, conductivity
    source_params = runtime_params.sources[self.SOURCE_ID]
    assert isinstance(source_params, RuntimeParams)

    # Calculate heat profile.
    absorbed_power = (
        source_params.P_total * source_params.absorption_fraction
    )
    profile = formulas.gaussian_profile(
        geo,
        center=source_params.gaussian_location,
        width=source_params.gaussian_width,
        total=absorbed_power,
    )
    source_ion = profile * (1 - source_params.electron_heat_fraction)
    source_el = profile * source_params.electron_heat_fraction
    return (source_ion, source_el)


class GenericIonElHeatSourceConfig(base.SourceConfigBase):
  """Configuration for the GenericIonElHeatSource.

  Attributes:
    gaussian_width: Gaussian width in normalized radial coordinate
    gaussian_location: Source Gaussian central location (in normalized r)
    P_total: Total heating: high default based on total ITER power including
      alphas
    electron_heat_fraction: Electron heating fraction
  """

  model_name: Annotated[Literal['gaussian'], torax_pydantic.JAX_STATIC] = (
      'gaussian'
  )
  gaussian_width: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.25)
  )
  gaussian_location: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  P_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      120e6
  )
  electron_heat_fraction: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.66666)
  )
  absorption_fraction: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    return RuntimeParams(
        **dataclasses.asdict(super().build_runtime_params(t)),
        gaussian_width=self.gaussian_width.get_value(t),
        gaussian_location=self.gaussian_location.get_value(t),
        P_total=self.P_total.get_value(t),
        electron_heat_fraction=self.electron_heat_fraction.get_value(t),
        absorption_fraction=self.absorption_fraction.get_value(t),
    )

  def build_source(self) -> GenericIonElectronHeatSource:
    return GenericIonElectronHeatSource()
