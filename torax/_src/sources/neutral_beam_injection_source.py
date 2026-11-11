# Copyright 2026 DeepMind Technologies Limited
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
"""Neutral beam injection (NBI) source."""

import dataclasses
from typing import Annotated, ClassVar, Literal

from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import solver_profiles as solver_profiles_lib
from torax._src.geometry import geometry
from torax._src.sources import base
from torax._src.sources import core_sources
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_state as source_state_lib
from torax._src.torax_pydantic import torax_pydantic


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class NeutralBeamInjectionSource(source.Source):
  """Neutral beam injection source."""

  SOURCE_ID: ClassVar[str] = 'nbi'
  POPULATED_SOURCE_OUTPUTS: ClassVar[
      tuple[core_sources.PopulatedSourceOutput, ...]
  ] = (
      core_sources.PopulatedSourceOutput.POWER_DENSITY_I,
      core_sources.PopulatedSourceOutput.POWER_DENSITY_E,
      core_sources.PopulatedSourceOutput.J_PARALLEL,
      core_sources.PopulatedSourceOutput.PARTICLE_SOURCE_E,
  )
  PRODUCES_FAST_IONS: ClassVar[bool] = True

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      solver_profiles: solver_profiles_lib.SolverProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[
      core_sources.SourceOutput,
      source_state_lib.SourceState,
  ]:
    del runtime_params, geo, solver_profiles, source_state
    raise NotImplementedError(
        'Model-based NBI source is not yet implemented; use'
        ' mode=Mode.PRESCRIBED or mode=Mode.ZERO.'
    )


class NeutralBeamInjectionSourceConfig(base.SourceConfigBase):
  """Configuration for NeutralBeamInjectionSource.

  Currently only prescribed or zero modes are supported. In `PRESCRIBED` mode,
  `prescribed_values` accepts `TimeVaryingArray` entries on the cell grid for
  the populated source output channels (`power_density_i` [W/m^3],
  `power_density_e` [W/m^3], `j_parallel` [A/m^2], `particle_source_e`
  [m^-3 s^-1]) as well as optional fast-ion species distributions via
  `prescribed_values.fast_ions`.

  Attributes:
    model_name: Discriminator field for Pydantic (`'prescribed'`).
    mode: Defines how the source values are computed. Defaults to
      `Mode.PRESCRIBED`.
  """

  model_name: Annotated[Literal['prescribed'], torax_pydantic.JAX_STATIC] = (
      'prescribed'
  )
  mode: Annotated[
      sources_runtime_params_lib.Mode, torax_pydantic.JAX_STATIC
  ] = sources_runtime_params_lib.Mode.PRESCRIBED

  def build_source(self) -> NeutralBeamInjectionSource:
    return NeutralBeamInjectionSource()
