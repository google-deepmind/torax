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
"""Pellet source for the n_e equation."""
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
from torax._src.sources import source_state as source_state_lib
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class PelletSource(source.Source):
  """Pellet source for the n_e equation."""

  SOURCE_ID: ClassVar[str] = 'pellet'
  AFFECTED_CORE_PROFILES: ClassVar[tuple[source.AffectedCoreProfile, ...]] = (
      source.AffectedCoreProfile.NE,
  )

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
      conductivity: conductivity_base.Conductivity | None,
      source_state: source_state_lib.SourceState,
      psidot: array_typing.FloatVectorCell | None = None,
  ) -> tuple[
      source_profiles.SourceOutput,
      source_state_lib.SourceState,
  ]:
    """Calculates external source term for n from pellets.

    Does not modify source_state; returns it unchanged.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      calculated_source_profiles: Source profiles calculated so far.
      conductivity: Conductivity model output.
      source_state: Source state carried over from the previous timestep. Unused
        by this model and returned unchanged.
      psidot: Unused.

    Returns:
      A tuple of (SourceOutput(profiles=(electron_density_profile,)),
      source_state).
    """
    del core_profiles, calculated_source_profiles, conductivity, psidot
    source_params = runtime_params.sources[self.SOURCE_ID]
    assert isinstance(source_params, RuntimeParams)
    profile = formulas.gaussian_profile(
        center=source_params.pellet_deposition_location,
        width=source_params.pellet_width,
        total=source_params.S_total,
        geo=geo,
    )
    return source_profiles.SourceOutput(profiles=(profile,)), source_state


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  pellet_width: array_typing.FloatScalar
  pellet_deposition_location: array_typing.FloatScalar
  S_total: array_typing.FloatScalar


class PelletSourceConfig(base.SourceConfigBase):
  """Pellet source for the n_e equation.

  Attributes:
    pellet_width: Gaussian width of pellet deposition [normalized radial coord]
    pellet_deposition_location: Gaussian center of pellet deposition [normalized
      radial coord]
    S_total: total pellet particles/s
    mode: Defines how the source values are computed (from a model, from a file,
      etc.)
  """

  model_name: Annotated[Literal['gaussian'], torax_pydantic.JAX_STATIC] = (
      'gaussian'
  )
  pellet_width: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )
  pellet_deposition_location: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.85)
  )
  S_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      2e22
  )

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    base_params = super().build_runtime_params(t)
    return RuntimeParams(
        **vars(base_params),
        pellet_width=self.pellet_width.get_value(t),
        pellet_deposition_location=self.pellet_deposition_location.get_value(t),
        S_total=self.S_total.get_value(t),
    )

  def build_source(self) -> PelletSource:
    return PelletSource()
