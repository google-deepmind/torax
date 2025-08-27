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
"""Generic particle source for the n_e equation."""
import dataclasses
from typing import Annotated, ClassVar, Literal

import chex
import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base
from torax._src.sources import formulas
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

# Default value for the model function to be used for the generic particle
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'gaussian'


# pylint: disable=invalid-name
def calc_generic_particle_source(
    dynamic_runtime_params_slice: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
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
          total=dynamic_source_runtime_params.S_total,
          geo=geo,
      ),
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericParticleSource(source.Source):
  """Neutral-beam injection source for the n_e equation."""

  SOURCE_NAME: ClassVar[str] = 'generic_particle'
  model_func: source.SourceProfileFunction = calc_generic_particle_source

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.NE,)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicParticleRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  particle_width: array_typing.FloatScalar
  deposition_location: array_typing.FloatScalar
  S_total: array_typing.FloatScalar


class GenericParticleSourceConfig(base.SourceModelBase):
  """Generic particle source for the n_e equation.

  Attributes:
    particle_width: particle source Gaussian width in normalized radial coord
    deposition_location: particle source Gaussian center in normalized radial
      coord
    S_total: total particle source particles/s
    mode: Defines how the source values are computed (from a model, from a file,
      etc.)
  """

  model_name: Annotated[Literal['gaussian'], torax_pydantic.JAX_STATIC] = (
      'gaussian'
  )
  particle_width: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.25)
  )
  deposition_location: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  S_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      1e22
  )
  mode: Annotated[runtime_params_lib.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params_lib.Mode.MODEL_BASED
  )

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calc_generic_particle_source

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicParticleRuntimeParams:
    return DynamicParticleRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
        particle_width=self.particle_width.get_value(t),
        deposition_location=self.deposition_location.get_value(t),
        S_total=self.S_total.get_value(t),
    )

  def build_source(self) -> GenericParticleSource:
    return GenericParticleSource(model_func=self.model_func)
