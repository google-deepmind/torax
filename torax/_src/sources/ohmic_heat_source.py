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
"""Ohmic heat source."""
import dataclasses
from typing import Annotated, ClassVar, Literal
import chex
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import psi_calculations
from torax._src.sources import base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import torax_pydantic

# Default value for the model function to be used for the ohmic heat
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'standard'


def ohmic_model_func(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    unused_source_name: str,
    core_profiles: state.CoreProfiles,
    calculated_source_profiles: source_profiles_lib.SourceProfiles | None,
    conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, ...]:
  """Returns the Ohmic source for electron heat equation."""
  if calculated_source_profiles is None:
    raise ValueError(
        'calculated_source_profiles is a required argument for'
        ' ohmic_model_func. This can occur if this source function is used in'
        ' an explicit source.'
    )

  if conductivity is None:
    raise ValueError(
        'conductivity is a required argument for ohmic_model_func. This can'
        ' occur if this source function is used in an explicit source.'
    )

  j_total, _, _ = psi_calculations.calc_j_total(
      geo,
      core_profiles.psi,
  )
  psi_sources = calculated_source_profiles.total_psi_sources(geo)
  psidot = psi_calculations.calculate_psidot_from_psi_sources(
      psi_sources=psi_sources,
      sigma=conductivity.sigma,
      resistivity_multiplier=runtime_params.numerics.resistivity_multiplier,
      psi=core_profiles.psi,
      geo=geo,
  )

  # Ohmic power is positive regardless of the sign of voltage and current.
  pohm = jnp.abs(j_total * psidot / (2 * jnp.pi * geo.R_major))
  return (pohm,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class OhmicHeatSource(source_lib.Source):
  """Ohmic heat source for electron heat equation.

  Pohm = jtor * psidot /(2*pi*R_major), related to electric power formula P =
  IV.
  """

  SOURCE_NAME: ClassVar[str] = 'ohmic'
  model_func: source_lib.SourceProfileFunction = ohmic_model_func

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(
      self,
  ) -> tuple[source_lib.AffectedCoreProfile, ...]:
    return (source_lib.AffectedCoreProfile.TEMP_EL,)


class OhmicHeatSourceConfig(base.SourceModelBase):
  """Configuration for the OhmicHeatSource."""

  model_name: Annotated[Literal['standard'], torax_pydantic.JAX_STATIC] = (
      'standard'
  )
  mode: Annotated[runtime_params_lib.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params_lib.Mode.MODEL_BASED
  )

  @property
  def model_func(self) -> source_lib.SourceProfileFunction:
    return ohmic_model_func

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> runtime_params_lib.RuntimeParams:
    return runtime_params_lib.RuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
    )

  def build_source(self) -> OhmicHeatSource:
    return OhmicHeatSource(model_func=self.model_func)
