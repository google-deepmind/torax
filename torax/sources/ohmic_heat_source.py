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
from typing import ClassVar, Literal

import chex
import jax.numpy as jnp
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.physics import psi_calculations
from torax.sources import base
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_profiles as source_profiles_lib


# Default value for the model function to be used for the ohmic heat
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'ohmic_model_func'


def ohmic_model_func(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    unused_source_name: str,
    core_profiles: state.CoreProfiles,
    calculated_source_profiles: source_profiles_lib.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Returns the Ohmic source for electron heat equation."""
  if calculated_source_profiles is None:
    raise ValueError(
        'calculated_source_profiles is a required argument for'
        ' ohmic_model_func. This can occur if this source function is used in'
        ' an explicit source.'
    )

  jtot, _, _ = psi_calculations.calc_jtot(
      geo,
      core_profiles.psi,
  )
  psi_sources = calculated_source_profiles.total_psi_sources(geo)
  sigma = calculated_source_profiles.j_bootstrap.sigma
  sigma_face = calculated_source_profiles.j_bootstrap.sigma_face
  psidot = psi_calculations.calculate_psidot_from_psi_sources(
      psi_sources=psi_sources,
      sigma=sigma,
      sigma_face=sigma_face,
      resistivity_multiplier=dynamic_runtime_params_slice.numerics.resistivity_mult,
      psi=core_profiles.psi,
      geo=geo,
  )
  pohm = jtot * psidot / (2 * jnp.pi * geo.Rmaj)
  return (pohm,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class OhmicHeatSource(source_lib.Source):
  """Ohmic heat source for electron heat equation.

  Pohm = jtor * psidot /(2*pi*Rmaj), related to electric power formula P = IV.
  """

  SOURCE_NAME: ClassVar[str] = 'ohmic_heat_source'
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
  model_function_name: Literal['ohmic_model_func'] = 'ohmic_model_func'
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source_lib.SourceProfileFunction:
    return ohmic_model_func

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> runtime_params_lib.DynamicRuntimeParams:
    return runtime_params_lib.DynamicRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
    )

  def build_source(self) -> OhmicHeatSource:
    return OhmicHeatSource(model_func=self.model_func)
