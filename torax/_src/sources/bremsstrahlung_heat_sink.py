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

# pylint: disable=invalid-name

"""Bremsstrahlung heat sink for electron heat equation.."""
import dataclasses
from typing import Annotated, ClassVar, Final, Literal

import chex
import jax
from jax import numpy as jnp
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

# Default value for the model function to be used for the Bremsstrahlung heat
# sink. This is also used as an identifier for the model function in the default
# source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: Final[str] = 'wesson'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  use_relativistic_correction: bool


def calc_bremsstrahlung(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
    use_relativistic_correction: bool = False,
) -> tuple[chex.Array, chex.Array]:
  """Calculate the Bremsstrahlung radiation power profile.

  Uses the model from Wesson, John, and David J. Campbell. Tokamaks. Vol. 149.
  An optional correction for relativistic effects from Stott PPCF 2005 can be
  enabled with the flag "use_relativistic_correction".

  Args:
      core_profiles (state.CoreProfiles): core plasma profiles.
      geo (geometry.Geometry): geometry object.
      use_relativistic_correction (bool, optional): Set to true to include the
        relativistic correction from Stott. Defaults to False.

  Returns:
      jax.Array: total bremsstrahlung radiation power [MW]
      jax.Array: bremsstrahlung radiation power profile [W/m^3]
  """
  n_e20 = core_profiles.n_e.face_value() / 1e20

  T_e_kev = core_profiles.T_e.face_value()

  P_brem_profile_face: jax.Array = (
      5.35e-3 * core_profiles.Z_eff_face * n_e20**2 * jnp.sqrt(T_e_kev)
  )  # MW/m^3

  def calc_relativistic_correction() -> jax.Array:
    # Apply the Stott relativistic correction.
    Tm = 511.0  # m_e * c**2 in keV
    correction = (1.0 + 2.0 * T_e_kev / Tm) * (
        1.0
        + (2.0 / core_profiles.Z_eff_face) * (1.0 - 1.0 / (1.0 + T_e_kev / Tm))
    )
    return correction

  # In MW/m^3
  P_brem_profile_face = jnp.where(
      use_relativistic_correction,
      P_brem_profile_face * calc_relativistic_correction(),
      P_brem_profile_face,
  )

  # In W/m^3
  P_brem_profile_cell = geometry.face_to_cell(P_brem_profile_face) * 1e6

  # In MW
  P_brem_total = math_utils.volume_integration(P_brem_profile_cell, geo)
  return P_brem_total, P_brem_profile_cell


def bremsstrahlung_model_func(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[chex.Array, ...]:
  """Model function for the Bremsstrahlung heat sink."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  _, P_brem_profile = calc_bremsstrahlung(
      core_profiles,
      geo,
      use_relativistic_correction=dynamic_source_runtime_params.use_relativistic_correction,
  )
  # As a sink, the power is negative.
  return (-1.0 * P_brem_profile,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class BremsstrahlungHeatSink(source.Source):
  """Brehmsstrahlung heat sink for electron heat equation."""

  SOURCE_NAME: ClassVar[str] = 'bremsstrahlung'
  model_func: source.SourceProfileFunction = bremsstrahlung_model_func

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL,)


class BremsstrahlungHeatSinkConfig(base.SourceModelBase):
  """Bremsstrahlung heat sink for electron heat equation.

  Attributes:
    use_relativistic_correction: Whether to use relativistic correction.
  """

  model_name: Annotated[Literal['wesson'], torax_pydantic.JAX_STATIC] = 'wesson'
  use_relativistic_correction: bool = False
  mode: Annotated[runtime_params_lib.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params_lib.Mode.MODEL_BASED
  )

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return bremsstrahlung_model_func

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> 'DynamicRuntimeParams':
    return DynamicRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
        use_relativistic_correction=self.use_relativistic_correction,
    )

  def build_source(self) -> BremsstrahlungHeatSink:
    return BremsstrahlungHeatSink(model_func=self.model_func)
