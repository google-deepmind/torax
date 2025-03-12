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
from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal

import chex
import jax
from jax import numpy as jnp
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import base
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles


class BremsstrahlungHeatSinkConfig(base.SourceModelBase):
  """Bremsstrahlung heat sink for electron heat equation.

  Attributes:
    use_relativistic_correction: Whether to use relativistic correction.
  """

  source_name: Literal['bremsstrahlung_heat_sink'] = 'bremsstrahlung_heat_sink'
  use_relativistic_correction: bool = False
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return bremsstrahlung_model_func

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> 'DynamicRuntimeParams':
    return DynamicRuntimeParams(
        prescribed_values=self.prescribed_values.get_value(t),
        use_relativistic_correction=self.use_relativistic_correction,
    )

  def build_source(self) -> BremsstrahlungHeatSink:
    return BremsstrahlungHeatSink(model_func=self.model_func)


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  use_relativistic_correction: bool


def calc_bremsstrahlung(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
    Zeff_face: chex.Array,
    nref: float,
    use_relativistic_correction: bool = False,
) -> tuple[chex.Array, chex.Array]:
  """Calculate the Bremsstrahlung radiation power profile.

  Uses the model from Wesson, John, and David J. Campbell. Tokamaks. Vol. 149.
  An optional correction for relativistic effects from Stott PPCF 2005 can be
  enabled with the flag "use_relativistic_correction".

  Args:
      core_profiles (state.CoreProfiles): core plasma profiles.
      geo (geometry.Geometry): geometry object.
      Zeff_face (float): effective charge number on face grid.
      nref (float): reference density.
      use_relativistic_correction (bool, optional): Set to true to include the
        relativistic correction from Stott. Defaults to False.

  Returns:
      jax.Array: total bremsstrahlung radiation power [MW]
      jax.Array: bremsstrahlung radiation power profile [W/m^3]
  """
  ne20 = (nref / 1e20) * core_profiles.ne.face_value()

  Te_kev = core_profiles.temp_el.face_value()

  P_brem_profile_face: jax.Array = (
      5.35e-3 * Zeff_face * ne20**2 * jnp.sqrt(Te_kev)
  )  # MW/m^3

  def calc_relativistic_correction() -> jax.Array:
    # Apply the Stott relativistic correction.
    Tm = 511.0  # m_e * c**2 in keV
    correction = (1.0 + 2.0 * Te_kev / Tm) * (
        1.0 + (2.0 / Zeff_face) * (1.0 - 1.0 / (1.0 + Te_kev / Tm))
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
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Model function for the Bremsstrahlung heat sink."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  _, P_brem_profile = calc_bremsstrahlung(
      core_profiles,
      geo,
      dynamic_runtime_params_slice.plasma_composition.Zeff_face,
      dynamic_runtime_params_slice.numerics.nref,
      use_relativistic_correction=dynamic_source_runtime_params.use_relativistic_correction,
  )
  # As a sink, the power is negative.
  return (-1.0 * P_brem_profile,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class BremsstrahlungHeatSink(source.Source):
  """Brehmsstrahlung heat sink for electron heat equation."""

  SOURCE_NAME: ClassVar[str] = 'bremsstrahlung_heat_sink'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'bremsstrahlung_model_func'
  model_func: source.SourceProfileFunction = bremsstrahlung_model_func

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL,)
