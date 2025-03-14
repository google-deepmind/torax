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

"""Collisional ion-electron heat source."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal

import chex
import jax
from jax import numpy as jnp
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.physics import collisions
from torax.sources import base
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles


# pylint: disable=invalid-name
class QeiSourceConfig(base.SourceModelBase):
  """Configuration for the QeiSource.

  Attributes:
    Qei_mult: multiplier for ion-electron heat exchange term for sensitivity
      testing
  """

  source_name: Literal['qei_source'] = 'qei_source'
  Qei_mult: float = 1.0
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> None:
    return None

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        prescribed_values=self.prescribed_values.get_value(t),
        Qei_mult=self.Qei_mult,
    )

  def build_source(self) -> QeiSource:
    return QeiSource(model_func=self.model_func)


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  Qei_mult: float


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class QeiSource(source.Source):
  """Collisional ion-electron heat source.

  This is a special-case source because it can provide both implicit and
  explicit terms in our solver. See sim.py for how this is used.
  """

  SOURCE_NAME: ClassVar[str] = 'qei_source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'model_based_qei'

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (
        source.AffectedCoreProfile.TEMP_ION,
        source.AffectedCoreProfile.TEMP_EL,
    )

  def get_qei(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> source_profiles.QeiInfo:
    """Computes the value of the source."""
    dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
        self.source_name
    ]
    return jax.lax.cond(
        static_runtime_params_slice.sources[self.source_name].mode
        == runtime_params_lib.Mode.MODEL_BASED.value,
        lambda: _model_based_qei(
            static_runtime_params_slice,
            dynamic_runtime_params_slice,
            dynamic_source_runtime_params,
            geo,
            core_profiles,
        ),
        lambda: source_profiles.QeiInfo.zeros(geo),
    )

  def get_value(
      self,
      source_type: int,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tuple[chex.Array, ...]:
    raise NotImplementedError('Call get_qei() instead.')

  def get_source_profile_for_affected_core_profile(
      self,
      profile: tuple[chex.Array, ...],
      affected_mesh_state: int,
      geo: geometry.Geometry,
  ) -> jax.Array:
    raise NotImplementedError('This method is not valid for QeiSource.')


def _model_based_qei(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> source_profiles.QeiInfo:
  """Computes Qei via the coll_exchange model."""
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  zeros = jnp.zeros_like(geo.rho_norm)
  qei_coef = collisions.coll_exchange(
      core_profiles=core_profiles,
      nref=dynamic_runtime_params_slice.numerics.nref,
      Qei_mult=dynamic_source_runtime_params.Qei_mult,
  )
  implicit_ii = -qei_coef
  implicit_ee = -qei_coef

  if (
      # if only a single heat equation is being evolved
      (
          static_runtime_params_slice.ion_heat_eq
          and not static_runtime_params_slice.el_heat_eq
      )
      or (
          static_runtime_params_slice.el_heat_eq
          and not static_runtime_params_slice.ion_heat_eq
      )
  ):
    explicit_i = qei_coef * core_profiles.temp_el.value
    explicit_e = qei_coef * core_profiles.temp_ion.value
    implicit_ie = zeros
    implicit_ei = zeros
  else:
    explicit_i = zeros
    explicit_e = zeros
    implicit_ie = qei_coef
    implicit_ei = qei_coef
  return source_profiles.QeiInfo(
      qei_coef=qei_coef,
      implicit_ii=implicit_ii,
      explicit_i=explicit_i,
      implicit_ee=implicit_ee,
      explicit_e=explicit_e,
      implicit_ie=implicit_ie,
      implicit_ei=implicit_ei,
  )


# pylint: enable=invalid-name
