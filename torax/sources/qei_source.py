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

import chex
import jax
from jax import numpy as jnp
from torax import config_slice
from torax import geometry
from torax import physics
from torax import state as state_lib
from torax.sources import source
from torax.sources import source_config


@chex.dataclass(frozen=True)
class QeiInfo:
  """Represents the source values coming from a QeiSource."""

  qei_coef: jnp.ndarray
  implicit_ii: jnp.ndarray
  explicit_i: jnp.ndarray
  implicit_ee: jnp.ndarray
  explicit_e: jnp.ndarray
  implicit_ie: jnp.ndarray
  implicit_ei: jnp.ndarray


@dataclasses.dataclass(frozen=True, kw_only=True)
class QeiSource(source.Source):
  """Collisional ion-electron heat source.

  This is a special-case source because it can provide both implicit and
  explicit terms in our solver. See sim.py for how this is used.
  """

  name: str = 'qei_source'

  supported_types: tuple[source_config.SourceType, ...] = (
      source_config.SourceType.MODEL_BASED,
      source_config.SourceType.ZERO,
  )

  # Don't include affected_mesh_states in the __init__ arguments.
  # Freeze this param. Qei is a special-case source and affects these equations
  # in a slightly different manner than the rest of the sources.
  affected_mesh_states: tuple[source.AffectedMeshStateAttribute, ...] = (
      dataclasses.field(
          init=False,
          default_factory=lambda: (
              source.AffectedMeshStateAttribute.TEMP_ION,
              source.AffectedMeshStateAttribute.TEMP_EL,
          ),
      )
  )

  def get_qei(
      self,
      source_type: int,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      geo: geometry.Geometry,
      sim_state: state_lib.ToraxSimState,
  ) -> QeiInfo:
    """Computes the value of the source."""
    source_type = self.check_source_type(source_type)
    return jax.lax.cond(
        source_type == source_config.SourceType.MODEL_BASED.value,
        lambda: _model_based_qei(
            dynamic_config_slice, static_config_slice, geo, sim_state
        ),
        lambda: _zero_qei(geo),
    )

  def get_value(
      self,
      source_type: int,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      sim_state: state_lib.ToraxSimState | None = None,
  ) -> QeiInfo:
    raise NotImplementedError('Call get_qei() instead.')

  def get_profile_for_affected_state(
      self,
      profile: chex.ArrayTree,
      affected_mesh_state: int,
      geo: geometry.Geometry,
  ) -> jnp.ndarray:
    raise NotImplementedError('This method is not valid for QeiSource.')


def _model_based_qei(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_lib.ToraxSimState,
) -> QeiInfo:
  """Computes Qei via the coll_exchange model."""
  zeros = jnp.zeros_like(geo.r_norm)
  qei_coef = physics.coll_exchange(
      state=sim_state.mesh_state,
      nref=dynamic_config_slice.nref,
      Ai=dynamic_config_slice.Ai,
      Qei_mult=dynamic_config_slice.Qei_mult,
  )
  implicit_ii = -qei_coef
  implicit_ee = -qei_coef

  if (
      # if only a single heat equation is being evolved
      (static_config_slice.ion_heat_eq and not static_config_slice.el_heat_eq)
      or (
          static_config_slice.el_heat_eq and not static_config_slice.ion_heat_eq
      )
  ):
    explicit_i = qei_coef * sim_state.mesh_state.temp_el.value
    explicit_e = qei_coef * sim_state.mesh_state.temp_ion.value
    implicit_ie = zeros
    implicit_ei = zeros
  else:
    explicit_i = zeros
    explicit_e = zeros
    implicit_ie = qei_coef
    implicit_ei = qei_coef
  return QeiInfo(
      qei_coef=qei_coef,
      implicit_ii=implicit_ii,
      explicit_i=explicit_i,
      implicit_ee=implicit_ee,
      explicit_e=explicit_e,
      implicit_ie=implicit_ie,
      implicit_ei=implicit_ei,
  )


def _zero_qei(
    geo: geometry.Geometry,
) -> QeiInfo:
  zeros = jnp.zeros_like(geo.r_norm)
  return QeiInfo(
      qei_coef=zeros,
      implicit_ii=zeros,
      explicit_i=zeros,
      implicit_ee=zeros,
      explicit_e=zeros,
      implicit_ie=zeros,
      implicit_ei=zeros,
  )
