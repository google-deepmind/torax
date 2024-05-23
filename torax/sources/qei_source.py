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
from torax import geometry
from torax import physics
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles


# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  # multiplier for ion-electron heat exchange term for sensitivity testing
  Qei_mult: float = 1.0

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  Qei_mult: float


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class QeiSource(source.Source):
  """Collisional ion-electron heat source.

  This is a special-case source because it can provide both implicit and
  explicit terms in our solver. See sim.py for how this is used.
  """

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )

  supported_modes: tuple[runtime_params_lib.Mode, ...] = (
      runtime_params_lib.Mode.MODEL_BASED,
      runtime_params_lib.Mode.ZERO,
  )

  # Don't include affected_core_profiles in the __init__ arguments.
  # Freeze this param. Qei is a special-case source and affects these equations
  # in a slightly different manner than the rest of the sources.
  affected_core_profiles: tuple[source.AffectedCoreProfile, ...] = (
      dataclasses.field(
          init=False,
          default_factory=lambda: (
              source.AffectedCoreProfile.TEMP_ION,
              source.AffectedCoreProfile.TEMP_EL,
          ),
      )
  )

  def get_qei(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> source_profiles.QeiInfo:
    """Computes the value of the source."""
    self.check_mode(dynamic_source_runtime_params.mode)
    return jax.lax.cond(
        dynamic_source_runtime_params.mode
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
      core_profiles: state.CoreProfiles | None = None,
  ) -> source_profiles.QeiInfo:
    raise NotImplementedError('Call get_qei() instead.')

  def get_source_profile_for_affected_core_profile(
      self,
      profile: chex.ArrayTree,
      affected_mesh_state: int,
      geo: geometry.Geometry,
  ) -> jnp.ndarray:
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
  zeros = jnp.zeros_like(geo.r_norm)
  qei_coef = physics.coll_exchange(
      core_profiles=core_profiles,
      nref=dynamic_runtime_params_slice.numerics.nref,
      Ai=dynamic_runtime_params_slice.plasma_composition.Ai,
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
