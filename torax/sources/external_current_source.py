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

"""External current source profile."""

from __future__ import annotations

import dataclasses

import chex
from jax import numpy as jnp
from jax.scipy import integrate
from torax import config_slice
from torax import geometry
from torax import state
from torax.sources import source
from torax.sources import source_config


_trapz = integrate.trapezoid


def calculate_Iext(  # pylint: disable=invalid-name
    dynamic_config_slice: config_slice.DynamicConfigSlice,
) -> float:
  """Calculates the total value of external current."""
  if dynamic_config_slice.use_absolute_jext:
    return dynamic_config_slice.Iext
  else:
    return dynamic_config_slice.Ip * dynamic_config_slice.fext


def calculate_jext_face(
    geo: geometry.Geometry,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
) -> jnp.ndarray:
  """Calculates the external current density profiles.

  Args:
    geo: Tokamak geometry.
    dynamic_config_slice: Parameter configuration at present timesteap.

  Returns:
    External current density profile along the face grid.
  """
  # pylint: disable=invalid-name
  Iext = calculate_Iext(dynamic_config_slice)
  # form of external current on face grid
  jextform_face = jnp.exp(
      -((geo.r_face_norm - dynamic_config_slice.rext) ** 2)
      / (2 * dynamic_config_slice.wext**2)
  )

  Cext = Iext * 1e6 / _trapz(jextform_face * geo.spr_face, geo.r_face)

  jext_face = Cext * jextform_face  # external current profile
  # pylint: enable=invalid-name
  return jext_face


def calculate_jext_hires(
    geo: geometry.CircularGeometry,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
) -> jnp.ndarray:
  """Calculates the external current density profile along the hires grid.

  Args:
    geo: Tokamak geometry.
    dynamic_config_slice: Parameter configuration at present timesteap.

  Returns:
    External current density profile along the hires cell grid.
  """
  # pylint: disable=invalid-name
  Iext = calculate_Iext(dynamic_config_slice)
  # calculate "External" current profile (e.g. ECCD)
  # form of external current on cell grid
  jextform_hires = jnp.exp(
      -((geo.r_hires_norm - dynamic_config_slice.rext) ** 2)
      / (2 * dynamic_config_slice.wext**2)
  )
  Cext_hires = Iext * 1e6 / _trapz(jextform_hires * geo.spr_hires, geo.r_hires)
  # External current profile on cell grid
  jext_hires = Cext_hires * jextform_hires
  # pylint: enable=invalid-name
  return jext_hires


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExternalCurrentSource(source.Source):
  """External current density source profile."""

  name: str = 'jext'

  supported_types: tuple[source_config.SourceType, ...] = (
      source_config.SourceType.FORMULA_BASED,
      source_config.SourceType.ZERO,
  )

  # Don't include affected_core_profiles in the __init__ arguments.
  # Freeze this param.
  affected_core_profiles: tuple[source.AffectedCoreProfile, ...] = (
      dataclasses.field(
          init=False,
          default_factory=lambda: (source.AffectedCoreProfile.PSI,),
      )
  )

  def get_value(
      self,
      source_type: int,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the external current density profile along face and cell grids."""
    source_type = self.check_source_type(source_type)
    profile = source.get_source_profiles(
        source_type=source_type,
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        core_profiles=core_profiles,
        # There is no model implementation.
        model_func=(
            lambda _0, _1, _2: source.ProfileType.FACE.get_zero_profile(geo)
        ),
        formula=lambda dcs, g, _: calculate_jext_face(
            g,
            dcs,
        ),
        output_shape=source.ProfileType.FACE.get_profile_shape(geo),
    )
    return profile, geometry.face_to_cell(profile)

  def jext_hires(
      self,
      source_type: int,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.CircularGeometry,
  ) -> jnp.ndarray:
    """Return the external current density profile along the hires cell grid."""
    source_type = self.check_source_type(source_type)
    return source.get_source_profiles(
        source_type=source_type,
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        core_profiles=None,
        # There is no model for this source.
        model_func=(lambda _0, _1, _2: jnp.zeros_like(geo.r_hires_norm)),
        formula=lambda dcs, g, _: calculate_jext_hires(
            g,
            dcs,
        ),
        output_shape=geo.r_hires_norm.shape,
    )

  def get_source_profile_for_affected_core_profile(
      self,
      profile: chex.ArrayTree,
      affected_core_profile: int,
      geo: geometry.Geometry,
  ) -> jnp.ndarray:
    return jnp.where(
        affected_core_profile in self.affected_core_profiles_ints,
        profile[0],  # the jext profile
        jnp.zeros_like(geo.r),
    )
