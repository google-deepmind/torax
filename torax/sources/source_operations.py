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

"""Functions for building source profiles in TORAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from torax import constants
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles


def sum_sources_psi(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
) -> jax.Array:
  """Computes psi source values for sim.calc_coeffs."""
  total = source_profile.j_bootstrap.j_bootstrap
  for source_name, source in source_models.psi_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=source_lib.AffectedCoreProfile.PSI.value,
        geo=geo,
    )
  mu0 = constants.CONSTANTS.mu0
  prefactor = 8 * geo.vpr * jnp.pi**2 * geo.B0 * mu0 * geo.Phib / geo.F**2
  scale_source = lambda src: -src * prefactor
  return scale_source(total)


def sum_sources_ne(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
) -> jax.Array:
  """Computes ne source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.rho)
  for source_name, source in source_models.ne_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=source_lib.AffectedCoreProfile.NE.value,
        geo=geo,
    )
  return total * geo.vpr


def sum_sources_temp_ion(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
) -> jax.Array:
  """Computes temp_ion source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.rho)
  for source_name, source in source_models.temp_ion_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=(source_lib.AffectedCoreProfile.TEMP_ION.value),
        geo=geo,
    )
  return total * geo.vpr


def sum_sources_temp_el(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: source_models_lib.SourceModels,
) -> jax.Array:
  """Computes temp_el source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.rho)
  for source_name, source in source_models.temp_el_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=(source_lib.AffectedCoreProfile.TEMP_EL.value),
        geo=geo,
    )
  return total * geo.vpr


def calc_and_sum_sources_psi(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes sum of psi sources for psi_dot calculation."""

  # TODO(b/335597108): Revisit how to calculate this once we enable more
  # expensive source functions that might not jittable (like file-based or
  # RPC-based sources).
  psi_profiles = source_profile_builders.build_standard_source_profiles(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
      calculate_anyway=True,
      affected_core_profiles=(source_lib.AffectedCoreProfile.PSI,),
  )
  total = 0
  for source_name, source in source_models.psi_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=psi_profiles[source_name],
        affected_core_profile=source_lib.AffectedCoreProfile.PSI.value,
        geo=geo,
    )
  static_bootstrap_runtime_params = static_runtime_params_slice.sources[
      source_models.j_bootstrap_name
  ]
  j_bootstrap_profiles = source_profile_builders.build_bootstrap_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      static_source_runtime_params=static_bootstrap_runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      j_bootstrap_source=source_models.j_bootstrap,
      calculate_anyway=True,
  )
  total += j_bootstrap_profiles.j_bootstrap

  mu0 = constants.CONSTANTS.mu0
  prefactor = 8 * geo.vpr * jnp.pi**2 * geo.B0 * mu0 * geo.Phib / geo.F**2
  scale_source = lambda src: -src * prefactor

  return (
      scale_source(total),
      j_bootstrap_profiles.sigma,
      j_bootstrap_profiles.sigma_face,
  )
