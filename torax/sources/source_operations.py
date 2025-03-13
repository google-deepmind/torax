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
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles as source_profiles_lib


def sum_sources_psi(
    geo: geometry.Geometry,
    source_profiles: source_profiles_lib.SourceProfiles,
) -> jax.Array:
  """Computes psi source values for sim.calc_coeffs."""
  total = source_profiles.j_bootstrap.j_bootstrap
  total += sum(source_profiles.psi.values())
  mu0 = constants.CONSTANTS.mu0
  prefactor = 8 * geo.vpr * jnp.pi**2 * geo.B0 * mu0 * geo.Phib / geo.F**2
  return -total * prefactor


def sum_sources_ne(
    geo: geometry.Geometry,
    source_profiles: source_profiles_lib.SourceProfiles,
) -> jax.Array:
  """Computes ne source values for sim.calc_coeffs."""
  total = sum(source_profiles.ne.values())
  return total * geo.vpr


def sum_sources_temp_ion(
    geo: geometry.Geometry,
    source_profiles: source_profiles_lib.SourceProfiles,
) -> jax.Array:
  """Computes temp_ion source values for sim.calc_coeffs."""
  total = sum(source_profiles.temp_ion.values())
  return total * geo.vpr


def sum_sources_temp_el(
    geo: geometry.Geometry,
    source_profiles: source_profiles_lib.SourceProfiles,
) -> jax.Array:
  """Computes temp_el source values for sim.calc_coeffs."""
  total = sum(source_profiles.temp_el.values())
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
  j_bootstrap_profiles = source_models.j_bootstrap.get_bootstrap(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )
  profiles = source_profiles_lib.SourceProfiles(
      j_bootstrap=j_bootstrap_profiles,
      psi={}, temp_el={}, temp_ion={}, ne={},
      qei=source_profiles_lib.QeiInfo.zeros(geo))
  source_profile_builders.build_standard_source_profiles(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      calculate_anyway=True,
      psi_only=True,
      calculated_source_profiles=profiles,
  )

  return (
      sum_sources_psi(geo, source_profiles=profiles),
      j_bootstrap_profiles.sigma,
      j_bootstrap_profiles.sigma_face,
  )
