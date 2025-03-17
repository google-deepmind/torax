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
import jax
import jax.numpy as jnp
from torax import constants
from torax.geometry import geometry
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
