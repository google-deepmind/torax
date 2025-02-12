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


"""Impurity radiation heat sink for electron heat equation based on constant fraction of total power density."""

import dataclasses

import chex
import jax
import jax.numpy as jnp
from torax import array_typing
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib

MODEL_FUNCTION_NAME = "radially_constant_fraction_of_Pin"


def radially_constant_fraction_of_Pin(  # pylint: disable=invalid-name
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    calculated_source_profiles: source_profiles_lib.SourceProfiles | None,
    source_models: source_models_lib.SourceModels,
) -> jax.Array:
  """Model function for radiation heat sink from impurities.

  This model represents a sink in the temp_el equation, whose value is a fixed %
  of the total heating power input.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Geometry object.
    source_name: Name of the source.
    core_profiles: Core profiles object.
    calculated_source_profiles: Source profiles which have already been
      calculated and can be used to avoid recomputing them.
    source_models: Source models object.

  Returns:
    The heat sink profile.
  """
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)

  # Based on source_models.sum_sources_temp_el and source_models.calc_and_sum
  # sources_psi, but only summing over heating *input* sources
  # (Pohm + Paux + Palpha + ...) and summing over *both* ion + electron heating

  def get_heat_source_profile(source: source_lib.Source) -> jax.Array:
    # TODO(b/381543891): Currently this recomputes the profile for each source,
    # which is inefficient. Refactor to avoid this.
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
    )
    return source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.TEMP_EL.value, geo
    ) + source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.TEMP_ION.value, geo
    )

  # Calculate the total power input to the heat equations
  heat_sources_and_sinks = (
      source_models.temp_el_sources | source_models.temp_ion_sources
  )
  heat_sources = {
      k: v for k, v in heat_sources_and_sinks.items() if "sink" not in k
  }
  source_profiles = jax.tree.map(
      get_heat_source_profile,
      list(heat_sources.values()),
  )
  Qtot_in = jnp.sum(jnp.stack(source_profiles), axis=0)
  Ptot_in = math_utils.cell_integration(Qtot_in * geo.vpr, geo)
  Vtot = geo.volume_face[-1]

  # Calculate the heat sink as a fraction of the total power input
  return (
      -dynamic_source_runtime_params.fraction_of_total_power_density
      * Ptot_in
      / Vtot
      * jnp.ones_like(geo.rho)
  )


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  fraction_of_total_power_density: runtime_params_lib.TimeInterpolatedInput = (
      0.1
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> "RuntimeParamsProvider":
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: RuntimeParams

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> "DynamicRuntimeParams":
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  fraction_of_total_power_density: array_typing.ScalarFloat
