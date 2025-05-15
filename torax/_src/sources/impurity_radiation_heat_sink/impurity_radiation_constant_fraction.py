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
from typing import Literal

import chex
import jax.numpy as jnp
from torax import array_typing
from torax import math_utils
from torax import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
def radially_constant_fraction_of_Pin(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_core_profiles: state.CoreProfiles,
    calculated_source_profiles: source_profiles_lib.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[chex.Array, ...]:
  """Model function for radiation heat sink from impurities."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)

  if calculated_source_profiles is None:
    raise ValueError(
        'calculated_source_profiles is a required argument for'
        ' `radially_constant_fraction_of_Pin`. This can occur if this source'
        ' function is used in an explicit source.'
    )

  # Based on source_models.sum_sources_T_e and source_models.calc_and_sum
  # sources_psi, but only summing over heating *input* sources
  # (Pohm + Paux + Palpha + ...) and summing over *both* ion + electron heating

  # TODO(b/383061556) Move away from using brittle source names to identify
  # sinks/sources.
  source_profiles = jnp.zeros_like(geo.rho)
  for source_name in calculated_source_profiles.T_e:
    if 'sink' not in source_name:
      source_profiles += calculated_source_profiles.T_e[source_name]
  for source_name in calculated_source_profiles.T_i:
    if 'sink' not in source_name:
      source_profiles += calculated_source_profiles.T_i[source_name]

  Q_total_in = source_profiles
  P_total_in = math_utils.volume_integration(Q_total_in, geo)

  # Calculate the heat sink as a fraction of the total power input
  return (
      -dynamic_source_runtime_params.fraction_P_heating
      * P_total_in
      / geo.volume_face[-1]
      * jnp.ones_like(geo.rho),
  )


class ImpurityRadiationHeatSinkConstantFractionConfig(base.SourceModelBase):
  """Configuration for the ImpurityRadiationHeatSink.

  Attributes:
    fraction_P_heating: Fraction of total power density to be absorbed by the
      impurity.
  """

  model_name: Literal['P_in_scaled_flat_profile'] = 'P_in_scaled_flat_profile'
  fraction_P_heating: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> 'DynamicRuntimeParams':
    return DynamicRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        fraction_P_heating=self.fraction_P_heating.get_value(t),
    )

  def build_source(
      self,
  ) -> impurity_radiation_heat_sink.ImpurityRadiationHeatSink:
    return impurity_radiation_heat_sink.ImpurityRadiationHeatSink(
        model_func=self.model_func
    )

  @property
  def model_func(self) -> source_lib.SourceProfileFunction:
    return radially_constant_fraction_of_Pin


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  fraction_P_heating: array_typing.ScalarFloat
