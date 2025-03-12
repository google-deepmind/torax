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
from __future__ import annotations

from typing import Literal

import chex
import jax.numpy as jnp
from torax import array_typing
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import base
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax.torax_pydantic import torax_pydantic

MODEL_FUNCTION_NAME = 'radially_constant_fraction_of_Pin'


def radially_constant_fraction_of_Pin(  # pylint: disable=invalid-name
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_core_profiles: state.CoreProfiles,
    calculated_source_profiles: source_profiles_lib.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Model function for radiation heat sink from impurities.

  This model represents a sink in the temp_el equation, whose value is a fixed %
  of the total heating power input.

  Args:
    unused_static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Geometry object.
    source_name: Name of the source.
    unused_core_profiles: Core profiles object.
    calculated_source_profiles: Source profiles which have already been
      calculated and can be used to avoid recomputing them.

  Returns:
    The heat sink profile.
  """
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

  # Based on source_models.sum_sources_temp_el and source_models.calc_and_sum
  # sources_psi, but only summing over heating *input* sources
  # (Pohm + Paux + Palpha + ...) and summing over *both* ion + electron heating

  # TODO(b/383061556) Move away from using brittle source names to identify
  # sinks/sources.
  source_profiles = jnp.zeros_like(geo.rho)
  for source_name in calculated_source_profiles.temp_el:
    if 'sink' not in source_name:
      source_profiles += calculated_source_profiles.temp_el[source_name]
  for source_name in calculated_source_profiles.temp_ion:
    if 'sink' not in source_name:
      source_profiles += calculated_source_profiles.temp_ion[source_name]

  Q_total_in = source_profiles
  P_total_in = math_utils.volume_integration(Q_total_in, geo)

  # Calculate the heat sink as a fraction of the total power input
  return (
      -dynamic_source_runtime_params.fraction_of_total_power_density
      * P_total_in
      / geo.volume_face[-1]
      * jnp.ones_like(geo.rho),
  )


class ImpurityRadiationHeatSinkConstantFractionConfig(base.SourceModelBase):
  """Configuration for the ImpurityRadiationHeatSink.

  Attributes:
    fraction_of_total_power_density: Fraction of total power density to be
      absorbed by the impurity.
  """
  source_name: Literal['impurity_radiation_heat_sink'] = (
      'impurity_radiation_heat_sink'
  )
  model_function_name: Literal['radially_constant_fraction_of_Pin'] = (
      'radially_constant_fraction_of_Pin'
  )
  fraction_of_total_power_density: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> 'DynamicRuntimeParams':
    return DynamicRuntimeParams(
        prescribed_values=self.prescribed_values.get_value(t),
        fraction_of_total_power_density=self.fraction_of_total_power_density.get_value(
            t
        ),
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
  fraction_of_total_power_density: array_typing.ScalarFloat
