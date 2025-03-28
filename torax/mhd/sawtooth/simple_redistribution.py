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

"""Simple redistribution model for sawteeth. Currently a no-op."""

import dataclasses
from typing import Literal
import chex
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.mhd.sawtooth import base_pydantic_model
from torax.mhd.sawtooth import runtime_params
from torax.mhd.sawtooth import sawtooth_model
from torax.torax_pydantic import torax_pydantic


class SimpleRedistribution(sawtooth_model.RedistributionModel):
  """Simple redistribution model."""

  def __call__(
      self,
      rho_norm_q1: array_typing.ScalarFloat,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreProfiles:
    # TODO(b/317360481): implement redistribution model. For now, does nothing.
    """Applies flattening redistribution.

    Args:
      rho_norm_q1: The radius of the q=1 surface.
      static_runtime_params_slice: Static runtime parameters.
      dynamic_runtime_params_slice: Dynamic runtime parameters.
      geo: Geometry object.
      core_profiles: Core plasma profiles *before* redistribution.

    Returns:
      Core plasma profiles *after* redistribution.
    """

    return core_profiles


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params.RedistributionDynamicRuntimeParams):
  """Dynamic runtime params for simple redistribution model.

  Attributes:
    mixing_radius_multiplier: Profile modification will be limited to a radius
      of mixing_radius_multiplier * rho_norm_at_q1.
  """

  mixing_radius_multiplier: array_typing.ScalarFloat


class SimpleRedistributionConfig(base_pydantic_model.RedistributionConfig):
  """Pydantic model for simple redistribution configuration."""

  redistribution_model_type: Literal['simple'] = 'simple'
  mixing_radius_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.1)
  )

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_dynamic_params(t))
    return DynamicRuntimeParams(
        **base_kwargs,
        mixing_radius_multiplier=self.mixing_radius_multiplier.get_value(t)
    )

  def build_redistribution_model(
      self,
  ) -> SimpleRedistribution:
    return SimpleRedistribution()
