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


from typing import Literal
import chex
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.mhd.sawtooth import runtime_params
from torax.mhd.sawtooth import sawtooth_model
from torax.torax_pydantic import torax_pydantic


class SimpleRedistribution(sawtooth_model.RedistributionModel):
  """Simple redistribution model."""

  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreProfiles:
    # TODO(b/317360481): implement redistribution model. For now, does nothing.
    return core_profiles


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params.RedistributionDynamicRuntimeParams):
  # TODO(b/317360481): implement redistribution model. For now, does nothing.
  """Dynamic runtime params for simple redistribution model."""


class SimpleRedistributionConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for simple redistribution configuration."""

  redistribution_model_type: Literal['simple'] = 'simple'

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    del t  # Unused.
    return DynamicRuntimeParams()

  def build_redistribution_model(
      self,
  ) -> SimpleRedistribution:
    return SimpleRedistribution()
