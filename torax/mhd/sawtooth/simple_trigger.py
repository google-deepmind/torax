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

"""Simple trigger model for sawteeth."""

from typing import Literal
import chex
from jax import numpy as jnp
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.mhd.sawtooth import runtime_params
from torax.mhd.sawtooth import sawtooth_model
from torax.torax_pydantic import torax_pydantic


class SimpleTrigger(sawtooth_model.TriggerModel):
  """Simple trigger model."""

  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> array_typing.ScalarBool:
    # TODO(b/317360481): implement trigger model
    return jnp.array(False)


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params.TriggerDynamicRuntimeParams):
  """Dynamic runtime params for simple trigger model.

  Attributes:
    s_critical: Critical shear value at q=1 for sawtooth triggering.
  """

  s_critical: array_typing.ScalarFloat


class SimpleTriggerConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for simple trigger configuration.

  Attributes:
    s_critical: Critical shear value.
  """

  s_critical: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )
  trigger_model_type: Literal['simple'] = 'simple'

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(s_critical=self.s_critical.get_value(t))

  def build_trigger_model(self) -> SimpleTrigger:
    return SimpleTrigger()
