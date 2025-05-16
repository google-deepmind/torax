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

"""Pydantic model for sawtooth configuration."""

from typing import Union
import chex
import pydantic
from torax._src.config import runtime_params_slice
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.mhd.sawtooth import sawtooth_model
from torax._src.mhd.sawtooth import simple_redistribution
from torax._src.mhd.sawtooth import simple_trigger
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.torax_pydantic import torax_pydantic
from torax.transport_model import transport_model as transport_model_lib


class SawtoothConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for sawtooth configuration.

  Attributes:
    trigger_model: Configuration for the trigger model.
    redistribution_model: Configuration for the redistribution model.
    crash_step_duration: Sawteeth crash period for extra timestep generated.
  """

  trigger_model: Union[simple_trigger.SimpleTriggerConfig] = pydantic.Field(
      discriminator='model_name'
  )

  redistribution_model: simple_redistribution.SimpleRedistributionConfig = (
      pydantic.Field(discriminator='model_name')
  )

  crash_step_duration: torax_pydantic.Second = 1e-3

  def build_model(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
      pedestal_model: pedestal_model_lib.PedestalModel,
  ) -> sawtooth_model.SawtoothModel:
    return sawtooth_model.SawtoothModel(
        static_runtime_params_slice=static_runtime_params_slice,
        trigger_model=self.trigger_model.build_trigger_model(),
        redistribution_model=self.redistribution_model.build_redistribution_model(),
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.DynamicRuntimeParams:
    return sawtooth_runtime_params.DynamicRuntimeParams(
        crash_step_duration=self.crash_step_duration,
        trigger_params=self.trigger_model.build_dynamic_params(t),
        redistribution_params=self.redistribution_model.build_dynamic_params(t),
    )
