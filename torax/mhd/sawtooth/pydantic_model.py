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
from torax.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax.mhd.sawtooth import sawtooth_model
from torax.mhd.sawtooth import simple_redistribution
from torax.mhd.sawtooth import simple_trigger
from torax.torax_pydantic import torax_pydantic


class SawtoothConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for sawtooth configuration.

  Attributes:
    trigger_model_config: Configuration for the trigger model.
    redistribution_model_config: Configuration for the redistribution model.
    crash_step_duration: Sawteeth crash period for extra timestep generated.
  """

  trigger_model_config: Union[simple_trigger.SimpleTriggerConfig] = (
      pydantic.Field(discriminator='trigger_model_type')
  )

  redistribution_model_config: (
      simple_redistribution.SimpleRedistributionConfig
  ) = pydantic.Field(discriminator='redistribution_model_type')

  crash_step_duration: torax_pydantic.Second = 1e-3

  def build_model(self) -> sawtooth_model.SawtoothModel:
    return sawtooth_model.SawtoothModel(
        trigger_model=self.trigger_model_config.build_trigger_model(),
        redistribution_model=self.redistribution_model_config.build_redistribution_model(),
    )

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.DynamicRuntimeParams:
    return sawtooth_runtime_params.DynamicRuntimeParams(
        crash_step_duration=self.crash_step_duration,
        trigger_params=self.trigger_model_config.build_dynamic_params(t),
        redistribution_params=self.redistribution_model_config.build_dynamic_params(
            t
        ),
    )
