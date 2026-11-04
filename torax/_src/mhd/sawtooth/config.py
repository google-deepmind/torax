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

"""Pydantic config for sawtooth configuration."""
import chex
import pydantic
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.mhd.sawtooth import sawtooth_models
from torax._src.mhd.sawtooth import simple_redistribution
from torax._src.mhd.sawtooth import simple_trigger
from torax._src.torax_pydantic import torax_pydantic


class SawtoothConfig(torax_pydantic.ConfigBase):
  """Pydantic config for sawtooth configuration.

  Attributes:
    trigger_model: Configuration for the sawtooth trigger model. Currently only
      the `'simple'` model is supported.
    redistribution_model: Configuration for the sawtooth profile redistribution
      model. Currently only the `'simple'` model is supported.
    crash_step_duration: Duration of a sawtooth crash step [s]. This is
      how much the solver time will be bumped forward during a crash. It is
      important that this is small enough to satisfy the assumption of timescale
      separation between the sawtooth crash and the transport timescale,
      otherwise issues may arise regarding conservation properties.
  """

  trigger_model: simple_trigger.SimpleTriggerConfig = pydantic.Field(
      discriminator='model_name'
  )

  redistribution_model: simple_redistribution.SimpleRedistributionConfig = (
      pydantic.Field(discriminator='model_name')
  )

  crash_step_duration: torax_pydantic.Second = 1e-5

  def build_models(self):
    return sawtooth_models.SawtoothModels(
        trigger_model=self.trigger_model.build_trigger_model(),
        redistribution_model=self.redistribution_model.build_redistribution_model(),
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.RuntimeParams:
    return sawtooth_runtime_params.RuntimeParams(
        crash_step_duration=self.crash_step_duration,
        trigger_params=self.trigger_model.build_runtime_params(t),
        redistribution_params=self.redistribution_model.build_runtime_params(t),
    )
