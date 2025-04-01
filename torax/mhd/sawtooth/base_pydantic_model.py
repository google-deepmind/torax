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

"""Base pydantic configs for trigger and redistribution models for sawtooth."""

import chex
from torax.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax.torax_pydantic import torax_pydantic


class RedistributionConfig(torax_pydantic.BaseModelFrozen):
  """Base config for all redistribution models.

  Attributes:
    flattening_factor: The factor by which the profile is flattened.
    Default is near 1.0 but not exactly 1.0, to avoid zero-gradients.
  """

  flattening_factor: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.01)
  )

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.RedistributionDynamicRuntimeParams:
    return sawtooth_runtime_params.RedistributionDynamicRuntimeParams(
        flattening_factor=self.flattening_factor.get_value(t),
    )


class TriggerConfig(torax_pydantic.BaseModelFrozen):
  """Base config for all trigger models.

  Attributes:
    minimum_radius: The minimum radius of the q=1 surface for triggering.
  """

  minimum_radius: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.05)
  )

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.TriggerDynamicRuntimeParams:
    return sawtooth_runtime_params.TriggerDynamicRuntimeParams(
        minimum_radius=self.minimum_radius.get_value(t),
    )
