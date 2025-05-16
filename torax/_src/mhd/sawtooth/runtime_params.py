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

"""Dynamic runtime params for sawtooth model."""

import chex
from torax._src import array_typing


@chex.dataclass(frozen=True)
class TriggerDynamicRuntimeParams:
  """Dynamic runtime params for trigger models.

  Attributes:
    minimum_radius: Minimum radius of q=1 surface for triggering [rho_norm].
  """

  minimum_radius: array_typing.ScalarFloat


@chex.dataclass(frozen=True)
class RedistributionDynamicRuntimeParams:
  """Dynamic runtime params for redistribution models.

  Attributes:
    flattening_factor: Ratio of "flat" profile between magnetic axis and q=1
      surface.
  """

  flattening_factor: array_typing.ScalarFloat


@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Dynamic runtime params for sawtooth model.

  Attributes:
    trigger_params: Dynamic runtime params for trigger models.
    redistribution_params: Dynamic runtime params for redistribution models.
    crash_step_duration: Sawtooth crash period for extra timestep generated.
  """

  trigger_params: TriggerDynamicRuntimeParams
  redistribution_params: RedistributionDynamicRuntimeParams
  crash_step_duration: array_typing.ScalarFloat
