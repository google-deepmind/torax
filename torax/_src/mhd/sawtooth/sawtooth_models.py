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

"""A container for sawtooth models."""
import dataclasses

from torax._src.mhd.sawtooth import redistribution_base
from torax._src.mhd.sawtooth import trigger_base


@dataclasses.dataclass
class SawtoothModels:
  """Container for sawtooth models."""
  trigger_model: trigger_base.TriggerModel
  redistribution_model: redistribution_base.RedistributionModel

  def __eq__(self, other: 'SawtoothModels') -> bool:
    return (
        self.trigger_model == other.trigger_model
        and self.redistribution_model == other.redistribution_model
    )

  def __hash__(self) -> int:
    return hash((
        self.trigger_model,
        self.redistribution_model,
    ))
