# Copyright 2026 DeepMind Technologies Limited
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

"""Cross-timestep state for stateful physics models and global plasma energy."""

from collections.abc import Mapping
import dataclasses
import jax
from torax._src import state
from torax._src.pedestal_model import pedestal_state
from torax._src.sources import source_state
from torax._src.transport_model import transport_state


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ModelState:
  """Cross-timestep state for stateful physics models and global plasma energy."""

  pedestal: pedestal_state.PedestalState
  transport: transport_state.TransportState
  sources: Mapping[str, source_state.SourceState]
  internal_plasma_energy: state.PlasmaInternalEnergy
