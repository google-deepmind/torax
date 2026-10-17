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

"""Cross-timestep state for transport models."""

from collections.abc import Mapping
import dataclasses
import jax


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ComponentTransportState:
  """Base class for component transport model cross-timestep state."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TransportState:
  """Cross-timestep state for the transport model."""

  core_transport_states: Mapping[str, ComponentTransportState] = (
      dataclasses.field(default_factory=dict)
  )
  pedestal_transport_states: Mapping[str, ComponentTransportState] = (
      dataclasses.field(default_factory=dict)
  )
