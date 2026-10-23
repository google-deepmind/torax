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

"""State for pedestal models."""

import dataclasses
import jax


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StructureState:
  """State for a pedestal structure model."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FormationState:
  """State for a pedestal formation model."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalState:
  """State for a pedestal model carried over from the previous timestep."""

  structure: StructureState = dataclasses.field(default_factory=StructureState)
  formation: FormationState = dataclasses.field(default_factory=FormationState)
