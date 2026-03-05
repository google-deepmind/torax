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

"""Fast ion physics classes."""

import dataclasses

import jax
from torax._src.fvm import cell_variable


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FastIon:
  """State of a fast ion species.

  Attributes:
    species: Species name (e.g. 'He3').
    source: Source name (e.g. 'ICRH').
    n: Density [m^-3].
    T: Temperature [keV].
  """

  species: str = dataclasses.field(metadata={'static': True})
  source: str = dataclasses.field(metadata={'static': True})
  n: cell_variable.CellVariable
  T: cell_variable.CellVariable
