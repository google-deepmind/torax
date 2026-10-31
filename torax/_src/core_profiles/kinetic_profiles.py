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

"""Kinetic profile dataclass for evolving and derived plasma states."""

import dataclasses
import jax
from torax._src.fvm import cell_variable

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class KineticProfiles:
  """Kinetic profiles.

  Attributes:
    T_i: Ion temperature [keV].
    T_e: Electron temperature [keV].
    n_e: Electron density [m^-3].
    n_i: Main ion density [m^-3].
    n_impurity: Impurity density of bundled impurity [m^-3].
  """

  T_i: cell_variable.CellVariable
  T_e: cell_variable.CellVariable
  n_e: cell_variable.CellVariable
  n_i: cell_variable.CellVariable
  n_impurity: cell_variable.CellVariable
