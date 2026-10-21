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
from typing import Self

import jax
from jax import numpy as jnp
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry


# All fast ion species that can be modeled/prescribed in TORAX.
FAST_ION_SPECIES: tuple[str, ...] = ('H', 'D', 'T', 'He3', 'He4')


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

  species: str = jax.tree.static()
  source: str = jax.tree.static()
  n: cell_variable.CellVariable
  T: cell_variable.CellVariable

  @classmethod
  def from_arrays(
      cls,
      species: str,
      source_name: str,
      geo: geometry.Geometry,
      n: jax.Array,
      T: jax.Array,
      n_right_face_constraint: jax.Array | None = None,
      T_right_face_constraint: jax.Array | None = None,
  ) -> Self:
    """Constructs a FastIon object from density and temperature arrays."""
    return cls(
        species=species,
        source=source_name,
        n=cell_variable.CellVariable(
            value=n,
            face_centers=geo.rho_face_norm,
            right_face_grad_constraint=None,
            right_face_constraint=(
                n_right_face_constraint
                if n_right_face_constraint is not None
                else jnp.zeros(())
            ),
        ),
        T=cell_variable.CellVariable(
            value=T,
            face_centers=geo.rho_face_norm,
            right_face_grad_constraint=None,
            right_face_constraint=(
                T_right_face_constraint
                if T_right_face_constraint is not None
                else jnp.zeros(())
            ),
        ),
    )
