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

from collections.abc import Mapping
import dataclasses
from typing import Self

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry


# All fast ion species that can be modeled/prescribed in TORAX.
FAST_ION_SPECIES: tuple[str, ...] = ('H', 'D', 'T', 'He3', 'He4')


def validate_fast_ion_species(species: str, source: str) -> None:
  """Validates that `species` is in `FAST_ION_SPECIES`."""
  if species not in FAST_ION_SPECIES:
    raise ValueError(
        f'Fast ion species {species!r} for source {source!r} is not a'
        f' supported species. Supported species: {FAST_ION_SPECIES}'
    )


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

  def __post_init__(self):
    validate_fast_ion_species(self.species, self.source)

  @classmethod
  def from_arrays(
      cls,
      species: str,
      source_name: str,
      geo: geometry.Geometry,
      n: array_typing.FloatVectorCell,
      T: array_typing.FloatVectorCell,
      n_right_face_constraint: array_typing.FloatScalar | None = None,
      T_right_face_constraint: array_typing.FloatScalar | None = None,
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


def calculate_n_impurity_thermal(
    n_impurity: cell_variable.CellVariable,
    impurity_fractions: Mapping[str, array_typing.FloatVectorCell],
    fast_ions: tuple[FastIon, ...],
) -> cell_variable.CellVariable:
  """True thermal impurity density without fast ions [m^-3].

  TODO(b/868852029): This assumes fast ions subtracted from thermal impurities
  are originally from thermal species. This needs to be updated when we
  introduce other sources of fast ions that do not deplete thermal species
  (e.g. fusion).

  Args:
    n_impurity: Impurity density CellVariable.
    impurity_fractions: Fractional abundances of individual impurity species.
    fast_ions: Tuple of fast ion populations.

  Returns:
    Thermal impurity density CellVariable.
  """
  n_impurity_thermal_value = n_impurity.value
  n_impurity_thermal_right = n_impurity.right_face_constraint
  for fast_ion in fast_ions:
    if fast_ion.species in impurity_fractions:
      n_impurity_thermal_value = n_impurity_thermal_value - fast_ion.n.value
      if (
          n_impurity_thermal_right is not None
          and fast_ion.n.right_face_constraint is not None
      ):
        n_impurity_thermal_right = (
            n_impurity_thermal_right - fast_ion.n.right_face_constraint
        )
  return cell_variable.CellVariable(
      value=n_impurity_thermal_value,
      face_centers=n_impurity.face_centers,
      right_face_constraint=n_impurity_thermal_right,
      right_face_grad_constraint=None,
  )

