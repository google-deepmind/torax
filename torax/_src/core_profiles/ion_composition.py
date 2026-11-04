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

"""Ion composition and charge state container and density calculation."""

from collections.abc import Mapping
import dataclasses
from typing import Self

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.core_profiles.plasma_composition import electron_density_ratios_zeff
from torax._src.core_profiles.plasma_composition import impurity_fractions as impurity_fractions_lib
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import charge_states

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class IonComposition:
  """Ion composition and charge state information.

  Attributes:
    impurity_fractions: Fractional abundances of individual impurity species on
      cell grid [dimensionless].
    main_ion_fractions: Fractional abundances of individual main ion species.
    Z_i: Main ion charge on cell grid [dimensionless].
    Z_i_face: Main ion charge on face grid [dimensionless].
    Z_impurity: Effective charge <Z^2> / <Z> of bundled impurity on cell grid
      [dimensionless].
    Z_impurity_face: Effective charge <Z^2> / <Z> of bundled impurity on face
      grid [dimensionless].
    Z_avg: Mean impurity charge <Z> = sum(f_k * Z_k) on cell grid
      [dimensionless].
    Z_avg_face: Mean impurity charge <Z> = sum(f_k * Z_k) on face grid
      [dimensionless].
    Z_impurity_per_species: Charge state for each individual impurity species on
      cell grid [dimensionless].
    Z_impurity_per_species_face: Charge state for each individual impurity
      species on face grid [dimensionless].
    A_i: Main ion mass [amu].
    A_impurity: Impurity mass on cell grid [amu].
    A_impurity_face: Impurity mass on face grid [amu].
    Z_eff: Effective charge on cell grid [dimensionless].
    Z_eff_face: Effective charge on face grid [dimensionless].
  """

  impurity_fractions: Mapping[str, array_typing.FloatVectorCell]
  main_ion_fractions: Mapping[str, array_typing.FloatScalar]
  Z_i: array_typing.FloatVectorCell
  Z_i_face: array_typing.FloatVectorFace
  Z_impurity: array_typing.FloatVectorCell
  Z_impurity_face: array_typing.FloatVectorFace
  Z_avg: array_typing.FloatVectorCell
  Z_avg_face: array_typing.FloatVectorFace
  Z_impurity_per_species: Mapping[str, array_typing.FloatVectorCell]
  Z_impurity_per_species_face: Mapping[str, array_typing.FloatVectorFace]
  A_i: array_typing.FloatScalar
  A_impurity: array_typing.FloatVectorCell
  A_impurity_face: array_typing.FloatVectorFace
  Z_eff: array_typing.FloatVectorCell
  Z_eff_face: array_typing.FloatVectorFace


def get_ion_composition_and_density(
    plasma_composition: plasma_composition_lib.RuntimeParams,
    geo: geometry.Geometry,
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
) -> tuple[
    IonComposition, cell_variable.CellVariable, cell_variable.CellVariable
]:
  """Computes ion composition, main ion density, and impurity density.

  Args:
    plasma_composition: Plasma composition runtime parameters.
    geo: Tokamak geometry.
    n_e: Electron density profile [m^-3].
    T_e: Electron temperature profile [keV].

  Returns:
    Tuple of `(ion_composition, n_i, n_impurity)`.
  """
  composition_cell = _IonCompositionIntermediates.from_grid(
      plasma_composition, T_e.value, is_face=False
  )
  composition_face = _IonCompositionIntermediates.from_grid(
      plasma_composition, T_e.face_value(), is_face=True
  )

  n_e_right_bc = n_e.get_right_bc()
  n_i = cell_variable.CellVariable(
      value=n_e.value * composition_cell.dilution_factor,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=None,
      right_face_constraint=(
          n_e_right_bc * composition_face.dilution_factor[-1]
      ),
  )
  n_impurity = cell_variable.CellVariable(
      value=_impurity_density_from_quasineutrality(
          n_e.value,
          n_i.value,
          composition_cell.Z_i,
          composition_cell.Z_impurity,
          composition_cell.dilution_factor,
      ),
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=None,
      right_face_constraint=_impurity_density_from_quasineutrality(
          n_e_right_bc,
          n_i.get_right_bc(),
          composition_face.Z_i[-1],
          composition_face.Z_impurity[-1],
          composition_face.dilution_factor[-1],
      ),
  )
  Z_eff_face = (
      composition_face.Z_i**2 * n_i.face_value()
      + composition_face.Z_impurity**2 * n_impurity.face_value()
  ) / n_e.face_value()

  ion_composition = IonComposition(
      impurity_fractions=composition_cell.impurity_fractions,
      main_ion_fractions=plasma_composition.main_ion.fractions,
      Z_i=composition_cell.Z_i,
      Z_i_face=composition_face.Z_i,
      Z_impurity=composition_cell.Z_impurity,
      Z_impurity_face=composition_face.Z_impurity,
      Z_avg=composition_cell.Z_avg,
      Z_avg_face=composition_face.Z_avg,
      Z_impurity_per_species=composition_cell.Z_impurity_per_species,
      Z_impurity_per_species_face=composition_face.Z_impurity_per_species,
      A_i=plasma_composition.main_ion.A_avg,
      A_impurity=composition_cell.A_impurity,
      A_impurity_face=composition_face.A_impurity,
      Z_eff=composition_cell.Z_eff,
      Z_eff_face=Z_eff_face,
  )
  return ion_composition, n_i, n_impurity


def _calculate_average_mass(
    fractions: Mapping[str, array_typing.FloatVector],
    A_override: array_typing.FloatScalar | None = None,
) -> array_typing.FloatVector:
  """Calculates the average atomic mass of an ion mixture."""
  if A_override is not None:
    first_fraction = next(iter(fractions.values()))
    return jnp.ones_like(first_fraction) * A_override
  return jnp.sum(
      jnp.asarray([
          constants.ION_PROPERTIES_DICT[ion].A * fraction
          for ion, fraction in fractions.items()
      ]),
      axis=0,
  )


def _calculate_dilution_from_Z_eff(
    Z_i: array_typing.FloatVector,
    Z_impurity: array_typing.FloatVector,
    Z_eff: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Calculates the main ion dilution factor n_i / n_e from Z_eff."""
  pure_plasma = Z_eff == Z_i
  denom = Z_i * (Z_impurity - Z_i)
  # Guard jnp.where dead branch against zero division to prevent NaN gradients.
  safe_denom = jnp.where(pure_plasma, 1.0, denom)
  return jnp.where(
      pure_plasma,
      1.0 / Z_i,
      (Z_impurity - Z_eff) / safe_denom,
  )


def _impurity_density_from_quasineutrality(
    n_e: array_typing.FloatVector | array_typing.FloatScalar,
    n_i: array_typing.FloatVector | array_typing.FloatScalar,
    Z_i: array_typing.FloatVector | array_typing.FloatScalar,
    Z_impurity: array_typing.FloatVector | array_typing.FloatScalar,
    dilution_factor: array_typing.FloatVector | array_typing.FloatScalar,
) -> jax.Array:
  """Calculates bundled impurity density (n_e - n_i * Z_i) / Z_impurity."""
  no_impurities = dilution_factor == 1.0 / Z_i
  # Guard jnp.where dead branch against zero division when Z_impurity == 0.
  safe_Z_impurity = jnp.where(no_impurities, 1.0, Z_impurity)
  return jnp.where(
      no_impurities,
      0.0,
      (n_e - n_i * Z_i) / safe_Z_impurity,
  )


@dataclasses.dataclass(frozen=True)
class _IonCompositionIntermediates:
  """Intermediate quantities resolved on a 1D grid (cell or face).

  Note: On the face grid (`is_face=True`), `Z_eff` is only used as an input to
  resolve `dilution_factor` and `impurity_fractions`; `IonComposition` computes
  `Z_eff_face` from the interpolated `CellVariable.face_value()` densities.
  """

  Z_i: array_typing.FloatVector
  Z_impurity_per_species: Mapping[str, array_typing.FloatVector]
  impurity_fractions: Mapping[str, array_typing.FloatVector]
  A_impurity: array_typing.FloatVector
  Z_avg: array_typing.FloatVector
  Z_impurity: array_typing.FloatVector
  Z_eff: array_typing.FloatVector
  dilution_factor: array_typing.FloatVector

  @classmethod
  def from_grid(
      cls,
      plasma_composition: plasma_composition_lib.RuntimeParams,
      T_e: array_typing.FloatVector,
      is_face: bool = False,
  ) -> Self:
    """Computes 1D ion composition on either the cell or face grid."""
    impurity_params = plasma_composition.impurity
    Z_eff = (
        plasma_composition.Z_eff_face if is_face else plasma_composition.Z_eff
    )

    Z_main_ion_per_species = charge_states.get_average_charge_state_per_species(
        T_e=T_e,
        ion_symbols=plasma_composition.main_ion_names,
        Z_override=plasma_composition.main_ion.Z_override,
    )
    Z_i = charge_states.get_effective_charge_state(
        Z_per_species=Z_main_ion_per_species,
        fractions=plasma_composition.main_ion.fractions,
    )
    Z_per_species = charge_states.get_average_charge_state_per_species(
        T_e=T_e,
        ion_symbols=plasma_composition.impurity_names,
        Z_override=impurity_params.Z_override,
    )
    match impurity_params:
      case impurity_fractions_lib.RuntimeParams():
        return cls.from_fractions(
            fractions=(
                impurity_params.fractions_face
                if is_face
                else impurity_params.fractions
            ),
            A_override=impurity_params.A_override,
            Z_i=Z_i,
            Z_per_species=Z_per_species,
            Z_eff=Z_eff,
        )
      case electron_density_ratios.RuntimeParams():
        return cls.from_n_e_ratios(
            n_e_ratios=(
                impurity_params.n_e_ratios_face
                if is_face
                else impurity_params.n_e_ratios
            ),
            fractions=(
                impurity_params.fractions_face
                if is_face
                else impurity_params.fractions
            ),
            A_override=impurity_params.A_override,
            Z_i=Z_i,
            Z_per_species=Z_per_species,
        )
      case electron_density_ratios_zeff.RuntimeParams():
        return cls.from_n_e_ratios_zeff(
            n_e_ratios=(
                impurity_params.n_e_ratios_face
                if is_face
                else impurity_params.n_e_ratios
            ),
            unknown_species=impurity_params.unknown_species,
            A_override=impurity_params.A_override,
            Z_i=Z_i,
            Z_per_species=Z_per_species,
            Z_eff=Z_eff,
        )
      case _:
        raise ValueError(f'Unknown impurity mode: {type(impurity_params)}')

  @classmethod
  def from_fractions(
      cls,
      fractions: Mapping[str, array_typing.FloatVector],
      A_override: array_typing.FloatScalar | None,
      Z_i: array_typing.FloatVector,
      Z_per_species: Mapping[str, array_typing.FloatVector],
      Z_eff: array_typing.FloatVector,
      dilution_factor: array_typing.FloatVector | None = None,
  ) -> Self:
    """Resolves 1D ion composition from impurity fractions and Z_eff."""
    A_avg = _calculate_average_mass(fractions, A_override)
    Z_avg = charge_states.get_average_charge_state(
        Z_per_species=Z_per_species,
        fractions=fractions,
    )
    Z_impurity = charge_states.get_effective_charge_state(
        Z_per_species=Z_per_species,
        fractions=fractions,
        Z_avg=Z_avg,
    )
    if dilution_factor is None:
      dilution_factor = _calculate_dilution_from_Z_eff(Z_i, Z_impurity, Z_eff)
    return cls(
        Z_i=Z_i,
        Z_impurity_per_species=Z_per_species,
        impurity_fractions=fractions,
        A_impurity=A_avg,
        Z_avg=Z_avg,
        Z_impurity=Z_impurity,
        Z_eff=Z_eff,
        dilution_factor=dilution_factor,
    )

  @classmethod
  def from_n_e_ratios(
      cls,
      n_e_ratios: Mapping[str, array_typing.FloatVector],
      fractions: Mapping[str, array_typing.FloatVector],
      A_override: array_typing.FloatScalar | None,
      Z_i: array_typing.FloatVector,
      Z_per_species: Mapping[str, array_typing.FloatVector],
      Z_eff: array_typing.FloatVector | None = None,
  ) -> Self:
    """Resolves 1D ion composition from prescribed impurity n_e ratios."""
    impurity_charge_fraction = charge_states.get_average_charge_state(
        Z_per_species, n_e_ratios
    )
    dilution_factor = (1.0 - impurity_charge_fraction) / Z_i
    if Z_eff is None:
      Z_eff = dilution_factor * Z_i**2 + jnp.sum(
          jnp.asarray([
              Z_per_species[ion] ** 2 * n_e_ratio
              for ion, n_e_ratio in n_e_ratios.items()
          ]),
          axis=0,
      )
    return cls.from_fractions(
        fractions=fractions,
        A_override=A_override,
        Z_i=Z_i,
        Z_per_species=Z_per_species,
        Z_eff=Z_eff,
        dilution_factor=dilution_factor,
    )

  @classmethod
  def from_n_e_ratios_zeff(
      cls,
      n_e_ratios: Mapping[str, array_typing.FloatVector | None],
      unknown_species: str,
      A_override: array_typing.FloatScalar | None,
      Z_i: array_typing.FloatVector,
      Z_per_species: Mapping[str, array_typing.FloatVector],
      Z_eff: array_typing.FloatVector,
  ) -> Self:
    """Resolves 1D ion composition when one impurity is constrained by Z_eff."""
    known_ratios = {
        symbol: ratio if ratio is not None else jnp.zeros_like(Z_i)
        for symbol, ratio in n_e_ratios.items()
    }
    Z_unknown = Z_per_species[unknown_species]
    sum_Z_n_ratio = charge_states.get_average_charge_state(
        Z_per_species, known_ratios
    )
    sum_Z2_n_ratio = jnp.sum(
        jnp.asarray([
            Z_per_species[symbol] ** 2 * ratio
            for symbol, ratio in known_ratios.items()
        ]),
        axis=0,
    )

    # Eliminate dilution_factor from quasineutrality and Z_eff:
    #   Z_i * dilution_factor + Z_unknown * r_unknown = 1 - sum_Z_n_ratio
    #   Z_i^2 * dilution_factor + Z_unknown^2 * r_unknown
    #       = Z_eff - sum_Z2_n_ratio
    denom = Z_unknown * (Z_unknown - Z_i)
    denom = jnp.where(
        jnp.abs(denom) < constants.CONSTANTS.eps,
        constants.CONSTANTS.eps,
        denom,
    )
    r_unknown = ((Z_eff - sum_Z2_n_ratio) - Z_i * (1.0 - sum_Z_n_ratio)) / denom

    n_e_ratios_all = {**known_ratios, unknown_species: r_unknown}
    fractions = electron_density_ratios.calculate_fractions_from_ratios(
        n_e_ratios_all
    )
    return cls.from_n_e_ratios(
        n_e_ratios=n_e_ratios_all,
        fractions=fractions,
        A_override=A_override,
        Z_i=Z_i,
        Z_per_species=Z_per_species,
        Z_eff=Z_eff,
    )
