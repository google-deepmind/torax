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

"""Ion composition and charge state container with constructor.

Forked / factored out from `torax._src.core_profiles.getters`.
"""

from __future__ import annotations

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

  Ion density profiles (`n_i`, `n_impurity`) live on `KineticProfiles`, whereas
  `IonComposition` holds species fractions, atomic masses, and charge states.
  `Z_i`, `Z_impurity`, and `Z_eff` hold the plasma-level and species-effective
  charges used across transport, resistivity, and MHD, while
  `charge_state_info` holds detailed per-impurity coronal charge-state
  distributions.

  Attributes:
    impurity_fractions: Fractional abundances of individual impurity species.
    main_ion_fractions: Fractional abundances of individual main ion species.
    Z_i: Main ion charge on cell grid [dimensionless].
    Z_i_face: Main ion charge on face grid [dimensionless].
    Z_impurity: Impurity charge of bundled impurity on cell grid
      [dimensionless].
    Z_impurity_face: Impurity charge of bundled impurity on face grid
      [dimensionless].
    A_i: Main ion mass [amu].
    A_impurity: Impurity mass on cell grid [amu].
    A_impurity_face: Impurity mass on face grid [amu].
    Z_eff: Effective charge on cell grid [dimensionless].
    Z_eff_face: Effective charge on face grid [dimensionless].
    charge_state_info: Container with averaged and per-species ion charge state
      information on cell grid.
    charge_state_info_face: Container with averaged and per-species ion charge
      state information on face grid.
  """

  impurity_fractions: Mapping[str, array_typing.FloatVector]
  main_ion_fractions: Mapping[str, array_typing.FloatScalar]
  Z_i: array_typing.FloatVectorCell
  Z_i_face: array_typing.FloatVectorFace
  Z_impurity: array_typing.FloatVectorCell
  Z_impurity_face: array_typing.FloatVectorFace
  A_i: array_typing.FloatScalar
  A_impurity: array_typing.FloatVectorCell
  A_impurity_face: array_typing.FloatVectorFace
  Z_eff: array_typing.FloatVectorCell
  Z_eff_face: array_typing.FloatVectorFace
  charge_state_info: charge_states.ChargeStateInfo
  charge_state_info_face: charge_states.ChargeStateInfo

  @classmethod
  def from_profiles(
      cls,
      plasma_composition: plasma_composition_lib.RuntimeParams,
      geo: geometry.Geometry,
      n_e: cell_variable.CellVariable,
      T_e: cell_variable.CellVariable,
  ) -> tuple[
      IonComposition, cell_variable.CellVariable, cell_variable.CellVariable
  ]:
    """Computes ion composition and densities from kinetic profiles."""
    return _compute_ion_composition(plasma_composition, geo, n_e, T_e)


def _calculate_main_ion_dilution_factor(
    Z_i: array_typing.FloatScalar,
    Z_impurity: array_typing.FloatVector,
    Z_eff: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Calculates the main ion dilution factor."""
  return (Z_impurity - Z_eff) / (Z_i * (Z_impurity - Z_i))


@dataclasses.dataclass(frozen=True)
class _IonProperties:
  """Helper container for holding ion calculation outputs."""

  A_impurity: array_typing.FloatVectorCell
  A_impurity_face: array_typing.FloatVectorFace
  Z_impurity: array_typing.FloatVectorCell
  Z_impurity_face: array_typing.FloatVectorFace
  Z_eff: array_typing.FloatVectorCell
  dilution_factor: array_typing.FloatVectorCell
  dilution_factor_edge: array_typing.FloatScalar
  impurity_fractions: Mapping[str, array_typing.FloatVectorCell]
  charge_state_info: charge_states.ChargeStateInfo
  charge_state_info_face: charge_states.ChargeStateInfo

  @classmethod
  def from_fractions(
      cls,
      impurity_params: impurity_fractions_lib.RuntimeParams,
      T_e: cell_variable.CellVariable,
      Z_i: array_typing.FloatVectorCell,
      Z_i_face: array_typing.FloatVectorFace,
      Z_eff_from_config: array_typing.FloatVectorCell,
      Z_eff_face_from_config: array_typing.FloatVectorFace,
  ) -> Self:
    """Calculates ion properties when impurity content is defined by fractions."""

    charge_state_info = charge_states.get_average_charge_state(
        T_e=T_e.value,
        fractions=impurity_params.fractions,
        Z_override=impurity_params.Z_override,
    )
    Z_impurity = charge_state_info.Z_mixture

    charge_state_info_face = charge_states.get_average_charge_state(
        T_e=T_e.face_value(),
        fractions=impurity_params.fractions_face,
        Z_override=impurity_params.Z_override,
    )
    Z_impurity_face = charge_state_info_face.Z_mixture

    Z_eff = Z_eff_from_config
    Z_eff_edge = Z_eff_face_from_config[-1]

    dilution_factor = jnp.where(
        Z_eff == 1.0,
        1.0,
        _calculate_main_ion_dilution_factor(Z_i, Z_impurity, Z_eff),
    )
    dilution_factor_edge = jnp.where(
        Z_eff_edge == 1.0,
        1.0,
        _calculate_main_ion_dilution_factor(
            Z_i_face[-1], Z_impurity_face[-1], Z_eff_edge
        ),
    )
    return cls(
        A_impurity=impurity_params.A_avg,
        A_impurity_face=impurity_params.A_avg_face,
        Z_impurity=Z_impurity,
        Z_impurity_face=Z_impurity_face,
        Z_eff=Z_eff,
        dilution_factor=dilution_factor,
        dilution_factor_edge=dilution_factor_edge,
        impurity_fractions=impurity_params.fractions,
        charge_state_info=charge_state_info,
        charge_state_info_face=charge_state_info_face,
    )

  @classmethod
  def from_n_e_ratios(
      cls,
      impurity_params: electron_density_ratios.RuntimeParams,
      T_e: cell_variable.CellVariable,
      Z_i: array_typing.FloatVectorCell,
      Z_i_face: array_typing.FloatVectorFace,
  ) -> Self:
    """Calculates ion properties when impurity is defined by n_e ratios."""
    average_charge_state = charge_states.get_average_charge_state(
        T_e=T_e.value,
        fractions=impurity_params.fractions,
        Z_override=impurity_params.Z_override,
    )
    average_charge_state_face = charge_states.get_average_charge_state(
        T_e=T_e.face_value(),
        fractions=impurity_params.fractions_face,
        Z_override=impurity_params.Z_override,
    )
    Z_impurity = average_charge_state.Z_mixture
    Z_impurity_face = average_charge_state_face.Z_mixture
    dilution_factor = (
        1
        - jnp.sum(
            jnp.array([
                average_charge_state.Z_per_species[ion] * n_e_ratio
                for ion, n_e_ratio in impurity_params.n_e_ratios.items()
            ]),
            axis=0,
        )
    ) / Z_i
    dilution_factor_edge = (
        1
        - jnp.sum(
            jnp.array([
                average_charge_state_face.Z_per_species[ion][-1] * n_e_ratio[-1]
                for ion, n_e_ratio in impurity_params.n_e_ratios_face.items()
            ]),
            axis=0,
        )
    ) / Z_i_face[-1]
    Z_eff = dilution_factor * Z_i**2 + jnp.sum(
        jnp.array([
            average_charge_state.Z_per_species[ion] ** 2 * n_e_ratio
            for ion, n_e_ratio in impurity_params.n_e_ratios.items()
        ]),
        axis=0,
    )

    return cls(
        A_impurity=impurity_params.A_avg,
        A_impurity_face=impurity_params.A_avg_face,
        Z_impurity=Z_impurity,
        Z_impurity_face=Z_impurity_face,
        Z_eff=Z_eff,
        dilution_factor=dilution_factor,
        dilution_factor_edge=dilution_factor_edge,
        impurity_fractions=impurity_params.fractions,
        charge_state_info=average_charge_state,
        charge_state_info_face=average_charge_state_face,
    )

  @classmethod
  def from_n_e_ratios_zeff(
      cls,
      impurity_params: electron_density_ratios_zeff.RuntimeParams,
      T_e: cell_variable.CellVariable,
      Z_i: array_typing.FloatVectorCell,
      Z_i_face: array_typing.FloatVectorFace,
      Z_eff_from_config: array_typing.FloatVectorCell,
      Z_eff_face_from_config: array_typing.FloatVectorFace,
  ) -> Self:
    """Calculates ion properties when one impurity is constrained by Z_eff."""
    impurity_symbols = tuple(impurity_params.n_e_ratios.keys())
    Z_per_species = jnp.stack([
        charge_states.calculate_average_charge_state_single_species(
            T_e.value, symbol
        )
        for symbol in impurity_symbols
    ])
    Z_per_species_face = jnp.stack([
        charge_states.calculate_average_charge_state_single_species(
            T_e.face_value(), symbol
        )
        for symbol in impurity_symbols
    ])

    unknown_species_index = impurity_symbols.index(
        impurity_params.unknown_species
    )
    Z_unknown = Z_per_species[unknown_species_index]
    Z_unknown_face = Z_per_species_face[unknown_species_index]

    n_e_ratios_known = jnp.array([
        impurity_params.n_e_ratios[symbol]
        if symbol != impurity_params.unknown_species
        else jnp.zeros_like(T_e.value)
        for symbol in impurity_symbols
    ])
    n_e_ratios_known_face = jnp.array([
        impurity_params.n_e_ratios_face[symbol]
        if symbol != impurity_params.unknown_species
        else jnp.zeros_like(T_e.face_value())
        for symbol in impurity_symbols
    ])

    sum_Z_n_ratio = jnp.sum(n_e_ratios_known * Z_per_species, axis=0)
    sum_Z2_n_ratio = jnp.sum(n_e_ratios_known * Z_per_species**2, axis=0)
    sum_Z_n_ratio_face = jnp.sum(
        n_e_ratios_known_face * Z_per_species_face, axis=0
    )
    sum_Z2_n_ratio_face = jnp.sum(
        n_e_ratios_known_face * Z_per_species_face**2, axis=0
    )

    def _solve_system(a1, a2, b1, b2, c1, c2):
      """Solves a 2x2 system of the form a1*x + b1*y = c1, a2*x + b2*y = c2."""
      det_A = a1 * b2 - a2 * b1
      det_A = jnp.where(
          jnp.abs(det_A) < constants.CONSTANTS.eps,
          constants.CONSTANTS.eps,
          det_A,
      )
      x = (b2 * c1 - b1 * c2) / det_A
      y = (a1 * c2 - a2 * c1) / det_A
      return x, y

    dilution_factor, r_unknown = _solve_system(
        a1=Z_i,
        b1=Z_unknown,
        a2=Z_i**2,
        b2=Z_unknown**2,
        c1=1.0 - sum_Z_n_ratio,
        c2=Z_eff_from_config - sum_Z2_n_ratio,
    )
    dilution_factor_face, r_unknown_face = _solve_system(
        a1=Z_i_face,
        b1=Z_unknown_face,
        a2=Z_i_face**2,
        b2=Z_unknown_face**2,
        c1=1.0 - sum_Z_n_ratio_face,
        c2=Z_eff_face_from_config - sum_Z2_n_ratio_face,
    )

    n_e_ratios_all_species = n_e_ratios_known.at[unknown_species_index, :].set(
        r_unknown
    )
    n_e_ratios_all_species_face = n_e_ratios_known_face.at[
        unknown_species_index, :
    ].set(r_unknown_face)

    n_e_ratios_mapping = {
        symbol: n_e_ratios_all_species[i]
        for i, symbol in enumerate(impurity_symbols)
    }
    n_e_ratios_mapping_face = {
        symbol: n_e_ratios_all_species_face[i]
        for i, symbol in enumerate(impurity_symbols)
    }

    fractions = electron_density_ratios.calculate_fractions_from_ratios(
        n_e_ratios_mapping
    )
    fractions_face = electron_density_ratios.calculate_fractions_from_ratios(
        n_e_ratios_mapping_face
    )

    if impurity_params.A_override is None:
      A_avg = jnp.sum(
          jnp.array([
              constants.ION_PROPERTIES_DICT[ion].A * fraction
              for ion, fraction in fractions.items()
          ]),
          axis=0,
      )
      A_avg_face = jnp.sum(
          jnp.array([
              constants.ION_PROPERTIES_DICT[ion].A * fraction
              for ion, fraction in fractions_face.items()
          ]),
          axis=0,
      )
    else:
      A_avg = jnp.ones_like(T_e.value) * impurity_params.A_override
      A_override_face = (
          impurity_params.A_override
          if impurity_params.A_override_face is None
          else impurity_params.A_override_face
      )
      A_avg_face = jnp.ones_like(T_e.face_value()) * A_override_face

    charge_state_info = charge_states.get_average_charge_state(
        T_e=T_e.value,
        fractions=fractions,
        Z_override=impurity_params.Z_override,
    )
    Z_impurity = charge_state_info.Z_mixture

    charge_state_info_face = charge_states.get_average_charge_state(
        T_e=T_e.face_value(),
        fractions=fractions_face,
        Z_override=impurity_params.Z_override,
    )
    Z_impurity_face = charge_state_info_face.Z_mixture

    return cls(
        A_impurity=A_avg,
        A_impurity_face=A_avg_face,
        Z_impurity=Z_impurity,
        Z_impurity_face=Z_impurity_face,
        Z_eff=Z_eff_from_config,
        dilution_factor=dilution_factor,
        dilution_factor_edge=dilution_factor_face[-1],
        impurity_fractions=fractions,
        charge_state_info=charge_state_info,
        charge_state_info_face=charge_state_info_face,
    )

  @classmethod
  def from_plasma_composition(
      cls,
      plasma_composition: plasma_composition_lib.RuntimeParams,
      T_e: cell_variable.CellVariable,
      Z_i: array_typing.FloatVectorCell,
      Z_i_face: array_typing.FloatVectorFace,
  ) -> Self:
    """Computes ion properties based on the configured impurity mode."""
    impurity_params = plasma_composition.impurity
    match impurity_params:
      case impurity_fractions_lib.RuntimeParams():
        return cls.from_fractions(
            impurity_params,
            T_e,
            Z_i,
            Z_i_face,
            plasma_composition.Z_eff,
            plasma_composition.Z_eff_face,
        )
      case electron_density_ratios.RuntimeParams():
        return cls.from_n_e_ratios(
            impurity_params,
            T_e,
            Z_i,
            Z_i_face,
        )
      case electron_density_ratios_zeff.RuntimeParams():
        return cls.from_n_e_ratios_zeff(
            impurity_params,
            T_e,
            Z_i,
            Z_i_face,
            plasma_composition.Z_eff,
            plasma_composition.Z_eff_face,
        )
      case _:
        raise ValueError(f'Unknown impurity mode: {type(impurity_params)}')


@jax.jit
def _compute_ion_composition(
    plasma_composition: plasma_composition_lib.RuntimeParams,
    geo: geometry.Geometry,
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
) -> tuple[
    IonComposition, cell_variable.CellVariable, cell_variable.CellVariable
]:
  """Updated ion density, charge state, and mass based on state and config."""
  Z_i = charge_states.get_average_charge_state(
      T_e=T_e.value,
      fractions=plasma_composition.main_ion.fractions,
      Z_override=plasma_composition.main_ion.Z_override,
  ).Z_mixture
  Z_i_face = charge_states.get_average_charge_state(
      T_e=T_e.face_value(),
      fractions=plasma_composition.main_ion.fractions,
      Z_override=plasma_composition.main_ion.Z_override,
  ).Z_mixture

  ion_properties = _IonProperties.from_plasma_composition(
      plasma_composition, T_e, Z_i, Z_i_face
  )

  n_e_right_bc = n_e.get_right_bc()

  n_i = cell_variable.CellVariable(
      value=n_e.value * ion_properties.dilution_factor,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=None,
      right_face_constraint=n_e_right_bc * ion_properties.dilution_factor_edge,
  )

  safe_Z_impurity = jnp.where(
      ion_properties.dilution_factor == 1.0,
      1.0,
      ion_properties.Z_impurity,
  )
  n_impurity_value = jnp.where(
      ion_properties.dilution_factor == 1.0,
      0.0,
      (n_e.value - n_i.value * Z_i) / safe_Z_impurity,
  )

  safe_Z_impurity_edge = jnp.where(
      ion_properties.dilution_factor_edge == 1.0,
      1.0,
      ion_properties.Z_impurity_face[-1],
  )
  n_impurity_right_face_constraint = jnp.where(
      ion_properties.dilution_factor_edge == 1.0,
      0.0,
      (n_e_right_bc - n_i.get_right_bc() * Z_i_face[-1]) / safe_Z_impurity_edge,
  )

  n_impurity = cell_variable.CellVariable(
      value=n_impurity_value,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=None,
      right_face_constraint=n_impurity_right_face_constraint,
  )

  Z_eff_face = _calculate_Z_eff(
      Z_i_face,
      ion_properties.Z_impurity_face,
      n_i.face_value(),
      n_impurity.face_value(),
      n_e.face_value(),
  )

  impurity_fractions_dict = {}
  for symbol in plasma_composition.impurity_names:
    fraction = ion_properties.impurity_fractions[symbol]
    if fraction.ndim == 0:
      impurity_fractions_dict[symbol] = jnp.full_like(n_e.value, fraction)
    else:
      impurity_fractions_dict[symbol] = fraction

  main_ion_fractions_dict = {}
  for symbol in plasma_composition.main_ion_names:
    fraction = plasma_composition.main_ion.fractions[symbol]
    main_ion_fractions_dict[symbol] = fraction

  ion_composition = IonComposition(
      impurity_fractions=impurity_fractions_dict,
      main_ion_fractions=main_ion_fractions_dict,
      Z_i=Z_i,
      Z_i_face=Z_i_face,
      Z_impurity=ion_properties.Z_impurity,
      Z_impurity_face=ion_properties.Z_impurity_face,
      A_i=plasma_composition.main_ion.A_avg,
      A_impurity=ion_properties.A_impurity,
      A_impurity_face=ion_properties.A_impurity_face,
      Z_eff=ion_properties.Z_eff,
      Z_eff_face=Z_eff_face,
      charge_state_info=ion_properties.charge_state_info,
      charge_state_info_face=ion_properties.charge_state_info_face,
  )
  return ion_composition, n_i, n_impurity


def _calculate_Z_eff(
    Z_i: array_typing.FloatVector,
    Z_impurity: array_typing.FloatVector,
    n_i: array_typing.FloatVector,
    n_impurity: array_typing.FloatVector,
    n_e: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Calculates Z_eff based on single effective impurity and main_ion."""
  return (Z_i**2 * n_i + Z_impurity**2 * n_impurity) / n_e
