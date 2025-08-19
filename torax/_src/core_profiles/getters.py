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

"""Functions for getting updated CellVariable objects for CoreProfiles."""
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.config import plasma_composition
from torax._src.config import profile_conditions
from torax._src.config import runtime_params_slice
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import charge_states
from torax._src.physics import formulas

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Ions:
  """Helper container for holding ion attributes."""

  n_i: cell_variable.CellVariable
  n_impurity: cell_variable.CellVariable
  impurity_fractions: array_typing.FloatVector
  Z_i: array_typing.FloatVectorCell
  Z_i_face: array_typing.FloatVectorFace
  Z_impurity: array_typing.FloatVectorCell
  Z_impurity_face: array_typing.FloatVectorFace
  A_i: array_typing.FloatScalar
  A_impurity: array_typing.FloatScalar
  Z_eff: array_typing.FloatVectorCell
  Z_eff_face: array_typing.FloatVectorFace


def get_updated_ion_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed ion temperature profiles."""
  T_i = cell_variable.CellVariable(
      value=dynamic_profile_conditions.T_i,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=dynamic_profile_conditions.T_i_right_bc,
      dr=geo.drho_norm,
  )
  return T_i


def get_updated_electron_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron temperature profiles."""
  T_e = cell_variable.CellVariable(
      value=dynamic_profile_conditions.T_e,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=dynamic_profile_conditions.T_e_right_bc,
      dr=geo.drho_norm,
  )
  return T_e


def get_updated_electron_density(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron density profiles."""

  # Greenwald density in m^-3.
  # Ip in MA. a_minor in m.
  nGW = (
      dynamic_profile_conditions.Ip
      / 1e6  # Convert to MA.
      / (jnp.pi * geo.a_minor**2)
      * 1e20
  )
  n_e_value = jnp.where(
      dynamic_profile_conditions.n_e_nbar_is_fGW,
      dynamic_profile_conditions.n_e * nGW,
      dynamic_profile_conditions.n_e,
  )
  # Calculate n_e_right_bc.
  n_e_right_bc = jnp.where(
      dynamic_profile_conditions.n_e_right_bc_is_fGW,
      dynamic_profile_conditions.n_e_right_bc * nGW,
      dynamic_profile_conditions.n_e_right_bc,
  )

  if dynamic_profile_conditions.normalize_n_e_to_nbar:
    face_left = n_e_value[0]  # Zero gradient boundary condition at left face.
    face_right = n_e_right_bc
    face_inner = (n_e_value[..., :-1] + n_e_value[..., 1:]) / 2.0
    n_e_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]],
    )
    # Find normalization factor such that desired line-averaged n_e is set.
    # Line-averaged electron density (nbar) is poorly defined. In general, the
    # definition is machine-dependent and even shot-dependent since it depends
    # on the usage of a specific interferometry chord. Furthermore, even if we
    # knew the specific chord used, its calculation would depend on magnetic
    # geometry information beyond what is available in StandardGeometry.
    # In lieu of a better solution, we use line-averaged electron density
    # defined on the outer midplane.
    a_minor_out = geo.R_out_face[-1] - geo.R_out_face[0]
    # find target nbar in absolute units
    target_nbar = jnp.where(
        dynamic_profile_conditions.n_e_nbar_is_fGW,
        dynamic_profile_conditions.nbar * nGW,
        dynamic_profile_conditions.nbar,
    )
    if not dynamic_profile_conditions.n_e_right_bc_is_absolute:
      # In this case, n_e_right_bc is taken from n_e and we also normalize it.
      C = target_nbar / (_trapz(n_e_face, geo.R_out_face) / a_minor_out)
      n_e_right_bc = C * n_e_right_bc
    else:
      # If n_e_right_bc is absolute, subtract off contribution from outer
      # face to get C we need to multiply the inner values with.
      nbar_from_n_e_face_inner = (
          _trapz(n_e_face[:-1], geo.R_out_face[:-1]) / a_minor_out
      )

      dr_edge = geo.R_out_face[-1] - geo.R_out_face[-2]

      C = (target_nbar - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
          nbar_from_n_e_face_inner + 0.5 * n_e_face[-2] * dr_edge / a_minor_out
      )
  else:
    C = 1

  n_e_value = C * n_e_value

  n_e = cell_variable.CellVariable(
      value=n_e_value,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=n_e_right_bc,
  )
  return n_e


@dataclasses.dataclass(frozen=True)
class _IonProperties:
  """Helper container for holding ion calculation outputs."""

  Z_impurity: array_typing.FloatVectorCell
  Z_impurity_face: array_typing.FloatVectorFace
  Z_eff: array_typing.FloatVectorCell
  dilution_factor: array_typing.FloatVectorCell
  dilution_factor_edge: array_typing.FloatScalar
  impurity_fractions: array_typing.FloatVector


def _get_ion_properties_from_fractions(
    impurity_symbols: tuple[str, ...],
    impurity_dynamic_params: plasma_composition.DynamicImpurityFractions,
    T_e: cell_variable.CellVariable,
    Z_i: array_typing.FloatVectorCell,
    Z_i_face: array_typing.FloatVectorFace,
    Z_eff_from_config: array_typing.FloatVectorCell,
    Z_eff_face_from_config: array_typing.FloatVectorFace,
) -> _IonProperties:
  """Calculates ion properties when impurity content is defined by fractions."""

  Z_impurity = charge_states.get_average_charge_state(
      ion_symbols=impurity_symbols,
      ion_mixture=impurity_dynamic_params,
      T_e=T_e.value,
  ).Z_mixture
  Z_impurity_face = charge_states.get_average_charge_state(
      ion_symbols=impurity_symbols,
      ion_mixture=impurity_dynamic_params,
      T_e=T_e.face_value(),
  ).Z_mixture

  Z_eff = Z_eff_from_config
  Z_eff_edge = Z_eff_face_from_config[-1]

  dilution_factor = jnp.where(
      Z_eff == 1.0,
      1.0,
      formulas.calculate_main_ion_dilution_factor(Z_i, Z_impurity, Z_eff),
  )
  dilution_factor_edge = jnp.where(
      Z_eff_edge == 1.0,
      1.0,
      formulas.calculate_main_ion_dilution_factor(
          Z_i_face[-1], Z_impurity_face[-1], Z_eff_edge
      ),
  )
  return _IonProperties(
      Z_impurity=Z_impurity,
      Z_impurity_face=Z_impurity_face,
      Z_eff=Z_eff,
      dilution_factor=dilution_factor,
      dilution_factor_edge=dilution_factor_edge,
      impurity_fractions=impurity_dynamic_params.fractions,
  )


def _get_ion_properties_from_n_e_ratios(
    impurity_symbols: tuple[str, ...],
    impurity_dynamic_params: plasma_composition.DynamicNeRatios,
    T_e: cell_variable.CellVariable,
    Z_i: array_typing.FloatVectorCell,
    Z_i_face: array_typing.FloatVectorFace,
) -> _IonProperties:
  """Calculates ion properties when impurity content is defined by n_e ratios."""
  impurity_mixture = plasma_composition.DynamicIonMixture(
      fractions=impurity_dynamic_params.fractions,
      A_avg=impurity_dynamic_params.A_avg,
      Z_override=impurity_dynamic_params.Z_override,
  )
  average_charge_state_cell = charge_states.get_average_charge_state(
      ion_symbols=impurity_symbols,
      ion_mixture=impurity_mixture,
      T_e=T_e.value,
  )
  average_charge_state_face = charge_states.get_average_charge_state(
      ion_symbols=impurity_symbols,
      ion_mixture=impurity_mixture,
      T_e=T_e.face_value(),
  )
  Z_impurity = average_charge_state_cell.Z_mixture
  Z_impurity_face = average_charge_state_face.Z_mixture
  # impurity_mixture.n_e_ratios has shape (n_species,).
  # Z_per_species has shape (n_species,) if T_e.value is a scalar, or
  # (n_species, n_grid) if T_e.value is an array.
  # We need to broadcast n_e_ratios for element-wise multiplication.
  # Reshape fractions to be broadcastable with Z_per_species.
  n_e_ratios_reshaped = jnp.reshape(
      impurity_dynamic_params.n_e_ratios,
      impurity_dynamic_params.fractions.shape
      + (1,) * (average_charge_state_cell.Z_per_species.ndim - 1),
  )
  dilution_factor = (
      1
      - jnp.sum(
          average_charge_state_cell.Z_per_species * n_e_ratios_reshaped,
          axis=0,
      )
      / Z_i
  )
  dilution_factor_edge = (
      1
      - jnp.sum(
          average_charge_state_face.Z_per_species[:, -1]
          * impurity_dynamic_params.n_e_ratios,
      )
      / Z_i_face[-1]
  )
  Z_eff = dilution_factor * Z_i**2 + jnp.sum(
      average_charge_state_cell.Z_per_species**2 * n_e_ratios_reshaped,
      axis=0,
  )
  return _IonProperties(
      Z_impurity=Z_impurity,
      Z_impurity_face=Z_impurity_face,
      Z_eff=Z_eff,
      dilution_factor=dilution_factor,
      dilution_factor_edge=dilution_factor_edge,
      impurity_fractions=impurity_mixture.fractions,
  )


# jitted since also used outside the solver
@jax_utils.jit
def get_updated_ions(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
) -> Ions:
  """Updated ion density, charge state, and mass based on state and config.

  Main ion and impurities are each treated as a single effective ion, but could
  be comparised of multiple species within an IonMixture. The main ion and
  impurity densities are calculated depending on the Z_eff constraint,
  quasineutrality, and the average impurity charge state which may be
  temperature dependent.

  Z_eff = (Z_i**2 * n_i + Z_impurity**2 * n_impurity)/n_e  ;
  n_impurity*Z_impurity + n_i*Z_i = n_e

  Args:
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Geometry of the tokamak.
    n_e: Electron density profile [m^-3].
    T_e: Electron temperature profile [keV].

  Returns:
    Ion container with the following attributes:
      n_i: Ion density profile [m^-3].
      n_impurity: Impurity density profile [m^-3].
      Z_i: Average charge state of main ion on cell grid [dimensionless].
        Typically just the average of the atomic numbers since these are
        normally low Z ions and can be assumed to be fully ionized.
      Z_i_face: Average charge state of main ion on face grid [dimensionless].
      Z_impurity: Average charge state of impurities on cell grid
      [dimensionless].
      Z_impurity_face: Average charge state of impurities on face grid
      [dimensionless].
      A_i: Average atomic number of main ion [amu].
      A_impurity: Average atomic number of impurities [amu].
  """

  Z_i = charge_states.get_average_charge_state(
      ion_symbols=dynamic_runtime_params_slice.plasma_composition.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.main_ion,
      T_e=T_e.value,
  ).Z_mixture
  Z_i_face = charge_states.get_average_charge_state(
      ion_symbols=dynamic_runtime_params_slice.plasma_composition.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.main_ion,
      T_e=T_e.face_value(),
  ).Z_mixture

  impurity_dynamic_params = (
      dynamic_runtime_params_slice.plasma_composition.impurity
  )
  impurity_mode = impurity_dynamic_params.impurity_mode

  match impurity_mode:
    case plasma_composition.IMPURITY_MODE_FRACTIONS:
      ion_properties = _get_ion_properties_from_fractions(
          dynamic_runtime_params_slice.plasma_composition.impurity_names,
          impurity_dynamic_params,
          T_e,
          Z_i,
          Z_i_face,
          dynamic_runtime_params_slice.plasma_composition.Z_eff,
          dynamic_runtime_params_slice.plasma_composition.Z_eff_face,
      )

    case plasma_composition.IMPURITY_MODE_NE_RATIOS:
      ion_properties = _get_ion_properties_from_n_e_ratios(
          dynamic_runtime_params_slice.plasma_composition.impurity_names,
          impurity_dynamic_params,
          T_e,
          Z_i,
          Z_i_face,
      )
    case _:
      # Not expected to be reached but needed to avoid linter errors.
      raise ValueError(f"Unknown impurity mode: {impurity_mode}")

  n_i = cell_variable.CellVariable(
      value=n_e.value * ion_properties.dilution_factor,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=n_e.right_face_constraint
      * ion_properties.dilution_factor_edge,
  )

  n_impurity_value = jnp.where(
      ion_properties.dilution_factor == 1.0,
      0.0,
      (n_e.value - n_i.value * Z_i) / ion_properties.Z_impurity,
  )

  n_impurity_right_face_constraint = jnp.where(
      ion_properties.dilution_factor_edge == 1.0,
      0.0,
      (n_e.right_face_constraint - n_i.right_face_constraint * Z_i_face[-1])
      / ion_properties.Z_impurity_face[-1],
  )

  n_impurity = cell_variable.CellVariable(
      value=n_impurity_value,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=n_impurity_right_face_constraint,
  )

  # Z_eff from plasma composition is imposed and can be passed to CoreProfiles.
  # However, we must recalculate Z_eff_face from the updated densities and
  # charge states since linearly interpolated Z_eff (which is what plasma
  # composition Z_eff_face is) would not be physically consistent.
  Z_eff_face = _calculate_Z_eff(
      Z_i_face,
      ion_properties.Z_impurity_face,
      n_i.face_value(),
      n_impurity.face_value(),
      n_e.face_value(),
  )

  return Ions(
      n_i=n_i,
      n_impurity=n_impurity,
      impurity_fractions=ion_properties.impurity_fractions,
      Z_i=Z_i,
      Z_i_face=Z_i_face,
      Z_impurity=ion_properties.Z_impurity,
      Z_impurity_face=ion_properties.Z_impurity_face,
      A_i=dynamic_runtime_params_slice.plasma_composition.main_ion.A_avg,
      A_impurity=dynamic_runtime_params_slice.plasma_composition.impurity.A_avg,
      Z_eff=ion_properties.Z_eff,
      Z_eff_face=Z_eff_face,
  )


def _calculate_Z_eff(
    Z_i: array_typing.FloatVector,
    Z_impurity: array_typing.FloatVector,
    n_i: array_typing.FloatVector,
    n_impurity: array_typing.FloatVector,
    n_e: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Calculates Z_eff based on impurity and main_ion."""
  return (Z_i**2 * n_i + Z_impurity**2 * n_impurity) / n_e
