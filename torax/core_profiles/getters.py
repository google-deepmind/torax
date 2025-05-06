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
import functools

import jax
from jax import numpy as jnp
from torax import array_typing
from torax import jax_utils
from torax.config import numerics
from torax.config import profile_conditions
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.physics import charge_states
from torax.physics import formulas

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def get_updated_ion_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed ion temperature profiles."""
  temp_ion = cell_variable.CellVariable(
      value=dynamic_profile_conditions.Ti,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=dynamic_profile_conditions.T_i_right_bc,
      dr=geo.drho_norm,
  )
  return temp_ion


def get_updated_electron_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron temperature profiles."""
  temp_el = cell_variable.CellVariable(
      value=dynamic_profile_conditions.Te,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=dynamic_profile_conditions.T_e_right_bc,
      dr=geo.drho_norm,
  )
  return temp_el


def get_updated_electron_density(
    dynamic_numerics: numerics.DynamicNumerics,
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron density profiles."""

  nGW = (
      dynamic_profile_conditions.I_total
      / (jnp.pi * geo.a_minor**2)
      * 1e20
      / dynamic_numerics.nref
  )
  n_e_value = jnp.where(
      dynamic_profile_conditions.n_e_is_fGW,
      dynamic_profile_conditions.n_e * nGW,
      dynamic_profile_conditions.n_e,
  )
  # Calculate n_e_bound_right.
  n_e_bound_right = jnp.where(
      dynamic_profile_conditions.n_e_bound_right_is_fGW,
      dynamic_profile_conditions.n_e_bound_right * nGW,
      dynamic_profile_conditions.n_e_bound_right,
  )

  if dynamic_profile_conditions.normalize_to_nbar:
    face_left = n_e_value[0]  # Zero gradient boundary condition at left face.
    face_right = n_e_bound_right
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
    Rmin_out = geo.R_out_face[-1] - geo.R_out_face[0]
    # find target nbar in absolute units
    target_nbar = jnp.where(
        dynamic_profile_conditions.n_e_is_fGW,
        dynamic_profile_conditions.nbar * nGW,
        dynamic_profile_conditions.nbar,
    )
    if not dynamic_profile_conditions.n_e_bound_right_is_absolute:
      # In this case, n_e_bound_right taken from n_e and we also normalize it.
      C = target_nbar / (_trapz(n_e_face, geo.R_out_face) / Rmin_out)
      n_e_bound_right = C * n_e_bound_right
    else:
      # If n_e_bound_right is absolute, subtract off contribution from outer
      # face to get C we need to multiply the inner values with.
      nbar_from_n_e_face_inner = (
          _trapz(n_e_face[:-1], geo.R_out_face[:-1]) / Rmin_out
      )

      dr_edge = geo.R_out_face[-1] - geo.R_out_face[-2]

      C = (target_nbar - 0.5 * n_e_face[-1] * dr_edge / Rmin_out) / (
          nbar_from_n_e_face_inner + 0.5 * n_e_face[-2] * dr_edge / Rmin_out
      )
  else:
    C = 1

  n_e_value = C * n_e_value

  n_e = cell_variable.CellVariable(
      value=n_e_value,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=n_e_bound_right,
  )
  return n_e


# jitted since also used outside the stepper
@functools.partial(
    jax_utils.jit, static_argnames=['static_runtime_params_slice']
)
def get_ion_density_and_charge_states(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    n_e: cell_variable.CellVariable,
    temp_el: cell_variable.CellVariable,
) -> tuple[
    cell_variable.CellVariable,
    cell_variable.CellVariable,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
]:
  """Updated ion densities based on state.

  Main ion and impurities are each treated as a single effective ion, but could
  be comparised of multiple species within an IonMixture. The main ion and
  impurity densities are calculated depending on the Zeff constraint,
  quasineutrality, and the average impurity charge state which may be
  temperature dependent.

  Zeff = (Zi**2 * ni + Zimp**2 * nimp)/n_e  ;  nimp*Zimp + ni*Zi = n_e

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Geometry of the tokamak.
    n_e: Electron density profile [nref].
    temp_el: Electron temperature profile [keV].

  Returns:
    ni: Ion density profile [nref].
    nimp: Impurity density profile [nref].
    Zi: Average charge state of main ion on cell grid [amu].
      Typically just the average of the atomic numbers since these are normally
      low Z ions and can be assumed to be fully ionized.
    Zi_face: Average charge state of main ion on face grid [amu].
    Zimp: Average charge state of impurities on cell grid [amu].
    Zimp_face: Average charge state of impurities on face grid [amu].
  """

  Zi, Zi_face, Zimp, Zimp_face = _get_charge_states(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      temp_el,
  )

  Zeff = dynamic_runtime_params_slice.plasma_composition.Zeff
  Zeff_face = dynamic_runtime_params_slice.plasma_composition.Zeff_face

  dilution_factor = formulas.calculate_main_ion_dilution_factor(Zi, Zimp, Zeff)
  dilution_factor_edge = formulas.calculate_main_ion_dilution_factor(
      Zi_face[-1], Zimp_face[-1], Zeff_face[-1]
  )

  ni = cell_variable.CellVariable(
      value=n_e.value * dilution_factor,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=n_e.right_face_constraint * dilution_factor_edge,
  )

  nimp = cell_variable.CellVariable(
      value=(n_e.value - ni.value * Zi) / Zimp,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=(
          n_e.right_face_constraint - ni.right_face_constraint * Zi_face[-1]
      )
      / Zimp_face[-1],
  )
  return ni, nimp, Zi, Zi_face, Zimp, Zimp_face


def _get_charge_states(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    temp_el: cell_variable.CellVariable,
) -> tuple[
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
]:
  """Updated charge states based on IonMixtures and electron temperature."""
  Zi = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.main_ion,
      Te=temp_el.value,
  )
  Zi_face = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.main_ion,
      Te=temp_el.face_value(),
  )

  Zimp = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.impurity_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.impurity,
      Te=temp_el.value,
  )
  Zimp_face = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.impurity_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.impurity,
      Te=temp_el.face_value(),
  )

  return Zi, Zi_face, Zimp, Zimp_face
