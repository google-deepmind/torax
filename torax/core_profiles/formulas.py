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

"""Common functions used for working with core profiles."""
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import constants
from torax import jax_utils
from torax.config import numerics
from torax.config import profile_conditions
from torax.fvm import cell_variable
from torax.geometry import geometry

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def get_updated_ion_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Updated ion temp. Used upon initialization and if temp_ion=False."""
  Ti_bound_right = jax_utils.error_if_not_positive(
      dynamic_profile_conditions.Ti_bound_right,
      'Ti_bound_right',
  )
  temp_ion = cell_variable.CellVariable(
      value=dynamic_profile_conditions.Ti,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Ti_bound_right,
      dr=geo.drho_norm,
  )

  return temp_ion


def get_updated_electron_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Updated electron temp. Used upon initialization and if temp_el=False."""
  Te_bound_right = jax_utils.error_if_not_positive(
      dynamic_profile_conditions.Te_bound_right,
      'Te_bound_right',
  )
  temp_el = cell_variable.CellVariable(
      value=dynamic_profile_conditions.Te,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Te_bound_right,
      dr=geo.drho_norm,
  )
  return temp_el


def get_ne(
    dynamic_numerics: numerics.DynamicNumerics,
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial or prescribed electron density profile at current timestep."""

  nGW = (
      dynamic_profile_conditions.Ip_tot
      / (jnp.pi * geo.Rmin**2)
      * 1e20
      / dynamic_numerics.nref
  )
  ne_value = jnp.where(
      dynamic_profile_conditions.ne_is_fGW,
      dynamic_profile_conditions.ne * nGW,
      dynamic_profile_conditions.ne,
  )
  # Calculate ne_bound_right.
  ne_bound_right = jnp.where(
      dynamic_profile_conditions.ne_bound_right_is_fGW,
      dynamic_profile_conditions.ne_bound_right * nGW,
      dynamic_profile_conditions.ne_bound_right,
  )

  if dynamic_profile_conditions.normalize_to_nbar:
    face_left = ne_value[0]  # Zero gradient boundary condition at left face.
    face_right = ne_bound_right
    face_inner = (ne_value[..., :-1] + ne_value[..., 1:]) / 2.0
    ne_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]],
    )
    # Find normalization factor such that desired line-averaged ne is set.
    # Line-averaged electron density (nbar) is poorly defined. In general, the
    # definition is machine-dependent and even shot-dependent since it depends
    # on the usage of a specific interferometry chord. Furthermore, even if we
    # knew the specific chord used, its calculation would depend on magnetic
    # geometry information beyond what is available in StandardGeometry.
    # In lieu of a better solution, we use line-averaged electron density
    # defined on the outer midplane.
    Rmin_out = geo.Rout_face[-1] - geo.Rout_face[0]
    # find target nbar in absolute units
    target_nbar = jnp.where(
        dynamic_profile_conditions.ne_is_fGW,
        dynamic_profile_conditions.nbar * nGW,
        dynamic_profile_conditions.nbar,
    )
    if not dynamic_profile_conditions.ne_bound_right_is_absolute:
      # In this case, ne_bound_right is taken from ne and we also normalize it.
      C = target_nbar / (_trapz(ne_face, geo.Rout_face) / Rmin_out)
      ne_bound_right = C * ne_bound_right
    else:
      # If ne_bound_right is absolute, subtract off contribution from outer
      # face to get C we need to multiply the inner values with.
      nbar_from_ne_face_inner = (
          _trapz(ne_face[:-1], geo.Rout_face[:-1]) / Rmin_out
      )

      dr_edge = geo.Rout_face[-1] - geo.Rout_face[-2]

      C = (target_nbar - 0.5 * ne_face[-1] * dr_edge / Rmin_out) / (
          nbar_from_ne_face_inner + 0.5 * ne_face[-2] * dr_edge / Rmin_out
      )
  else:
    C = 1

  ne_value = C * ne_value

  ne = cell_variable.CellVariable(
      value=ne_value,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=jnp.array(ne_bound_right),
  )
  return ne


def calculate_psi_grad_constraint_from_Ip_tot(
    Ip_tot: array_typing.ScalarFloat,
    geo: geometry.Geometry,
) -> jax.Array:
  """Calculates the gradient constraint on the poloidal flux (psi) from Ip."""
  return (
      Ip_tot
      * 1e6
      * (16 * jnp.pi**3 * constants.CONSTANTS.mu0 * geo.Phib)
      / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
  )


# TODO(b/377225415): generalize to arbitrary number of ions.
def get_main_ion_dilution_factor(
    Zi: array_typing.ScalarFloat,
    Zimp: array_typing.ArrayFloat,
    Zeff: array_typing.ArrayFloat,
) -> jax.Array:
  """Calculates the main ion dilution factor based on a single assumed impurity and general main ion charge."""
  return (Zimp - Zeff) / (Zi * (Zimp - Zi))

