# Copyright 2025 DeepMind Technologies Limited
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
"""Calculations related to the rotation of the plasma."""

from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.neoclassical import formulas as neoclassical_formulas
from torax._src.physics import psi_calculations


# pylint: disable=invalid-name
def _calculate_radial_electric_field(
    pressure_thermal_i: cell_variable.CellVariable,
    toroidal_angular_velocity: cell_variable.CellVariable,
    poloidal_velocity: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    Z_i_face: array_typing.FloatVector,
    B_pol_face: array_typing.FloatVector,
    B_tor_face: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Calculates the radial electric field Er.

  Er = (1 / (Zi * e * ni)) * dpi/dr - v_phi * B_theta + v_theta * B_phi

  Args:
    pressure_thermal_i: Pressure profile as a cell variable.
    toroidal_angular_velocity: Toroidal velocity profile as a cell variable.
    poloidal_velocity: Poloidal velocity profile as a cell variable.
    n_i: Main ion density profile as a cell variable.
    Z_i_face: Main ion charge on the face grid.
    B_pol_face: Flux-surface-averaged poloidal magnetic field on the face grid.
    B_tor_face: Flux-surface-averaged toroidal magnetic field on the face grid.
    geo: Geometry object.

  Returns:
    Er: Radial electric field [V/m] on the cell grid.
  """
  # Calculate dpi/dr with respect to a midplane-averaged radial coordinate.
  dpi_dr = pressure_thermal_i.face_grad(
      x=geo.r_mid, x_left=geo.r_mid_face[0], x_right=geo.r_mid_face[-1]
  )

  # Calculate Er
  denominator = Z_i_face * constants.CONSTANTS.q_e * n_i.face_value()
  Er = (
      math_utils.safe_divide(jnp.array(1.0), denominator) * dpi_dr
      - toroidal_angular_velocity.face_value()
      * geo.R_major_profile_face
      * B_pol_face
      + poloidal_velocity.face_value() * B_tor_face
  )
  return cell_variable.CellVariable(
      value=geometry.face_to_cell(Er),
      face_centers=geo.rho_face_norm,
      right_face_constraint=Er[-1],
      right_face_grad_constraint=None,
  )


def _calculate_v_ExB(
    Er_face: array_typing.FloatVectorFace,
    B_total_face: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the ExB velocity, on the face grid."""
  B_total_face = jnp.maximum(B_total_face, constants.CONSTANTS.eps)
  return jnp.where(B_total_face > 0, Er_face / B_total_face, 0.0)


def calculate_rotation(
    T_i: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    q_face: array_typing.FloatVectorFace,
    Z_eff_face: array_typing.FloatVectorFace,
    Z_i_face: array_typing.FloatVector,
    toroidal_angular_velocity: cell_variable.CellVariable,
    pressure_thermal_i: cell_variable.CellVariable,
    geo: geometry.Geometry,
    poloidal_velocity_multiplier: array_typing.FloatScalar = 1.0,
):
  """Calculates quantities related to the rotation of the plasma.

  Args:
    T_i: Ion temperature profile as a cell variable.
    psi: Poloidal flux profile as a cell variable.
    n_i: Main ion density profile as a cell variable.
    q_face: Safety factor on the face grid.
    Z_eff_face: Effective charge on the face grid.
    Z_i_face: Main ion charge on the face grid.
    toroidal_angular_velocity: Toroidal velocity profile as a cell variable.
    pressure_thermal_i: Pressure profile as a cell variable.
    geo: Geometry object.
    poloidal_velocity_multiplier: A multiplier to apply to the poloidal
      velocity.

  Returns:
    v_ExB: ExB velocity profile on the face grid [m/s].
    Er: Radial electric field as a cell variable [V/m] .
    poloidal_velocity: Poloidal velocity as a cell variable [m/s].
  """

  # Flux surface average of `B_phi = F/R`.
  B_tor_face = geo.F_face / geo.R_major_profile_face  # Tesla

  # flux-surface-averaged B_theta.
  B_pol_squared_face = psi_calculations.calc_bpol_squared(
      geo, psi
  )  # On the face grid.
  B_pol_face = jnp.sqrt(B_pol_squared_face)  # Tesla
  B_total_squared_face = B_pol_squared_face + B_tor_face**2
  B_total_face = jnp.sqrt(B_total_squared_face)

  poloidal_velocity = neoclassical_formulas.calculate_poloidal_velocity(
      T_i=T_i,
      n_i=n_i.face_value(),
      q=q_face,
      Z_eff=Z_eff_face,
      Z_i=Z_i_face,
      B_tor=B_tor_face,
      B_total_squared=B_total_squared_face,
      geo=geo,
      poloidal_velocity_multiplier=poloidal_velocity_multiplier,
  )

  Er = _calculate_radial_electric_field(
      pressure_thermal_i=pressure_thermal_i,
      toroidal_angular_velocity=toroidal_angular_velocity,
      poloidal_velocity=poloidal_velocity,
      n_i=n_i,
      Z_i_face=Z_i_face,
      B_pol_face=B_pol_face,
      B_tor_face=B_tor_face,
      geo=geo,
  )

  v_ExB = _calculate_v_ExB(Er.face_value(), B_total_face)

  return v_ExB, Er, poloidal_velocity
