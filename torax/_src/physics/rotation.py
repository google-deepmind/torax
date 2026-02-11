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
import dataclasses

from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.neoclassical.formulas import formulas
from torax._src.physics import psi_calculations


# pylint: disable=invalid-name
@dataclasses.dataclass(frozen=True)
class RotationOutput:
  """Structured output from rotation calculations.

  Attributes:
    v_ExB: Total ExB velocity profile on the face grid [m/s].
    v_ExB_poloidal_and_pressure: v_ExB from poloidal rotation + pressure
      gradient contributions:
      (dpi/dr / (Zi * e * ni) + v_theta * B_phi) / B_total.
    v_ExB_toroidal: v_ExB from toroidal velocity contribution:
      -v_phi * B_theta / B_total.
    Er: Total radial electric field as a cell variable [V/m].
    poloidal_velocity: Poloidal velocity as a cell variable [m/s].
  """

  v_ExB: array_typing.FloatVectorFace
  v_ExB_poloidal_and_pressure: array_typing.FloatVectorFace
  v_ExB_toroidal: array_typing.FloatVectorFace
  Er: cell_variable.CellVariable
  poloidal_velocity: cell_variable.CellVariable


def _calculate_radial_electric_field(
    pressure_total_i: cell_variable.CellVariable,
    toroidal_angular_velocity: cell_variable.CellVariable,
    poloidal_velocity: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    Z_i_face: array_typing.FloatVector,
    B_pol_face: array_typing.FloatVector,
    B_tor_face: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> tuple[
    cell_variable.CellVariable,
    array_typing.FloatVectorFace,
    array_typing.FloatVectorFace,
]:
  """Calculates the radial electric field Er with separate components.

  Er = (1 / (Zi * e * ni)) * dpi/dr - v_phi * B_theta + v_theta * B_phi

  We split Er into:
  - Er_poloidal_and_pressure: (1 / (Zi * e * ni)) * dpi/dr + v_theta * B_phi
    This contribution is from the pressure gradient and poloidal rotation.
  - Er_toroidal: -v_phi * B_theta
    This contribution is from toroidal velocity.

  Args:
    pressure_total_i: Ion pressure profile (thermal + fast) as a cell variable.
    toroidal_angular_velocity: Toroidal velocity profile as a cell variable.
    poloidal_velocity: Poloidal velocity profile as a cell variable.
    n_i: Main ion density profile as a cell variable.
    Z_i_face: Main ion charge on the face grid.
    B_pol_face: Flux-surface-averaged poloidal magnetic field on the face grid.
    B_tor_face: Flux-surface-averaged toroidal magnetic field on the face grid.
    geo: Geometry object.

  Returns:
    Er: Radial electric field [V/m] as a CellVariable.
    Er_poloidal_and_pressure_face: Er contribution from pressure gradient and
      poloidal rotation [V/m] on face grid.
    Er_toroidal_face: Er contribution from toroidal velocity [V/m] on face grid.
  """
  # Calculate dpi/dr with respect to a midplane-averaged radial coordinate.
  dpi_dr = pressure_total_i.face_grad(
      x=geo.r_mid, x_left=geo.r_mid_face[0], x_right=geo.r_mid_face[-1]
  )

  # Calculate Er
  denominator = Z_i_face * constants.CONSTANTS.q_e * n_i.face_value()

  Er_poloidal_and_pressure_face = (
      math_utils.safe_divide(jnp.array(1.0), denominator) * dpi_dr
      + poloidal_velocity.face_value() * B_tor_face
  )

  Er_toroidal_face = (
      -toroidal_angular_velocity.face_value()
      * geo.R_major_profile_face
      * B_pol_face
  )

  Er_face = Er_poloidal_and_pressure_face + Er_toroidal_face

  Er = cell_variable.CellVariable(
      value=geometry.face_to_cell(Er_face),
      face_centers=geo.rho_face_norm,
      right_face_constraint=Er_face[-1],
      right_face_grad_constraint=None,
  )
  return Er, Er_poloidal_and_pressure_face, Er_toroidal_face


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
    pressure_total_i: cell_variable.CellVariable,
    geo: geometry.Geometry,
    poloidal_velocity_multiplier: array_typing.FloatScalar = 1.0,
) -> RotationOutput:
  """Calculates quantities related to the rotation of the plasma.

  Args:
    T_i: Ion temperature profile as a cell variable.
    psi: Poloidal flux profile as a cell variable.
    n_i: Main ion density profile as a cell variable.
    q_face: Safety factor on the face grid.
    Z_eff_face: Effective charge on the face grid.
    Z_i_face: Main ion charge on the face grid.
    toroidal_angular_velocity: Toroidal velocity profile as a cell variable.
    pressure_total_i: Total ion pressure (thermal + fast) as a cell variable.
    geo: Geometry object.
    poloidal_velocity_multiplier: A multiplier to apply to the poloidal
      velocity.

  Returns:
    RotationOutput with v_ExB, separated v_ExB components, Er, and
    poloidal_velocity.
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

  poloidal_velocity = formulas.calculate_poloidal_velocity(
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

  Er, Er_poloidal_and_pressure_face, Er_toroidal_face = (
      _calculate_radial_electric_field(
          pressure_total_i=pressure_total_i,
          toroidal_angular_velocity=toroidal_angular_velocity,
          poloidal_velocity=poloidal_velocity,
          n_i=n_i,
          Z_i_face=Z_i_face,
          B_pol_face=B_pol_face,
          B_tor_face=B_tor_face,
          geo=geo,
      )
  )

  v_ExB = _calculate_v_ExB(Er.face_value(), B_total_face)
  v_ExB_poloidal_and_pressure = _calculate_v_ExB(
      Er_poloidal_and_pressure_face, B_total_face
  )
  v_ExB_toroidal = _calculate_v_ExB(Er_toroidal_face, B_total_face)

  return RotationOutput(
      v_ExB=v_ExB,
      v_ExB_poloidal_and_pressure=v_ExB_poloidal_and_pressure,
      v_ExB_toroidal=v_ExB_toroidal,
      Er=Er,
      poloidal_velocity=poloidal_velocity,
  )
