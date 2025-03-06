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

"""Calculations related to derived quantities from poloidal flux (psi).

Functions:
    - update_jtot_q_face_s_face: Updates core profiles with
      psi-derived quantities.
    - calc_q: Calculates the q-profile (q).
    - calc_jtot: Calculate flux-surface-averaged toroidal current density.
    - calc_s: Calculates magnetic shear (s).
    - calc_s_rmid: Calculates magnetic shear (s), using midplane r as radial
      coordinate.
    - calc_Wpol: Calculates total magnetic energy (Wpol).
    - calc_li3: Calculates normalized internal inductance li3 (ITER convention).
    - calc_q95: Calculates the q-profile at 95% of the normalized poloidal flux.
    - calculate_psi_grad_constraint_from_Ip_tot: Calculates the gradient
      constraint on the poloidal flux (psi) from Ip.
    - _calc_bpol2: Calculates square of poloidal field (Bp).
"""

import dataclasses

import chex
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import constants
from torax import jax_utils
from torax import state
from torax.fvm import cell_variable
from torax.geometry import geometry

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


@jax_utils.jit
def update_jtot_q_face_s_face(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> state.CoreProfiles:
  """Updates core profiles with psi-derived quantities.

  Args:
    geo: Geometry object.
    core_profiles: Core plasma profiles.

  Returns:
    Updated core profiles with new jtot, jtot_face, Ip_profile_face, q_face,
    and s_face.
  """

  jtot, jtot_face, Ip_profile_face = calc_jtot(geo, core_profiles.psi)
  q_face, _ = calc_q(geo=geo, psi=core_profiles.psi)
  s_face = calc_s(geo, core_profiles.psi)

  currents = dataclasses.replace(
      core_profiles.currents,
      jtot=jtot,
      jtot_face=jtot_face,
      Ip_profile_face=Ip_profile_face,
  )
  new_core_profiles = dataclasses.replace(
      core_profiles,
      currents=currents,
      q_face=q_face,
      s_face=s_face,
  )
  return new_core_profiles


def calc_q(
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
) -> tuple[chex.Array, chex.Array]:
  """Calculates the q-profile (q) given current (jtot) and poloidal flux (psi).

  Args:
    geo: Magnetic geometry.
    psi: Poloidal flux.

  Returns:
    q_face: q at faces.
    q: q at cell centers.
  """
  # iota is standard terminology for 1/q
  inv_iota = jnp.abs(
      (2 * geo.Phib * geo.rho_face_norm[1:]) / psi.face_grad()[1:]
  )

  # Use L'Hôpital's rule to calculate iota on-axis, with psi_face_grad()[0]=0.
  inv_iota0 = jnp.expand_dims(
      jnp.abs((2 * geo.Phib * geo.drho_norm) / psi.face_grad()[1]), 0
  )

  q_face = jnp.concatenate([inv_iota0, inv_iota])
  q_face *= geo.q_correction_factor
  q = geometry.face_to_cell(q_face)

  return q_face, q


def calc_jtot(
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
) -> tuple[chex.Array, chex.Array, chex.Array]:
  """Calculate flux-surface-averaged toroidal current density from poloidal flux.

  Calculation based on jtot = dI/dS

  Args:
    geo: Torus geometry.
    psi: Poloidal flux.

  Returns:
    jtot: total current density [A/m2] on cell grid
    jtot_face: total current density [A/m2] on face grid
    Ip_profile_face: cumulative total plasma current profile [A] on face grid
  """

  # pylint: disable=invalid-name
  Ip_profile_face = (
      psi.face_grad()
      * geo.g2g3_over_rhon_face
      * geo.F_face
      / geo.Phib
      / (16 * jnp.pi**3 * constants.CONSTANTS.mu0)
  )

  dI_tot_drhon = jnp.gradient(Ip_profile_face, geo.rho_face_norm)

  jtot_face_bulk = dI_tot_drhon[1:] / geo.spr_face[1:]

  # Set on-axis jtot according to L'Hôpital's rule, noting that I[0]=S[0]=0.
  jtot_face_axis = Ip_profile_face[1] / geo.area_face[1]

  jtot_face = jnp.concatenate([jnp.array([jtot_face_axis]), jtot_face_bulk])
  jtot = geometry.face_to_cell(jtot_face)

  return jtot, jtot_face, Ip_profile_face


def calc_s(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  """Calculates magnetic shear (s) from poloidal flux (psi).

  Args:
    geo: Torus geometry.
    psi: Poloidal flux.

  Returns:
    s_face: Magnetic shear, on the face grid.
  """

  # iota (1/q) should have a /2*Phib but we drop it since will cancel out in
  # the s calculation.
  iota_scaled = jnp.abs((psi.face_grad()[1:] / geo.rho_face_norm[1:]))

  # on-axis iota_scaled from L'Hôpital's rule = dpsi_face_grad / drho_norm
  # Using expand_dims to make it compatible with jnp.concatenate
  iota_scaled0 = jnp.expand_dims(
      jnp.abs(psi.face_grad()[1] / geo.drho_norm), axis=0
  )

  iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])

  s_face = (
      -geo.rho_face_norm
      * jnp.gradient(iota_scaled, geo.rho_face_norm)
      / iota_scaled
  )

  return s_face


def calc_s_rmid(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  """Calculates magnetic shear (s) from poloidal flux (psi).

  Version taking the derivative of iota with respect to the midplane r,
  in line with expectations from circular-derived models like QuaLiKiz.

  Args:
    geo: Torus geometry.
    psi: Poloidal flux.

  Returns:
    s_face: Magnetic shear, on the face grid.
  """

  # iota (1/q) should have a /2*Phib but we drop it since will cancel out in
  # the s calculation.
  iota_scaled = jnp.abs((psi.face_grad()[1:] / geo.rho_face_norm[1:]))

  # on-axis iota_scaled from L'Hôpital's rule = dpsi_face_grad / drho_norm
  # Using expand_dims to make it compatible with jnp.concatenate
  iota_scaled0 = jnp.expand_dims(
      jnp.abs(psi.face_grad()[1] / geo.drho_norm), axis=0
  )

  iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])

  rmid_face = (geo.Rout_face - geo.Rin_face) * 0.5

  s_face = -rmid_face * jnp.gradient(iota_scaled, rmid_face) / iota_scaled

  return s_face


def _calc_bpol2(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  r"""Calculates square of poloidal field (Bp) from poloidal flux (psi).

  An identity for the poloidal magnetic field is:
  :math:`B_p = 1/R \partial \psi / \partial \rho (\nabla \rho \times e_\phi)`

  Where :math:`e_\phi` is the unit vector pointing in the toroidal direction.

  Args:
    geo: Torus geometry.
    psi: Poloidal flux.

  Returns:
    bpol2_face: Square of poloidal magnetic field, on the face grid.
  """
  bpol2_bulk = (
      (psi.face_grad()[1:] / (2 * jnp.pi)) ** 2
      * geo.g2_face[1:]
      / geo.vpr_face[1:] ** 2
  )
  bpol2_axis = jnp.array([0.0])
  bpol2_face = jnp.concatenate([bpol2_axis, bpol2_bulk])
  return bpol2_face


def calc_Wpol(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  """Calculates total magnetic energy (Wpol) from poloidal flux (psi)."""
  bpol2 = _calc_bpol2(geo, psi)
  Wpol = _trapz(bpol2 * geo.vpr_face, geo.rho_face_norm) / (
      2 * constants.CONSTANTS.mu0
  )
  return Wpol


def calc_li3(
    Rmaj: jax.Array,
    Wpol: jax.Array,
    Ip_total: jax.Array,
) -> jax.Array:
  """Calculates li3 based on a formulation using Wpol.

  Normalized internal inductance is defined as:
  li = <Bpol^2>_V / <Bpol^2>_LCFS where <>_V is a volume average and <>_LCFS is
  the average at the last closed flux surface.

  We use the ITER convention for normalized internal inductance defined as:
  li3 = 2*V*<Bpol^2>_V / (mu0^2 Ip^2*Rmaj) = 4 * Wpol / (mu0 Ip^2*Rmaj)

  Ip (total plasma current) enters through the integral form of Ampere's law.
  Since Wpol also corresponds to a volume integral of the poloidal field, we
  can define li3 with respect to Wpol.

  Args:
    Rmaj: Major radius.
    Wpol: Total magnetic energy.
    Ip_total: Total plasma current.

  Returns:
    li3: Normalized internal inductance, ITER convention.
  """
  return 4 * Wpol / (constants.CONSTANTS.mu0 * Ip_total**2 * Rmaj)


def calc_q95(
    psi_norm_face: array_typing.ArrayFloat,
    q_face: array_typing.ArrayFloat,
) -> array_typing.ScalarFloat:
  """Calculates q95 from the q profile and the normalized poloidal flux.

  Args:
    psi_norm_face: normalized poloidal flux
    q_face: safety factor on the face grid

  Returns:
    q95: q at 95% of the normalized poloidal flux
  """
  q95 = jnp.interp(0.95, psi_norm_face, q_face)

  return q95


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
