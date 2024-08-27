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

"""Physics calculations.

This module contains problem-specific calculations that set up e.g.
coefficients on terms in a differential equation, as opposed to more
general differential equation solver functionality.
"""
import dataclasses

import chex
import jax
from jax import numpy as jnp
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import state
from torax.fvm import cell_variable
from torax.geometry import Geometry  # pylint: disable=g-importing-member

_trapz = jax.scipy.integrate.trapezoid


# Many variable names in this file use scientific or mathematical notation, so
# disable pylint complaints.
# pylint: disable=invalid-name


def get_main_ion_dilution_factor(
    Zimp: float,
    Zeff: float,
) -> float:
  return (Zimp - Zeff) / (Zimp - 1)


@jax_utils.jit
def update_jtot_q_face_s_face(
    geo: Geometry,
    core_profiles: state.CoreProfiles,
    q_correction_factor: float,
) -> state.CoreProfiles:
  """Updates jtot, jtot_face, q_face, and s_face."""

  jtot, jtot_face = calc_jtot_from_psi(
      geo,
      core_profiles.psi,
  )
  q_face, _ = calc_q_from_jtot_psi(
      geo=geo,
      psi=core_profiles.psi,
      jtot_face=jtot_face,
      q_correction_factor=q_correction_factor,
  )
  s_face = calc_s_from_psi(
      geo,
      core_profiles.psi,
  )
  currents = dataclasses.replace(
      core_profiles.currents, jtot=jtot, jtot_face=jtot_face
  )
  new_core_profiles = dataclasses.replace(
      core_profiles,
      currents=currents,
      q_face=q_face,
      s_face=s_face,
  )
  return new_core_profiles


def coll_exchange(
    core_profiles: state.CoreProfiles,
    nref: float,
    Ai: float,
    Qei_mult: float,
) -> jax.Array:
  """Computes collisional ion-electron heat exchange coefficient.

  Args:
    core_profiles: Core plasma profiles.
    nref: Reference value for normalization
    Ai: amu of main ion (if multiple isotope, make average)
    Qei_mult: multiplier for ion-electron heat exchange term

  Returns:
    Qei_coeff: ion-electron collisional heat exchange coefficient.
  """
  n_scale = nref / 1e20
  lam_ei = (
      15.2
      - 0.5 * jnp.log(core_profiles.ne.value * n_scale)
      + jnp.log(core_profiles.temp_el.value)
  )
  # collisionality
  log_tau_e = (
      jnp.log(12 * jnp.pi**1.5 / (core_profiles.ne.value * nref * lam_ei))
      - 4 * jnp.log(constants.CONSTANTS.qe)
      + 0.5 * jnp.log(constants.CONSTANTS.me / 2.0)
      + 2 * jnp.log(constants.CONSTANTS.epsilon0)
      + 1.5 * jnp.log(core_profiles.temp_el.value * constants.CONSTANTS.keV2J)
  )
  # pylint: disable=invalid-name
  log_Qei_coef = (
      jnp.log(Qei_mult * 1.5 * core_profiles.ne.value * nref)
      + jnp.log(constants.CONSTANTS.keV2J / (Ai * constants.CONSTANTS.mp))
      + jnp.log(2 * constants.CONSTANTS.me)
      - log_tau_e
  )
  Qei_coef = jnp.exp(log_Qei_coef)
  return Qei_coef


def internal_boundary(
    geo: Geometry,
    Ped_top: jax.Array,
    set_pedestal: jax.Array,
) -> jax.Array:
  # Create Boolean mask FiPy CellVariable with True where the internal boundary
  # condition is
  # find index closest to pedestal top.
  idx = jnp.abs(geo.rho_norm - Ped_top).argmin()
  mask_np = jnp.zeros(len(geo.rho), dtype=bool)
  mask_np = jnp.where(set_pedestal, mask_np.at[idx].set(True), mask_np)
  return mask_np


def calc_q_from_jtot_psi(
    geo: Geometry,
    psi: cell_variable.CellVariable,
    jtot_face: jax.Array,
    q_correction_factor: float,
) -> tuple[chex.Array, chex.Array]:
  """Calculates the q-profile (q) given current (jtot) and poloidal flux (psi).

  We don't simply pass a `CoreProfiles` instance because this needs to be called
  before the first `CoreProfiles` is constructed; the output of this function is
  used to populate the `q_face` field of the first `CoreProfiles`.

  Args:
    geo: Magnetic geometry.
    psi: Poloidal flux.
    jtot_face: Total toroidal current density on face grid.
    q_correction_factor: ad-hoc fix for non-physical circular geometry model
      such that q(r=a) = 3 for standard ITER parameters;

  Returns:
    q_face: q at faces.
    q: q at cell centers.
  """
  # We calculate iota on the face grid but omit face 0, so inv_iota[0]
  # corresponds to face 1.
  # iota on face 0 is unused in this function, and would need to be implemented
  # as a special case.
  inv_iota = jnp.abs(
      (2 * geo.Phib * geo.rho_face_norm[1:]) / psi.face_grad()[1:]
  )
  # use on-axis definition of q (Wesson 2004, Eq 3.48)
  q0 = 2 * geo.B0 / (constants.CONSTANTS.mu0 * jtot_face[0] * geo.Rmaj)
  q0 = jnp.expand_dims(q0, 0)
  q_face = jnp.concatenate([q0, inv_iota])
  q_face *= jnp.where(
      geo.geometry_type == geometry.GeometryType.CIRCULAR.value,
      q_correction_factor,
      1,
  )
  q = geometry.face_to_cell(q_face)

  return q_face, q


def calc_jtot_from_psi(
    geo: Geometry,
    psi: cell_variable.CellVariable,
) -> tuple[chex.Array, chex.Array]:
  """Calculates current (jtot) from poloidal flux (psi).

  Args:
    geo: Torus geometry.
    psi: Poloidal flux.

  Returns:
    jtot: total current density (Amps / m^2) on cell grid
    jtot_face: total current density (Amps / m^2) on face grid
  """

  # inside flux surface on face grid
  # pylint: disable=invalid-name
  I_tot = (
      psi.face_grad()
      * geo.g2g3_over_rhon_face
      * geo.F_face
      / geo.Phib
      / (16 * jnp.pi**3 * constants.CONSTANTS.mu0)
  )

  dI_tot_drhon = jnp.gradient(I_tot, geo.rho_face_norm)

  jtot_face_bulk = dI_tot_drhon[1:] / geo.spr_face[1:]

  # For now set on-axis to the same as the second grid point, due to 0/0
  # division.
  jtot_face_axis = jtot_face_bulk[0]

  jtot_face = jnp.concatenate([jnp.array([jtot_face_axis]), jtot_face_bulk])
  jtot = geometry.face_to_cell(jtot_face)

  return jtot, jtot_face


def calc_s_from_psi(
    geo: Geometry, psi: cell_variable.CellVariable
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


def calc_s_from_psi_rmid(
    geo: Geometry, psi: cell_variable.CellVariable
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


def calc_nu_star(
    geo: Geometry,
    core_profiles: state.CoreProfiles,
    nref: float,
    Zeff: float,
    coll_mult: float,
) -> jax.Array:
  """Calculates nu star.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.
    nref: Reference value for normalization
    Zeff: Effective ion charge.
    coll_mult: Collisionality multiplier in QLKNN for sensitivity testing.

  Returns:
    nu_star: on face grid.
  """

  temp_electron_var = core_profiles.temp_el
  temp_electron_face = temp_electron_var.face_value()
  raw_ne_var = core_profiles.ne
  raw_ne_face = raw_ne_var.face_value()

  # Coulomb constant and collisionality. Wesson 2nd edition p661-663:
  # Lambde(:) = 15.2_DBL - 0.5_DBL*LOG(0.1_DBL*Nex(:)) + LOG(Tex(:))
  lambde = (
      15.2
      - 0.5 * jnp.log(0.1 * raw_ne_face / 1e20 * nref)
      + jnp.log(temp_electron_face)
  )
  # ion_electron collision formula
  nu_e = (
      1
      / 1.09e-3
      * Zeff
      * (raw_ne_face / 1e19 * nref)
      * lambde
      / (temp_electron_face) ** 1.5
      * coll_mult
  )

  # calculate bounce time
  epsilon = geo.rho_face / geo.Rmaj
  # to avoid divisions by zero
  epsilon = jnp.clip(epsilon, constants.CONSTANTS.eps)
  tau_bounce = (
      core_profiles.q_face
      * geo.Rmaj
      / (
          epsilon**1.5
          * jnp.sqrt(
              temp_electron_face
              * constants.CONSTANTS.keV2J
              / constants.CONSTANTS.me
          )
      )
  )
  # due to pathological on-axis epsilon=0 term
  tau_bounce = tau_bounce.at[0].set(tau_bounce[1])

  # calculate normalized collisionality
  nustar = nu_e * tau_bounce

  return nustar


# pylint: enable=invalid-name
