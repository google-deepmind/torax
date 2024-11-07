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
from torax import array_typing
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
    Zeff: jax.Array,
) -> jax.Array:
  return (Zimp - Zeff) / (Zimp - 1)


@jax_utils.jit
def update_jtot_q_face_s_face(
    geo: Geometry,
    core_profiles: state.CoreProfiles,
    q_correction_factor: float,
) -> state.CoreProfiles:
  """Updates jtot, jtot_face, q_face, and s_face."""

  jtot, jtot_face, Ip_profile_face = calc_jtot_from_psi(
      geo,
      core_profiles.psi,
  )
  q_face, _ = calc_q_from_psi(
      geo=geo,
      psi=core_profiles.psi,
      q_correction_factor=q_correction_factor,
  )
  s_face = calc_s_from_psi(
      geo,
      core_profiles.psi,
  )
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


def coll_exchange(
    core_profiles: state.CoreProfiles,
    nref: float,
    Qei_mult: float,
) -> jax.Array:
  """Computes collisional ion-electron heat exchange coefficient.

  Args:
    core_profiles: Core plasma profiles.
    nref: Reference value for normalization
    Qei_mult: multiplier for ion-electron heat exchange term

  Returns:
    Qei_coeff: ion-electron collisional heat exchange coefficient.
  """
  # Calculate Coulomb logarithm
  lambda_ei = _calculate_lambda_ei(
      core_profiles.temp_el.value, core_profiles.ne.value * nref
  )
  # ion-electron collisionality for Zeff=1. Ion charge and multiple ion effects
  # are included in the Qei_coef calculation below.
  log_tau_e_Z1 = _calculate_log_tau_e_Z1(
      core_profiles.temp_el.value,
      core_profiles.ne.value * nref,
      lambda_ei,
  )
  # pylint: disable=invalid-name

  weighted_Zeff = _calculate_weighted_Zeff(core_profiles)

  log_Qei_coef = (
      jnp.log(Qei_mult * 1.5 * core_profiles.ne.value * nref)
      + jnp.log(constants.CONSTANTS.keV2J / constants.CONSTANTS.mp)
      + jnp.log(2 * constants.CONSTANTS.me)
      + jnp.log(weighted_Zeff)
      - log_tau_e_Z1
  )
  Qei_coef = jnp.exp(log_Qei_coef)
  return Qei_coef


# TODO(b/377225415): generalize to arbitrary number of ions.
def _calculate_weighted_Zeff(
    core_profiles: state.CoreProfiles,
) -> jax.Array:
  """Calculates ion mass weighted Zeff. Used for collisional heat exchange."""
  return (
      core_profiles.ni.value * core_profiles.Zi**2 / core_profiles.Ai
      + core_profiles.nimp.value * core_profiles.Zimp**2 / core_profiles.Aimp
  ) / core_profiles.ne.value


def _calculate_log_tau_e_Z1(
    temp_el: jax.Array,
    ne: jax.Array,
    lambda_ei: jax.Array,
) -> jax.Array:
  """Calculates log of electron-ion collision time for Z=1 plasma.

  See Wesson 3rd edition p729. Extension to multiple ions is context dependent
  and implemented in calling functions.

  Args:
    temp_el: Electron temperature in keV.
    ne: Electron density in m^-3.
    lambda_ei: Coulomb logarithm.

  Returns:
    Log of electron-ion collision time.
  """
  return (
      jnp.log(12 * jnp.pi**1.5 / (ne * lambda_ei))
      - 4 * jnp.log(constants.CONSTANTS.qe)
      + 0.5 * jnp.log(constants.CONSTANTS.me / 2.0)
      + 2 * jnp.log(constants.CONSTANTS.epsilon0)
      + 1.5 * jnp.log(temp_el * constants.CONSTANTS.keV2J)
  )


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


def calc_q_from_psi(
    geo: Geometry,
    psi: cell_variable.CellVariable,
    q_correction_factor: float,
) -> tuple[chex.Array, chex.Array]:
  """Calculates the q-profile (q) given current (jtot) and poloidal flux (psi).

  We don't simply pass a `CoreProfiles` instance because this needs to be called
  before the first `CoreProfiles` is constructed; the output of this function is
  used to populate the `q_face` field of the first `CoreProfiles`.

  Args:
    geo: Magnetic geometry.
    psi: Poloidal flux.
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

  # Use L'Hôpital's rule to calculate iota on-axis, with psi_face_grad()[0]=0.
  inv_iota0 = jnp.expand_dims(
      jnp.abs((2 * geo.Phib * geo.drho_norm) / psi.face_grad()[1]), 0
  )

  q_face = jnp.concatenate([inv_iota0, inv_iota])
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
) -> tuple[chex.Array, chex.Array, chex.Array]:
  """Calculates FSA toroidal current density (jtot) from poloidal flux (psi).

  Calculation based on jtot = dI/dS

  Args:
    geo: Torus geometry.
    psi: Poloidal flux.

  Returns:
    jtot: total current density [A/m2] on cell grid
    jtot_face: total current density [A/m2] on face grid
    Ip_profile_face: cumulative total plasma current profile [A] on face grid
  """

  # inside flux surface on face grid
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
    Zeff_face: jax.Array,
    coll_mult: float,
) -> jax.Array:
  """Calculates nu star.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.
    nref: Reference value for normalization
    Zeff_face: Effective ion charge on face grid.
    coll_mult: Collisionality multiplier in QLKNN for sensitivity testing.

  Returns:
    nu_star: on face grid.
  """

  # Calculate Coulomb logarithm
  lambda_ei_face = _calculate_lambda_ei(
      core_profiles.temp_el.face_value(), core_profiles.ne.face_value() * nref
  )

  # ion_electron collisionality
  log_tau_e_Z1 = _calculate_log_tau_e_Z1(
      core_profiles.temp_el.face_value(),
      core_profiles.ne.face_value() * nref,
      lambda_ei_face,
  )

  nu_e = 1 / jnp.exp(log_tau_e_Z1) * Zeff_face * coll_mult

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
              core_profiles.temp_el.face_value()
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


def _calculate_lambda_ei(
    temp_el: jax.Array,
    ne: jax.Array,
) -> jax.Array:
  """Calculates Coulomb logarithm for electron-ion collisions.

  See Wesson 3rd edition p727.

  Args:
    temp_el: Electron temperature in keV.
    ne: Electron density in m^-3.

  Returns:
    Coulomb logarithm.
  """
  return 15.2 - 0.5 * jnp.log(ne / 1e20) + jnp.log(temp_el)


def fast_ion_fractional_heating_formula(
    birth_energy: float | array_typing.ArrayFloat,
    temp_el: array_typing.ArrayFloat,
    fast_ion_mass: float,
) -> array_typing.ArrayFloat:
  """Returns the fraction of heating that goes to the ions.

  From eq. 5 and eq. 26 in Mikkelsen Nucl. Tech. Fusion 237 4 1983.
  Note there is a typo in eq. 26  where a `2x` term is missing in the numerator
  of the log.

  Args:
    birth_energy: Birth energy of the fast ions in keV.
    temp_el: Electron temperature.
    fast_ion_mass: Mass of the fast ions in amu.

  Returns:
    The fraction of heating that goes to the ions.
  """
  critical_energy = 10 * fast_ion_mass * temp_el  # Eq. 5.
  energy_ratio = birth_energy / critical_energy

  # Eq. 26.
  x_squared = energy_ratio
  x = jnp.sqrt(x_squared)
  frac_i = (
      2
      * (
          (1 / 6) * jnp.log((1.0 - x + x_squared) / (1.0 + 2.0 * x + x_squared))
          + (jnp.arctan((2.0 * x - 1.0) / jnp.sqrt(3)) + jnp.pi / 6)
          / jnp.sqrt(3)
      )
      / x_squared
  )
  return frac_i


# pylint: enable=invalid-name
