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

_DEUTERIUM_MASS_AMU = 2.014

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


def calculate_plh_scaling_factor(
    geo: Geometry,
    core_profiles: state.CoreProfiles,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Calculates the H-mode transition power scaling in low and high density branches.

  See Y.R. Martin and Tomonori Takizuka.
  "Power requirement for accessing the H-mode in ITER."
  Journal of Physics: Conference Series. Vol. 123. No. 1. IOP Publishing, 2008.

  Only valid for hydrogenic isotopes and mixtures (H, D, T).
  Includes a simple inverse scaling of the factor to average isotope mass.

  For an overview see U Plank, U., et al. "Overview of L-to H-mode transition
  experiments at ASDEX Upgrade."
  Plasma Physics and Controlled Fusion 65.1 (2022): 014001.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.

  Returns:
    Tuple of: P_LH scaling factor for high and low density branches, and the
      density corresponding to the P_LH minimum.
  """

  line_avg_ne = _calculate_line_avg_density(geo, core_profiles)
  # LH transition power for deuterium, in W. Eq 3 from Martin 2008.
  P_LH_hi_dens_D = (
      2.15
      * (line_avg_ne / 1e20) ** 0.782
      * geo.B0**0.772
      * geo.Rmin**0.975
      * geo.Rmaj**0.999
      * 1e6
  )

  # Scale to average isotope mass.
  P_LH_hi_dens = P_LH_hi_dens_D * _DEUTERIUM_MASS_AMU / core_profiles.Ai

  # Calculate low density branch of P_LH (in units of nref) from Eq 3 Ryter 2014
  ne_min_P_LH = (
      0.7
      * (core_profiles.currents.Ip_profile_face[-1] / 1e6) ** 0.34
      * geo.Rmin**-0.95
      * geo.B0**0.62
      * (geo.Rmaj / geo.Rmin) ** 0.4
      * 1e19
      / core_profiles.nref
  )
  P_LH_low_dens = (
      0.36
      * (core_profiles.currents.Ip_profile_face[-1] / 1e6) ** 0.27
      * geo.B0**1.25
      * geo.Rmaj**1.23
      * (geo.Rmaj / geo.Rmin) ** 0.08
      * 1e6
  )
  return P_LH_hi_dens, P_LH_low_dens, ne_min_P_LH


def calculate_scaling_law_confinement_time(
    geo: Geometry,
    core_profiles: state.CoreProfiles,
    Ploss: jax.Array,
    scaling_law: str,
) -> jax.Array:
  """Calculates the thermal energy confinement time for a given empirical scaling law.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.
    Ploss: Plasma power loss in MW.
    scaling_law: Scaling law to use.

  Returns:
    Thermal energy confinement time in s.
  """
  scaling_params = {
      'H98': {
          # H98 empirical confinement scaling law:
          # ITER Physics Expert Groups on Confinement and Transport and
          # Confinement Modelling and Database, Nucl. Fusion 39 2175, 1999
          # Doyle et al, Nucl. Fusion 47 (2007) S18–S127, Eq 30
          'prefactor': 0.0562,
          'Ip_exponent': 0.93,
          'B_exponent': 0.15,
          'line_avg_ne_exponent': 0.41,
          'Ploss_exponent': -0.69,
          'R_exponent': 1.97,
          'inverse_aspect_ratio_exponent': 0.58,
          'elongation_exponent': 0.78,
          'effective_mass_exponent': 0.19,
          'triangularity_exponent': 0.0,
      },
      'H97L': {
          # From the ITER L-mode confinement database.
          # S.M. Kaye et al 1997 Nucl. Fusion 37 1303, Eq 7
          'prefactor': 0.023,
          'Ip_exponent': 0.96,
          'B_exponent': 0.03,
          'line_avg_ne_exponent': 0.4,
          'Ploss_exponent': -0.73,
          'R_exponent': 1.83,
          'inverse_aspect_ratio_exponent': -0.06,
          'elongation_exponent': 0.64,
          'effective_mass_exponent': 0.20,
          'triangularity_exponent': 0.0,
      },
      'H20': {
          # Updated ITER H-mode confinement database, using full dataset.
          # G. Verdoolaege et al 2021 Nucl. Fusion 61 076006, Eq 7
          'prefactor': 0.053,
          'Ip_exponent': 0.98,
          'B_exponent': 0.22,
          'line_avg_ne_exponent': 0.24,
          'Ploss_exponent': -0.669,
          'R_exponent': 1.71,
          'inverse_aspect_ratio_exponent': 0.35,
          'elongation_exponent': 0.80,
          'effective_mass_exponent': 0.20,
          'triangularity_exponent': 0.36,  # (1+delta)^exponent
      },
  }

  if scaling_law not in scaling_params:
    raise ValueError(f'Unknown scaling law: {scaling_law}')

  params = scaling_params[scaling_law]

  Ip = core_profiles.currents.Ip_profile_face[-1] / 1e6  # in MA
  B = geo.B0
  line_avg_ne = _calculate_line_avg_density(geo, core_profiles) / 1e19
  R = geo.Rmaj
  inverse_aspect_ratio = geo.Rmin / geo.Rmaj

  # Effective elongation definition. This is a different definition than
  # the standard definition used in geo.elongation.
  elongation = geo.area_face[-1] / (jnp.pi * geo.Rmin**2)
  # TODO(b/317360834): extend when multiple ions are supported.
  effective_mass = core_profiles.Ai
  triangularity = geo.delta_face[-1]

  tau_scaling = (
      params['prefactor']
      * Ip**params['Ip_exponent']
      * B**params['B_exponent']
      * line_avg_ne**params['line_avg_ne_exponent']
      * Ploss**params['Ploss_exponent']
      * R**params['R_exponent']
      * inverse_aspect_ratio**params['inverse_aspect_ratio_exponent']
      * elongation**params['elongation_exponent']
      * effective_mass**params['effective_mass_exponent']
      * (1 + triangularity)**params['triangularity_exponent']
  )
  return tau_scaling


def _calculate_line_avg_density(
    geo: Geometry,
    core_profiles: state.CoreProfiles,
) -> jax.Array:
  """Calculates line-averaged electron density.

  Line-averaged electron density is poorly defined. In general, the definition
  is machine-dependent and even shot-dependent since it depends on the usage of
  a specific interferometry chord. Furthermore, even if we knew the specific
  chord used, its calculation would depend on magnetic geometry information
  beyond what is available in StandardGeometry. In lieu of a better solution, we
  use line-averaged electron density defined on the outer midplane.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.

  Returns:
    Line-averaged electron density.
  """
  Rmin_out = geo.Rout_face[-1] - geo.Rout_face[0]
  line_avg_ne = (
      core_profiles.nref
      * _trapz(core_profiles.ne.face_value(), geo.Rout_face)
      / Rmin_out
  )
  return line_avg_ne


# pylint: enable=invalid-name
