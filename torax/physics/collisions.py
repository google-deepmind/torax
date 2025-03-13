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

"""Physics calculations related to collisional quantities.

Functions:
    - coll_exchange: Computes the collisional ion-electron heat exchange
      coefficient (equipartion).
    - calc_nu_star: Calculates the nu_star parameter: the electron-ion collision
      frequency normalized by bounce frequency.
    - fast_ion_fractional_heating_formula: Returns the fraction of heating that
      goes to the ions according to Stix 1975 analyticlal formulas.
    - _calculate_lambda_ei: Calculates the Coulomb logarithm for electron-ion
      collisions.
    - _calculate_weighted_Zeff: Calculates ion mass weighted Zeff used in
      the equipartion calculation.
    - _calculate_log_tau_e_Z1: Calculates log of electron-ion collision time for
      Z=1 plasma.
"""

import jax
from jax import numpy as jnp
from torax import array_typing
from torax import constants
from torax import state
from torax.geometry import geometry

# pylint: disable=invalid-name


def coll_exchange(
    core_profiles: state.CoreProfiles,
    nref: float,
    Qei_mult: float,
) -> jax.Array:
  """Computes collisional ion-electron heat exchange coefficient (equipartion).

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


def calc_nu_star(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    nref: float,
    Zeff_face: jax.Array,
    coll_mult: float,
) -> jax.Array:
  """Calculates nustar.

    Electron-ion collision frequency normalized by bounce frequency.

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
