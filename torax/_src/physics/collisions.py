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
    - _calculate_log_lambda_ei: Calculates the Coulomb logarithm for
    electron-ion
      collisions.
    - calculate_log_lambda_ii: Calculates the Coulomb logarithm for ion-ion
      collisions.
    - _calculate_weighted_Z_eff: Calculates ion mass weighted Z_eff used in
      the equipartion calculation.
    - _calculate_log_tau_e_Z1: Calculates log of electron-ion collision time for
      Z=1 plasma.
"""

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.geometry import geometry

# pylint: disable=invalid-name


def coll_exchange(
    core_profiles: state.CoreProfiles,
    Qei_multiplier: float,
) -> jax.Array:
  """Computes collisional ion-electron heat exchange coefficient (equipartion).

  Args:
    core_profiles: Core plasma profiles.
    Qei_multiplier: multiplier for ion-electron heat exchange term

  Returns:
    Qei_coeff: ion-electron collisional heat exchange coefficient.
  """
  # Calculate Coulomb logarithm
  log_lambda_ei = calculate_log_lambda_ei(
      core_profiles.T_e.value, core_profiles.n_e.value
  )
  # ion-electron collisionality for Z_eff=1. Ion charge and multiple ion effects
  # are included in the Qei_coef calculation below.
  log_tau_e_Z1 = _calculate_log_tau_e_Z1(
      core_profiles.T_e.value,
      core_profiles.n_e.value,
      log_lambda_ei,
  )
  # pylint: disable=invalid-name

  weighted_Z_eff = _calculate_weighted_Z_eff(core_profiles)

  log_Qei_coef = (
      jnp.log(Qei_multiplier * 1.5 * core_profiles.n_e.value)
      + jnp.log(constants.CONSTANTS.keV2J / constants.CONSTANTS.mp)
      + jnp.log(2 * constants.CONSTANTS.me)
      + jnp.log(weighted_Z_eff)
      - log_tau_e_Z1
  )
  Qei_coef = jnp.exp(log_Qei_coef)
  return Qei_coef


def calc_nu_star(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    collisionality_multiplier: float,
) -> jax.Array:
  """Calculates nustar.

    Electron-ion collision frequency normalized by bounce frequency.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.
    collisionality_multiplier: Collisionality multiplier in QLKNN for
      sensitivity testing.

  Returns:
    nu_star: on face grid.
  """

  # Calculate Coulomb logarithm
  log_lambda_ei_face = calculate_log_lambda_ei(
      core_profiles.T_e.face_value(),
      core_profiles.n_e.face_value(),
  )

  # ion_electron collisionality
  log_tau_e_Z1 = _calculate_log_tau_e_Z1(
      core_profiles.T_e.face_value(),
      core_profiles.n_e.face_value(),
      log_lambda_ei_face,
  )

  nu_e = (
      1
      / jnp.exp(log_tau_e_Z1)
      * core_profiles.Z_eff_face
      * collisionality_multiplier
  )

  # calculate bounce time
  epsilon = geo.rho_face / geo.R_major
  # to avoid divisions by zero
  epsilon = jnp.clip(epsilon, constants.CONSTANTS.eps)
  tau_bounce = (
      core_profiles.q_face
      * geo.R_major
      / (
          epsilon**1.5
          * jnp.sqrt(
              core_profiles.T_e.face_value()
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
    T_e: array_typing.ArrayFloat,
    fast_ion_mass: float,
) -> array_typing.ArrayFloat:
  """Returns the fraction of heating that goes to the ions.

  From eq. 5 and eq. 26 in Mikkelsen Nucl. Tech. Fusion 237 4 1983.
  Note there is a typo in eq. 26  where a `2x` term is missing in the numerator
  of the log.

  Args:
    birth_energy: Birth energy of the fast ions in keV.
    T_e: Electron temperature.
    fast_ion_mass: Mass of the fast ions in amu.

  Returns:
    The fraction of heating that goes to the ions.
  """
  critical_energy = 10 * fast_ion_mass * T_e  # Eq. 5.
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


def calculate_log_lambda_ei(
    T_e: jax.Array,
    n_e: jax.Array,
) -> jax.Array:
  """Calculates Coulomb logarithm for electron-ion collisions.

  See Wesson 3rd edition p727.

  Args:
    T_e: Electron temperature in keV.
    n_e: Electron density in m^-3.

  Returns:
    Coulomb logarithm.
  """
  # Rescale T_e to eV for specific form of formula.
  T_e_ev = T_e * 1e3
  return 31.3 - 0.5 * jnp.log(n_e) + jnp.log(T_e_ev)


def calculate_log_lambda_ii(
    T_i: jax.Array,
    n_i: jax.Array,
    Z_i: jax.Array,
) -> jax.Array:
  """Calculates Coulomb logarithm for ion-ion collisions.

  Formula 18e in Sauter PoP 1999. See also NRL formulary 2013, page 34.

  Args:
    T_i: Electron temperature in keV.
    n_i: Electron density in m^-3.
    Z_i: Ion charge [dimensionless].

  Returns:
    Coulomb logarithm.
  """
  # Rescale T_i to eV for specific form of formula.
  T_i_ev = T_i * 1e3
  return 30.0 - 0.5 * jnp.log(n_i) + 1.5 * jnp.log(T_i_ev) - 3.0 * jnp.log(Z_i)


# TODO(b/377225415): generalize to arbitrary number of ions.
def _calculate_weighted_Z_eff(
    core_profiles: state.CoreProfiles,
) -> jax.Array:
  """Calculates ion mass weighted Z_eff. Used for collisional heat exchange."""
  return (
      core_profiles.n_i.value * core_profiles.Z_i**2 / core_profiles.A_i
      + core_profiles.n_impurity.value
      * core_profiles.Z_impurity**2
      / core_profiles.A_impurity
  ) / core_profiles.n_e.value


def _calculate_log_tau_e_Z1(
    T_e: jax.Array,
    n_e: jax.Array,
    log_lambda_ei: jax.Array,
) -> jax.Array:
  """Calculates log of electron-ion collision time for Z=1 plasma.

  See Wesson 3rd edition p729. Extension to multiple ions is context dependent
  and implemented in calling functions.

  Args:
    T_e: Electron temperature in keV.
    n_e: Electron density in m^-3.
    log_lambda_ei: Coulomb logarithm.

  Returns:
    Log of electron-ion collision time.
  """
  return (
      jnp.log(12 * jnp.pi**1.5 / (n_e * log_lambda_ei))
      - 4 * jnp.log(constants.CONSTANTS.qe)
      + 0.5 * jnp.log(constants.CONSTANTS.me / 2.0)
      + 2 * jnp.log(constants.CONSTANTS.epsilon0)
      + 1.5 * jnp.log(T_e * constants.CONSTANTS.keV2J)
  )
