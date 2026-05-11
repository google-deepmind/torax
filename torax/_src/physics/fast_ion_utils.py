# Copyright 2026 DeepMind Technologies Limited
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

"""Fast ion utility functions."""

import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src import math_utils
from torax._src.physics import collisions


# pylint: disable=invalid-name


def _nu_epsilon(
    m_a_amu: float,
    Z_a: float,
    T_a_keV: jax.Array,
    m_b_amu: float,
    Z_b: float,
    n_b_m3: jax.Array,
    T_b_keV: jax.Array,
    ln_lambda: jax.Array,
) -> jax.Array:
  """NRL Formulary energy exchange rate nu_epsilon [Hz].

  See NRL Plasma Formulary, page 34.

  Args:
    m_a_amu: Mass of species a [amu].
    Z_a: Charge number of species a.
    T_a_keV: Temperature of species a [keV].
    m_b_amu: Mass of species b [amu].
    Z_b: Charge number of species b.
    n_b_m3: Density of species b [m^-3].
    T_b_keV: Temperature of species b [keV].
    ln_lambda: Coulomb logarithm.

  Returns:
    Energy exchange rate [Hz].
  """

  n_b_cm3 = n_b_m3 / 1.0e6
  T_a_ev = T_a_keV * 1000.0
  T_b_ev = T_b_keV * 1000.0

  # The formulary uses cgs units. We convert the constant to use amu for
  # masses to avoid tiny values which can cause numerical issues.
  coeff = 1.8e-19 / jnp.sqrt(constants.CONSTANTS.m_amu * 1e3)

  num = (
      coeff
      * jnp.sqrt(m_a_amu * m_b_amu)
      * Z_a**2
      * Z_b**2
      * n_b_cm3
      * ln_lambda
  )
  denom = jnp.power(m_b_amu * T_a_ev + m_a_amu * T_b_ev, 1.5)
  return jnp.asarray(math_utils.safe_divide(num=num, denom=denom, eps=1e-7))


def _compute_T_tail(
    P_density_W: jax.Array,
    T_e: jax.Array,
    n_e: jax.Array,
    n_total: jax.Array,
    charge_number: float,
    mass_number: float,
) -> jax.Array:
  """Computes the effective tail temperature via the Stix xi parameter.

  Uses the Spitzer slowing-down time on electrons (tau_s) and the Stix
  parameter xi to compute T_tail = T_e * (1 + xi) [Stix, Nuc. Fus. 1975].

  The slowing-down time uses:
    tau_s [s] = 6.27e8 * A * (T_e[eV])^1.5 / (Z^2 * n_e[cm^-3] * ln_lambda)
  [Stix, Plasma Physics 14, 367 (1972), formula 16].

  Args:
    P_density_W: Absolute power density [W/m^3].
    T_e: Electron temperature [keV].
    n_e: Electron density [m^-3].
    n_total: Total minority density [m^-3].
    charge_number: Charge number of the minority species.
    mass_number: Mass number of the minority species.

  Returns:
    T_tail: Effective tail temperature [keV].
  """
  log_lambda_ei = collisions.calculate_log_lambda_ei(T_e, n_e)

  T_e_eV = T_e * 1000.0
  n_e_cm3 = n_e / 1.0e6

  tau_s = math_utils.safe_divide(
      num=6.27e8 * mass_number * jnp.power(T_e_eV, 1.5),
      denom=charge_number**2 * n_e_cm3 * log_lambda_ei,
      eps=1e-7,
  )
  T_e_J = T_e * constants.CONSTANTS.keV_to_J
  energy_density = 1.5 * n_total * T_e_J
  # Accroding to Stix 1972 (page 374), the energy_slowing_down_time is half
  # the Spitzer slowing-down time.
  energy_slowing_down_time = 0.5 * tau_s

  xi = math_utils.safe_divide(
      num=P_density_W * energy_slowing_down_time, denom=energy_density, eps=1e-7
  )

  return T_e * (1.0 + xi)


def bimaxwellian_split(
    power_deposition: jax.Array,
    T_e: jax.Array,
    n_e: jax.Array,
    T_i: jax.Array,
    n_i: jax.Array,
    minority_concentration: jax.Array | float,
    P_total_W: float,
    charge_number: float,
    mass_number: float,
    bulk_ion_mass: float,
    Z_i: float,
    n_impurity: jax.Array,
    Z_impurity: float,
    A_impurity: float,
) -> tuple[jax.Array, jax.Array]:
  """Returns (n_tail, T_tail) using the Power Balance Closure.

  Splits a minority species density into a bulk thermal component and a
  high-energy tail component based on Stix theory power balance.

  Unlike the simplified Stix model, this implementation includes energy transfer
  to bulk ions and impurities via the NRL Formulary nu_epsilon rate, making it
  more accurate when T_tail is close to the critical energy.

  Args:
    power_deposition: Power deposition profile [MW/m^3 / MW_in]. Normalized per
      MW of input power.
    T_e: Electron temperature profile [keV].
    n_e: Electron density profile [m^-3].
    T_i: Ion temperature profile [keV].
    n_i: Main ion density profile [m^-3].
    minority_concentration: Minority species fractional concentration
      (n_minority/n_e).
    P_total_W: Total absolute power absorbed [W].
    charge_number: Charge number of the minority species (e.g. 2 for He3).
    mass_number: Mass number of the minority species (e.g. 3.016 for He3).
    bulk_ion_mass: Mass of the bulk main ion species [amu] (e.g. 2.014 for D).
    Z_i: Charge number of the bulk main ion species.
    n_impurity: Impurity density profile [m^-3].
    Z_impurity: Charge number of the impurity species.
    A_impurity: Mass number of the impurity species [amu].

  Returns:
    Tuple containing:
      n_tail: Density of the fast tail component [m^-3].
      T_tail: Temperature of the fast tail component [keV].
  """
  consts = constants.CONSTANTS

  n_total = n_e * minority_concentration

  P_density_W = power_deposition * (P_total_W)

  me_amu = consts.m_e / consts.m_amu

  T_tail = _compute_T_tail(
      P_density_W=P_density_W,
      T_e=T_e,
      n_e=n_e,
      n_total=n_total,
      charge_number=charge_number,
      mass_number=mass_number,
  )

  log_lambda_ei = collisions.calculate_log_lambda_ei(T_e, n_e)

  nu_tail_e = _nu_epsilon(
      mass_number,
      charge_number,
      T_tail,
      me_amu,
      1.0,
      n_e,
      T_e,
      log_lambda_ei,
  )

  log_lambda_ii = collisions.calculate_log_lambda_ii(
      T_tail, n_i, jnp.asarray(Z_i)
  )
  nu_tail_i = _nu_epsilon(
      mass_number,
      charge_number,
      T_tail,
      bulk_ion_mass,
      Z_i,
      n_i,
      T_i,
      log_lambda_ii,
  )

  log_lambda_impurity = collisions.calculate_log_lambda_ii(
      T_tail, jnp.maximum(n_impurity, 1.0), jnp.asarray(Z_impurity)
  )
  nu_tail_impurity = _nu_epsilon(
      mass_number,
      charge_number,
      T_tail,
      A_impurity,
      Z_impurity,
      n_impurity,
      T_i,
      log_lambda_impurity,
  )

  energy_loss_rate_per_particle = (
      1.5
      * consts.keV_to_J
      * (
          nu_tail_e * (T_tail - T_e)
          + nu_tail_i * (T_tail - T_i)
          + nu_tail_impurity * (T_tail - T_i)
      )
  )

  # TODO(b/512078510): Choose a reasonable eps value for safe_divide here.
  n_tail = math_utils.safe_divide(
      num=P_density_W, denom=energy_loss_rate_per_particle, eps=1e-7
  )
  n_tail = jnp.clip(n_tail, 0.0, n_total * 0.99)

  n_tail = jnp.where(P_density_W <= 1.0e-6, 0.0, n_tail)
  T_tail = jnp.where(P_density_W <= 1.0e-6, T_i, T_tail)

  return n_tail, T_tail
