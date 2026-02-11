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
    Z_a: int,
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
  T_a_eV = T_a_keV * 1000.0
  T_b_eV = T_b_keV * 1000.0

  num = (
      1.8e-19
      * jnp.sqrt(m_a_amu * m_b_amu)
      * Z_a**2
      * Z_b**2
      * n_b_cm3
      * ln_lambda
  )
  denom = jnp.power(m_b_amu * T_a_eV + m_a_amu * T_b_eV, 1.5)
  return num / (denom + constants.CONSTANTS.eps)


def bimaxwellian_split(
    power_deposition: jax.Array,
    T_e: jax.Array,
    n_e: jax.Array,
    T_i: jax.Array,
    minority_concentration: jax.Array | float,
    P_total_W: float,
    charge_number: int,
    mass_number: float,
) -> tuple[jax.Array, jax.Array]:
  """Returns (n_tail, T_tail) using the Power Balance Closure.

  Splits a minority species density into a bulk thermal component and a
  high-energy tail component based on Stix theory power balance.

  Unlike the simplified Stix model, this implementation includes energy transfer
  to bulk ions via the NRL Formulary nu_epsilon rate, making it more accurate
  when T_tail is close to the critical energy.

  Args:
    power_deposition: Power deposition profile [MW/m^3 / MW_in]. Normalized per
      MW of input power.
    T_e: Electron temperature profile [keV].
    n_e: Electron density profile [m^-3].
    T_i: Ion temperature profile [keV].
    minority_concentration: Minority species fractional concentration
      (n_minority/n_e).
    P_total_W: Total absolute power absorbed [W].
    charge_number: Charge number of the minority species (e.g. 2 for He3).
    mass_number: Mass number of the minority species (e.g. 3.016 for He3).

  Returns:
    Tuple containing:
      n_tail: Density of the fast tail component [m^-3].
      T_tail: Temperature of the fast tail component [keV].
  """
  consts = constants.CONSTANTS

  n_total = n_e * minority_concentration

  p_dens_MW = power_deposition * (P_total_W / 1.0e6)
  p_abs_w = p_dens_MW * 1.0e6

  m_bulk = 2.014
  z_bulk = 1.0
  n_bulk = n_e
  me_amu = consts.m_e / consts.m_amu

  log_lam_e = collisions.calculate_log_lambda_ei(T_e, n_e)

  T_e_eV = T_e * 1000.0
  n_e_cm3 = n_e / 1.0e6

  tau_s = (6.27e8 * mass_number * jnp.power(T_e_eV, 1.5)) / (
      charge_number**2 * n_e_cm3 * log_lam_e + consts.eps
  )

  T_e_J = T_e * consts.keV_to_J
  xi = p_abs_w * tau_s / (1.5 * n_total * T_e_J + consts.eps)

  T_tail = T_e * (1.0 + xi)

  nu_tail_e = _nu_epsilon(
      mass_number,
      charge_number,
      T_tail,
      me_amu,
      1.0,
      n_e,
      T_e,
      log_lam_e,
  )

  log_lam_i = collisions.calculate_log_lambda_ii(
      T_tail, n_bulk, jnp.array(z_bulk)
  )
  nu_tail_i = _nu_epsilon(
      mass_number,
      charge_number,
      T_tail,
      m_bulk,
      z_bulk,
      n_bulk,
      T_i,
      log_lam_i,
  )

  energy_loss_per_particle = (
      1.5
      * consts.keV_to_J
      * (nu_tail_e * (T_tail - T_e) + nu_tail_i * (T_tail - T_i))
  )

  n_tail = math_utils.safe_divide(p_abs_w, energy_loss_per_particle)

  n_tail = jnp.clip(n_tail, 0.0, n_total * 0.99)

  T_tail = jnp.where(p_abs_w <= 1.0e-6, T_i, T_tail)

  return n_tail, T_tail
