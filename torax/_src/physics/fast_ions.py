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

"""Fast ion physics classes."""

import dataclasses

import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src import math_utils
from torax._src.fvm import cell_variable


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FastIon:
  """State of a fast ion species.

  Attributes:
    species: Species name (e.g. 'He3').
    source: Source name (e.g. 'ICRH').
    n: Density [m^-3].
    T: Temperature [keV].
  """

  species: str = dataclasses.field(metadata={'static': True})
  source: str = dataclasses.field(metadata={'static': True})
  n: cell_variable.CellVariable
  T: cell_variable.CellVariable


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
  keV_to_J = constants.CONSTANTS.keV_to_J

  n_total = n_e * minority_concentration

  # Calculate T_tail (The high-energy slope from Stix)
  # power_deposition is normalized (per MW input).
  # Calculate absolute power density in MW/m^3.
  # P_total_W is in Watts, convert to MW: P_total_W / 1e6
  # Result p_dens_mw is in MW/m^3.
  p_dens_mw = power_deposition * (P_total_W / 1.0e6)

  n_e20 = n_e / 1.0e20

  # Stix parameter xi [Stix, Nucl. Fusion 15, 737 (1975)].
  # xi = (0.24 * sqrt(T_e) * A_f * p_dens) / (n_e20^2 * Z_f^2 * c_fast)
  xi = (0.24 * jnp.sqrt(T_e) * mass_number * p_dens_mw) / (
      n_e20**2 * charge_number**2 * minority_concentration
  )

  T_tail = T_e * (1.0 + xi)
  T_bulk = T_i

  # Spitzer slowing-down time on electrons:
  # tau_s = 6.27e8 * A_f * T_e[eV]^1.5 / (Z_f^2 * n_e[cm^-3] * ln_lambda)
  # Converting to T_e[keV] and n_e20 [1e20 m^-3] with ln_lambda ~ 15:
  # 6.27e8 * 1000^1.5 / 1e14 / 15 ~ 0.013.
  tau_s = (0.013 * mass_number * T_e**1.5) / (charge_number**2 * n_e20)

  # Solve for n_tail using Power Balance:
  # P_abs (W/m^3) = 1.5 * n_tail * (T_tail - T_bulk) * e / tau_s
  p_abs_w = p_dens_mw * 1.0e6  # MW/m^3 -> W/m^3

  dT_joules = (T_tail - T_bulk) * keV_to_J

  n_tail = math_utils.safe_divide(p_abs_w * tau_s, 1.5 * dT_joules)

  # Constraints and Particle Conservation
  # Tail density cannot exceed 99% of total He3
  n_tail = jnp.clip(n_tail, 0.0, n_total * 0.99)

  return n_tail, T_tail
