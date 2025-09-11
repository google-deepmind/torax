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

"""Collisional radiative model for charge states and cooling rates.

From A. A. Mavrin (2017):
Radiative Cooling Rates for Low-Z Impurities in Non-coronal Equilibrium State.
J Fusion Energ (2017) 36:161-172
DOI 10.1007/s10894-017-0136-z
"""

import enum
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src.edge import mavrin_2017_charge_states_data
from torax._src.edge import mavrin_2017_cooling_rate_data

# pylint: disable=invalid-name

_NE_TAU_CORONAL_LIMIT = 1e19


class MavrinVariable(enum.StrEnum):
  Z = 'Z'
  LZ = 'LZ'


def calculate_mavrin_2017(
    T_e: array_typing.FloatVector,
    ne_tau: array_typing.FloatScalar,
    ion_symbol: str,
    variable: MavrinVariable,
) -> array_typing.FloatVector:
  """Calculates the average charge state of an impurity based on a polynomial fit.

  Polynomial fit range is ~1 eV-15 keV, which is within the typical
  bounds of tokamak edge plasmas. For safety, inputs are clipped to avoid
  polynomial extrapolation outside the precise per-species ranges.

  Args:
    T_e: Electron temperature [keV].
    ne_tau: The non-coronal parameter, being the product of electron density and
      impurity residence time [m^-3 s]. High values of ne_tau correspond to the
      coronal limit.
    ion_symbol: Species to calculate average charge state for.
    variable: The variable to calculate, either 'Z' (charge states) or 'LZ'
      (radiative cooling rates).

  Returns:
    Either average charge states or cooling rate, depending on the variable.
  """

  match variable:
    case MavrinVariable.Z:
      cr_module = mavrin_2017_charge_states_data
    case MavrinVariable.LZ:
      cr_module = mavrin_2017_cooling_rate_data
    case _:
      allowed_variables = ', '.join([v.name for v in MavrinVariable])
      raise ValueError(
          f'Invalid fit variable: {variable}. Allowed variables are:'
          f' {allowed_variables}'
      )

  if ion_symbol not in cr_module.COEFFS.keys():
    raise ValueError(
        f'Invalid ion symbol: {ion_symbol}. Allowed symbols are:'
        f' {cr_module.COEFFS.keys()}'
    )

  # Mavrin 2017 formulas are constructed for [eV] temperature input
  T_e_ev = T_e * 1e3

  # Avoid extrapolating fitted polynomial out of bounds.
  min_temp = cr_module.MIN_TEMPERATURES[ion_symbol]
  max_temp = cr_module.MAX_TEMPERATURES[ion_symbol]
  T_e_ev = jnp.clip(T_e_ev, min_temp, max_temp)
  # Residence parameter capped at 10^19, which is the coronal limit.
  ne_tau = jnp.clip(ne_tau, a_max=_NE_TAU_CORONAL_LIMIT)

  # Gather coefficients for each temperature
  interval_indices = jnp.searchsorted(
      cr_module.TEMPERATURE_INTERVALS[ion_symbol], T_e_ev
  )
  coeffs_in_range = jnp.take(
      cr_module.COEFFS[ion_symbol], interval_indices, axis=1
  )

  X = jnp.log10(T_e_ev)
  Y = jnp.log10(ne_tau / _NE_TAU_CORONAL_LIMIT)

  # 2D polynomial from Mavrin 2017, Eq. 8
  log10_variable = (
      coeffs_in_range[0]
      + coeffs_in_range[1] * X
      + coeffs_in_range[2] * Y
      + coeffs_in_range[3] * X**2
      + coeffs_in_range[4] * X * Y
      + coeffs_in_range[5] * Y**2
      + coeffs_in_range[6] * X**3
      + coeffs_in_range[7] * X**2 * Y
      + coeffs_in_range[8] * X * Y**2
      + coeffs_in_range[9] * Y**3
  )

  return 10**log10_variable


def calculate_L_INT(
    start_temp: array_typing.FloatScalar,
    stop_temp: array_typing.FloatScalar,
    ne_tau: array_typing.FloatScalar,
    ion_symbol: str,
    resolution: int = 100,
) -> array_typing.FloatScalar:
  """Calculates the integral of Lz * sqrt(Te) from start_temp to stop_temp.

  This is the L_INT term from the extended Lengyel model.

  Equation 34 from T. Body, et al., Nucl. Fusion 65 (2025) 086002

  Args:
    start_temp: The starting electron temperature for the integration [keV].
    stop_temp: The stopping electron temperature for the integration [keV].
    ne_tau: The non-coronal parameter [m^-3 s].
    ion_symbol: The impurity species.
    resolution: The number of points to use in the numerical integration.

  Returns:
    The integrated cooling rate [eV^1.5 * m^3 * W].
  """
  # Convert start and stop temperatures to eV for the integration grid
  start_temp_eV = start_temp * 1e3
  stop_temp_eV = stop_temp * 1e3

  # Create a logarithmically spaced grid of electron temperatures in eV
  electron_temp_ev = jnp.logspace(
      jnp.log10(start_temp_eV), jnp.log10(stop_temp_eV), num=resolution
  )

  # Convert the grid back to keV to call the existing Mavrin model function
  electron_temp = electron_temp_ev / 1e3

  # Calculate Lz at each temperature point on the grid
  Lz_values = calculate_mavrin_2017(
      T_e=electron_temp,
      ne_tau=ne_tau,
      ion_symbol=ion_symbol,
      variable=MavrinVariable.LZ,
  )

  # The integrand is Lz * sqrt(Te)
  Lz_sqrt_Te = Lz_values * jnp.sqrt(electron_temp_ev)

  # Perform trapezoidal integration
  Lint = jax.scipy.integrate.trapezoid(y=Lz_sqrt_Te, x=electron_temp_ev)

  return Lint
