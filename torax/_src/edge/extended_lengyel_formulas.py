# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper physics formulas for the extended Lengyel model."""

from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.edge import extended_lengyel_defaults

# pylint: disable=invalid-name


def _temperature_fit_function(
    target_electron_temp: array_typing.FloatScalar,
    params: extended_lengyel_defaults._FitParams,
) -> array_typing.FloatScalar:
  """A general form for divertor loss functions in terms of target temperature.

  Equation 33 from Stangeby, 2018, PPCF 60 044022.

  Args:
    target_electron_temp: Electron temperature at the target [eV].
    params: Fit parameters for the function.

  Returns:
    The value of the fit function.
  """
  return 1.0 - params.amplitude * jnp.power(
      1.0 - jnp.exp(-target_electron_temp / params.width), params.shape
  )


def calc_momentum_loss_in_convection_layer(
    target_electron_temp: array_typing.FloatScalar,
) -> array_typing.FloatScalar:
  """Calculates the momentum loss in the convection layer."""
  return _temperature_fit_function(
      target_electron_temp,
      extended_lengyel_defaults.TEMPERATURE_FIT_PARAMS['momentum_loss'],
  )


def calc_density_ratio_in_convection_layer(
    target_electron_temp: array_typing.FloatScalar,
) -> array_typing.FloatScalar:
  """Calculates the density ratio in the convection layer."""
  return _temperature_fit_function(
      target_electron_temp,
      extended_lengyel_defaults.TEMPERATURE_FIT_PARAMS['density_ratio'],
  )


def calc_power_loss_in_convection_layer(
    target_electron_temp: array_typing.FloatScalar,
) -> array_typing.FloatScalar:
  """Calculates the power loss in the convection layer."""
  return _temperature_fit_function(
      target_electron_temp,
      extended_lengyel_defaults.TEMPERATURE_FIT_PARAMS['power_loss'],
  )


def calc_alpha_t(
    separatrix_electron_density: array_typing.FloatScalar,
    separatrix_electron_temp: array_typing.FloatScalar,
    cylindrical_safety_factor: array_typing.FloatScalar,
    major_radius: array_typing.FloatScalar,
    average_ion_mass: array_typing.FloatScalar,
    Z_eff: array_typing.FloatScalar,
    mean_ion_charge_state: array_typing.FloatScalar,
    ion_to_electron_temp_ratio: array_typing.FloatScalar = 1.0,
) -> array_typing.FloatScalar:
  """Calculate the turbulence broadening parameter alpha_t.

  Equation 9 from T. Eich et al. Nuclear Fusion, 60(5), 056016. (2020),
  with an additional factor of an ion_to_electron_temp_ratio.

  Args:
    separatrix_electron_density: electron density at the separatrix [m^-3].
    separatrix_electron_temp: electron temperature at the separatrix [keV].
    cylindrical_safety_factor: cylindrical safety factor [dimensionless].
    major_radius: major radius [m].
    average_ion_mass: average ion mass [amu].
    Z_eff: effective ion charge [dimensionless].
    mean_ion_charge_state: mean ion charge state [dimensionless]. Defined as
      n_e/(sum_i n_i).
    ion_to_electron_temp_ratio: ratio of ion to electron temperature.

  Returns:
    alpha_t: the turbulence parameter alpha_t.
  """
  eV_to_J = constants.CONSTANTS.keV2J / 1000.0
  separatrix_electron_temp_ev = separatrix_electron_temp * 1e3
  average_ion_mass_kg = average_ion_mass * constants.CONSTANTS.m_amu

  # Variant from Verdoolaege et al., 2021 Nucl. Fusion 61 076006.
  # Differs from Wesson 3rd edition p727 by a small absolute value of 0.1.
  coulomb_logarithm = (
      30.9
      - 0.5 * jnp.log(separatrix_electron_density)
      + jnp.log(separatrix_electron_temp_ev)
  )

  # Plasma ion sound speed. Differs from that stated in Eich 2020 by the
  # inclusion of the mean ion charge state.
  ion_sound_speed = jnp.sqrt(
      mean_ion_charge_state
      * separatrix_electron_temp_ev
      * eV_to_J
      / average_ion_mass_kg
  )

  # electron-electron collision frequency. Equation B1 from Eich 2020.
  log_nu_ee = (
      jnp.log(4.0 / 3.0)
      + 0.5 * jnp.log(2.0 * jnp.pi)
      + jnp.log(separatrix_electron_density)
      + 4 * jnp.log(constants.CONSTANTS.qe)
      + jnp.log(coulomb_logarithm)
      - 2 * jnp.log(4.0 * jnp.pi * constants.CONSTANTS.epsilon0)
      - 0.5 * jnp.log(constants.CONSTANTS.me)
      - 1.5 * jnp.log(separatrix_electron_temp_ev * eV_to_J)
  )

  nu_ee = jnp.exp(log_nu_ee)

  # Z_eff correction to transform electron-electron collisions to ion-electron
  # collisions. Equation B2 in Eich 2020
  Z_eff_correction = (1.0 - 0.569) * jnp.exp(
      -(((Z_eff - 1.0) / 3.25) ** 0.85)
  ) + 0.569

  nu_ei = nu_ee * Z_eff_correction * Z_eff

  # Equation 9 from Eich 2020, with an additional factor of an
  # ion_to_electron_temp_ratio.
  alpha_t = (
      1.02
      * nu_ei
      / ion_sound_speed
      * (1.0 * constants.CONSTANTS.me / average_ion_mass_kg)
      * cylindrical_safety_factor**2
      * major_radius
      * (1.0 + ion_to_electron_temp_ratio / mean_ion_charge_state)
  )

  return alpha_t
