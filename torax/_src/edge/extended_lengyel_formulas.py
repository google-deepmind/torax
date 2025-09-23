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

from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.edge import collisional_radiative_models
from torax._src.edge import extended_lengyel_defaults

# pylint: disable=invalid-name


def _temperature_fit_function(
    target_electron_temp: array_typing.FloatScalar,
    params: extended_lengyel_defaults._FitParams,
) -> jax.Array:
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
) -> jax.Array:
  """Calculates the momentum loss in the convection layer."""
  return _temperature_fit_function(
      target_electron_temp,
      extended_lengyel_defaults.TEMPERATURE_FIT_PARAMS['momentum_loss'],
  )


def calc_density_ratio_in_convection_layer(
    target_electron_temp: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the ratio n_e_target/n_e_cc in the convection layer."""
  return _temperature_fit_function(
      target_electron_temp,
      extended_lengyel_defaults.TEMPERATURE_FIT_PARAMS['density_ratio'],
  )


def calc_power_loss_in_convection_layer(
    target_electron_temp: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the power loss in the convection layer."""
  return _temperature_fit_function(
      target_electron_temp,
      extended_lengyel_defaults.TEMPERATURE_FIT_PARAMS['power_loss'],
  )


def calc_shaping_factor(
    elongation_psi95: array_typing.FloatScalar,
    triangularity_psi95: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the separatrix flux surface shaping factor.

  Used for calculations related to magnetic geometry at the separatrix.

  See Equation 56 from T. Body et al 2025 Nucl. Fusion 65 086002,
  and T. Eich et al. Nuclear Fusion 60 056016 (2020) for details.

  Args:
    elongation_psi95: Elongation at psiN=0.95.
    triangularity_psi95: Triangularity at psiN=0.95.

  Returns:
    The flux surface shaping factor.
  """
  return jnp.sqrt(
      (
          1.0
          + elongation_psi95**2
          * (1.0 + 2.0 * triangularity_psi95**2 - 1.2 * triangularity_psi95**3)
      )
      / 2.0
  )


def calc_separatrix_average_poloidal_field(
    plasma_current: array_typing.FloatScalar,
    minor_radius: array_typing.FloatScalar,
    shaping_factor: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the average poloidal field at the separatrix.

  Used for calculations related to magnetic geometry at the separatrix.

  See equation 52 from T. Body et al 2025 Nucl. Fusion 65 086002,
  and T. Eich et al. Nuclear Fusion 60 056016 (2020) for details.

  Args:
    plasma_current: Plasma current [A].
    minor_radius: Minor radius from magnetic axis to outboard midplane [m].
    shaping_factor: Flux surface shaping factor.

  Returns:
    The average poloidal field at the separatrix [T].
  """
  poloidal_circumference = 2.0 * jnp.pi * minor_radius * shaping_factor
  return constants.CONSTANTS.mu_0 * plasma_current / poloidal_circumference


def calc_cylindrical_safety_factor(
    magnetic_field_on_axis: array_typing.FloatScalar,
    separatrix_average_poloidal_field: array_typing.FloatScalar,
    shaping_factor: array_typing.FloatScalar,
    minor_radius: array_typing.FloatScalar,
    major_radius: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the cylindrical safety factor.

  The cylindrical safety factor is a characteristic safety-factor value at the
  plasma edge, used as part of the determination of turbulence drive for the
  turbulence broadening parameter alpha_t.

  See equation 55 from T. Body et al 2025 Nucl. Fusion 65 086002,
  and T. Eich et al. Nuclear Fusion 60 056016 (2020) for details.

  Args:
    magnetic_field_on_axis: B-field at magnetic axis [T].
    separatrix_average_poloidal_field: Average poloidal magnetic field at the
      separatrix [T].
    shaping_factor: Flux surface shaping factor.
    minor_radius: Minor radius from magnetic axis to outboard midplane [m].
    major_radius: Major radius of magnetic axis [m].

  Returns:
    The cylindrical safety factor.
  """
  return jnp.array(
      magnetic_field_on_axis
      / separatrix_average_poloidal_field
      * minor_radius
      / major_radius
      * shaping_factor
  )


def calc_fieldline_pitch_at_omp(
    magnetic_field_on_axis: array_typing.FloatScalar,
    plasma_current: array_typing.FloatScalar,
    major_radius: array_typing.FloatScalar,
    minor_radius: array_typing.FloatScalar,
    elongation_psi95: array_typing.FloatScalar,
    triangularity_psi95: array_typing.FloatScalar,
    ratio_of_upstream_to_average_poloidal_field: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the fieldline pitch at the outboard midplane."""
  consts = constants.CONSTANTS

  # Calibrated shape factor for calculating flux surface circumference.
  # Body NF 2025 Eq 56.
  shaping_factor = jnp.sqrt(
      (
          1.0
          + elongation_psi95**2
          * (1.0 + 2.0 * triangularity_psi95**2 - 1.2 * triangularity_psi95**3)
      )
      / 2.0
  )

  # Body NF 2025 Eq 52. Note the paper equation has a typo. Using minor radius
  # is correct.
  poloidal_circumference = 2.0 * jnp.pi * minor_radius * shaping_factor
  separatrix_average_poloidal_field = (
      consts.mu_0 * plasma_current / poloidal_circumference
  )

  upstream_poloidal_field = (
      ratio_of_upstream_to_average_poloidal_field
      * separatrix_average_poloidal_field
  )

  # Using 1/R dependence of toroidal field.
  upstream_toroidal_field = magnetic_field_on_axis * (
      major_radius / (major_radius + minor_radius)
  )

  # Calculate pitch at omp as ratio of total to poloidal field.
  return (
      jnp.sqrt(upstream_toroidal_field**2 + upstream_poloidal_field**2)
      / upstream_poloidal_field
  )


def calc_electron_temp_at_cc_interface(
    target_electron_temp: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the electron temperature at the convection/conduction interface.

  This function determines the electron temperature at the boundary between the
  convection-dominated sheath/divertor region and the upstream conduction layer.
  It uses empirical fit functions for momentum and density loss within the
  convection layer, which depend on the electron temperature at the divertor
  target. The formula relates the interface temperature to the target
  temperature modified by these loss factors.

  See section 4 of T. Body et al 2025 Nucl. Fusion 65 086002 for details.

  Args:
    target_electron_temp: Electron temperature at the divertor target [eV].

  Returns:
    The electron temperature at the convection/conduction interface [eV].
  """
  momentum_loss = calc_momentum_loss_in_convection_layer(target_electron_temp)
  density_loss = calc_density_ratio_in_convection_layer(target_electron_temp)
  return target_electron_temp / ((1.0 - momentum_loss) / (2.0 * density_loss))


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
      * constants.CONSTANTS.eV_to_J
      / average_ion_mass_kg
  )

  # electron-electron collision frequency. Equation B1 from Eich 2020.
  log_nu_ee = (
      jnp.log(4.0 / 3.0)
      + 0.5 * jnp.log(2.0 * jnp.pi)
      + jnp.log(separatrix_electron_density)
      + 4 * jnp.log(constants.CONSTANTS.q_e)
      + jnp.log(coulomb_logarithm)
      - 2 * jnp.log(4.0 * jnp.pi * constants.CONSTANTS.epsilon_0)
      - 0.5 * jnp.log(constants.CONSTANTS.m_e)
      - 1.5 * jnp.log(separatrix_electron_temp_ev * constants.CONSTANTS.eV_to_J)
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
      * (1.0 * constants.CONSTANTS.m_e / average_ion_mass_kg)
      * cylindrical_safety_factor**2
      * major_radius
      * (1.0 + ion_to_electron_temp_ratio / mean_ion_charge_state)
  )

  return alpha_t


def calculate_q_parallel(
    *,
    separatrix_electron_temp: array_typing.FloatScalar,
    average_ion_mass: array_typing.FloatScalar,
    separatrix_average_poloidal_field: array_typing.FloatScalar,
    alpha_t: array_typing.FloatScalar,
    ratio_of_upstream_to_average_poloidal_field: array_typing.FloatScalar,
    fraction_of_PSOL_to_divertor: array_typing.FloatScalar,
    major_radius: array_typing.FloatScalar,
    minor_radius: array_typing.FloatScalar,
    power_crossing_separatrix: array_typing.FloatScalar,
    fieldline_pitch_at_omp: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the parallel heat flux density.

  For the flux-tube assumed in the extended Lengyel model.
  See T. Body et al 2025 Nucl. Fusion 65 086002 for details.


  Args:
    separatrix_electron_temp: Electron temperature at the separatrix [eV].
    average_ion_mass: Average ion mass [amu].
    separatrix_average_poloidal_field: Average poloidal magnetic field at the
      separatrix [T].
    alpha_t: Turbulence broadening parameter alpha_t.
    ratio_of_upstream_to_average_poloidal_field: Bpol_omp / Bpol_avg.
    fraction_of_PSOL_to_divertor: Fraction of PSOL to divertor [dimensionless].
    major_radius: Major radius [m].
    minor_radius: Minor radius [m].
    power_crossing_separatrix: Power crossing the separatrix [W].
    fieldline_pitch_at_omp: Ratio of total to poloidal magnetic field at the
      outboard midplane.

  Returns:
    q_parallel: Parallel heat flux density [W/m^2].
  """

  # Body NF 2025 Eq 53.
  separatrix_average_rho_s_pol = (
      jnp.sqrt(
          separatrix_electron_temp
          * average_ion_mass
          * constants.CONSTANTS.m_amu
          / constants.CONSTANTS.q_e
      )
      / separatrix_average_poloidal_field
  )

  # Body NF 2025 Eq 49.
  separatrix_average_lambda_q = (
      0.6 * (1.0 + 2.1 * alpha_t**1.7) * separatrix_average_rho_s_pol
  )

  # Scaling lambda_q by the scalings of the average and upstream toroidal and
  # poloidal fields. Body NF 2025 Eq 50.
  ratio_of_upstream_to_average_lambda_q = (
      ratio_of_upstream_to_average_poloidal_field
      * (major_radius + minor_radius)
      / major_radius
  )
  lambda_q_outboard_midplane = (
      separatrix_average_lambda_q / ratio_of_upstream_to_average_lambda_q
  )

  # Power reduction for the fraction of power inside one e-folding length
  # (lambda_q).
  fraction_of_power_entering_flux_tube = (
      1.0 - 1.0 / jnp.e
  ) * fraction_of_PSOL_to_divertor

  # Parallel heat flux at the target.
  # Body NF 2025 Eq 48.

  q_parallel = (
      power_crossing_separatrix
      * fraction_of_power_entering_flux_tube
      / (
          2.0
          * jnp.pi
          * (major_radius + minor_radius)
          * lambda_q_outboard_midplane
      )
      * fieldline_pitch_at_omp
  )

  return q_parallel


def calc_Z_eff(
    *,
    c_z: array_typing.FloatScalar,
    T_e: array_typing.FloatScalar,
    Z_i: array_typing.FloatScalar,
    ne_tau: array_typing.FloatScalar,
    seed_impurity_weights: Mapping[str, array_typing.FloatScalar],
    fixed_impurity_concentrations: Mapping[str, array_typing.FloatScalar],
) -> jax.Array:
  """Helper function to calculate Z_eff in the extended Lengyel model.

  Z_eff is the effective ion charge, defined as sum(n_i * Z_i^2) / n_e.
  This function calculates Z_eff based on contributions from a background plasma
  (with Z=Zi) and specified seeded and fixed impurities, using the Mavrin 2017
  collisional-radiative model to determine the mean charge state of each
  impurity species. Quasineutrality is also used as a constraint, where
  sum(n_i * Z_i)/n_e = 1.

  Args:
    c_z: Concentration of the total seeded impurity species.
    T_e: Electron temperature [keV].
    Z_i: Main ion charge.
    ne_tau: The non-coronal parameter, being the product of electron density and
      impurity residence time [m^-3 s].
    seed_impurity_weights: Mapping from ion symbol (e.g., 'C') to its relative
      weight within the seeded impurity mix.
    fixed_impurity_concentrations: Mapping from ion symbol to its absolute
      concentration (n_z / n_e).

  Returns:
    The effective ion charge Z_eff [dimensionless].
  """
  # Initializations
  Z_eff = 0.0
  dilution_factor = 0.0
  # Contribution from seeded impurities, with n_e_ratio = c_z*weight.
  for key, weight in seed_impurity_weights.items():
    Z_impurity_per_species = collisional_radiative_models.calculate_mavrin_2017(
        T_e=jnp.array([T_e]),
        ne_tau=ne_tau,
        ion_symbol=key,
        variable=collisional_radiative_models.MavrinVariable.Z,
    )
    Z_eff += Z_impurity_per_species**2 * c_z * weight
    dilution_factor += Z_impurity_per_species * c_z * weight
  # Contribution from fixed impurities, with n_e_ratio=concentration
  for key, concentration in fixed_impurity_concentrations.items():
    Z_impurity_per_species = collisional_radiative_models.calculate_mavrin_2017(
        T_e=jnp.array([T_e]),
        ne_tau=ne_tau,
        ion_symbol=key,
        variable=collisional_radiative_models.MavrinVariable.Z,
    )
    Z_eff += Z_impurity_per_species**2 * concentration
    dilution_factor += Z_impurity_per_species * concentration
  # Contribution from main ions
  n_i = (1 - dilution_factor) / Z_i
  Z_eff += n_i * Z_i**2
  return Z_eff[0]  # Return scalar for extended-lengyel.
