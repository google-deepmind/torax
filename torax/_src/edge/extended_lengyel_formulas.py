# Copyright 2025 DeepMind Technologies Limited
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

"""A collection of helper physics formulas for the extended Lengyel model.

This module provides self-contained functions for calculating various physical
quantities and empirical fits used throughout the extended Lengyel model. These
functions are the building blocks for the more complex, interconnected model
logic defined in `divertor_sol_1d.py`.

This includes:
- Empirical fits for momentum, density, and power loss in the convection layer.
- Formulas related to magnetic geometry (e.g., shaping factor, safety factor).
- Calculation of effective ion charge (Z_eff) based on impurity concentrations.
- Empirical scaling for divertor enrichment.
"""

from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.edge import collisional_radiative_models
from torax._src.edge import extended_lengyel_defaults

# pylint: disable=invalid-name


def _temperature_fit_function(
    T_e_target: array_typing.FloatScalar,
    params: extended_lengyel_defaults._FitParams,
) -> jax.Array:
  """A general form for divertor loss functions in terms of target temperature.

  Equation 33 from Stangeby, 2018, PPCF 60 044022.
  https://doi.org/10.1088/1361-6587/aaacf6

  Args:
    T_e_target: Electron temperature at the target [eV].
    params: Fit parameters for the function.

  Returns:
    The value of the fit function.
  """
  return 1.0 - params.amplitude * jnp.power(
      1.0 - jnp.exp(-T_e_target / params.width), params.shape
  )


def calc_momentum_loss_in_convection_layer(
    T_e_target: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the momentum loss in the convection layer."""
  return _temperature_fit_function(
      T_e_target,
      extended_lengyel_defaults.TEMPERATURE_FIT_PARAMS['momentum_loss'],
  )


def calc_density_ratio_in_convection_layer(
    T_e_target: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the ratio n_e_target/n_e_cc in the convection layer."""
  return _temperature_fit_function(
      T_e_target,
      extended_lengyel_defaults.TEMPERATURE_FIT_PARAMS['density_ratio'],
  )


def calc_power_loss_in_convection_layer(
    T_e_target: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the power loss in the convection layer."""
  return _temperature_fit_function(
      T_e_target,
      extended_lengyel_defaults.TEMPERATURE_FIT_PARAMS['power_loss'],
  )


def calc_shaping_factor(
    elongation_psi95: array_typing.FloatScalar,
    triangularity_psi95: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the separatrix flux surface shaping factor.

  Used for calculations related to magnetic geometry at the separatrix.

  See Equation 56 from T. Body et al 2025 Nucl. Fusion 65 086002,
  https://doi.org/10.1088/1741-4326/ade4d9

  and T. Eich et al. Nuclear Fusion 60 056016 (2020) for details.
  https://doi.org/10.1088/1741-4326/ab7a66

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
  https://doi.org/10.1088/1741-4326/ade4d9

  and T. Eich et al. Nuclear Fusion 60 056016 (2020) for details.
  https://doi.org/10.1088/1741-4326/ab7a66

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
  https://doi.org/10.1088/1741-4326/ade4d9

  and T. Eich et al. Nuclear Fusion 60 056016 (2020) for details.
  https://doi.org/10.1088/1741-4326/ab7a66

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
    ratio_bpol_omp_to_bpol_avg: array_typing.FloatScalar,
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
      ratio_bpol_omp_to_bpol_avg * separatrix_average_poloidal_field
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


def calc_Z_eff(
    *,
    c_z: array_typing.FloatScalar,
    T_e: array_typing.FloatScalar,
    Z_i: array_typing.FloatScalar,
    # TODO(b/434175938): (v2) Rename to n_e_tau for consistency.
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
    Z_impurity_per_species = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            T_e=jnp.array([T_e]),
            ne_tau=ne_tau,
            ion_symbol=key,
        )
    )
    Z_eff += Z_impurity_per_species**2 * c_z * weight
    dilution_factor += Z_impurity_per_species * c_z * weight
  # Contribution from fixed impurities, with n_e_ratio=concentration
  for key, concentration in fixed_impurity_concentrations.items():
    Z_impurity_per_species = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            T_e=jnp.array([T_e]),
            ne_tau=ne_tau,
            ion_symbol=key,
        )
    )
    Z_eff += Z_impurity_per_species**2 * concentration
    dilution_factor += Z_impurity_per_species * concentration
  # Contribution from main ions
  n_i = (1 - dilution_factor) / Z_i
  Z_eff += n_i * Z_i**2
  return Z_eff[0]  # Return scalar for extended-lengyel.


def calc_enrichment_kallenbach(
    pressure_neutral_divertor: array_typing.FloatScalar,
    ion_symbol: str,
    enrichment_multiplier: array_typing.FloatScalar = 1.0,
) -> jax.Array:
  """Calculate divertor enrichment according to regression from Kallenbach 2024.

  A. Kallenbach et al 2024 Nucl. Fusion 64 056003
  DOI: 10.1088/1741-4326/ad3139
  See figure 8.

  enrichment = 41.0 * Z^-0.5 * p0^-0.4 * (E_ionization_z / E_ionization_D)^-5.8

  Args:
    pressure_neutral_divertor: Divertor neutral pressure [Pa].
    ion_symbol: Symbol of the impurity ion.
    enrichment_multiplier: Multiplier to adjust the empirical scaling.

  Returns:
    Enrichment factor (c_divertor / c_core).
  """

  if ion_symbol not in constants.ION_PROPERTIES_DICT:
    raise ValueError(
        f'Invalid ion symbol in enrichment calculation: {ion_symbol}.'
        f' Allowed symbols are: {constants.ION_SYMBOLS}'
    )

  properties = constants.ION_PROPERTIES_DICT[ion_symbol]
  Z = properties.Z
  E_ionization_Z = properties.E_ionization
  E_ionization_D = constants.ION_PROPERTIES_DICT['D'].E_ionization

  # Avoid division by zero if pressure is very low
  p0 = jnp.maximum(pressure_neutral_divertor, constants.CONSTANTS.eps)

  enrichment = (
      41.0 * Z**-0.5 * p0**-0.4 * (E_ionization_Z / E_ionization_D) ** -5.8
  )

  return enrichment * enrichment_multiplier
