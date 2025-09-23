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

"""Implementation of the extended Lengyel model from Body et al. NF 2025."""

import dataclasses
from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_formulas

# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=unused-variable


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExtendedLengyelOutputs:
  """Outputs from the extended Lengyel model.

  Attributes:
    neutral_pressure_in_divertor: Neutral pressure in the divertor [Pa].
    alpha_t: Turbulence broadenign parameter alpha_t.
    q_parallel: Parallel heat flux [W/m^2].
    heat_flux_perp_to_target: Heat flux perpendicular to the target [W/m^2].
    separatrix_electron_temp: Electron temperature at the separatrix [keV].
    separatrix_z_effective: Z_eff at the separatrix.
    impurity_concentrations: A mapping from ion symbol to its n_e_ratio.
  """

  neutral_pressure_in_divertor: jax.Array
  alpha_t: jax.Array
  q_parallel: jax.Array
  heat_flux_perp_to_target: jax.Array
  separatrix_electron_temp: jax.Array
  separatrix_z_effective: jax.Array
  impurity_concentrations: Mapping[str, jax.Array]


# TODO(b/446608829)
# 1. Consider renaming variables to match the rest of TORAX
# 2. Consider repackaging the inputs into single or multiple dataclasses for
#    better readability.
def run_extended_lengyel_model(
    *,
    target_electron_temp: array_typing.FloatScalar,
    power_crossing_separatrix: array_typing.FloatScalar,
    separatrix_electron_density: array_typing.FloatScalar,
    seed_impurity_weights: Mapping[str, array_typing.FloatScalar],
    fixed_impurity_concentrations: Mapping[str, array_typing.FloatScalar],
    magnetic_field_on_axis: array_typing.FloatScalar,
    plasma_current: array_typing.FloatScalar,
    parallel_connection_length: array_typing.FloatScalar,
    divertor_parallel_length: array_typing.FloatScalar,
    major_radius: array_typing.FloatScalar,
    minor_radius: array_typing.FloatScalar,
    elongation_psi95: array_typing.FloatScalar,
    triangularity_psi95: array_typing.FloatScalar,
    average_ion_mass: array_typing.FloatScalar,
    divertor_broadening_factor: array_typing.FloatScalar = (
        extended_lengyel_defaults.DIVERTOR_BROADENING_FACTOR
    ),
    ratio_of_upstream_to_average_poloidal_field: array_typing.FloatScalar = (
        extended_lengyel_defaults.RATIO_UPSTREAM_TO_AVG_BPOL
    ),
    ne_tau: array_typing.FloatScalar = extended_lengyel_defaults.NE_TAU,
    sheath_heat_transmission_factor: array_typing.FloatScalar = (
        extended_lengyel_defaults.SHEATH_HEAT_TRANSMISSION_FACTOR
    ),
    target_angle_of_incidence: array_typing.FloatScalar = extended_lengyel_defaults.TARGET_ANGLE_OF_INCIDENCE,
    fraction_of_P_SOL_to_divertor: array_typing.FloatScalar = extended_lengyel_defaults.FRACTION_OF_PSOL_TO_DIVERTOR,
    SOL_conduction_fraction: array_typing.FloatScalar = extended_lengyel_defaults.SOL_CONDUCTION_FRACTION,
    ratio_of_molecular_to_ion_mass: array_typing.FloatScalar = extended_lengyel_defaults.RATIO_MOLECULAR_TO_ION_MASS,
    wall_temperature: array_typing.FloatScalar = extended_lengyel_defaults.WALL_TEMPERATURE,
    separatrix_mach_number: array_typing.FloatScalar = extended_lengyel_defaults.SEPARATRIX_MACH_NUMBER,
    separatrix_ratio_of_ion_to_electron_temp: array_typing.FloatScalar = (
        extended_lengyel_defaults.SEPARATRIX_RATIO_ION_TO_ELECTRON_TEMP
    ),
    separatrix_ratio_of_electron_to_ion_density: array_typing.FloatScalar = (
        extended_lengyel_defaults.SEPARATRIX_RATIO_ELECTRON_TO_ION_DENSITY
    ),
    target_ratio_of_ion_to_electron_temp: array_typing.FloatScalar = (
        extended_lengyel_defaults.TARGET_RATIO_ION_TO_ELECTRON_TEMP
    ),
    target_ratio_of_electron_to_ion_density: array_typing.FloatScalar = (
        extended_lengyel_defaults.TARGET_RATIO_ELECTRON_TO_ION_DENSITY
    ),
    target_mach_number: array_typing.FloatScalar = extended_lengyel_defaults.TARGET_MACH_NUMBER,
    toroidal_flux_expansion: array_typing.FloatScalar = extended_lengyel_defaults.TOROIDAL_FLUX_EXPANSION,
    inner_loop_iterations: int = extended_lengyel_defaults.INNER_LOOP_ITERATIONS,
    outer_loop_iterations: int = extended_lengyel_defaults.OUTER_LOOP_ITERATIONS,
) -> ExtendedLengyelOutputs:
  """Calculate the impurity concentration required for detachment.

  Args:
    target_electron_temp: Desired electron temperature at sheath entrance [eV].
    power_crossing_separatrix: Power crossing separatrix [W].
    separatrix_electron_density: Electron density at outboard midplane [m^-3].
    seed_impurity_weights: Mapping from ion symbol to fractions of seeded
      impurities. Total impurity n_e_ratio (c_z) is calculated by the model.
      c_z*seed_impurity_weights thus forms an output of the model.
    fixed_impurity_concentrations: Mapping from ion symbol to fixed
      concentrations (n_e_ratio) of background impurities.
    magnetic_field_on_axis: B-field at magnetic axis [T].
    plasma_current: Plasma current [A].
    parallel_connection_length: From target to outboard midplane [m].
    divertor_parallel_length: From target to X-point [m].
    major_radius: Major radius of magnetic axis [m].
    minor_radius: Minor radius from magnetic axis to outboard midplane [m].
    elongation_psi95: Elongation at psiN=0.95.
    triangularity_psi95: Triangularity at psiN=0.95.
    average_ion_mass: Average main-ion mass [amu].
    divertor_broadening_factor: lambda_INT / lambda_q.
    ratio_of_upstream_to_average_poloidal_field: Bpol_omp / Bpol_avg.
    ne_tau: Product of electron density and ion residence time [s m^-3].
    sheath_heat_transmission_factor: Sheath heat transmission factor gamma.
    target_angle_of_incidence: Angle between fieldline and target [degrees].
    fraction_of_P_SOL_to_divertor: Fraction of power to outer divertor.
    SOL_conduction_fraction: Fraction of power carried by conduction.
    ratio_of_molecular_to_ion_mass: Ratio of molecular to ion mass.
    wall_temperature: Divertor wall temperature [K].
    separatrix_mach_number: Mach number at separatrix.
    separatrix_ratio_of_ion_to_electron_temp: Ti/Te at separatrix.
    separatrix_ratio_of_electron_to_ion_density: ne/ni at separatrix.
    target_ratio_of_ion_to_electron_temp: Ti/Te at target.
    target_ratio_of_electron_to_ion_density: ne/ni at target.
    target_mach_number: Mach number at target.
    toroidal_flux_expansion: Toroidal flux expansion factor.
    inner_loop_iterations: Number of iterations for the inner loop.
    outer_loop_iterations: Number of iterations for the outer loop.

  Returns:
    An ExtendedLengyelOutputs object with the calculated values.
  """
  # WIP function body. Dummy return values for now.

  # --------------------------------------- #
  # ---------- 1. Pre-processing ---------- #
  # --------------------------------------- #

  shaping_factor = extended_lengyel_formulas.calc_shaping_factor(
      elongation_psi95=elongation_psi95,
      triangularity_psi95=triangularity_psi95,
  )
  separatrix_average_poloidal_field = (
      extended_lengyel_formulas.calc_separatrix_average_poloidal_field(
          plasma_current=plasma_current,
          minor_radius=minor_radius,
          shaping_factor=shaping_factor,
      )
  )
  cylindrical_safety_factor = (
      extended_lengyel_formulas.calc_cylindrical_safety_factor(
          magnetic_field_on_axis=magnetic_field_on_axis,
          separatrix_average_poloidal_field=separatrix_average_poloidal_field,
          shaping_factor=shaping_factor,
          minor_radius=minor_radius,
          major_radius=major_radius,
      )
  )
  fieldline_pitch_at_omp = extended_lengyel_formulas.calc_fieldline_pitch_at_omp(
      magnetic_field_on_axis=magnetic_field_on_axis,
      plasma_current=plasma_current,
      major_radius=major_radius,
      minor_radius=minor_radius,
      elongation_psi95=elongation_psi95,
      triangularity_psi95=triangularity_psi95,
      ratio_of_upstream_to_average_poloidal_field=ratio_of_upstream_to_average_poloidal_field,
  )
  electron_temp_at_cc_interface = (
      extended_lengyel_formulas.calc_electron_temp_at_cc_interface(
          target_electron_temp=target_electron_temp
      )
  )

  # We are considering the flux tube within the extent of the first lambda_q
  # (e-folding length) into the Scrape Off Layer (SOL) and divertor.

  fraction_of_power_entering_flux_tube = (
      1.0 - 1.0 / jnp.e
  ) * fraction_of_P_SOL_to_divertor

  # Initialize values for iterative solver.
  separatrix_electron_temp = jnp.array(0.1)
  alpha_t = jnp.array(0.0)
  divertor_Z_eff = 1.0

  # --------------------------------------- #
  # -------- 2. Iterative Solver----------- #
  # --------------------------------------- #

  # Not implemented yet.

  return ExtendedLengyelOutputs(
      neutral_pressure_in_divertor=jnp.array(0.0),
      alpha_t=alpha_t,
      q_parallel=jnp.array(0.0),
      heat_flux_perp_to_target=jnp.array(0.0),
      separatrix_electron_temp=separatrix_electron_temp,
      separatrix_z_effective=jnp.array(0.0),
      impurity_concentrations={},
  )
