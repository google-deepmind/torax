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
import enum
from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.edge import collisional_radiative_models
from torax._src.edge import divertor_sol_1d as divertor_sol_1d_lib
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_formulas

# pylint: disable=invalid-name
# pylint: disable=unused-variable


# Scale factors for physics calculations to avoid numerical issues in fp32.
_LINT_SCALE_FACTOR = 1e30
_DENSITY_SCALE_FACTOR = 1e-20

# _LINT_SCALE_FACTOR * _DENSITY_SCALE_FACTOR**2
_LINT_K_INVERSE_SCALE_FACTOR = 1e-10


class SolveCzStatus(enum.IntEnum):
  """Status of the _solve_for_c_z_prefactor calculation.

  Attributes:
    SUCCESS: The calculation was successful.
    Q_DIV_SQUARED_NEGATIVE: q_div_squared was negative. This is unphysical and
      indicates that the required power loss is too high for the given plasma
      parameters. This can happen, for example, if the target temperature is too
      low for the given upstream heat flux.
  """

  SUCCESS = 0
  Q_DIV_SQUARED_NEGATIVE = 1


class ComputationMode(enum.StrEnum):
  """Computation modes for the extended Lengyel model.

  Attributes:
    FORWARD: Calculate impurity concentrations for a given target temperature.
    INVERSE: Calculate target temperature for a given impurity concentration.
  """

  FORWARD = 'forward'
  INVERSE = 'inverse'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExtendedLengyelOutputs:
  """Outputs from the extended Lengyel model.

  Attributes:
    target_electron_temp: Electron temperature at sheath entrance [eV].
    neutral_pressure_in_divertor: Neutral pressure in the divertor [Pa].
    alpha_t: Turbulence broadening factor alpha_t.
    q_parallel: Parallel heat flux [W/m^2].
    heat_flux_perp_to_target: Heat flux perpendicular to the target [W/m^2].
    separatrix_electron_temp: Electron temperature at the separatrix [keV].
    separatrix_Z_eff: Z_eff at the separatrix.
    impurity_concentrations: A mapping from ion symbol to its n_e_ratio.
  """

  target_electron_temp: jax.Array
  neutral_pressure_in_divertor: jax.Array
  alpha_t: jax.Array
  q_parallel: jax.Array
  heat_flux_perp_to_target: jax.Array
  separatrix_electron_temp: jax.Array
  separatrix_Z_eff: jax.Array
  impurity_concentrations: Mapping[str, jax.Array]


# TODO(b/446608829)
# 1. Consider renaming variables to match the rest of TORAX
# 2. Consider repackaging the inputs into single or multiple dataclasses for
#    better readability.
def run_extended_lengyel_model(
    *,
    power_crossing_separatrix: array_typing.FloatScalar,
    separatrix_electron_density: array_typing.FloatScalar,
    fixed_impurity_concentrations: Mapping[str, array_typing.FloatScalar],
    main_ion_charge: array_typing.FloatScalar,
    magnetic_field_on_axis: array_typing.FloatScalar,
    plasma_current: array_typing.FloatScalar,
    parallel_connection_length: array_typing.FloatScalar,
    divertor_parallel_length: array_typing.FloatScalar,
    major_radius: array_typing.FloatScalar,
    minor_radius: array_typing.FloatScalar,
    elongation_psi95: array_typing.FloatScalar,
    triangularity_psi95: array_typing.FloatScalar,
    average_ion_mass: array_typing.FloatScalar,
    target_electron_temp: array_typing.FloatScalar | None = None,
    seed_impurity_weights: Mapping[str, array_typing.FloatScalar] | None = None,
    computation_mode: ComputationMode = ComputationMode.FORWARD,
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
    iterations: int = extended_lengyel_defaults.ITERATIONS,
) -> ExtendedLengyelOutputs:
  """Calculate the impurity concentration required for detachment.

  Args:
    power_crossing_separatrix: Power crossing separatrix [W].
    separatrix_electron_density: Electron density at outboard midplane [m^-3].
    fixed_impurity_concentrations: Mapping from ion symbol to fixed
      concentrations (n_e_ratio) of background impurities.
    main_ion_charge: Average main ion charge [dimensionless].
    magnetic_field_on_axis: B-field at magnetic axis [T].
    plasma_current: Plasma current [A].
    parallel_connection_length: From target to outboard midplane [m].
    divertor_parallel_length: From target to X-point [m].
    major_radius: Major radius of magnetic axis [m].
    minor_radius: Minor radius from magnetic axis to outboard midplane [m].
    elongation_psi95: Elongation at psiN=0.95.
    triangularity_psi95: Triangularity at psiN=0.95.
    average_ion_mass: Average main-ion mass [amu].
    target_electron_temp: For inverse mode, desired electron temperature at
      sheath entrance [eV].
    seed_impurity_weights: For inverse mode, Mapping from ion symbol to
      fractions of seeded impurities. Total impurity n_e_ratio (c_z) is
      calculated by the model. c_z_prefactor*seed_impurity_weights thus forms an
      output of the model.
    computation_mode: The computation mode for the model. See ComputationMode
      for details.
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
    iterations: Number of iterations for fixed point solver.

  Returns:
    An ExtendedLengyelOutputs object with the calculated values.
  """
  # WIP function body. Dummy return values for now.

  # --------------------------------------- #
  # ---------- 1. Pre-processing ---------- #
  # --------------------------------------- #

  _validate_inputs_for_computation_mode(
      computation_mode, target_electron_temp, seed_impurity_weights
  )

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

  # Initialize values for iterative solver.
  alpha_t = 0.0
  c_z_prefactor = 0.0
  kappa_e = extended_lengyel_defaults.KAPPA_E_0
  q_parallel = 1e6  # arbitrary value for initialization.
  c_z = 0.0
  separatrix_Z_eff = 1.0

  sol_state = divertor_sol_1d_lib.DivertorSOL1D(
      q_parallel=q_parallel,
      alpha_t=alpha_t,
      c_z_prefactor=c_z_prefactor,
      kappa_e=kappa_e,
      seed_impurity_weights=seed_impurity_weights,
      fixed_impurity_concentrations=fixed_impurity_concentrations,
      ne_tau=ne_tau,
      main_ion_charge=main_ion_charge,
      target_electron_temp=target_electron_temp,
      SOL_conduction_fraction=SOL_conduction_fraction,
      divertor_broadening_factor=divertor_broadening_factor,
      divertor_parallel_length=divertor_parallel_length,
      parallel_connection_length=parallel_connection_length,
      separatrix_mach_number=separatrix_mach_number,
      separatrix_electron_density=separatrix_electron_density,
      separatrix_ratio_of_ion_to_electron_temp=separatrix_ratio_of_ion_to_electron_temp,
      separatrix_ratio_of_electron_to_ion_density=separatrix_ratio_of_electron_to_ion_density,
      average_ion_mass=average_ion_mass,
      sheath_heat_transmission_factor=sheath_heat_transmission_factor,
      target_mach_number=target_mach_number,
      target_ratio_of_ion_to_electron_temp=target_ratio_of_ion_to_electron_temp,
      target_ratio_of_electron_to_ion_density=target_ratio_of_electron_to_ion_density,
      toroidal_flux_expansion=toroidal_flux_expansion,
  )

  # --------------------------------------- #
  # -------- 2. Iterative Solver----------- #
  # --------------------------------------- #

  for _ in range(iterations):
    # Calculate new value of q_parallel modified by alpha_t broadening.
    sol_state.q_parallel = extended_lengyel_formulas.calculate_q_parallel(
        separatrix_electron_temp=sol_state.separatrix_electron_temp,
        average_ion_mass=sol_state.average_ion_mass,
        separatrix_average_poloidal_field=separatrix_average_poloidal_field,
        alpha_t=sol_state.alpha_t,
        ratio_of_upstream_to_average_poloidal_field=ratio_of_upstream_to_average_poloidal_field,
        fraction_of_PSOL_to_divertor=fraction_of_P_SOL_to_divertor,
        minor_radius=minor_radius,
        major_radius=major_radius,
        power_crossing_separatrix=power_crossing_separatrix,
        fieldline_pitch_at_omp=fieldline_pitch_at_omp,
    )

    # Solve for the impurity concentration required to achieve the target
    # temperature for a given q_parallel. This also updates the divertor and
    # separatrix Z_eff values in sol_state, used downstream.
    sol_state.c_z_prefactor, _ = _solve_for_c_z_prefactor(sol_state=sol_state)

    # Update alpha_t for the next loop iteration. Impacts q_parallel.
    sol_state.alpha_t = extended_lengyel_formulas.calc_alpha_t(
        separatrix_electron_density=sol_state.separatrix_electron_density,
        separatrix_electron_temp=sol_state.separatrix_electron_temp / 1e3,
        cylindrical_safety_factor=cylindrical_safety_factor,
        major_radius=major_radius,
        average_ion_mass=sol_state.average_ion_mass,
        Z_eff=sol_state.separatrix_Z_eff,
        mean_ion_charge_state=1.0,
    )

    # Update kappa_e for the next loop iteration. Impacts q_parallel and
    # temperatures upstream from target.
    sol_state.kappa_e = extended_lengyel_formulas.calc_kappa_e(
        sol_state.divertor_Z_eff
    )

  # --------------------------------------- #
  # -------- 3. Post-processing ----------- #
  # --------------------------------------- #

  neutral_pressure_in_divertor, heat_flux_perp_to_target = (
      _calc_post_processed_outputs(
          target_electron_temp=sol_state.target_electron_temp,
          average_ion_mass=sol_state.average_ion_mass,
          parallel_heat_flux_at_target=sol_state.parallel_heat_flux_at_target,
          sheath_heat_transmission_factor=sheath_heat_transmission_factor,
          ratio_of_molecular_to_ion_mass=ratio_of_molecular_to_ion_mass,
          wall_temperature=wall_temperature,
          target_angle_of_incidence=target_angle_of_incidence,
      )
  )

  return ExtendedLengyelOutputs(
      target_electron_temp=sol_state.target_electron_temp,
      neutral_pressure_in_divertor=neutral_pressure_in_divertor,
      alpha_t=sol_state.alpha_t,
      q_parallel=sol_state.q_parallel,
      heat_flux_perp_to_target=heat_flux_perp_to_target,
      separatrix_electron_temp=sol_state.separatrix_electron_temp / 1e3,
      separatrix_Z_eff=sol_state.separatrix_Z_eff,
      impurity_concentrations=sol_state.impurity_concentrations,
  )


def _validate_inputs_for_computation_mode(
    computation_mode: ComputationMode,
    target_electron_temp: array_typing.FloatScalar | None,
    seed_impurity_weights: Mapping[str, array_typing.FloatScalar] | None,
):
  """Validates inputs based on the specified computation mode."""
  if computation_mode == ComputationMode.FORWARD:
    if target_electron_temp is not None:
      raise ValueError(
          'Target electron temperature must not be provided for forward'
          ' computation.'
      )
    if seed_impurity_weights is not None:
      raise ValueError(
          'Seed impurity weights must not be provided for forward computation.'
      )
  elif computation_mode == ComputationMode.INVERSE:
    if target_electron_temp is None:
      raise ValueError(
          'Target electron temperature must be provided for inverse'
          ' computation.'
      )
    if seed_impurity_weights is None:
      raise ValueError(
          'Seed impurity weights must be provided for inverse computation.'
      )
  else:
    raise ValueError(f'Unknown computation mode: {computation_mode}')


def _calc_post_processed_outputs(
    target_electron_temp: array_typing.FloatScalar,
    average_ion_mass: array_typing.FloatScalar,
    parallel_heat_flux_at_target: array_typing.FloatScalar,
    sheath_heat_transmission_factor: array_typing.FloatScalar,
    ratio_of_molecular_to_ion_mass: array_typing.FloatScalar,
    wall_temperature: array_typing.FloatScalar,
    target_angle_of_incidence: array_typing.FloatScalar,
) -> tuple[jax.Array, jax.Array]:
  """Calculates post-processed outputs for the extended Lengyel model."""
  sound_speed_at_target = jnp.sqrt(
      2.0
      * target_electron_temp
      * constants.CONSTANTS.eV_to_J
      / (average_ion_mass * constants.CONSTANTS.m_amu)
  )

  # From equation 22 of Body NF 2025.
  electron_density_at_target = parallel_heat_flux_at_target / (
      sheath_heat_transmission_factor
      * target_electron_temp
      * constants.CONSTANTS.eV_to_J
      * sound_speed_at_target
  )

  # From equation 57 of Body NF 2025.
  log_flux_density_to_pascals_factor = 0.5 * (
      jnp.log(2.0)
      - jnp.log(jnp.pi)
      - jnp.log(ratio_of_molecular_to_ion_mass)
      - jnp.log(average_ion_mass)
      - jnp.log(constants.CONSTANTS.m_amu)
      - jnp.log(constants.CONSTANTS.k_B)
      - jnp.log(wall_temperature)
  )

  flux_density_to_pascals_factor = jnp.exp(log_flux_density_to_pascals_factor)

  parallel_ion_flux_to_target = (
      electron_density_at_target * sound_speed_at_target
  )
  parallel_to_perp_factor = jnp.sin(jnp.deg2rad(target_angle_of_incidence))

  neutral_pressure_in_divertor = (
      parallel_ion_flux_to_target
      * parallel_to_perp_factor
      / flux_density_to_pascals_factor
  )

  heat_flux_perp_to_target = (
      parallel_heat_flux_at_target * parallel_to_perp_factor
  )
  return neutral_pressure_in_divertor, heat_flux_perp_to_target


def _solve_for_c_z_prefactor(
    sol_state: divertor_sol_1d_lib.DivertorSOL1D,
) -> tuple[jax.Array, jax.Array]:
  """Solves the extended Lengyel model for the required impurity concentration.

  This function implements the extended Lengyel model in inverse mode,
  calculating the seeded impurity concentration (`c_z`) needed to achieve the
  an input target temperature consistent with a given set of plasma parameters.

  See Section 5 of T. Body et al 2025 Nucl. Fusion 65 086002 for the derivation.

  Args:
    sol_state: A DivertorSOL1D object containing the plasma parameters.

  Returns:
      c_z_prefactor: The scaling factor for the seeded impurity concentrations.
        To be multiplied by seed_impurity_weights to get each seeded impurity
        concentration.
      status: A SolveCzStatus enum indicating the outcome of the calculation.
  """
  # Temperatures must be in keV for the L_INT calculation.
  cc_temp_keV = sol_state.electron_temp_at_cc_interface / 1000.0
  div_temp_keV = sol_state.divertor_entrance_electron_temp / 1000.0
  sep_temp_keV = sol_state.separatrix_electron_temp / 1000.0

  # Calculate integrated radiation terms (L_INT) for seeded impurities.
  # See Eq. 34 in Body et al. 2025.
  Ls_cc_div = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_state.seed_impurity_weights,
          start_temp=cc_temp_keV,
          stop_temp=div_temp_keV,
          ne_tau=sol_state.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Ls_cc_u = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_state.seed_impurity_weights,
          start_temp=cc_temp_keV,
          stop_temp=sep_temp_keV,
          ne_tau=sol_state.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Ls_div_u = Ls_cc_u - Ls_cc_div

  # Calculate integrated radiation terms for fixed background impurities.
  Lf_cc_div = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_state.fixed_impurity_concentrations,
          start_temp=cc_temp_keV,
          stop_temp=div_temp_keV,
          ne_tau=sol_state.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Lf_cc_u = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_state.fixed_impurity_concentrations,
          start_temp=cc_temp_keV,
          stop_temp=sep_temp_keV,
          ne_tau=sol_state.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Lf_div_u = Lf_cc_u - Lf_cc_div

  # Define shorthand variables for clarity, matching the paper's notation.
  qu = sol_state.q_parallel
  qcc = sol_state.parallel_heat_flux_at_cc_interface
  b = sol_state.divertor_broadening_factor
  # `k` is a lumped parameter from the Lengyel model derivation.
  # See Eq. 33 in Body et al. 2025.
  k = (
      2.0
      * sol_state.kappa_e
      * (sol_state.separatrix_electron_density * _DENSITY_SCALE_FACTOR) ** 2
      * sol_state.separatrix_electron_temp**2
  )

  # Calculate the squared parallel heat flux at the divertor entrance.
  # This formula is derived by combining the Lengyel equations for the region
  # above and below the divertor entrance. See Eq. 40 in Body et al. 2025.
  q_div_squared = (
      Ls_div_u
      / _LINT_SCALE_FACTOR
      * (qcc**2 + k * Lf_cc_div / _LINT_K_INVERSE_SCALE_FACTOR)
      + (Ls_cc_div / _LINT_SCALE_FACTOR)
      * (qu**2 - k * Lf_div_u / _LINT_K_INVERSE_SCALE_FACTOR)
  ) / (Ls_div_u / _LINT_SCALE_FACTOR / b**2 + Ls_cc_div / _LINT_SCALE_FACTOR)

  # Check for unphysical result.
  status = jnp.where(
      q_div_squared < 0.0,
      SolveCzStatus.Q_DIV_SQUARED_NEGATIVE,
      SolveCzStatus.SUCCESS,
  )

  # Calculate the required seeded impurity concentration `c_z`.
  # See Eq. 42 in Body et al. 2025.
  c_z_prefactor = (
      (qu**2 + (1.0 / b**2 - 1.0) * q_div_squared - qcc**2)
      / (k * Ls_cc_u / _LINT_K_INVERSE_SCALE_FACTOR)
  ) - (Lf_cc_u / Ls_cc_u)

  return c_z_prefactor, status
