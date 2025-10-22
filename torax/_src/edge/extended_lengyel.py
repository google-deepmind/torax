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
import functools
from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.edge import divertor_sol_1d as divertor_sol_1d_lib
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_formulas
from torax._src.edge import extended_lengyel_solvers

# pylint: disable=invalid-name


class ComputationMode(enum.StrEnum):
  """Computation modes for the extended Lengyel model.

  Attributes:
    FORWARD: Calculate impurity concentrations for a given target temperature.
    INVERSE: Calculate target temperature for a given impurity concentration.
  """

  FORWARD = 'forward'
  INVERSE = 'inverse'


class SolverMode(enum.StrEnum):
  """Solver modes for the extended Lengyel model.

  Attributes:
    FIXED_STEP: A simple fixed-step iterative solver.
    NEWTON_RAPHSON: A Newton-Raphson solver (not yet implemented).
  """

  FIXED_STEP = 'fixed_step'
  NEWTON_RAPHSON = 'newton_raphson'


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
    seed_impurity_concentrations: A mapping from ion symbol to its n_e_ratio.
  """

  target_electron_temp: jax.Array
  neutral_pressure_in_divertor: jax.Array
  alpha_t: jax.Array
  q_parallel: jax.Array
  heat_flux_perp_to_target: jax.Array
  separatrix_electron_temp: jax.Array
  separatrix_Z_eff: jax.Array
  seed_impurity_concentrations: Mapping[str, jax.Array]


# TODO(b/446608829)
# 1. Consider renaming variables to match the rest of TORAX
# 2. Consider repackaging the inputs into single or multiple dataclasses for
#    better readability.
@functools.partial(
    jax.jit,
    static_argnames=[
        'computation_mode',
        'solver_mode',
    ],
)
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
    solver_mode: SolverMode = SolverMode.FIXED_STEP,
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
    fixed_step_iterations: int = extended_lengyel_defaults.FIXED_STEP_ITERATIONS,
    newton_raphson_iterations: int = extended_lengyel_defaults.NEWTON_RAPHSON_ITERATIONS,
    newton_raphson_tol: float = extended_lengyel_defaults.NEWTON_RAPHSON_TOL,
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
    solver_mode: The solver mode for the model. See SolverMode for details.
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
    fixed_step_iterations: Number of iterations for fixed step solver.
    newton_raphson_iterations: Number of iterations for Newton-Raphson solver.
    newton_raphson_tol: Tolerance for Newton-Raphson solver.

  Returns:
    An ExtendedLengyelOutputs object with the calculated values.
  """

  # --------------------------------------- #
  # ---------- 1. Pre-processing ---------- #
  # --------------------------------------- #

  if seed_impurity_weights is None:
    seed_impurity_weights = {}

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

  params = divertor_sol_1d_lib.ExtendedLengyelParameters(
      major_radius=major_radius,
      minor_radius=minor_radius,
      separatrix_average_poloidal_field=separatrix_average_poloidal_field,
      fieldline_pitch_at_omp=fieldline_pitch_at_omp,
      cylindrical_safety_factor=cylindrical_safety_factor,
      power_crossing_separatrix=power_crossing_separatrix,
      ratio_of_upstream_to_average_poloidal_field=ratio_of_upstream_to_average_poloidal_field,
      fraction_of_P_SOL_to_divertor=fraction_of_P_SOL_to_divertor,
      SOL_conduction_fraction=SOL_conduction_fraction,
      target_angle_of_incidence=target_angle_of_incidence,
      ratio_of_molecular_to_ion_mass=ratio_of_molecular_to_ion_mass,
      wall_temperature=wall_temperature,
      seed_impurity_weights=seed_impurity_weights,
      fixed_impurity_concentrations=fixed_impurity_concentrations,
      ne_tau=ne_tau,
      main_ion_charge=main_ion_charge,
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

  # Initialize values for iterative solver.
  alpha_t_init = 0.1
  c_z_prefactor_init = 0.0
  kappa_e_init = extended_lengyel_defaults.KAPPA_E_0
  separatrix_electron_temp_init = 100.0  # [eV], needed to initialize q_parallel
  q_parallel_init = extended_lengyel_formulas.calculate_q_parallel(
      separatrix_electron_temp=separatrix_electron_temp_init,
      average_ion_mass=params.average_ion_mass,
      separatrix_average_poloidal_field=params.separatrix_average_poloidal_field,
      alpha_t=alpha_t_init,
      ratio_of_upstream_to_average_poloidal_field=params.ratio_of_upstream_to_average_poloidal_field,
      fraction_of_PSOL_to_divertor=params.fraction_of_P_SOL_to_divertor,
      minor_radius=params.minor_radius,
      major_radius=params.major_radius,
      power_crossing_separatrix=params.power_crossing_separatrix,
      fieldline_pitch_at_omp=params.fieldline_pitch_at_omp,
  )

  if computation_mode == ComputationMode.INVERSE:
    target_electron_temp_init = target_electron_temp  # from input
  elif computation_mode == ComputationMode.FORWARD:
    target_electron_temp_init = 2.0  # eV
  else:
    raise ValueError(f'Unknown computation mode: {computation_mode}')

  initial_state = divertor_sol_1d_lib.ExtendedLengyelState(
      q_parallel=q_parallel_init,
      alpha_t=alpha_t_init,
      c_z_prefactor=c_z_prefactor_init,
      kappa_e=kappa_e_init,
      target_electron_temp=target_electron_temp_init,
  )

  initial_sol_model = divertor_sol_1d_lib.DivertorSOL1D(
      params=params,
      state=initial_state,
  )

  # --------------------------------------- #
  # -------- 2. Iterative Solver----------- #
  # --------------------------------------- #

  solver_key = (computation_mode, solver_mode)

  # ComputationMode enum is a static variable so can use standard flow.
  if solver_key == (ComputationMode.INVERSE, SolverMode.FIXED_STEP):
    output_sol_model = extended_lengyel_solvers.inverse_mode_fixed_step_solver(
        sol_model=initial_sol_model,
        iterations=fixed_step_iterations,
    )
  elif solver_key == (ComputationMode.FORWARD, SolverMode.FIXED_STEP):
    output_sol_model = extended_lengyel_solvers.forward_mode_fixed_step_solver(
        sol_model=initial_sol_model,
        iterations=fixed_step_iterations,
    )
  elif solver_key == (ComputationMode.INVERSE, SolverMode.NEWTON_RAPHSON):
    raise NotImplementedError(
        'Newton-Raphson solver is not yet implemented for inverse mode.'
    )
  elif solver_key == (ComputationMode.FORWARD, SolverMode.NEWTON_RAPHSON):
    output_sol_model, _ = extended_lengyel_solvers.forward_mode_newton_solver(
        initial_sol_model=initial_sol_model,
        maxiter=newton_raphson_iterations,
        tol=newton_raphson_tol,
    )
  else:
    raise ValueError(
        'Invalid computation and solver mode combination:'
        f' {computation_mode}, {solver_mode}'
    )

  # --------------------------------------- #
  # -------- 3. Post-processing ----------- #
  # --------------------------------------- #

  neutral_pressure_in_divertor, heat_flux_perp_to_target = (
      _calc_post_processed_outputs(
          sol_model=output_sol_model,
      )
  )

  return ExtendedLengyelOutputs(
      target_electron_temp=output_sol_model.state.target_electron_temp,
      neutral_pressure_in_divertor=neutral_pressure_in_divertor,
      alpha_t=output_sol_model.state.alpha_t,
      q_parallel=output_sol_model.state.q_parallel,
      heat_flux_perp_to_target=heat_flux_perp_to_target,
      separatrix_electron_temp=output_sol_model.separatrix_electron_temp / 1e3,
      separatrix_Z_eff=output_sol_model.separatrix_Z_eff,
      seed_impurity_concentrations=output_sol_model.seed_impurity_concentrations,
  )


def _validate_inputs_for_computation_mode(
    computation_mode: ComputationMode,
    target_electron_temp: array_typing.FloatScalar,
    seed_impurity_weights: Mapping[str, array_typing.FloatScalar],
):
  """Validates inputs based on the specified computation mode."""
  if computation_mode == ComputationMode.FORWARD:
    if target_electron_temp is not None:
      raise ValueError(
          'Target electron temperature must not be provided for forward'
          ' computation.'
      )
    if seed_impurity_weights:
      raise ValueError(
          'Seed impurity weights must not be provided for forward computation.'
      )
  elif computation_mode == ComputationMode.INVERSE:
    if target_electron_temp is None:
      raise ValueError(
          'Target electron temperature must be provided for inverse'
          ' computation.'
      )
    if not seed_impurity_weights:
      raise ValueError(
          'Seed impurity weights must be provided for inverse computation.'
      )
  else:
    raise ValueError(f'Unknown computation mode: {computation_mode}')


def _calc_post_processed_outputs(
    sol_model: divertor_sol_1d_lib.DivertorSOL1D,
) -> tuple[jax.Array, jax.Array]:
  """Calculates post-processed outputs for the extended Lengyel model."""
  sound_speed_at_target = jnp.sqrt(
      2.0
      * sol_model.state.target_electron_temp
      * constants.CONSTANTS.eV_to_J
      / (sol_model.params.average_ion_mass * constants.CONSTANTS.m_amu)
  )

  # From equation 22 of Body NF 2025.
  electron_density_at_target = sol_model.parallel_heat_flux_at_target / (
      sol_model.params.sheath_heat_transmission_factor
      * sol_model.state.target_electron_temp
      * constants.CONSTANTS.eV_to_J
      * sound_speed_at_target
  )

  # From equation 57 of Body NF 2025.
  log_flux_density_to_pascals_factor = 0.5 * (
      jnp.log(2.0)
      - jnp.log(jnp.pi)
      - jnp.log(sol_model.params.ratio_of_molecular_to_ion_mass)
      - jnp.log(sol_model.params.average_ion_mass)
      - jnp.log(constants.CONSTANTS.m_amu)
      - jnp.log(constants.CONSTANTS.k_B)
      - jnp.log(sol_model.params.wall_temperature)
  )

  flux_density_to_pascals_factor = jnp.exp(log_flux_density_to_pascals_factor)

  parallel_ion_flux_to_target = (
      electron_density_at_target * sound_speed_at_target
  )
  parallel_to_perp_factor = jnp.sin(
      jnp.deg2rad(sol_model.params.target_angle_of_incidence)
  )

  neutral_pressure_in_divertor = (
      parallel_ion_flux_to_target
      * parallel_to_perp_factor
      / flux_density_to_pascals_factor
  )

  heat_flux_perp_to_target = (
      sol_model.parallel_heat_flux_at_target * parallel_to_perp_factor
  )
  return neutral_pressure_in_divertor, heat_flux_perp_to_target
