# Copyright 2025 DeepMind Technologies Limited
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

"""Standalone implementation of extended Lengyel from Body et al. NF 2025."""

import dataclasses
import functools
from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src.edge import base
from torax._src.edge import divertor_sol_1d as divertor_sol_1d_lib
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_formulas
from torax._src.edge import extended_lengyel_solvers

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExtendedLengyelOutputs(base.EdgeModelOutputs):
  """Outputs from the extended Lengyel model on top of the base class outputs.

  Attributes:
    alpha_t: Turbulence broadening factor alpha_t.
    kappa_e: Electron heat conductivity prefactor [W/(m*eV^3.5)].
    c_z_prefactor: Impurity concentration prefactor [dimensionless].
    Z_eff_separatrix: Z_eff at the separatrix.
    seed_impurity_concentrations: A mapping from ion symbol to its n_e_ratio.
    solver_status: Status of the solver.
    calculated_enrichment: A mapping from ion symbol to its enrichment factor as
      calculated by the Kallenbach model.
  """

  alpha_t: jax.Array
  kappa_e: jax.Array
  c_z_prefactor: jax.Array
  Z_eff_separatrix: jax.Array
  seed_impurity_concentrations: Mapping[str, jax.Array]
  solver_status: extended_lengyel_solvers.ExtendedLengyelSolverStatus
  calculated_enrichment: Mapping[str, jax.Array]


@functools.partial(
    jax.jit,
    static_argnames=[
        'computation_mode',
        'solver_mode',
    ],
)
def run_extended_lengyel_standalone(
    *,
    power_crossing_separatrix: array_typing.FloatScalar,
    separatrix_electron_density: array_typing.FloatScalar,
    fixed_impurity_concentrations: Mapping[str, array_typing.FloatScalar],
    main_ion_charge: array_typing.FloatScalar,
    magnetic_field_on_axis: array_typing.FloatScalar,
    plasma_current: array_typing.FloatScalar,
    connection_length_target: array_typing.FloatScalar,
    connection_length_divertor: array_typing.FloatScalar,
    major_radius: array_typing.FloatScalar,
    minor_radius: array_typing.FloatScalar,
    elongation_psi95: array_typing.FloatScalar,
    triangularity_psi95: array_typing.FloatScalar,
    average_ion_mass: array_typing.FloatScalar,
    mean_ion_charge_state: array_typing.FloatScalar,
    T_e_target: array_typing.FloatScalar | None = None,
    seed_impurity_weights: Mapping[str, array_typing.FloatScalar] | None = None,
    computation_mode: extended_lengyel_enums.ComputationMode = extended_lengyel_enums.ComputationMode.FORWARD,
    solver_mode: extended_lengyel_enums.SolverMode = extended_lengyel_enums.SolverMode.FIXED_POINT,
    divertor_broadening_factor: array_typing.FloatScalar = (
        extended_lengyel_defaults.DIVERTOR_BROADENING_FACTOR
    ),
    ratio_bpol_omp_to_bpol_avg: array_typing.FloatScalar = (
        extended_lengyel_defaults.RATIO_BPOL_OMP_TO_BPOL_AVG
    ),
    ne_tau: array_typing.FloatScalar = extended_lengyel_defaults.NE_TAU,
    sheath_heat_transmission_factor: array_typing.FloatScalar = (
        extended_lengyel_defaults.SHEATH_HEAT_TRANSMISSION_FACTOR
    ),
    angle_of_incidence_target: array_typing.FloatScalar = extended_lengyel_defaults.ANGLE_OF_INCIDENCE_TARGET,
    fraction_of_P_SOL_to_divertor: array_typing.FloatScalar = extended_lengyel_defaults.FRACTION_OF_PSOL_TO_DIVERTOR,
    SOL_conduction_fraction: array_typing.FloatScalar = extended_lengyel_defaults.SOL_CONDUCTION_FRACTION,
    ratio_of_molecular_to_ion_mass: array_typing.FloatScalar = extended_lengyel_defaults.RATIO_MOLECULAR_TO_ION_MASS,
    T_wall: array_typing.FloatScalar = extended_lengyel_defaults.T_WALL,
    mach_separatrix: array_typing.FloatScalar = extended_lengyel_defaults.MACH_SEPARATRIX,
    T_i_T_e_ratio_separatrix: array_typing.FloatScalar = (
        extended_lengyel_defaults.T_I_T_E_RATIO_SEPARATRIX
    ),
    n_e_n_i_ratio_separatrix: array_typing.FloatScalar = (
        extended_lengyel_defaults.N_E_N_I_RATIO_SEPARATRIX
    ),
    T_i_T_e_ratio_target: array_typing.FloatScalar = (
        extended_lengyel_defaults.T_I_T_E_RATIO_TARGET
    ),
    n_e_n_i_ratio_target: array_typing.FloatScalar = (
        extended_lengyel_defaults.N_E_N_I_RATIO_TARGET
    ),
    mach_target: array_typing.FloatScalar = extended_lengyel_defaults.MACH_TARGET,
    toroidal_flux_expansion: array_typing.FloatScalar = extended_lengyel_defaults.TOROIDAL_FLUX_EXPANSION,
    fixed_point_iterations: int | None = None,
    newton_raphson_iterations: int = extended_lengyel_defaults.NEWTON_RAPHSON_ITERATIONS,
    newton_raphson_tol: float = extended_lengyel_defaults.NEWTON_RAPHSON_TOL,
    enrichment_model_multiplier: array_typing.FloatScalar = 1.0,
    diverted: bool = True,
    initial_guess: (
        divertor_sol_1d_lib.ExtendedLengyelInitialGuess | None
    ) = None,
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
    connection_length_target: From target to outboard midplane [m].
    connection_length_divertor: From target to X-point [m].
    major_radius: Major radius of magnetic axis [m].
    minor_radius: Minor radius from magnetic axis to outboard midplane [m].
    elongation_psi95: Elongation at psiN=0.95 [dimensionless].
    triangularity_psi95: Triangularity at psiN=0.95 [dimensionless]..
    average_ion_mass: Average main-ion mass [amu] defined as sum(m_i*n_i)/n_e.
    mean_ion_charge_state: Mean ion charge state [dimensionless]. Defined as
      n_e/(sum_i n_i).
    T_e_target: For inverse mode, desired electron temperature at sheath
      entrance [eV].
    seed_impurity_weights: For inverse mode, Mapping from ion symbol to
      fractions of seeded impurities. Total impurity n_e_ratio (c_z) is
      calculated by the model. c_z_prefactor*seed_impurity_weights thus forms an
      output of the model.
    computation_mode: The computation mode for the model. See ComputationMode
      for details.
    solver_mode: The solver mode for the model. See SolverMode for details.
    divertor_broadening_factor: lambda_INT / lambda_q  [dimensionless].
    ratio_bpol_omp_to_bpol_avg: Bpol_omp / Bpol_avg  [dimensionless].
    ne_tau: Product of electron density and ion residence time [s m^-3].
    sheath_heat_transmission_factor: Sheath heat transmission factor gamma
      [dimensionless].
    angle_of_incidence_target: Angle between fieldline and target [degrees].
    fraction_of_P_SOL_to_divertor: Fraction of power to outer divertor
      [dimensionless].
    SOL_conduction_fraction: Fraction of power carried by conduction
      [dimensionless].
    ratio_of_molecular_to_ion_mass: Ratio of molecular to ion mass
      [dimensionless].
    T_wall: Divertor wall temperature [K].
    mach_separatrix: Mach number at separatrix [dimensionless].
    T_i_T_e_ratio_separatrix: Ti/Te at separatrix [dimensionless].
    n_e_n_i_ratio_separatrix: ne/ni at separatrix [dimensionless].
    T_i_T_e_ratio_target: Ti/Te at target [dimensionless].
    n_e_n_i_ratio_target: ne/ni at target [dimensionless].
    mach_target: Mach number at target [dimensionless].
    toroidal_flux_expansion: Toroidal flux expansion factor [dimensionless].
    fixed_point_iterations: Number of iterations for fixed step solver. If None,
      then a default value is used based on the solver mode: different defaults
      for hybrid and fixed-step solvers. For Newton-Raphson, this argument is
      ignored and remains None if inputted as None.
    newton_raphson_iterations: Number of iterations for Newton-Raphson solver.
    newton_raphson_tol: Tolerance for Newton-Raphson solver.
    enrichment_model_multiplier: Multiplier for the Kallenbach enrichment model.
    diverted: Whether we are in diverted geometry or not.
    initial_guess: Initial guess for the iterative solver state variables.

  Returns:
    An ExtendedLengyelOutputs object with the calculated values and solver
    status.
  """

  # --------------------------------------- #
  # ---------- 1. Pre-processing ---------- #
  # --------------------------------------- #

  if seed_impurity_weights is None:
    seed_impurity_weights = {}

  _validate_inputs_for_computation_mode(
      computation_mode, T_e_target, seed_impurity_weights
  )

  if fixed_point_iterations is None:
    if solver_mode == extended_lengyel_enums.SolverMode.HYBRID:
      fixed_point_iterations = (
          extended_lengyel_defaults.HYBRID_FIXED_POINT_ITERATIONS
      )
    else:
      fixed_point_iterations = extended_lengyel_defaults.FIXED_POINT_ITERATIONS

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
  fieldline_pitch_at_omp = (
      extended_lengyel_formulas.calc_fieldline_pitch_at_omp(
          magnetic_field_on_axis=magnetic_field_on_axis,
          plasma_current=plasma_current,
          major_radius=major_radius,
          minor_radius=minor_radius,
          elongation_psi95=elongation_psi95,
          triangularity_psi95=triangularity_psi95,
          ratio_bpol_omp_to_bpol_avg=ratio_bpol_omp_to_bpol_avg,
      )
  )

  params = divertor_sol_1d_lib.ExtendedLengyelParameters(
      major_radius=major_radius,
      minor_radius=minor_radius,
      separatrix_average_poloidal_field=separatrix_average_poloidal_field,
      fieldline_pitch_at_omp=fieldline_pitch_at_omp,
      cylindrical_safety_factor=cylindrical_safety_factor,
      power_crossing_separatrix=power_crossing_separatrix,
      ratio_bpol_omp_to_bpol_avg=ratio_bpol_omp_to_bpol_avg,
      fraction_of_P_SOL_to_divertor=fraction_of_P_SOL_to_divertor,
      SOL_conduction_fraction=SOL_conduction_fraction,
      angle_of_incidence_target=angle_of_incidence_target,
      ratio_of_molecular_to_ion_mass=ratio_of_molecular_to_ion_mass,
      T_wall=T_wall,
      seed_impurity_weights=seed_impurity_weights,
      fixed_impurity_concentrations=fixed_impurity_concentrations,
      ne_tau=ne_tau,
      main_ion_charge=main_ion_charge,
      mean_ion_charge_state=mean_ion_charge_state,
      divertor_broadening_factor=divertor_broadening_factor,
      connection_length_divertor=connection_length_divertor,
      connection_length_target=connection_length_target,
      mach_separatrix=mach_separatrix,
      separatrix_electron_density=separatrix_electron_density,
      T_i_T_e_ratio_separatrix=T_i_T_e_ratio_separatrix,
      n_e_n_i_ratio_separatrix=n_e_n_i_ratio_separatrix,
      average_ion_mass=average_ion_mass,
      sheath_heat_transmission_factor=sheath_heat_transmission_factor,
      mach_target=mach_target,
      T_i_T_e_ratio_target=T_i_T_e_ratio_target,
      n_e_n_i_ratio_target=n_e_n_i_ratio_target,
      toroidal_flux_expansion=toroidal_flux_expansion,
  )

  # Initialize values for iterative solver.
  if initial_guess is not None:
    alpha_t_init = initial_guess.alpha_t
    kappa_e_init = initial_guess.kappa_e
    T_e_separatrix_init = initial_guess.T_e_separatrix
    # q_parallel is calculated from T_e_separatrix, not passed directly.
    q_parallel_init = divertor_sol_1d_lib.calc_q_parallel(
        params=params,
        T_e_separatrix=T_e_separatrix_init,
        alpha_t=alpha_t_init,
    )

    if computation_mode == extended_lengyel_enums.ComputationMode.INVERSE:
      T_e_target_init = T_e_target  # from input
      assert isinstance(initial_guess, divertor_sol_1d_lib.InverseInitialGuess)
      c_z_prefactor_init = initial_guess.c_z_prefactor
    elif computation_mode == extended_lengyel_enums.ComputationMode.FORWARD:
      assert isinstance(initial_guess, divertor_sol_1d_lib.ForwardInitialGuess)
      T_e_target_init = initial_guess.T_e_target
      # Not used as an evolved variable in forward mode.
      c_z_prefactor_init = extended_lengyel_defaults.DEFAULT_C_Z_PREFACTOR_INIT
    else:
      raise ValueError(f'Unknown computation mode: {computation_mode}')

  else:
    alpha_t_init = extended_lengyel_defaults.DEFAULT_ALPHA_T_INIT
    c_z_prefactor_init = extended_lengyel_defaults.DEFAULT_C_Z_PREFACTOR_INIT
    kappa_e_init = extended_lengyel_defaults.KAPPA_E_0
    T_e_separatrix_init = (
        extended_lengyel_defaults.DEFAULT_T_E_SEPARATRIX_INIT
    )  # [eV]
    q_parallel_init = divertor_sol_1d_lib.calc_q_parallel(
        params=params,
        T_e_separatrix=T_e_separatrix_init,
        alpha_t=alpha_t_init,
    )

    if computation_mode == extended_lengyel_enums.ComputationMode.INVERSE:
      T_e_target_init = T_e_target  # from input
    elif computation_mode == extended_lengyel_enums.ComputationMode.FORWARD:
      T_e_target_init = (
          extended_lengyel_defaults.DEFAULT_T_E_TARGET_INIT_FORWARD
      )  # eV.
    else:
      raise ValueError(f'Unknown computation mode: {computation_mode}')

  initial_state = divertor_sol_1d_lib.ExtendedLengyelState(
      q_parallel=q_parallel_init,
      alpha_t=alpha_t_init,
      c_z_prefactor=c_z_prefactor_init,
      kappa_e=kappa_e_init,
      T_e_target=T_e_target_init,
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
  match solver_key:
    case (
        extended_lengyel_enums.ComputationMode.INVERSE,
        extended_lengyel_enums.SolverMode.FIXED_POINT,
    ):
      output_sol_model, solver_status = (
          extended_lengyel_solvers.inverse_mode_fixed_point_solver(
              initial_sol_model=initial_sol_model,
              iterations=fixed_point_iterations,
          )
      )
    case (
        extended_lengyel_enums.ComputationMode.FORWARD,
        extended_lengyel_enums.SolverMode.FIXED_POINT,
    ):
      output_sol_model, solver_status = (
          extended_lengyel_solvers.forward_mode_fixed_point_solver(
              initial_sol_model=initial_sol_model,
              iterations=fixed_point_iterations,
          )
      )
    case (
        extended_lengyel_enums.ComputationMode.INVERSE,
        extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
    ):
      output_sol_model, solver_status = (
          extended_lengyel_solvers.inverse_mode_newton_solver(
              initial_sol_model=initial_sol_model,
              maxiter=newton_raphson_iterations,
              tol=newton_raphson_tol,
          )
      )
    case (
        extended_lengyel_enums.ComputationMode.FORWARD,
        extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
    ):
      output_sol_model, solver_status = (
          extended_lengyel_solvers.forward_mode_newton_solver(
              initial_sol_model=initial_sol_model,
              maxiter=newton_raphson_iterations,
              tol=newton_raphson_tol,
          )
      )
    case (
        extended_lengyel_enums.ComputationMode.INVERSE,
        extended_lengyel_enums.SolverMode.HYBRID,
    ):
      output_sol_model, solver_status = (
          extended_lengyel_solvers.inverse_mode_hybrid_solver(
              initial_sol_model=initial_sol_model,
              fixed_point_iterations=fixed_point_iterations,
              newton_raphson_iterations=newton_raphson_iterations,
              newton_raphson_tol=newton_raphson_tol,
          )
      )
    case (
        extended_lengyel_enums.ComputationMode.FORWARD,
        extended_lengyel_enums.SolverMode.HYBRID,
    ):
      output_sol_model, solver_status = (
          extended_lengyel_solvers.forward_mode_hybrid_solver(
              initial_sol_model=initial_sol_model,
              fixed_point_iterations=fixed_point_iterations,
              newton_raphson_iterations=newton_raphson_iterations,
              newton_raphson_tol=newton_raphson_tol,
          )
      )
    case _:
      raise ValueError(
          'Invalid computation and solver mode combination:'
          f' {computation_mode}, {solver_mode}'
      )

  # --------------------------------------- #
  # -------- 3. Post-processing ----------- #
  # --------------------------------------- #

  pressure_neutral_divertor, q_perpendicular_target = (
      _calc_post_processed_outputs(
          sol_model=output_sol_model,
      )
  )

  calculated_enrichment = {}
  all_impurities = set(fixed_impurity_concentrations.keys()) | set(
      seed_impurity_weights.keys()
  )
  for species in all_impurities:
    # For limited geometry, enrichment factor is 1.0.
    calculated_enrichment[species] = jnp.where(
        diverted,
        extended_lengyel_formulas.calc_enrichment_kallenbach(
            pressure_neutral_divertor=pressure_neutral_divertor,
            ion_symbol=species,
            enrichment_multiplier=enrichment_model_multiplier,
        ),
        jnp.array(1.0, dtype=jax_utils.get_dtype()),
    )

  return ExtendedLengyelOutputs(
      T_e_target=output_sol_model.state.T_e_target,
      pressure_neutral_divertor=pressure_neutral_divertor,
      alpha_t=output_sol_model.state.alpha_t,
      kappa_e=output_sol_model.state.kappa_e,
      c_z_prefactor=output_sol_model.state.c_z_prefactor,
      q_parallel=output_sol_model.state.q_parallel,
      q_perpendicular_target=q_perpendicular_target,
      T_e_separatrix=output_sol_model.T_e_separatrix / 1e3,
      Z_eff_separatrix=output_sol_model.Z_eff_separatrix,
      seed_impurity_concentrations=output_sol_model.seed_impurity_concentrations,
      solver_status=solver_status,
      calculated_enrichment=calculated_enrichment,
  )


def _validate_inputs_for_computation_mode(
    computation_mode: extended_lengyel_enums.ComputationMode,
    T_e_target: array_typing.FloatScalar,
    seed_impurity_weights: Mapping[str, array_typing.FloatScalar],
):
  """Validates inputs based on the specified computation mode."""
  if computation_mode == extended_lengyel_enums.ComputationMode.FORWARD:
    if T_e_target is not None:
      raise ValueError(
          'Target electron temperature must not be provided for forward'
          ' computation.'
      )
    if seed_impurity_weights:
      raise ValueError(
          'Seed impurity weights must not be provided for forward computation.'
      )
  elif computation_mode == extended_lengyel_enums.ComputationMode.INVERSE:
    if T_e_target is None:
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
      * sol_model.state.T_e_target
      * constants.CONSTANTS.eV_to_J
      / (sol_model.params.average_ion_mass * constants.CONSTANTS.m_amu)
  )

  # From equation 22 of Body NF 2025.
  electron_density_at_target = sol_model.parallel_heat_flux_at_target / (
      sol_model.params.sheath_heat_transmission_factor
      * sol_model.state.T_e_target
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
      - jnp.log(sol_model.params.T_wall)
  )

  flux_density_to_pascals_factor = jnp.exp(log_flux_density_to_pascals_factor)

  parallel_ion_flux_to_target = (
      electron_density_at_target * sound_speed_at_target
  )
  parallel_to_perp_factor = jnp.sin(
      jnp.deg2rad(sol_model.params.angle_of_incidence_target)
  )

  pressure_neutral_divertor = (
      parallel_ion_flux_to_target
      * parallel_to_perp_factor
      / flux_density_to_pascals_factor
  )

  q_perpendicular_target = (
      sol_model.parallel_heat_flux_at_target * parallel_to_perp_factor
  )
  return pressure_neutral_divertor, q_perpendicular_target
