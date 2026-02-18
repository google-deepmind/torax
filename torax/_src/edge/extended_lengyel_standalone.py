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

from __future__ import annotations

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
from torax._src.solver import jax_root_finding

# pylint: disable=invalid-name


def _roots_are_distinct(
    diffs: array_typing.FloatVector,
    ref_values: array_typing.FloatVector,
) -> jax.Array:
  """Returns a boolean array indicating if roots are distinct.

  Args:
    diffs: Differences between consecutive sorted roots.
    ref_values: The sorted roots themselves, used for relative tolerance check.

  Returns:
    A boolean array indicating if roots are distinct.
  """
  threshold = (
      extended_lengyel_defaults.MULTISTART_ROOT_ATOL
      + extended_lengyel_defaults.MULTISTART_ROOT_RTOL * jnp.abs(ref_values)
  )
  return diffs > threshold


def _extract_solver_metrics(
    status: extended_lengyel_solvers.ExtendedLengyelSolverStatus,
) -> tuple[
    array_typing.IntVector, array_typing.FloatVector, array_typing.IntVector
]:
  """Extracts solver metrics from status, handling different solver types."""
  numerics = status.numerics_outcome

  # Default values
  iterations = jnp.array(-1, dtype=jax_utils.get_int_dtype())
  residual = jnp.array(0.0, dtype=jax_utils.get_dtype())
  error = jnp.array(0, dtype=jax_utils.get_int_dtype())

  if isinstance(numerics, jax_root_finding.RootMetadata):
    iterations = numerics.iterations
    # Keep full residual for detailed analysis
    residual = numerics.residual
    error = numerics.error

  return iterations, residual, error


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
    roots: The full batch of results from all initial guesses in forward mode.
      This is a pytree of the same structure as ExtendedLengyelOutputs, where
      each leaf has an additional leading dimension of size `num_guesses`. This
      field can be None if multistart is not used or not supported (e.g. inverse
      mode).
    multiple_roots_found: Boolean flag indicating if multiple distinct, valid
      roots were found in forward mode. This field is None for Inverse Mode,
      which only has a single solution.
  """

  alpha_t: jax.Array
  kappa_e: jax.Array
  c_z_prefactor: jax.Array
  Z_eff_separatrix: jax.Array
  seed_impurity_concentrations: Mapping[str, jax.Array]
  solver_status: extended_lengyel_solvers.ExtendedLengyelSolverStatus
  calculated_enrichment: Mapping[str, jax.Array]
  roots: ExtendedLengyelOutputs | None = None
  multiple_roots_found: jax.Array | None = None

  # TODO(b/323504363): b/446608829 - Simplify this function where viable.
  def get_unique_roots(self) -> ExtendedLengyelOutputs | None:
    """Identifies and returns unique valid roots from solver results.

    Returns:
      ExtendedLengyelOutputs containing unique valid roots, or None if no roots
      are available (e.g. inverse mode).
      Leaves have shape (time, n_roots, ...) or (n_roots, ...) if time is
      absent.
      Roots are sorted by T_e_target value. Padding (nan/inf) is added if
      fewer than max_roots are found.
    """
    if self.roots is None:
      return None
    roots = self.roots

    # Collect all array fields from `roots` to be processed.
    # This ensures all relevant output fields are handled.
    fields_to_compress = {}
    for field in dataclasses.fields(roots):
      if field.name in ['roots', 'multiple_roots_found', 'solver_status']:
        continue  # Skip recursive field and internal flags
      value = getattr(roots, field.name)
      if isinstance(value, array_typing.Array):
        fields_to_compress[field.name] = jnp.asarray(value)
      elif isinstance(value, Mapping):
        for k, v in value.items():
          if isinstance(v, array_typing.Array):
            fields_to_compress[f'{field.name}_{k}'] = jnp.asarray(v)

    ref_shape = jnp.asarray(roots.T_e_target).shape

    iterations, residual, error = _extract_solver_metrics(roots.solver_status)
    if isinstance(
        roots.solver_status.numerics_outcome, jax_root_finding.RootMetadata
    ):
      fields_to_compress['solver_iterations'] = iterations
      fields_to_compress['solver_residual'] = residual
      fields_to_compress['solver_error'] = error
      fields_to_compress['solver_last_tau'] = (
          roots.solver_status.numerics_outcome.last_tau
      )
    else:
      # Fixed-point solver. Only error is relevant.
      fields_to_compress['solver_error'] = jnp.broadcast_to(error, ref_shape)
    fields_to_compress['solver_physics_outcome'] = jnp.broadcast_to(
        jnp.asarray(roots.solver_status.physics_outcome), ref_shape
    )

    # Determine shapes and axes. Input could be (time, n_roots) or (n_roots,),
    # depending on whether we are processing a stacked or unstacked case.
    # An unstacked case may be generated if using this method on a standalone
    # ExtendedLengyelOutputs instance. The stacked case is generated when
    # processing time-dependent outputs.
    T_e = jnp.asarray(roots.T_e_target)
    has_time = T_e.ndim > 1

    T_e_prepared = fields_to_compress['T_e_target']

    # Analyze along roots axis
    # If we stacked, T is axis 0, G is axis 1. Roots axis is 1.
    # If we didn't stack (1D), G is axis 0. Roots axis is 0.
    roots_axis = 1 if has_time else 0

    # Sort indices
    # Move NaNs to inf for sorting
    T_e_for_sorting = jnp.where(jnp.isnan(T_e_prepared), jnp.inf, T_e_prepared)
    sorted_indices = jnp.argsort(T_e_for_sorting, axis=roots_axis)

    # Compress Logic
    sorted_T_e = jnp.take_along_axis(
        T_e_for_sorting, sorted_indices, axis=roots_axis
    )

    # Difference along roots axis to find duplicates
    diff = jnp.diff(sorted_T_e, axis=roots_axis, prepend=-jnp.inf)

    is_unique = jnp.logical_and(
        _roots_are_distinct(diff, sorted_T_e),
        sorted_T_e != jnp.inf,
    )

    # Pack unique roots to the start
    # argsort sorts ascending, so invert the is_unique mask before sorting to
    # get the unique roots at the start
    packing_indices = jnp.argsort(~is_unique, axis=roots_axis, stable=True)

    # Final indices into the VALID/SORTED array
    final_indices = jnp.take_along_axis(
        sorted_indices, packing_indices, axis=roots_axis
    )

    # Final Mask (reordered)
    final_valid_mask = jnp.take_along_axis(
        is_unique, packing_indices, axis=roots_axis
    )

    result_dict = {}

    for name, data in fields_to_compress.items():
      # Expand final_indices to match data dimensions for broadcasting

      current_indices = final_indices
      current_mask = final_valid_mask

      extra_dims = data.ndim - final_indices.ndim
      if extra_dims > 0:
        # Append singleton dimensions to indices and mask
        expand_axes = tuple(range(data.ndim - extra_dims, data.ndim))
        current_indices = jnp.expand_dims(current_indices, axis=expand_axes)
        current_mask = jnp.expand_dims(current_mask, axis=expand_axes)

      # Reorder
      packed = jnp.take_along_axis(data, current_indices, axis=roots_axis)

      # Apply mask
      if jnp.issubdtype(packed.dtype, jnp.floating):
        nan_value = jnp.nan
      else:
        # Use -1 for integer types (iterations, error, physics_outcome)
        nan_value = -1

      masked = jnp.where(current_mask, packed, nan_value)
      result_dict[name] = masked

    n_unique_per_time = jnp.sum(final_valid_mask, axis=roots_axis)
    max_unique = int(jnp.max(n_unique_per_time))
    max_unique = max(max_unique, 1)
    trim_slice = [slice(None)] * final_valid_mask.ndim
    trim_slice[roots_axis] = slice(0, max_unique)
    trim_slice = tuple(trim_slice)
    result_dict = {k: v[trim_slice] for k, v in result_dict.items()}

    if 'solver_iterations' in result_dict:
      last_tau = result_dict.get('solver_last_tau')
      if last_tau is None:
        last_tau = jnp.zeros_like(
            result_dict['solver_iterations'], dtype=jnp.float32
        )
      new_numerics_outcome = jax_root_finding.RootMetadata(
          iterations=result_dict['solver_iterations'].astype(jnp.int32),
          residual=result_dict['solver_residual'],
          error=result_dict['solver_error'].astype(jnp.int32),
          last_tau=last_tau,
      )
    else:
      new_numerics_outcome = roots.solver_status.numerics_outcome
    new_solver_status = extended_lengyel_solvers.ExtendedLengyelSolverStatus(
        physics_outcome=result_dict['solver_physics_outcome'],  # pytype: disable=wrong-arg-types
        numerics_outcome=new_numerics_outcome,
    )

    # Reconstruct seed_impurity_concentrations and calculated_enrichment
    new_seed_impurity_concentrations = {
        k: result_dict[f'seed_impurity_concentrations_{k}']
        for k in roots.seed_impurity_concentrations or {}
    }

    new_calculated_enrichment = {
        k: result_dict[f'calculated_enrichment_{k}']
        for k in roots.calculated_enrichment or {}
    }

    # Create the new ExtendedLengyelOutputs instance
    # We populate it with the compressed arrays.
    # Use dynamic construction to satisfy Pytype and handle all fields.
    output_args = {}
    valid_fields = {f.name for f in dataclasses.fields(self)}
    for name, val in result_dict.items():
      if name in valid_fields:
        output_args[name] = val

    output_args['solver_status'] = new_solver_status
    output_args['seed_impurity_concentrations'] = (
        new_seed_impurity_concentrations
    )
    output_args['calculated_enrichment'] = new_calculated_enrichment
    output_args['roots'] = None
    output_args['multiple_roots_found'] = None

    return self.__class__(**output_args)


@functools.partial(
    jax.jit,
    static_argnames=[
        'computation_mode',
        'solver_mode',
        'multistart_num_guesses',
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
    multistart_num_guesses: int = extended_lengyel_defaults.MULTISTART_NUM_GUESSES,
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
    multistart_num_guesses: Number of initial guesses for multistart solver.
      Only used in forward mode.
    enrichment_model_multiplier: Multiplier for the Kallenbach enrichment model.
    diverted: Whether we are in diverted geometry or not.
    initial_guess: Initial guess for the iterative solver state variables.

  Returns:
    An ExtendedLengyelOutputs object with the calculated values and solver
    status.
  """

  # 1. Pre-processing
  if seed_impurity_weights is None:
    seed_impurity_weights = {}

  _validate_inputs_for_computation_mode(
      computation_mode, T_e_target, seed_impurity_weights
  )

  params = _construct_parameters(
      power_crossing_separatrix=power_crossing_separatrix,
      separatrix_electron_density=separatrix_electron_density,
      fixed_impurity_concentrations=fixed_impurity_concentrations,
      main_ion_charge=main_ion_charge,
      magnetic_field_on_axis=magnetic_field_on_axis,
      plasma_current=plasma_current,
      connection_length_target=connection_length_target,
      connection_length_divertor=connection_length_divertor,
      major_radius=major_radius,
      minor_radius=minor_radius,
      elongation_psi95=elongation_psi95,
      triangularity_psi95=triangularity_psi95,
      average_ion_mass=average_ion_mass,
      mean_ion_charge_state=mean_ion_charge_state,
      seed_impurity_weights=seed_impurity_weights,
      divertor_broadening_factor=divertor_broadening_factor,
      ratio_bpol_omp_to_bpol_avg=ratio_bpol_omp_to_bpol_avg,
      ne_tau=ne_tau,
      sheath_heat_transmission_factor=sheath_heat_transmission_factor,
      angle_of_incidence_target=angle_of_incidence_target,
      fraction_of_P_SOL_to_divertor=fraction_of_P_SOL_to_divertor,
      SOL_conduction_fraction=SOL_conduction_fraction,
      ratio_of_molecular_to_ion_mass=ratio_of_molecular_to_ion_mass,
      T_wall=T_wall,
      mach_separatrix=mach_separatrix,
      T_i_T_e_ratio_separatrix=T_i_T_e_ratio_separatrix,
      n_e_n_i_ratio_separatrix=n_e_n_i_ratio_separatrix,
      T_i_T_e_ratio_target=T_i_T_e_ratio_target,
      n_e_n_i_ratio_target=n_e_n_i_ratio_target,
      mach_target=mach_target,
      toroidal_flux_expansion=toroidal_flux_expansion,
  )

  initial_sol_model = _get_initial_sol_model(
      params=params,
      initial_guess=initial_guess,
      computation_mode=computation_mode,
      T_e_target_input=T_e_target,
  )

  # Resolve default iterations if needed
  if fixed_point_iterations is None:
    if solver_mode == extended_lengyel_enums.SolverMode.HYBRID:
      fixed_point_iterations = (
          extended_lengyel_defaults.HYBRID_FIXED_POINT_ITERATIONS
      )
    else:
      fixed_point_iterations = extended_lengyel_defaults.FIXED_POINT_ITERATIONS

  # 2. Solver Execution
  if computation_mode == extended_lengyel_enums.ComputationMode.FORWARD:
    # Forward mode may have multiple solutions, so we use multistart.
    return _run_forward_mode_multistart(
        initial_sol_model=initial_sol_model,
        solver_mode=solver_mode,
        fixed_point_iterations=fixed_point_iterations,
        newton_raphson_iterations=newton_raphson_iterations,
        newton_raphson_tol=newton_raphson_tol,
        diverted=diverted,
        enrichment_model_multiplier=enrichment_model_multiplier,
        multistart_num_guesses=multistart_num_guesses,
    )
  else:
    # Inverse mode always has a single solution so we use a single solver call.
    return _run_single_solver(
        initial_sol_model=initial_sol_model,
        computation_mode=computation_mode,
        solver_mode=solver_mode,
        fixed_point_iterations=fixed_point_iterations,
        newton_raphson_iterations=newton_raphson_iterations,
        newton_raphson_tol=newton_raphson_tol,
        diverted=diverted,
        enrichment_model_multiplier=enrichment_model_multiplier,
    )


# TODO(b/323504363): b/446608829 - Restructure functions and flow in this module for
# readability.
def _construct_parameters(
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
    seed_impurity_weights: Mapping[str, array_typing.FloatScalar],
    divertor_broadening_factor: array_typing.FloatScalar,
    ratio_bpol_omp_to_bpol_avg: array_typing.FloatScalar,
    ne_tau: array_typing.FloatScalar,
    sheath_heat_transmission_factor: array_typing.FloatScalar,
    angle_of_incidence_target: array_typing.FloatScalar,
    fraction_of_P_SOL_to_divertor: array_typing.FloatScalar,
    SOL_conduction_fraction: array_typing.FloatScalar,
    ratio_of_molecular_to_ion_mass: array_typing.FloatScalar,
    T_wall: array_typing.FloatScalar,
    mach_separatrix: array_typing.FloatScalar,
    T_i_T_e_ratio_separatrix: array_typing.FloatScalar,
    n_e_n_i_ratio_separatrix: array_typing.FloatScalar,
    T_i_T_e_ratio_target: array_typing.FloatScalar,
    n_e_n_i_ratio_target: array_typing.FloatScalar,
    mach_target: array_typing.FloatScalar,
    toroidal_flux_expansion: array_typing.FloatScalar,
) -> divertor_sol_1d_lib.ExtendedLengyelParameters:
  """Constructs ExtendedLengyelParameters with derived physics values."""
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

  return divertor_sol_1d_lib.ExtendedLengyelParameters(
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

<<<<<<< HEAD

def _get_initial_sol_model(
    params: divertor_sol_1d_lib.ExtendedLengyelParameters,
    initial_guess: divertor_sol_1d_lib.ExtendedLengyelInitialGuess | None,
    computation_mode: extended_lengyel_enums.ComputationMode,
    T_e_target_input: array_typing.FloatScalar | None,
) -> divertor_sol_1d_lib.DivertorSOL1D:
  """Constructs the initial DivertorSOL1D model from params and guess."""
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
      T_e_target_init = T_e_target_input
      assert isinstance(initial_guess, divertor_sol_1d_lib.InverseInitialGuess)
      c_z_prefactor_init = initial_guess.c_z_prefactor
    elif computation_mode == extended_lengyel_enums.ComputationMode.FORWARD:
      assert isinstance(initial_guess, divertor_sol_1d_lib.ForwardInitialGuess)
      T_e_target_init = initial_guess.T_e_target
      c_z_prefactor_init = extended_lengyel_defaults.DEFAULT_C_Z_PREFACTOR_INIT
    else:
      raise ValueError(f'Unknown computation mode: {computation_mode}')

  else:
    alpha_t_init = extended_lengyel_defaults.DEFAULT_ALPHA_T_INIT
    c_z_prefactor_init = extended_lengyel_defaults.DEFAULT_C_Z_PREFACTOR_INIT
    kappa_e_init = extended_lengyel_defaults.DEFAULT_KAPPA_E_INIT
    T_e_separatrix_init = extended_lengyel_defaults.DEFAULT_T_E_SEPARATRIX_INIT
=======
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
>>>>>>> 85479b3d (Fix IMAS required terms check and improve Extended Lengyel edge model)
    q_parallel_init = divertor_sol_1d_lib.calc_q_parallel(
        params=params,
        T_e_separatrix=T_e_separatrix_init,
        alpha_t=alpha_t_init,
    )

    if computation_mode == extended_lengyel_enums.ComputationMode.INVERSE:
<<<<<<< HEAD
      T_e_target_init = T_e_target_input
    elif computation_mode == extended_lengyel_enums.ComputationMode.FORWARD:
      T_e_target_init = (
          extended_lengyel_defaults.DEFAULT_T_E_TARGET_INIT_FORWARD
      )
=======
      T_e_target_init = T_e_target  # from input
    elif computation_mode == extended_lengyel_enums.ComputationMode.FORWARD:
      T_e_target_init = (
          extended_lengyel_defaults.DEFAULT_T_E_TARGET_INIT_FORWARD
      )  # eV.
>>>>>>> 85479b3d (Fix IMAS required terms check and improve Extended Lengyel edge model)
    else:
      raise ValueError(f'Unknown computation mode: {computation_mode}')

  initial_state = divertor_sol_1d_lib.ExtendedLengyelState(
      q_parallel=q_parallel_init,
      alpha_t=alpha_t_init,
      c_z_prefactor=c_z_prefactor_init,
      kappa_e=kappa_e_init,
      T_e_target=T_e_target_init,
  )

  return divertor_sol_1d_lib.DivertorSOL1D(
      params=params,
      state=initial_state,
  )


def _execute_solver(
    sol_model_input: divertor_sol_1d_lib.DivertorSOL1D,
    computation_mode: extended_lengyel_enums.ComputationMode,
    solver_mode: extended_lengyel_enums.SolverMode,
    fixed_point_iterations: int,
    newton_raphson_iterations: int,
    newton_raphson_tol: float,
) -> tuple[
    divertor_sol_1d_lib.DivertorSOL1D,
    extended_lengyel_solvers.ExtendedLengyelSolverStatus,
]:
  """Executes the appropriate solver based on modes."""
  match (computation_mode, solver_mode):
    case (
        extended_lengyel_enums.ComputationMode.INVERSE,
        extended_lengyel_enums.SolverMode.FIXED_POINT,
    ):
      return extended_lengyel_solvers.inverse_mode_fixed_point_solver(
          initial_sol_model=sol_model_input,
          iterations=fixed_point_iterations,
      )
    case (
        extended_lengyel_enums.ComputationMode.FORWARD,
        extended_lengyel_enums.SolverMode.FIXED_POINT,
    ):
      return extended_lengyel_solvers.forward_mode_fixed_point_solver(
          initial_sol_model=sol_model_input,
          iterations=fixed_point_iterations,
      )
    case (
        extended_lengyel_enums.ComputationMode.INVERSE,
        extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
    ):
      return extended_lengyel_solvers.inverse_mode_newton_solver(
          initial_sol_model=sol_model_input,
          maxiter=newton_raphson_iterations,
          tol=newton_raphson_tol,
      )
    case (
        extended_lengyel_enums.ComputationMode.FORWARD,
        extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
    ):
      return extended_lengyel_solvers.forward_mode_newton_solver(
          initial_sol_model=sol_model_input,
          maxiter=newton_raphson_iterations,
          tol=newton_raphson_tol,
      )
    case (
        extended_lengyel_enums.ComputationMode.INVERSE,
        extended_lengyel_enums.SolverMode.HYBRID,
    ):
      return extended_lengyel_solvers.inverse_mode_hybrid_solver(
          initial_sol_model=sol_model_input,
          fixed_point_iterations=fixed_point_iterations,
          newton_raphson_iterations=newton_raphson_iterations,
          newton_raphson_tol=newton_raphson_tol,
      )
    case (
        extended_lengyel_enums.ComputationMode.FORWARD,
        extended_lengyel_enums.SolverMode.HYBRID,
    ):
      return extended_lengyel_solvers.forward_mode_hybrid_solver(
          initial_sol_model=sol_model_input,
          fixed_point_iterations=fixed_point_iterations,
          newton_raphson_iterations=newton_raphson_iterations,
          newton_raphson_tol=newton_raphson_tol,
      )
    case _:
      raise ValueError(
          'Invalid computation and solver mode combination:'
          f' {computation_mode}, {solver_mode}'
      )


def _run_forward_mode_multistart(
    initial_sol_model: divertor_sol_1d_lib.DivertorSOL1D,
    solver_mode: extended_lengyel_enums.SolverMode,
    fixed_point_iterations: int,
    newton_raphson_iterations: int,
    newton_raphson_tol: float,
    diverted: bool,
    enrichment_model_multiplier: array_typing.FloatScalar,
    multistart_num_guesses: int,
) -> ExtendedLengyelOutputs:
  """Runs the forward mode solver with multi-start logic."""

  # 1. Generate grid of guesses
  T_grid = jnp.logspace(
      jnp.log10(extended_lengyel_defaults.MULTISTART_T_E_TARGET_MIN),
      jnp.log10(extended_lengyel_defaults.MULTISTART_T_E_TARGET_MAX),
      num=multistart_num_guesses - 1,
  )
  # Alternate alpha_t between MIN and MAX
  alpha_grid = jnp.where(
      jnp.arange(multistart_num_guesses - 1) % 2 == 0,
      extended_lengyel_defaults.MULTISTART_ALPHA_T_VALUES[0],
      extended_lengyel_defaults.MULTISTART_ALPHA_T_VALUES[1],
  )

  initial_state = initial_sol_model.state

  all_T_target_guesses = jnp.concatenate(
      [jnp.asarray([initial_state.T_e_target]), T_grid]
  )
  all_alpha_t_guesses = jnp.concatenate(
      [jnp.asarray([initial_state.alpha_t]), alpha_grid]
  )

  def _process_single_guess(T_e_target, alpha_t):
    # a. Make guess
    state_guess = dataclasses.replace(
        initial_state, T_e_target=T_e_target, alpha_t=alpha_t
    )
    model = dataclasses.replace(initial_sol_model, state=state_guess)

    # b. Execute solver
    model_out, status = _execute_solver(
        model,
        extended_lengyel_enums.ComputationMode.FORWARD,
        solver_mode,
        fixed_point_iterations,
        newton_raphson_iterations,
        newton_raphson_tol,
    )

    # c. Post-process output
    pressure_neutral_divertor, q_perpendicular_target = (
        _calc_post_processed_outputs(sol_model=model_out)
    )
    calculated_enrichment = {}
    params = model_out.params
    all_impurities = set(params.fixed_impurity_concentrations.keys()) | set(
        params.seed_impurity_weights.keys()
    )
    for impurity in all_impurities:
      calculated_enrichment[impurity] = jnp.where(
          diverted,
          extended_lengyel_formulas.calc_enrichment_kallenbach(
              pressure_neutral_divertor=pressure_neutral_divertor,
              ion_symbol=impurity,
              enrichment_multiplier=enrichment_model_multiplier,
          ),
          jnp.array(1.0, dtype=jax_utils.get_dtype()),
      )

    is_valid = (
        status.numerics_outcome.error != 1
        if isinstance(status.numerics_outcome, jax_root_finding.RootMetadata)
        else jnp.array(True)
    )

    output = ExtendedLengyelOutputs(
        T_e_target=model_out.state.T_e_target,
        pressure_neutral_divertor=pressure_neutral_divertor,
        alpha_t=model_out.state.alpha_t,
        kappa_e=model_out.state.kappa_e,
        c_z_prefactor=model_out.state.c_z_prefactor,
        q_parallel=model_out.state.q_parallel,
        q_perpendicular_target=q_perpendicular_target,
        T_e_separatrix=model_out.T_e_separatrix / 1e3,
        Z_eff_separatrix=model_out.Z_eff_separatrix,
        seed_impurity_concentrations=model_out.seed_impurity_concentrations,
        solver_status=status,
        calculated_enrichment=calculated_enrichment,
        roots=None,
        multiple_roots_found=jnp.array(False),
    )
    return output, is_valid

  batch_outputs, valid_mask = jax.vmap(_process_single_guess)(
      all_T_target_guesses, all_alpha_t_guesses
  )

  # 4. Selection Logic. Find viable solutions.
  nominal_valid = valid_mask[0]

  # 5. Set nominal solution (index 0) if valid, else argmin of distance.
  def _distance_squared(T_e):
    return (jnp.log(T_e) - jnp.log(initial_state.T_e_target)) ** 2

  distances = jax.vmap(_distance_squared)(batch_outputs.T_e_target)
  distances = jnp.where(valid_mask, distances, jnp.inf)

  nominal_idx = jax.lax.cond(
      nominal_valid,
      lambda: 0,
      lambda: jnp.argmin(distances),
  )

  nominal_output = jax.tree_util.tree_map(
      lambda x: x[nominal_idx], batch_outputs
  )

  # 6. Identify Multiple distinct and valid Roots
  valid_Ts = jnp.where(valid_mask, batch_outputs.T_e_target, jnp.nan)
  sorted_valid_Ts = jnp.sort(valid_Ts)
  diffs = jnp.diff(sorted_valid_Ts)
  is_gap = _roots_are_distinct(diffs, sorted_valid_Ts[:-1])
  num_valid = jnp.sum(valid_mask)
  has_gaps = jnp.sum(is_gap, where=jnp.isfinite(diffs)) > 0
  multiple_roots_found = jnp.logical_and(has_gaps, num_valid > 1)

  return dataclasses.replace(
      nominal_output,
      roots=batch_outputs,
      multiple_roots_found=multiple_roots_found,
  )


def _run_single_solver(
    initial_sol_model: divertor_sol_1d_lib.DivertorSOL1D,
    computation_mode: extended_lengyel_enums.ComputationMode,
    solver_mode: extended_lengyel_enums.SolverMode,
    fixed_point_iterations: int,
    newton_raphson_iterations: int,
    newton_raphson_tol: float,
    diverted: bool,
    enrichment_model_multiplier: array_typing.FloatScalar,
) -> ExtendedLengyelOutputs:
  """Runs a single solver instance (e.g. for Inverse mode)."""
  output_sol_model, solver_status = _execute_solver(
      initial_sol_model,
      computation_mode,
      solver_mode,
      fixed_point_iterations,
      newton_raphson_iterations,
      newton_raphson_tol,
  )

  pressure_neutral_divertor, q_perpendicular_target = (
      _calc_post_processed_outputs(sol_model=output_sol_model)
  )
  calculated_enrichment = {}
  params = output_sol_model.params
  all_impurities = set(params.fixed_impurity_concentrations.keys()) | set(
      params.seed_impurity_weights.keys()
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
      roots=None,
      multiple_roots_found=jnp.array(False)
      if computation_mode == extended_lengyel_enums.ComputationMode.FORWARD
      else None,
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
