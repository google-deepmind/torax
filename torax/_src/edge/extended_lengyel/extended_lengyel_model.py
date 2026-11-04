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

"""Implementation of extended_lengyel instance of EdgeModel."""

import dataclasses
import logging
from typing import Mapping
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.edge import base
from torax._src.edge import edge_state as edge_state_lib
from torax._src.edge import runtime_params as edge_runtime_params_lib
from torax._src.edge.extended_lengyel import divertor_sol_1d as divertor_sol_1d_lib
from torax._src.edge.extended_lengyel import extended_lengyel_defaults
from torax._src.edge.extended_lengyel import extended_lengyel_enums
from torax._src.edge.extended_lengyel import extended_lengyel_formulas
from torax._src.edge.extended_lengyel import extended_lengyel_solvers
from torax._src.edge.extended_lengyel import extended_lengyel_standalone
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.physics import psi_calculations
from torax._src.solver import jax_root_finding
from torax._src.sources import core_sources as core_sources_lib


# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class InitialGuessRuntimeParams:
  """Runtime parameters for the initial guess."""

  alpha_t: array_typing.FloatScalar
  kappa_e: array_typing.FloatScalar
  T_e_separatrix: array_typing.FloatScalar
  T_e_target: array_typing.FloatScalar
  c_z_prefactor: array_typing.FloatScalar
  use_previous_step_as_guess: bool = jax.tree.static()

  def build_initial_guess(
      self,
      computation_mode: extended_lengyel_enums.ComputationMode,
  ) -> divertor_sol_1d_lib.ExtendedLengyelInitialGuess:
    """Builds the solver initial-guess dataclass for `computation_mode`."""
    return divertor_sol_1d_lib.build_initial_guess(
        computation_mode,
        alpha_t=self.alpha_t,
        kappa_e=self.kappa_e,
        T_e_separatrix=self.T_e_separatrix,
        T_e_target=self.T_e_target,
        c_z_prefactor=self.c_z_prefactor,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExtendedLengyelEdgeState(edge_state_lib.EdgeState):
  """Cross-timestep state for ExtendedLengyelModel.

  Attributes:
    pressure_neutral_divertor: Divertor neutral pressure [Pa] from the previous
      step.
    warm_start_guess: Solver initial guess (`ForwardInitialGuess` or
      `InverseInitialGuess`, with temperatures in [eV]) from the previous step.
    has_viable_warm_start: Whether `warm_start_guess` came from a viable
      previous solve.
  """

  pressure_neutral_divertor: jax.Array
  warm_start_guess: divertor_sol_1d_lib.ExtendedLengyelInitialGuess
  has_viable_warm_start: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RuntimeParams(edge_runtime_params_lib.RuntimeParams):
  """Runtime parameters for the extended Lengyel edge model."""

  # See extended_lengyel_standalone.py for documentation of these parameters.

  # --- Control Parameters ---
  computation_mode: extended_lengyel_enums.ComputationMode = jax.tree.static()
  solver_mode: extended_lengyel_enums.SolverMode = jax.tree.static()
  impurity_sot: extended_lengyel_enums.FixedImpuritySourceOfTruth = (
      jax.tree.static()
  )
  fixed_point_iterations: int
  fixed_point_relaxation_factor: float
  newton_raphson_iterations: int
  newton_raphson_tol: float
  # Number of guesses for forward mode multistart solver.
  multistart_num_guesses: int = jax.tree.static()

  # --- Physical Parameters ---
  n_e_tau: array_typing.FloatScalar
  divertor_broadening_factor: array_typing.FloatScalar
  ratio_bpol_omp_to_bpol_avg: array_typing.FloatScalar
  sheath_heat_transmission_factor: array_typing.FloatScalar
  fraction_of_P_SOL_to_divertor: array_typing.FloatScalar
  SOL_conduction_fraction: array_typing.FloatScalar
  ratio_of_molecular_to_ion_mass: array_typing.FloatScalar
  T_wall: array_typing.FloatScalar
  mach_separatrix: array_typing.FloatScalar
  T_i_T_e_ratio_separatrix: array_typing.FloatScalar
  n_e_n_i_ratio_separatrix: array_typing.FloatScalar
  T_i_T_e_ratio_target: array_typing.FloatScalar
  n_e_n_i_ratio_target: array_typing.FloatScalar
  mach_target: array_typing.FloatScalar

  # --- Geometry Parameters ---
  connection_length_target: array_typing.FloatScalar | None
  connection_length_divertor: array_typing.FloatScalar | None
  toroidal_flux_expansion: array_typing.FloatScalar
  angle_of_incidence_target: array_typing.FloatScalar
  diverted: array_typing.BoolScalar | None

  # --- Impurity parameters ---
  seed_impurity_weights: Mapping[str, array_typing.FloatScalar] | None
  fixed_impurity_concentrations: Mapping[str, array_typing.FloatScalar]
  enrichment_factor: Mapping[str, array_typing.FloatScalar] | None
  use_enrichment_model: bool = jax.tree.static()
  enrichment_model_multiplier: array_typing.FloatScalar

  # --- Optional parameter for inverse mode ---
  T_e_target: array_typing.FloatScalar | None

  # --- Initial Guess ---
  initial_guess: InitialGuessRuntimeParams


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _ResolvedGeometricParams:
  """Container for parameters after resolution between geometry and edge config."""

  connection_length_target: array_typing.FloatScalar
  connection_length_divertor: array_typing.FloatScalar
  toroidal_flux_expansion: array_typing.FloatScalar
  angle_of_incidence_target: array_typing.FloatScalar
  ratio_bpol_omp_to_bpol_avg: array_typing.FloatScalar
  divertor_broadening_factor: array_typing.FloatScalar


@dataclasses.dataclass(frozen=True, eq=False)
class ExtendedLengyelModel(base.EdgeModel):
  """Adapter for running the extended Lengyel model within TORAX."""

  def initial_state(
      self,
      edge_runtime_params: edge_runtime_params_lib.RuntimeParams,
  ) -> ExtendedLengyelEdgeState:
    """Returns the initial cross-timestep ExtendedLengyelEdgeState."""
    assert isinstance(
        edge_runtime_params, RuntimeParams
    ), 'Edge parameters must be of type ExtendedLengyelModel.RuntimeParams'

    return ExtendedLengyelEdgeState(
        pressure_neutral_divertor=jnp.asarray(
            extended_lengyel_defaults.INITIAL_PRESSURE_NEUTRAL_DIVERTOR,
            dtype=jax_utils.get_dtype(),
        ),
        warm_start_guess=edge_runtime_params.initial_guess.build_initial_guess(
            edge_runtime_params.computation_mode
        ),
        has_viable_warm_start=jnp.asarray(False, dtype=jnp.bool_),
    )

  def __call__(
      self,
      edge_runtime_params: edge_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: core_profiles_lib.CoreProfiles,
      core_sources: core_sources_lib.CoreSources,
      edge_state: edge_state_lib.EdgeState,
  ) -> tuple[
      extended_lengyel_standalone.ExtendedLengyelOutputs,
      ExtendedLengyelEdgeState,
  ]:
    """Runs the extended Lengyel model using current TORAX state."""

    edge_params = edge_runtime_params
    assert isinstance(
        edge_params, RuntimeParams
    ), 'Edge parameters must be of type ExtendedLengyelModel.RuntimeParams'
    assert isinstance(
        geo, standard_geometry.StandardGeometry
    ), 'Geometry must be of type StandardGeometry'
    assert isinstance(edge_state, ExtendedLengyelEdgeState)

    # Determine diverted status. For MEQ geometries, this is provided by the
    # geometry object. For other types, it must be provided in the config.
    # The config validation ensures that one and only one of these is valid.

    diverted = _get_diverted(geo, edge_params)

    # Extract and resolve geometric parameters, handling precedence and
    # warnings.
    resolved_geo_params = _resolve_geometric_parameters(
        geo, core_profiles, edge_params, diverted
    )

    # Calculate normalized poloidal flux (psi_norm) on the face grid.
    # Used to interpolate geometry quantities at psi_norm = 0.95.
    psi_norm_face = core_profiles.psi_norm_face

    # Interpolate elongation and triangularity at psi_norm = 0.95
    elongation_psi95 = jnp.interp(0.95, psi_norm_face, geo.elongation_face)
    triangularity_psi95 = jnp.interp(0.95, psi_norm_face, geo.delta_face)

    # Extract plasma state parameters from CoreProfiles at the LCFS
    separatrix_electron_density = core_profiles.kinetic.n_e.right_face_value[0]

    # Calculate ion properties
    n_i_sep = core_profiles.kinetic.n_i.right_face_value[0]
    n_imp_sep = core_profiles.kinetic.n_impurity.right_face_value[0]
    A_i_sep = core_profiles.ions.A_i
    A_imp_sep = core_profiles.ions.A_impurity_face[-1]
    mean_ion_charge_state = separatrix_electron_density / (n_i_sep + n_imp_sep)
    average_ion_mass = (
        A_i_sep * n_i_sep + A_imp_sep * n_imp_sep
    ) / separatrix_electron_density

    # Calculate total power crossing the separatrix from core sources.
    # This sums all electron and ion heating sources (and sinks).
    # Note: `total_sources` returns dPower/drho_norm (already multiplied by V').
    dP_e_drho_norm = core_sources.total_sources('power_density_e', geo)
    dP_i_drho_norm = core_sources.total_sources('power_density_i', geo)
    # Integrate over rho_norm to get total power out of the separatrix [W].
    P_SOL_total = math_utils.cell_integration(
        dP_e_drho_norm + dP_i_drho_norm, geo
    )

    # Extract power crossing separatrix
    # Ensure strictly positive power for log calculations in the model.
    power_crossing_separatrix = jnp.maximum(
        P_SOL_total, constants.CONSTANTS.eps
    )

    fixed_impurity_concentrations = edge_params.fixed_impurity_concentrations
    # If the source of truth for fixed impurities is the core, calculate the
    # edge concentrations from the core ratios.
    if (
        edge_params.impurity_sot
        == extended_lengyel_enums.FixedImpuritySourceOfTruth.CORE
    ):
      # Initialization
      fixed_impurity_concentrations = {}
      impurity_params = runtime_params.plasma_composition.impurity
      # Only support use of extended Lengyel for n_e_ratios impurity mode.
      # TODO(b/446608829): Support other modes for forward mode core SoT.
      assert isinstance(impurity_params, electron_density_ratios.RuntimeParams)

      for species, ratio_face in impurity_params.n_e_ratios_face.items():
        # Skip if it's a seeded impurity
        if (
            edge_params.seed_impurity_weights
            and species in edge_params.seed_impurity_weights
        ):
          continue

        # Calculate edge concentration: c_edge = c_core_lcfs * enrichment_factor
        if edge_params.use_enrichment_model:
          pressure_neutral_divertor = edge_state.pressure_neutral_divertor

          enrichment = jnp.where(
              diverted,
              extended_lengyel_formulas.calc_enrichment_kallenbach(
                  pressure_neutral_divertor=pressure_neutral_divertor,
                  ion_symbol=species,
                  enrichment_multiplier=edge_params.enrichment_model_multiplier,
              ),
              1.0,  # Limited geometry: no divertor enrichment.
          )
        elif edge_params.enrichment_factor is not None:
          enrichment = edge_params.enrichment_factor[species]
        else:
          raise ValueError(
              'enrichment_factor must be provided when use_enrichment_model is'
              ' False.'
          )
        fixed_impurity_concentrations[species] = ratio_face[-1] * enrichment

    # Determine initial guesses
    initial_guess = _get_initial_guess(edge_params, edge_state)

    # Call the standalone runner with combined parameters
    outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
        # Dynamic state from TORAX
        power_crossing_separatrix=power_crossing_separatrix,
        separatrix_electron_density=separatrix_electron_density,
        main_ion_charge=core_profiles.ions.Z_i_face[-1],
        mean_ion_charge_state=mean_ion_charge_state,
        magnetic_field_on_axis=geo.B_0,
        plasma_current=core_profiles.poloidal_flux.I_p_profile_face[-1],
        major_radius=geo.R_major,
        minor_radius=geo.a_minor,
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
        average_ion_mass=average_ion_mass,
        # Geometry parameters that may come from geometry or RuntimeParams
        connection_length_target=resolved_geo_params.connection_length_target,
        connection_length_divertor=resolved_geo_params.connection_length_divertor,
        toroidal_flux_expansion=resolved_geo_params.toroidal_flux_expansion,
        angle_of_incidence_target=resolved_geo_params.angle_of_incidence_target,
        ratio_bpol_omp_to_bpol_avg=resolved_geo_params.ratio_bpol_omp_to_bpol_avg,
        divertor_broadening_factor=resolved_geo_params.divertor_broadening_factor,
        # Configurable parameters from RuntimeParams
        T_e_target=edge_params.T_e_target,
        seed_impurity_weights=edge_params.seed_impurity_weights,
        fixed_impurity_concentrations=fixed_impurity_concentrations,
        computation_mode=edge_params.computation_mode,
        solver_mode=edge_params.solver_mode,
        n_e_tau=edge_params.n_e_tau,
        sheath_heat_transmission_factor=edge_params.sheath_heat_transmission_factor,
        fraction_of_P_SOL_to_divertor=edge_params.fraction_of_P_SOL_to_divertor,
        SOL_conduction_fraction=edge_params.SOL_conduction_fraction,
        ratio_of_molecular_to_ion_mass=edge_params.ratio_of_molecular_to_ion_mass,
        T_wall=edge_params.T_wall,
        mach_separatrix=edge_params.mach_separatrix,
        T_i_T_e_ratio_separatrix=edge_params.T_i_T_e_ratio_separatrix,
        n_e_n_i_ratio_separatrix=edge_params.n_e_n_i_ratio_separatrix,
        T_i_T_e_ratio_target=edge_params.T_i_T_e_ratio_target,
        n_e_n_i_ratio_target=edge_params.n_e_n_i_ratio_target,
        mach_target=edge_params.mach_target,
        fixed_point_iterations=edge_params.fixed_point_iterations,
        fixed_point_relaxation_factor=edge_params.fixed_point_relaxation_factor,
        newton_raphson_iterations=edge_params.newton_raphson_iterations,
        newton_raphson_tol=edge_params.newton_raphson_tol,
        multistart_num_guesses=edge_params.multistart_num_guesses,
        enrichment_model_multiplier=edge_params.enrichment_model_multiplier,
        diverted=diverted,
        use_enrichment_model=edge_params.use_enrichment_model,
        enrichment_factor=edge_params.enrichment_factor,
        impurity_sot=edge_params.impurity_sot,
        initial_guess=initial_guess,
    )

    updated_state = ExtendedLengyelEdgeState(
        pressure_neutral_divertor=jnp.asarray(
            outputs.pressure_neutral_divertor, dtype=jax_utils.get_dtype()
        ),
        warm_start_guess=outputs.build_initial_guess(
            edge_params.computation_mode
        ),
        has_viable_warm_start=jnp.asarray(
            _outputs_are_viable(outputs), dtype=jnp.bool_
        ),
    )
    return outputs, updated_state


def _resolve_geometric_parameters(
    geo: standard_geometry.StandardGeometry,
    core_profiles: core_profiles_lib.CoreProfiles,
    edge_params: RuntimeParams,
    diverted: array_typing.BoolScalar,
) -> _ResolvedGeometricParams:
  """Resolves geometric parameters from Geometry or RuntimeParams."""

  # Extract potential values from Geometry if existing
  geo_L_par = geo.connection_length_target
  geo_L_div = geo.connection_length_divertor
  geo_alpha = geo.angle_of_incidence_target
  if geo.R_target is not None and geo.R_OMP is not None:
    geo_flux_exp = geo.R_target / geo.R_OMP
  else:
    geo_flux_exp = None
  if geo.B_pol_OMP is not None:
    bpol_avg_lcfs = jnp.sqrt(
        psi_calculations.calc_bpol_squared(geo, core_profiles.poloidal_flux.psi)
    )[-1]
    geo_ratio_bpol = jnp.abs(geo.B_pol_OMP) / bpol_avg_lcfs
  else:
    geo_ratio_bpol = None

  # If limited, set divertor broadening to 1.0.
  broadening = jnp.where(diverted, edge_params.divertor_broadening_factor, 1.0)

  # Resolve basic geometric parameters
  L_par = _resolve_param(
      'connection_length_target',
      geo_L_par,
      edge_params.connection_length_target,
  )
  L_div = _resolve_param(
      'connection_length_divertor',
      geo_L_div,
      edge_params.connection_length_divertor,
  )
  alpha_incidence = _resolve_param(
      'angle_of_incidence_target',
      geo_alpha,
      edge_params.angle_of_incidence_target,
  )
  flux_expansion = _resolve_param(
      'toroidal_flux_expansion',
      geo_flux_exp,
      edge_params.toroidal_flux_expansion,
  )
  ratio_bpol = _resolve_param(
      'ratio_bpol_omp_to_bpol_avg',
      geo_ratio_bpol,
      edge_params.ratio_bpol_omp_to_bpol_avg,
  )

  return _ResolvedGeometricParams(
      connection_length_target=L_par,
      connection_length_divertor=L_div,
      toroidal_flux_expansion=flux_expansion,
      angle_of_incidence_target=alpha_incidence,
      ratio_bpol_omp_to_bpol_avg=ratio_bpol,
      divertor_broadening_factor=broadening,
  )


def _resolve_param(
    name: str,
    geo_val: array_typing.FloatScalar | None,
    config_val: array_typing.FloatScalar | None,
) -> array_typing.FloatScalar:
  """Helper to resolve a single parameter with logging.

  This function determines the definitive value for a geometric parameter that
  can be provided by either the `Geometry` object or the `RuntimeParams` config.
  The precedence is as follows:

  1.  If a valid value (non-zero, non-NaN) is available from `geo_val`, it is
      always preferred.
  2.  If `geo_val` is present but invalid (0 or NaN), `config_val` is used as a
      fallback if available.
  3.  If `geo_val` is `None`, `config_val` is used.
  4.  If both are `None`, or if `geo_val` is invalid and `config_val` is None,
      an error is raised if TORAX errors are enabled.

  Args:
    name: Variable name.
    geo_val: The value from the `Geometry` object.
    config_val: The value from the `RuntimeParams` config.

  Returns:
    The resolved parameter value.

  Raises:
    RunTimeError: If both `geo_val` and `config_val` are `None` or if `geo_val`
      is invalid and `config_val` is `None`.
  """
  match (geo_val, config_val):
    case (g_val, c_val) if g_val is not None:
      # Check if geometry value is valid (non-zero and non-NaN).
      is_valid = jnp.logical_and(
          jnp.not_equal(g_val, 0.0), jnp.logical_not(jnp.isnan(g_val))
      )

      if c_val is not None:
        return jax.lax.select(is_valid, g_val, c_val)
      else:
        # No fallback provided. Raise error if g_val is invalid.
        # This will raise a RuntimeError at runtime if is_valid is False and
        # TORAX errors are enabled.
        return jax_utils.error_if(
            jnp.asarray(g_val),
            jnp.logical_not(is_valid),
            f"ExtendedLengyelModel: Geometry parameter '{name}' is invalid"
            ' (0 or NaN) and no fallback value provided in'
            ' ExtendedLengyelConfig.',
        )

    case (None, c_val) if c_val is not None:
      logging.warning(
          "ExtendedLengyelModel: Parameter '%s' not found in Geometry. Using"
          ' Config value. It is recommended to use a Geometry source that'
          ' provides this parameter for full self-consistency.',
          name,
      )
      return c_val
    case (None, None):
      raise ValueError(
          f"ExtendedLengyelModel: Parameter '{name}' must be provided either"
          ' via the Geometry or the ExtendedLengyelConfig.'
      )
    case _:
      raise ValueError(
          'ExtendedLengyelModel: Unexpected state resolving parameter'
          f" '{name}'."
      )


def _get_diverted(
    geo: standard_geometry.StandardGeometry,
    edge_params: RuntimeParams,
) -> array_typing.BoolScalar:
  """Determines diverted status."""
  # To avoid None values in the jnp.where, use safe defaults.
  # The pydantic config validation ensures that the values that override the
  # None are never used.
  safe_meq_diverted = geo.diverted if geo.diverted is not None else False
  safe_params_diverted = (
      edge_params.diverted if edge_params.diverted is not None else False
  )
  is_meq = geo.geometry_type == geometry.GeometryType.MEQ
  return jnp.where(is_meq, safe_meq_diverted, safe_params_diverted)


def _outputs_are_viable(
    outputs: extended_lengyel_standalone.ExtendedLengyelOutputs,
) -> jax.Array:
  """Returns whether outputs are viable to use as next step's warm start."""
  # Check if solver had a "bad" Physics outcome. If so, do not use
  # it. Note that a "bad" PhysicsOutcome is not necessarily an error; it can
  # be a valid physical state (e.g., detachment). However, if reattaching,
  # then it's better not to start from such a cold state.
  bad_physics = (
      outputs.solver_status.physics_outcome
      != extended_lengyel_solvers.PhysicsOutcome.SUCCESS
  )
  # Numerics outcome: for NR solver, error=1 means not converged.
  # For fixed-point solver, it's always SUCCESS.
  numerics_outcome = outputs.solver_status.numerics_outcome
  if isinstance(numerics_outcome, jax_root_finding.RootMetadata):
    # NR error code: 0=fine, 1=not converged, 2=coarse (acceptable).
    bad_numerics = numerics_outcome.error == 1
  else:
    # Fixed-point always succeeds
    bad_numerics = False
  return jnp.logical_and(
      outputs.valid_for_core_update,
      jnp.logical_not(jnp.logical_or(bad_physics, bad_numerics)),
  )


def _get_initial_guess(
    edge_params: RuntimeParams,
    edge_state: ExtendedLengyelEdgeState,
) -> divertor_sol_1d_lib.ExtendedLengyelInitialGuess:
  """Resolves the initial guess for the solver state variables.

  Uses `edge_state.warm_start_guess` if `use_previous_step_as_guess` is
  enabled and `edge_state.has_viable_warm_start` is True; otherwise falls
  back to the configured `edge_params.initial_guess` (overrides or
  physics-based defaults).

  Note: When a step has a bad solver outcome (Newton-Raphson error=1, or
  physics_outcome != SUCCESS) or is not valid for core updates (e.g.,
  diverted=False), `has_viable_warm_start` is False so the previous outputs
  are not used as a warm start. This prevents the solver from inheriting a
  bad state that can drag it into cold root basins in subsequent steps.

  Args:
    edge_params: Runtime parameters for the edge model.
    edge_state: Cross-timestep state from the previous step.

  Returns:
    Resolved initial guess object.
  """
  config_guess = edge_params.initial_guess.build_initial_guess(
      edge_params.computation_mode
  )
  if not edge_params.initial_guess.use_previous_step_as_guess:
    return config_guess
  return jax.lax.cond(
      edge_state.has_viable_warm_start,
      lambda: edge_state.warm_start_guess,
      lambda: config_guess,
  )
