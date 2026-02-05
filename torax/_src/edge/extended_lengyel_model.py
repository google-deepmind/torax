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
import enum
import logging
from typing import Mapping
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.edge import base
from torax._src.edge import divertor_sol_1d as divertor_sol_1d_lib
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import extended_lengyel_standalone
from torax._src.edge import runtime_params as edge_runtime_params
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.physics import psi_calculations
from torax._src.solver import jax_root_finding
from torax._src.sources import source_profiles as source_profiles_lib


# pylint: disable=invalid-name
class FixedImpuritySourceOfTruth(enum.StrEnum):
  """Source of truth for fixed impurity concentrations when using an edge model.

  Determines how impurity concentrations are handled between the core plasma
  simulation and the edge model.

  Attributes:
    CORE: * The core impurity profiles are the source of truth. * The edge
      model's impurity concentrations are derived from the core values at the
      last closed flux surface: `c_edge = c_core_face[-1] * enrichment_factor`.
    EDGE: * The edge model's `fixed_impurity_concentrations` are the source of
      truth. * The core impurity profiles (n_e_ratios) are scaled to match the
      values determined by the edge model. runtime_params still sets the profile
        shape: `c_core = c_core / c_core_face[-1] * c_edge / enrichment_factor`.

  Note: For seeded impurities in the extended Lengyel edge model, the source of
  truth is always the edge model, regardless of this setting. This enum only
  controls the behavior for fixed impurities in that case.
  """

  CORE = 'core'
  EDGE = 'edge'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class InitialGuessRuntimeParams:
  """Runtime parameters for the initial guess."""

  alpha_t: array_typing.FloatScalar
  alpha_t_provided: array_typing.BoolScalar
  kappa_e: array_typing.FloatScalar
  kappa_e_provided: array_typing.BoolScalar
  T_e_separatrix: array_typing.FloatScalar
  T_e_separatrix_provided: array_typing.BoolScalar
  T_e_target: array_typing.FloatScalar
  T_e_target_provided: array_typing.BoolScalar
  c_z_prefactor: array_typing.FloatScalar
  c_z_prefactor_provided: array_typing.BoolScalar
  use_previous_step_as_guess: bool = dataclasses.field(
      metadata={'static': True}
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(edge_runtime_params.RuntimeParams):
  """Runtime parameters for the extended Lengyel edge model."""

  # See extended_lengyel_standalone.py for documentation of these parameters.

  # --- Control Parameters ---
  computation_mode: extended_lengyel_enums.ComputationMode = dataclasses.field(
      metadata={'static': True}
  )
  solver_mode: extended_lengyel_enums.SolverMode = dataclasses.field(
      metadata={'static': True}
  )
  impurity_sot: FixedImpuritySourceOfTruth = dataclasses.field(
      metadata={'static': FixedImpuritySourceOfTruth.CORE}
  )
  # Not static to allow rapid sensitivity checking of edge-model impact.
  update_temperatures: array_typing.BoolScalar
  update_impurities: array_typing.BoolScalar
  fixed_point_iterations: int
  newton_raphson_iterations: int
  newton_raphson_tol: float

  # --- Physical Parameters ---
  # TODO(b/434175938): (v2) Rename to n_e_tau for consistency.
  ne_tau: array_typing.FloatScalar
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
  enrichment_factor: Mapping[str, array_typing.FloatScalar]
  use_enrichment_model: bool = dataclasses.field(metadata={'static': True})
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

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_sources: source_profiles_lib.SourceProfiles,
      previous_edge_outputs: base.EdgeModelOutputs | None = None,
  ) -> extended_lengyel_standalone.ExtendedLengyelOutputs:
    """Runs the extended Lengyel model using current TORAX state."""

    edge_params = runtime_params.edge
    assert isinstance(
        edge_params, RuntimeParams
    ), 'Edge parameters must be of type ExtendedLengyelModel.RuntimeParams'
    assert isinstance(
        geo, standard_geometry.StandardGeometry
    ), 'Geometry must be of type StandardGeometry'

    # Determine diverted status. For FBT geometries, this is provided by the
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
    psi_face = core_profiles.psi.face_value()
    psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])

    # Interpolate elongation and triangularity at psi_norm = 0.95
    elongation_psi95 = jnp.interp(0.95, psi_norm_face, geo.elongation_face)
    triangularity_psi95 = jnp.interp(0.95, psi_norm_face, geo.delta_face)

    # Extract plasma state parameters from CoreProfiles at the LCFS
    separatrix_electron_density = core_profiles.n_e.face_value()[-1]

    # Calculate ion properties
    n_i_sep = core_profiles.n_i.face_value()[-1]
    n_imp_sep = core_profiles.n_impurity.face_value()[-1]
    A_i_sep = core_profiles.A_i
    A_imp_sep = core_profiles.A_impurity_face[-1]
    mean_ion_charge_state = separatrix_electron_density / (n_i_sep + n_imp_sep)
    average_ion_mass = (
        A_i_sep * n_i_sep + A_imp_sep * n_imp_sep
    ) / separatrix_electron_density

    # Calculate total power crossing the separatrix from core sources.
    # This sums all electron and ion heating sources (and sinks).
    # Note: `total_sources` returns dPower/drho_norm (already multiplied by V').
    dP_e_drho_norm = core_sources.total_sources('T_e', geo)
    dP_i_drho_norm = core_sources.total_sources('T_i', geo)
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
    if edge_params.impurity_sot == FixedImpuritySourceOfTruth.CORE:
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
        # Enrichment factor exists for all species (validated in config)
        fixed_impurity_concentrations[species] = (
            ratio_face[-1] * edge_params.enrichment_factor[species]
        )

    # Determine initial guesses
    initial_guess = _get_initial_guess(edge_params, previous_edge_outputs)

    # Call the standalone runner with combined parameters
    return extended_lengyel_standalone.run_extended_lengyel_standalone(
        # Dynamic state from TORAX
        power_crossing_separatrix=power_crossing_separatrix,
        separatrix_electron_density=separatrix_electron_density,
        main_ion_charge=core_profiles.Z_i_face[-1],
        mean_ion_charge_state=mean_ion_charge_state,
        magnetic_field_on_axis=geo.B_0,
        plasma_current=core_profiles.Ip_profile_face[-1],
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
        ne_tau=edge_params.ne_tau,
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
        newton_raphson_iterations=edge_params.newton_raphson_iterations,
        newton_raphson_tol=edge_params.newton_raphson_tol,
        enrichment_model_multiplier=edge_params.enrichment_model_multiplier,
        diverted=diverted,
        initial_guess=initial_guess,
    )


def _resolve_geometric_parameters(
    geo: standard_geometry.StandardGeometry,
    core_profiles: state.CoreProfiles,
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
        psi_calculations.calc_bpol_squared(geo, core_profiles.psi)
    )[-1]
    geo_ratio_bpol = jnp.abs(geo.B_pol_OMP) / jnp.sqrt(bpol_avg_lcfs)
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
            g_val,
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


def _get_diverted(
    geo: standard_geometry.StandardGeometry,
    edge_params: RuntimeParams,
) -> array_typing.BoolScalar:
  """Determines diverted status."""
  # To avoid None values in the jnp.where, use safe defaults.
  # The pydantic config validation ensures that the values that override the
  # None are never used.
  safe_fbt_diverted = geo.diverted if geo.diverted is not None else False
  safe_params_diverted = (
      edge_params.diverted if edge_params.diverted is not None else False
  )
  is_fbt = geo.geometry_type == geometry.GeometryType.FBT
  return jnp.where(is_fbt, safe_fbt_diverted, safe_params_diverted)


def _get_initial_guess(
    edge_params: RuntimeParams,
    previous_edge_outputs: base.EdgeModelOutputs | None = None,
) -> divertor_sol_1d_lib.ExtendedLengyelInitialGuess:
  """Resolves the initial guess for the solver state variables.

  Priorities:
  1. Previous simulation state (warm start), if available, enabled, and the
     previous solver had a good outcome (converged and no physics errors).
  2. Configuration overrides (if provided).
  3. Physics-based defaults.

  Note: If the previous solver had a bad outcome (Newton-Raphson error=1, or
  physics_outcome != SUCCESS), the previous outputs are NOT used as a warm
  start. This prevents the solver from inheriting a bad state that can drag
  it into cold root basins in subsequent time steps.

  Args:
    edge_params: Runtime parameters for the edge model.
    previous_edge_outputs: Outputs from the previous time step.

  Returns:
    Resolved initial guess object.
  """
  initial_guess = edge_params.initial_guess

  # Check if we have valid previous outputs for warm start.
  # `has_previous` is a Python bool for attribute access safety.
  # `viable_previous` is a JAX bool for solver status, used in jnp.where.
  has_previous = previous_edge_outputs is not None

  # Default: usable if exists.
  viable_previous = jnp.array(True, dtype=jnp.bool_)
  if has_previous:
    assert isinstance(
        previous_edge_outputs,
        extended_lengyel_standalone.ExtendedLengyelOutputs,
    )
    # Check if previous solver had a "bad" Physics outcome. If so, do not use
    # it. Note that a "bad" PhysicsOutcome is not necessarily an error; it can
    # be a valid physical state (e.g., detachment). However, if reattaching,
    # then it's better not to start from such a cold state.
    bad_physics = (
        previous_edge_outputs.solver_status.physics_outcome
        != extended_lengyel_solvers.PhysicsOutcome.SUCCESS
    )
    # Numerics outcome: for NR solver, error=1 means not converged.
    # For fixed-point solver, it's always SUCCESS.
    numerics_outcome = previous_edge_outputs.solver_status.numerics_outcome
    if isinstance(numerics_outcome, jax_root_finding.RootMetadata):
      # NR error code: 0=fine, 1=not converged, 2=coarse (acceptable).
      bad_numerics = numerics_outcome.error == 1
    else:
      # Fixed-point always succeeds
      bad_numerics = False
    viable_previous = jnp.logical_not(
        jnp.logical_or(bad_physics, bad_numerics)
    )

  # Warm start logic: use previous only if enabled AND we have previous outputs
  # AND those previous outputs are usable (good solver outcome).
  use_previous = (
      initial_guess.use_previous_step_as_guess
      and has_previous
      and viable_previous
  )

  # Defaults
  alpha_t_default = jnp.array(
      extended_lengyel_defaults.DEFAULT_ALPHA_T_INIT,
      dtype=jax_utils.get_dtype(),
  )
  kappa_e_default = jnp.array(
      extended_lengyel_defaults.KAPPA_E_0, dtype=jax_utils.get_dtype()
  )
  T_e_separatrix_default = jnp.array(
      extended_lengyel_defaults.DEFAULT_T_E_SEPARATRIX_INIT,
      dtype=jax_utils.get_dtype(),
  )
  T_e_target_default = jnp.array(
      extended_lengyel_defaults.DEFAULT_T_E_TARGET_INIT_FORWARD,
      dtype=jax_utils.get_dtype(),
  )
  c_z_default = jnp.array(
      extended_lengyel_defaults.DEFAULT_C_Z_PREFACTOR_INIT,
      dtype=jax_utils.get_dtype(),
  )

  # Helper for priority resolution: Previous > Config > Default
  # When previous_val is None (no previous outputs), we use 0.0 as a safe
  # placeholder. This value is never selected because use_previous is False
  # when there are no previous outputs.
  def _resolve(previous_val, config_val, config_provided, default_val):
    safe_previous = previous_val if previous_val is not None else 0.0
    val_config_or_default = jnp.where(config_provided, config_val, default_val)
    return jnp.where(use_previous, safe_previous, val_config_or_default)

  # Resolve common variables
  alpha_t = _resolve(
      previous_edge_outputs.alpha_t if has_previous else None,
      initial_guess.alpha_t,
      initial_guess.alpha_t_provided,
      alpha_t_default,
  )
  kappa_e = _resolve(
      previous_edge_outputs.kappa_e if has_previous else None,
      initial_guess.kappa_e,
      initial_guess.kappa_e_provided,
      kappa_e_default,
  )

  # T_e_separatrix: convert previous value from keV to eV
  T_e_separatrix = _resolve(
      previous_edge_outputs.T_e_separatrix * 1e3 if has_previous else None,
      initial_guess.T_e_separatrix,
      initial_guess.T_e_separatrix_provided,
      T_e_separatrix_default,
  )

  # Resolve mode-specific variables
  if (
      edge_params.computation_mode
      == extended_lengyel_enums.ComputationMode.FORWARD
  ):
    T_e_target = _resolve(
        previous_edge_outputs.T_e_target if has_previous else None,
        initial_guess.T_e_target,
        initial_guess.T_e_target_provided,
        T_e_target_default,
    )
    return divertor_sol_1d_lib.ForwardInitialGuess(
        alpha_t=alpha_t,
        kappa_e=kappa_e,
        T_e_separatrix=T_e_separatrix,
        T_e_target=T_e_target,
    )
  else:
    c_z_prefactor = _resolve(
        previous_edge_outputs.c_z_prefactor if has_previous else None,
        initial_guess.c_z_prefactor,
        initial_guess.c_z_prefactor_provided,
        c_z_default,
    )
    return divertor_sol_1d_lib.InverseInitialGuess(
        alpha_t=alpha_t,
        kappa_e=kappa_e,
        T_e_separatrix=T_e_separatrix,
        c_z_prefactor=c_z_prefactor,
    )
