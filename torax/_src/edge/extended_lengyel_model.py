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
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.edge import base
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_standalone
from torax._src.edge import runtime_params as edge_runtime_params
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.physics import psi_calculations
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name


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
  update_temperatures: array_typing.BoolScalar
  fixed_step_iterations: int
  newton_raphson_iterations: int
  newton_raphson_tol: float

  # --- Physical Parameters ---
  ne_tau: array_typing.FloatScalar
  divertor_broadening_factor: array_typing.FloatScalar
  ratio_bpol_omp_to_bpol_avg: array_typing.FloatScalar
  sheath_heat_transmission_factor: array_typing.FloatScalar
  fraction_of_P_SOL_to_divertor: array_typing.FloatScalar
  SOL_conduction_fraction: array_typing.FloatScalar
  ratio_of_molecular_to_ion_mass: array_typing.FloatScalar
  wall_temperature: array_typing.FloatScalar
  separatrix_mach_number: array_typing.FloatScalar
  separatrix_ratio_of_ion_to_electron_temp: array_typing.FloatScalar
  separatrix_ratio_of_electron_to_ion_density: array_typing.FloatScalar
  target_ratio_of_ion_to_electron_temp: array_typing.FloatScalar
  target_ratio_of_electron_to_ion_density: array_typing.FloatScalar
  target_mach_number: array_typing.FloatScalar

  # --- Geometry Parameters ---
  parallel_connection_length: array_typing.FloatScalar | None
  divertor_parallel_length: array_typing.FloatScalar | None
  toroidal_flux_expansion: array_typing.FloatScalar
  target_angle_of_incidence: array_typing.FloatScalar

  # --- Impurity parameters ---
  seed_impurity_weights: Mapping[str, array_typing.FloatScalar] | None
  fixed_impurity_concentrations: Mapping[str, array_typing.FloatScalar]
  enrichment_factor: Mapping[str, array_typing.FloatScalar]

  # --- Optional parameter for inverse mode ---
  target_electron_temp: array_typing.FloatScalar | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _ResolvedGeometricParams:
  """Container for parameters after resolution between geometry and edge config."""

  parallel_connection_length: array_typing.FloatScalar
  divertor_parallel_length: array_typing.FloatScalar
  toroidal_flux_expansion: array_typing.FloatScalar
  target_angle_of_incidence: array_typing.FloatScalar
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
  ) -> extended_lengyel_standalone.ExtendedLengyelOutputs:
    """Runs the extended Lengyel model using current TORAX state."""

    edge_params = runtime_params.edge
    assert isinstance(
        edge_params, RuntimeParams
    ), 'Edge parameters must be of type ExtendedLengyelModel.RuntimeParams'
    assert isinstance(
        geo, standard_geometry.StandardGeometry
    ), 'Geometry must be of type StandardGeometry'

    # Extract geometric parameters from TORAX geometry.

    # Extract and resolve geometric parameters, handling precedence and
    # warnings.
    resolved_geo_params = _resolve_geometric_parameters(
        geo, core_profiles, edge_params
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
        parallel_connection_length=resolved_geo_params.parallel_connection_length,
        divertor_parallel_length=resolved_geo_params.divertor_parallel_length,
        toroidal_flux_expansion=resolved_geo_params.toroidal_flux_expansion,
        target_angle_of_incidence=resolved_geo_params.target_angle_of_incidence,
        ratio_bpol_omp_to_bpol_avg=resolved_geo_params.ratio_bpol_omp_to_bpol_avg,
        divertor_broadening_factor=resolved_geo_params.divertor_broadening_factor,
        # Configurable parameters from RuntimeParams
        target_electron_temp=edge_params.target_electron_temp,
        seed_impurity_weights=edge_params.seed_impurity_weights,
        fixed_impurity_concentrations=edge_params.fixed_impurity_concentrations,
        computation_mode=edge_params.computation_mode,
        solver_mode=edge_params.solver_mode,
        ne_tau=edge_params.ne_tau,
        sheath_heat_transmission_factor=edge_params.sheath_heat_transmission_factor,
        fraction_of_P_SOL_to_divertor=edge_params.fraction_of_P_SOL_to_divertor,
        SOL_conduction_fraction=edge_params.SOL_conduction_fraction,
        ratio_of_molecular_to_ion_mass=edge_params.ratio_of_molecular_to_ion_mass,
        wall_temperature=edge_params.wall_temperature,
        separatrix_mach_number=edge_params.separatrix_mach_number,
        separatrix_ratio_of_ion_to_electron_temp=edge_params.separatrix_ratio_of_ion_to_electron_temp,
        separatrix_ratio_of_electron_to_ion_density=edge_params.separatrix_ratio_of_electron_to_ion_density,
        target_ratio_of_ion_to_electron_temp=edge_params.target_ratio_of_ion_to_electron_temp,
        target_ratio_of_electron_to_ion_density=edge_params.target_ratio_of_electron_to_ion_density,
        target_mach_number=edge_params.target_mach_number,
        fixed_step_iterations=edge_params.fixed_step_iterations,
        newton_raphson_iterations=edge_params.newton_raphson_iterations,
        newton_raphson_tol=edge_params.newton_raphson_tol,
    )


def _resolve_geometric_parameters(
    geo: standard_geometry.StandardGeometry,
    core_profiles: state.CoreProfiles,
    edge_params: RuntimeParams,
) -> _ResolvedGeometricParams:
  """Resolves geometric parameters from Geometry or RuntimeParams."""

  # Extract potential values from Geometry if existing
  geo_L_par = geo.connection_length_target
  geo_L_div = geo.connection_length_divertor
  geo_alpha = geo.target_angle_of_incidence
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
  if not geo.diverted:
    broadening = 1.0
  else:
    broadening = edge_params.divertor_broadening_factor

  # Resolve basic geometric parameters
  L_par = _resolve_param(
      'parallel_connection_length',
      geo_L_par,
      edge_params.parallel_connection_length,
  )
  L_div = _resolve_param(
      'divertor_parallel_length',
      geo_L_div,
      edge_params.divertor_parallel_length,
  )
  alpha_incidence = _resolve_param(
      'target_angle_of_incidence',
      geo_alpha,
      edge_params.target_angle_of_incidence,
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
      parallel_connection_length=L_par,
      divertor_parallel_length=L_div,
      toroidal_flux_expansion=flux_expansion,
      target_angle_of_incidence=alpha_incidence,
      ratio_bpol_omp_to_bpol_avg=ratio_bpol,
      divertor_broadening_factor=broadening,
  )


def _resolve_param(
    name: str,
    geo_val: array_typing.FloatScalar | None,
    config_val: array_typing.FloatScalar | None,
) -> array_typing.FloatScalar:
  """Helper to resolve a single parameter with logging."""
  match (geo_val, config_val):
    case (g_val, c_val) if g_val is not None:
      if c_val is not None:
        # Logging won't cause spamming under jit since will only be logged
        # during tracing.
        logging.warning(
            "ExtendedLengyelModel: Parameter '%s' found in both Geometry and"
            ' Config. Using Geometry value. Config value ignored.',
            name,
        )
      return g_val
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
