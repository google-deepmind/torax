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
from typing import Mapping
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.edge import base
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_standalone
from torax._src.edge import runtime_params as edge_runtime_params
from torax._src.geometry import geometry
from torax._src.output_tools import post_processing

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
  fixed_step_iterations: int
  newton_raphson_iterations: int
  newton_raphson_tol: float

  # --- Physical Parameters ---
  ne_tau: array_typing.FloatScalar
  divertor_broadening_factor: array_typing.FloatScalar
  ratio_of_upstream_to_average_poloidal_field: array_typing.FloatScalar
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
  seed_impurity_weights: Mapping[str, array_typing.FloatScalar]
  fixed_impurity_concentrations: Mapping[str, array_typing.FloatScalar]
  enrichment_factor: Mapping[str, array_typing.FloatScalar]

  # --- Optional parameter for inverse mode ---
  target_electron_temp: array_typing.FloatScalar | None


@dataclasses.dataclass(frozen=True, eq=False)
class ExtendedLengyelModel(base.EdgeModel):
  """Adapter for running the extended Lengyel model within TORAX."""

  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      post_processed_outputs: post_processing.PostProcessedOutputs,
  ) -> extended_lengyel_standalone.ExtendedLengyelOutputs:
    """Runs the extended Lengyel model using current TORAX state."""

    edge_params = runtime_params.edge
    assert isinstance(
        edge_params, RuntimeParams
    ), 'Edge parameters must be of type ExtendedLengyelModel.RuntimeParams'

    # Extract geometric parameters from TORAX geometry.

    # TODO(b/446608829) Add support for new optional geometry parameters such
    # as connection lengths and divertor broadening factor.

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

    # Extract power crossing separatrix
    # Ensure strictly positive power for log calculations in the model.
    power_crossing_separatrix = jnp.maximum(
        post_processed_outputs.P_SOL_total, 1e-3
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
        # Configurable parameters from RuntimeParams
        parallel_connection_length=edge_params.parallel_connection_length,
        divertor_parallel_length=edge_params.divertor_parallel_length,
        target_electron_temp=edge_params.target_electron_temp,
        seed_impurity_weights=edge_params.seed_impurity_weights,
        fixed_impurity_concentrations=edge_params.fixed_impurity_concentrations,
        computation_mode=edge_params.computation_mode,
        solver_mode=edge_params.solver_mode,
        divertor_broadening_factor=edge_params.divertor_broadening_factor,
        ratio_of_upstream_to_average_poloidal_field=edge_params.ratio_of_upstream_to_average_poloidal_field,
        ne_tau=edge_params.ne_tau,
        sheath_heat_transmission_factor=edge_params.sheath_heat_transmission_factor,
        target_angle_of_incidence=edge_params.target_angle_of_incidence,
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
        toroidal_flux_expansion=edge_params.toroidal_flux_expansion,
        fixed_step_iterations=edge_params.fixed_step_iterations,
        newton_raphson_iterations=edge_params.newton_raphson_iterations,
        newton_raphson_tol=edge_params.newton_raphson_tol,
    )
