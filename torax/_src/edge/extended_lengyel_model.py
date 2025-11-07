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
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.edge import base
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_solvers
from torax._src.geometry import geometry

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(base.RuntimeParams):
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
class ExtendedLengyelOutputs(base.EdgeModelOutputsBase):
  """Additional outputs from the extended Lengyel model.

  Attributes:
    alpha_t: Turbulence broadening factor alpha_t.
    separatrix_Z_eff: Z_eff at the separatrix.
    seed_impurity_concentrations: A mapping from ion symbol to its n_e_ratio.
    solver_status: Status of the solver.
  """

  alpha_t: jax.Array
  separatrix_Z_eff: jax.Array
  seed_impurity_concentrations: Mapping[str, jax.Array]
  solver_status: extended_lengyel_solvers.ExtendedLengyelSolverStatus


@dataclasses.dataclass(frozen=True, eq=False)
class ExtendedLengyelModel(base.EdgeModelBase):

  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> ExtendedLengyelOutputs:
    # TODO(b/446608829) - to be completed in a later PR.
    raise NotImplementedError
