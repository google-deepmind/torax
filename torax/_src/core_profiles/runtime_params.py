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

"""Runtime parameters for core profiles."""

import dataclasses
import enum

import jax
from torax._src import array_typing
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PrescribedFastIonData:
  """Evaluated prescribed fast ion data for a single species at time t.

  This is the JAX-compatible runtime counterpart of ``PrescribedFastIon``.
  It holds concrete array values evaluated at a specific time ``t``, and is
  stored on ``RuntimeParams`` for use inside JIT-compiled simulation steps.

  Attributes:
    source: Source name (e.g. 'icrh').
    species: Species name (e.g. 'He3').
    n: Prescribed density profile [m^-3].
    n_right_bc: Right boundary condition for density [m^-3].
    T: Prescribed temperature profile [keV].
    T_right_bc: Right boundary condition for temperature [keV].
  """

  source: str = dataclasses.field(metadata={'static': True})
  species: str = dataclasses.field(metadata={'static': True})
  n: array_typing.FloatVector
  n_right_bc: array_typing.FloatScalar
  T: array_typing.FloatVector
  T_right_bc: array_typing.FloatScalar


class InitialPsiMode(enum.StrEnum):
  """How to calculate the initial psi value."""

  PROFILE_CONDITIONS = 'profile_conditions'
  GEOMETRY = 'geometry'
  J = 'j'


class NeBoundaryConditionMode(enum.StrEnum):
  """Mode for the electron density right boundary condition.

  Attributes:
    PRESCRIBED: The boundary condition is prescribed directly via `n_e_right_bc`
      or taken from the `n_e` profile at rho_norm=1.
    DENSITY_FRACTION: The boundary condition is computed as `n_e(reference_rho,
      t) * multiplier`, where `reference_rho` and `multiplier` are
      user-specified. t is the time at the beginning of each time step interval.
  """

  PRESCRIBED = 'prescribed'
  DENSITY_FRACTION = 'density_fraction'


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParams:
  """Prescribed values and boundary conditions for the core profiles."""

  Ip: array_typing.FloatScalar
  v_loop_lcfs: array_typing.FloatScalar
  T_i_right_bc: array_typing.FloatScalar
  T_e_right_bc: array_typing.FloatScalar
  # Temperature profiles defined on the cell grid.
  T_e: array_typing.FloatVector
  T_i: array_typing.FloatVector
  # If provided as array, Psi profile defined on the cell grid.
  psi: array_typing.FloatVector | None
  psidot: array_typing.FloatVector | None
  toroidal_angular_velocity: array_typing.FloatVector | None
  toroidal_angular_velocity_right_bc: array_typing.FloatScalar | None
  # Electron density profile on the cell grid.
  n_e: array_typing.FloatVector
  nbar: array_typing.FloatScalar
  n_e_nbar_is_fGW: bool
  n_e_right_bc: array_typing.FloatScalar
  n_e_right_bc_is_fGW: bool
  n_e_right_bc_mode: NeBoundaryConditionMode = dataclasses.field(
      metadata={'static': True}
  )
  n_e_right_bc_reference_rho: array_typing.FloatScalar | None
  n_e_right_bc_multiplier: array_typing.FloatScalar | None
  internal_boundary_conditions: (
      internal_boundary_conditions_lib.InternalBoundaryConditions
  )
  current_profile_nu: float
  initial_j_is_total_current: bool = dataclasses.field(
      metadata={'static': True}
  )
  initial_psi_from_j: bool = dataclasses.field(metadata={'static': True})
  normalize_n_e_to_nbar: bool = dataclasses.field(metadata={'static': True})
  use_v_loop_lcfs_boundary_condition: bool = dataclasses.field(
      metadata={'static': True}
  )
  n_e_right_bc_is_absolute: bool = dataclasses.field(metadata={'static': True})
  initial_psi_mode: InitialPsiMode = dataclasses.field(
      metadata={'static': True}
  )
  prescribed_fast_ions: tuple[PrescribedFastIonData, ...] = ()
