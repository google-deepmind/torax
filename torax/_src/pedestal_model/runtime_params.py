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

"""Dataclass representing runtime parameter inputs to the pedestal models."""

import dataclasses
import enum

import jax
from torax._src import array_typing

# pylint: disable=invalid-name


@enum.unique
class Mode(enum.Enum):
  """Defines how the pedestal is generated."""

  # The pedestal is set by modifying the transport coefficients.
  ADAPTIVE_TRANSPORT = "ADAPTIVE_TRANSPORT"

  # The pedestal is set by modifying the state equations and directly setting
  # internal Dirichlet boundary conditions at a grid point or partial profile.
  INTERNAL_BOUNDARY_CONDITION = "INTERNAL_BOUNDARY_CONDITION"


# Store confinement mode as an int so that it is a valid JAX dynamic type.
class ConfinementMode(enum.IntEnum):
  L_MODE = 0
  H_MODE = 1
  TRANSITIONING_TO_H_MODE = 2
  TRANSITIONING_TO_L_MODE = 3


@enum.unique
class PedestalProfileForm(enum.StrEnum):
  """Controls the shape of internal boundary conditions in the pedestal region.

  Attributes:
    SET_AT_PED_TOP: Pedestal values are pinned at a single grid cell nearest to
      rho_norm_ped_top.
    MTANH: A smooth modified-tanh (mtanh) profile is applied across the pedestal
      region from the pedestal top to the separatrix, following the mtanh
      parameterization of Snyder et al., Nucl. Fusion 44 (2004) 320.
  """

  SET_AT_PED_TOP = "SET_AT_PED_TOP"
  MTANH = "MTANH"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StructureRuntimeParams:
  """Base runtime params for pedestal structure models."""

  pedestal_profile_form: PedestalProfileForm = dataclasses.field(
      metadata={"static": True}
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FormationRuntimeParams:
  """Base runtime params for pedestal formation models.

  Attributes:
    base_multiplier: Transport decrease multiplier in ADAPTIVE_TRANSPORT mode.
    initial_confinement_mode: Optional explicit initial confinement mode
      (L_MODE or H_MODE) at t_initial. When None, inferred from the formation
      model's transition conditions at t_initial.
  """

  base_multiplier: array_typing.FloatScalar
  initial_confinement_mode: ConfinementMode | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SaturationRuntimeParams:
  """Runtime params for pedestal saturation models."""

  steepness: array_typing.FloatScalar
  offset: array_typing.FloatScalar
  base_multiplier: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Input params for the pedestal model.

  Attributes:
    mode: Defines how the pedestal is generated.
    transition_time_width: Duration of the L-H or H-L transition ramp [s].
      During a transition, pedestal values are linearly interpolated between
      L-mode and H-mode values over this time window.
    explicit_pedestal: When True (default), the pedestal model is evaluated once
      per timestep before the solver and its output (T_ped, n_ped, rho_ped_top)
      is frozen during the solve. When False, the pedestal model is
      re-evaluated whenever the solver recomputes the PDE coefficients (e.g.
      each Newton-Raphson iteration or predictor-corrector step), coupling
      pedestal physics to the implicit solve. Note: for ADAPTIVE_TRANSPORT mode,
      transport multipliers are always re-evaluated implicitly with current
      profiles regardless of this setting, since the saturation model feedback
      loop requires implicit coupling.
    structure: Runtime params for the pedestal structure model.
    formation: Runtime params for the formation model.
    saturation: Runtime params for the saturation model.
    chi_max: Maximum effective thermal diffusion coefficient [m^2/s].
    D_e_max: Maximum effective particle diffusion coefficient [m^2/s].
    V_e_max: Maximum effective particle pinch velocity [m/s].
    V_e_min: Minimum effective particle pinch velocity [m/s].
    pedestal_top_smoothing_width: Width of the smoothing kernel at the pedestal
      top.
    initial_L_mode_bc_multiplier: Multiplier applied to the LCFS boundary
      condition values (`T_i`, `T_e`, `n_e` at the right face) to seed the
      L-mode pedestal-top baselines at `t_initial` when the simulation starts
      directly in H-mode without L-mode profile data.
  """

  mode: Mode = dataclasses.field(metadata={"static": True})
  transition_time_width: array_typing.FloatScalar
  explicit_pedestal: bool = dataclasses.field(metadata={"static": True})
  structure: StructureRuntimeParams
  formation: FormationRuntimeParams
  saturation: SaturationRuntimeParams
  chi_max: array_typing.FloatScalar
  D_e_max: array_typing.FloatScalar
  V_e_max: array_typing.FloatScalar
  V_e_min: array_typing.FloatScalar
  pedestal_top_smoothing_width: array_typing.FloatScalar
  initial_L_mode_bc_multiplier: array_typing.FloatScalar
