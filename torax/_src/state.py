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

"""Classes defining the TORAX state that evolves over time."""

import dataclasses
import enum
from typing import Self

from absl import logging
import jax
from torax._src import array_typing
from torax._src.core_profiles import profiles
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib

# pylint: disable=invalid-name


# TODO(b/572586768): Remove alias in next CL after migrating callers.
Conductivity = profiles.Conductivity
CoreProfiles = profiles.CoreProfiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CoreTransport:
  """Composite coefficients for plasma transport.

  Attributes:
    total: Combined 4-channel net transport coefficients (turbulent +
      neoclassical + pereverzev).
    turbulent: Turbulent transport output containing total and per-model
      outputs.
    neoclassical: Neoclassical transport output.
    pereverzev: Pereverzev-Corrigan transport output.
  """

  total: transport_coeffs_lib.TransportCoeffs
  turbulent: transport_coeffs_lib.TurbulentTransport
  neoclassical: transport_coeffs_lib.NeoclassicalTransport
  pereverzev: transport_coeffs_lib.PereverzevTransport | None = None

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> Self:
    """Returns a CoreTransport with all zeros. Useful for initializing."""
    return cls(
        total=transport_coeffs_lib.TransportCoeffs.zeros(geo),
        turbulent=transport_coeffs_lib.TurbulentTransport.zeros(geo),
        neoclassical=transport_coeffs_lib.NeoclassicalTransport.zeros(geo),
        pereverzev=None,
    )

  def to_output_dict(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Converts CoreTransport into an OutputVar mapping for the profiles node."""
    return {
        **self.turbulent.to_output_dict(context),
        **self.neoclassical.to_output_dict(context),
    }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SolverNumericOutputs:
  """Numerical quantities related to the solver.

  Attributes:
    outer_solver_iterations: Number of iterations performed in the outer loop of
      the solver.
    solver_error_state: 0 if solver converged with fine tolerance for this step
      1 if solver did not converge for this step (was above coarse tol) 2 if
      solver converged within coarse tolerance. Allowed to pass with a warning.
      Occasional error=2 has low impact on final sim state.
    inner_solver_iterations: Total number of iterations performed in the solver
      across all iterations of the solver.
    sawtooth_crash: True if a sawtooth model is active and the solver step
      corresponds to a sawtooth crash step.
  """

  outer_solver_iterations: array_typing.IntScalar
  solver_error_state: array_typing.IntScalar
  inner_solver_iterations: array_typing.IntScalar
  sawtooth_crash: array_typing.BoolScalar

  def to_output_dict(
      self, context: output_grid_context.OutputGridContext
  ) -> dict[str, output_grid_context.OutputVar]:
    """Serializes stacked SolverNumericOutputs into a dictionary of OutputVar tuples."""
    return {
        output_keys.OUTER_SOLVER_ITERATIONS: context.pack(
            output_keys.OUTER_SOLVER_ITERATIONS, self.outer_solver_iterations
        ),
        output_keys.INNER_SOLVER_ITERATIONS: context.pack(
            output_keys.INNER_SOLVER_ITERATIONS, self.inner_solver_iterations
        ),
        output_keys.SOLVER_ERROR_STATE: context.pack(
            output_keys.SOLVER_ERROR_STATE, self.solver_error_state
        ),
        output_keys.SAWTOOTH_CRASH: context.pack(
            output_keys.SAWTOOTH_CRASH, self.sawtooth_crash
        ),
    }


# TODO(b/434175938): change to StrEnum
@enum.unique
class SimError(enum.Enum):
  """Integer enum for sim error handling."""

  NO_ERROR = 0
  NAN_DETECTED = 1
  QUASINEUTRALITY_BROKEN = 2
  NEGATIVE_CORE_PROFILES = 3
  REACHED_MIN_DT = 4
  LOW_TEMPERATURE_COLLAPSE = 5
  DID_NOT_REACH_T_FINAL = 6

  def log_error(self):
    match self:
      case SimError.NEGATIVE_CORE_PROFILES:
        logging.error("""
            Simulation stopped due to negative values in core profiles.
            """)
      case SimError.NAN_DETECTED:
        logging.error("""
            Simulation stopped due to NaNs in state and/or post processed outputs.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.QUASINEUTRALITY_BROKEN:
        logging.error("""
            Simulation stopped due to quasineutrality being violated.
            Possible cause is bad handling of impurity species.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.REACHED_MIN_DT:
        logging.error("""
            Simulation stopped because the adaptive time step became too small.
            A common cause of vanishing timesteps is due to the nonlinear solver
            tending to negative densities or temperatures. This often arises
            through physical reasons like radiation collapse, or unphysical
            configuration such as impurity densities incompatible with physical
            quasineutrality. Check the output file for near-zero temperatures or
            densities at the last valid step.
            """)
      case SimError.LOW_TEMPERATURE_COLLAPSE:
        logging.error("""
          Simulation stopped because ion or electron temperature fell below the
          configured minimum threshold. This is usually caused by radiative
          collapse. Output file contains all profiles up to the last valid step.
          """)
      case SimError.DID_NOT_REACH_T_FINAL:
        logging.error("""
            Simulation did not reach the final timestep t_final while running
            the jitted run loop where max steps is bounded. This is because
            max_steps was exhausted before the simulation reached t_final,
            likely due to adaptive time stepping and the simulation needing a
            high number of steps to converege.
            Consider increasing max_steps or investigating convergence issues.
            """)
      case SimError.NO_ERROR:
        pass
      case _:
        raise ValueError(f'Unknown SimError: {self}')


class SimStatus(enum.StrEnum):
  """String enum for simulation output file status.

  This indicates the state of the simulation when the output file was written.
  """

  COMPLETED = 'completed'  # Simulation completed successfully to t_final
  CHECKPOINT = 'checkpoint'  # Intermediate checkpoint (not yet used)
  ERROR = 'error'  # Simulation stopped due to an error condition
