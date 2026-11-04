# Copyright 2026 DeepMind Technologies Limited
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

"""Integer enum for sim error handling."""

import enum

from absl import logging


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
            Simulation stopped due to NaNs in state and/or post processed
            outputs.
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
