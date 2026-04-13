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

"""The FromPreviousTimeStepCalculator class.

Steps through time based on the previous time step.
"""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.time_step_calculator import time_step_calculator
from torax._src.time_step_calculator import time_step_calculator_state as time_step_calculator_state_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FromPreviousTimeStepCalculatorState(
    time_step_calculator_state_lib.TimeStepCalculatorState
):
  """State for the FromPreviousTimeStepCalculator.

  Attributes:
    previous_non_sawtooth_dt: The duration of the most recent time step that is
      not a sawtooth.
  """

  previous_non_sawtooth_dt: jax.Array


class FromPreviousTimeStepCalculator(time_step_calculator.TimeStepCalculator):
  """TimeStepCalculator based on the previous time step."""

  def initial_state(
      self, runtime_params: runtime_params_lib.RuntimeParams
  ) -> FromPreviousTimeStepCalculatorState:
    """Returns the initial state for the time step calculator."""
    return FromPreviousTimeStepCalculatorState(
        previous_non_sawtooth_dt=jnp.array(runtime_params.numerics.fixed_dt)
    )

  def get_updated_state(
      self,
      sim_state: sim_state_lib.SimState,
  ) -> time_step_calculator_state_lib.TimeStepCalculatorState:
    """Returns the updated state for the time step calculator."""
    assert isinstance(
        sim_state.time_step_calculator_state,
        FromPreviousTimeStepCalculatorState,
    )
    # If the last step was a sawtooth crash, then the previous non-sawtooth dt
    # is the dt from before the crash. Otherwise, it is the dt from the most
    # recent time step.
    return FromPreviousTimeStepCalculatorState(
        previous_non_sawtooth_dt=jnp.where(
            sim_state.solver_numeric_outputs.sawtooth_crash,
            sim_state.time_step_calculator_state.previous_non_sawtooth_dt,
            sim_state.dt,
        ),
    )

  def _next_dt(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      sim_state: sim_state_lib.SimState,
  ) -> jax.Array:
    """Calculates the next time step duration based on the previous time step.

    The next timestep dt is the most recent non-sawtooth dt multiplied by the
    dt reduction factor, clipped to the min and max dt values.

    For example, if the previous non-sawtooth dt is 0.01 and the dt reduction
    factor is 2.0, then the next dt is 0.02.

    Args:
      runtime_params: Input runtime parameters for the current timestep.
      sim_state: State of the simulation.

    Returns:
      dt: Scalar time step duration.
    """
    assert isinstance(
        sim_state.time_step_calculator_state,
        FromPreviousTimeStepCalculatorState,
    )

    dt = (
        sim_state.time_step_calculator_state.previous_non_sawtooth_dt
        * runtime_params.numerics.dt_reduction_factor
    )
    dt = jnp.clip(dt, max=runtime_params.numerics.max_dt)
    return dt

  def __eq__(self, other) -> bool:
    return isinstance(other, type(self))

  def __hash__(self) -> int:
    return hash(type(self))
