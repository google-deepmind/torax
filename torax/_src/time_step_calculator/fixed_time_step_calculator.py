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

"""The FixedTimeStepCalculator class.

Steps through time using a constant time step.
"""

import jax
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.time_step_calculator import time_step_calculator


def get_time_grid(
    t_initial: float,
    t_final: float,
    fixed_dt: float,
    exact_t_final: bool,
    tolerance: float,
) -> np.ndarray:
  """Returns all times visited by a simulation with a constant fixed dt.

  Mirrors `TimeStepCalculator.is_done` and `TimeStepCalculator.next_dt`,
  accumulating `t + dt` in the simulation dtype so the returned times match
  the times reached by the simulation loop to floating point precision.

  Args:
    t_initial: Simulation start time [s].
    t_final: Simulation end time [s].
    fixed_dt: Constant time step [s]. Must be positive.
    exact_t_final: Whether the final step is shortened to land on `t_final`.
    tolerance: Tolerance within `t_final` at which the simulation is done.

  Returns:
    1D array of times, starting with `t_initial`.
  """
  if fixed_dt <= 0.0:
    raise ValueError(f'fixed_dt must be positive, got {fixed_dt}.')
  dtype = jax_utils.get_np_dtype()
  t = dtype(t_initial)
  t_final = dtype(t_final)
  fixed_dt = dtype(fixed_dt)
  times = [t]
  while t < t_final - tolerance:
    dt = fixed_dt
    if exact_t_final and t < t_final < t + dt:
      dt = t_final - t
    t = t + dt
    times.append(t)
  return np.asarray(times, dtype=dtype)


class FixedTimeStepCalculator(time_step_calculator.TimeStepCalculator):
  """TimeStepCalculator based on constant time steps."""

  def _next_dt(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      sim_state: sim_state_lib.SimState,
  ) -> jax.Array:
    """Returns the fixed time step duration."""
    del sim_state
    return jnp.array(runtime_params.numerics.fixed_dt)

  def __eq__(self, other) -> bool:
    return isinstance(other, type(self))

  def __hash__(self) -> int:
    return hash(type(self))
