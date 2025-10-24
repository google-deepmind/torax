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

"""The PhasedTimeStepCalculator class.

Steps through time using different fixed time steps for different time windows.

Example:
  For a simulation from 0.5s to 15.0s with different time steps for different
  phases (as described in issue #1663):
  
  ```python
  from torax._src.config import numerics
  from torax._src.time_step_calculator import pydantic_model
  
  config = numerics.Numerics(
      t_initial=0.5,
      t_final=15.0,
      fixed_dt=0.1,  # Fallback value
      phased_dt_windows=((0.5, 3.0), (3.0, 13.0), (13.0, 15.0)),
      phased_dt_values=(0.2, 0.02, 0.1),
      time_step_calculator=pydantic_model.TimeStepCalculator(
          calculator_type=pydantic_model.TimeStepCalculatorType.PHASED
      )
  )
  ```
  
  This configuration will use:
  - dt=0.2 for t ∈ [0.5, 3.0)
  - dt=0.02 for t ∈ [3.0, 13.0)
  - dt=0.1 for t ∈ [13.0, 15.0]
"""

import jax
from jax import numpy as jnp
from torax._src import state as state_module
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.time_step_calculator import time_step_calculator


class PhasedTimeStepCalculator(time_step_calculator.TimeStepCalculator):
  """TimeStepCalculator based on time-windowed fixed time steps.
  
  This calculator allows different fixed time steps to be used in different
  phases of a simulation. Time windows must be contiguous and non-overlapping,
  as validated in the Numerics configuration.
  
  If the current time falls outside all defined windows, the calculator falls
  back to using numerics.fixed_dt.
  
  The last window includes its endpoint to properly handle the final simulation
  time (t_final).
  """

  def _next_dt(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state_module.CoreProfiles,
      core_transport: state_module.CoreTransport,
  ) -> jax.Array:
    """Returns the time step duration for the current time.
    
    Args:
      runtime_params: Runtime parameters including numerics configuration with
        phased_dt_windows and phased_dt_values.
      geo: Geometry (unused, required by interface).
      core_profiles: Core profiles containing current simulation time.
      core_transport: Core transport (unused, required by interface).
      
    Returns:
      Time step duration as a JAX array scalar.
    """
    current_time = core_profiles.t

    # Get the time windows and corresponding dt values
    time_windows = runtime_params.numerics.phased_dt_windows
    dt_values = runtime_params.numerics.phased_dt_values

    # Default to fixed_dt if no phased configuration is provided
    if time_windows is None or dt_values is None:
      return jnp.array(runtime_params.numerics.fixed_dt)

    # Convert to JAX arrays for vectorized operations
    # This approach is JAX-compatible and handles all windows simultaneously
    starts = jnp.array([w[0] for w in time_windows])
    ends = jnp.array([w[1] for w in time_windows])
    dt_array = jnp.array(dt_values)
    
    # Find which window contains current_time
    # For the last window, include the endpoint (<=) to properly handle t_final
    num_windows = len(time_windows)
    is_last_window = jnp.arange(num_windows) == (num_windows - 1)
    
    # Check if current_time is in each window
    # Last window: [start, end], other windows: [start, end)
    in_windows = jnp.where(
        is_last_window,
        (current_time >= starts) & (current_time <= ends),
        (current_time >= starts) & (current_time < ends)
    )
    
    # Get the index of the matching window
    # argmax returns the index of first True value (or 0 if all False)
    has_match = jnp.any(in_windows)
    window_idx = jnp.argmax(in_windows)
    
    # Select the appropriate dt value
    # If a window matches, use its dt; otherwise use fixed_dt as fallback
    dt = jnp.where(
        has_match,
        dt_array[window_idx],
        runtime_params.numerics.fixed_dt
    )
    
    return dt

  def __eq__(self, other) -> bool:
    """Check equality based on type.
    
    Args:
      other: Object to compare with.
      
    Returns:
      True if other is also a PhasedTimeStepCalculator.
    """
    return isinstance(other, type(self))

  def __hash__(self) -> int:
    """Hash based on type.
    
    Returns:
      Hash value for this calculator type.
    """
    return hash(type(self))