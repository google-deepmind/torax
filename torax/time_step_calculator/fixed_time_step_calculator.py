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
from torax import state as state_module
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.time_step_calculator import time_step_calculator

# Dummy state and type for compatibility with time_step_calculator base class
STATE = None
State = type(STATE)


class FixedTimeStepCalculator(time_step_calculator.TimeStepCalculator[State]):
  """TimeStepCalculator based on constant time steps.

  Attributes:
    config: General configuration parameters.
  """

  def initial_state(self):
    return STATE

  def next_dt(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state_module.CoreProfiles,
      time_step_calculator_state: State,
      core_transport: state_module.CoreTransport,
  ) -> tuple[jax.Array, State]:
    """Calculates the next time step duration.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry for the tokamak being simulated.
      core_profiles: Current core plasma profiles.
      time_step_calculator_state: None, for compatibility with
        TimeStepCalculator base class.
      core_transport: Used to calculate chi, which determines maximum step size.

    Returns:
      dt: Scalar time step duration.
    """

    dt = jnp.array(dynamic_runtime_params_slice.numerics.fixed_dt)

    return dt, STATE
