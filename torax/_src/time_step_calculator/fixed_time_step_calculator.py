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
from torax._src import state as state_module
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.time_step_calculator import time_step_calculator


class FixedTimeStepCalculator(time_step_calculator.TimeStepCalculator):
  """TimeStepCalculator based on constant time steps."""

  def _next_dt(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state_module.CoreProfiles,
      core_transport: state_module.CoreTransport,
  ) -> jax.Array:
    """Returns the fixed time step duration."""
    del geo, core_profiles, core_transport
    return jnp.array(runtime_params.numerics.fixed_dt)

  def __eq__(self, other) -> bool:
    return isinstance(other, type(self))

  def __hash__(self) -> int:
    return hash(type(self))
