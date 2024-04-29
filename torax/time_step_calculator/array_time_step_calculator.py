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

"""The ArrayTimeStepCalculator class.

A TimeStepCalculator that iterates over entries in an array.
"""

from typing import Union

import chex
import jax
from jax import numpy as jnp
from torax import geometry
from torax import state as state_module
from torax.config import runtime_params_slice
from torax.time_step_calculator import time_step_calculator

State = int


# TODO(b/337844885). Remove the array option and make fixed_dt time-dependent
# instead.
class ArrayTimeStepCalculator(time_step_calculator.TimeStepCalculator[State]):
  """TimeStepCalculator that iterates over entries in an array."""

  def __init__(self, arr: chex.Array):
    super().__init__()
    self.arr = jnp.asarray(arr)

  def initial_state(self) -> State:
    return 0

  def not_done(
      self,
      t: Union[float, jax.Array],
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      state: State,
  ) -> Union[jax.Array, bool]:
    """Returns True until the whole array has been visited, then False."""
    del (
        t,
        dynamic_runtime_params_slice,
    )  # Unused for this type of TimeStepCalculator.
    idx = state
    return idx < self.arr.shape[0] - 1

  def next_dt(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state_module.CoreProfiles,
      time_step_calculator_state: State,
      core_transport: state_module.CoreTransport,
  ) -> tuple[jax.Array, State]:
    """Returns the next diff between consecutive array entries."""
    del (
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        core_transport,
    )  # Unused.
    idx = time_step_calculator_state
    idx += 1
    return self.arr[idx] - self.arr[idx - 1], idx
