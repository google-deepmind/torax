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

import jax
from jax import numpy as jnp
from torax import config_slice
from torax import geometry
from torax import state as state_module
from torax.time_step_calculator import time_step_calculator
from torax.transport_model import transport_model as transport_model_lib

State = int


class ArrayTimeStepCalculator(time_step_calculator.TimeStepCalculator[State]):
  """TimeStepCalculator that iterates over entries in an array."""

  def __init__(self, arr: jnp.ndarray):
    super().__init__()
    self.arr = arr

  def initial_state(self) -> State:
    return 0

  def not_done(
      self,
      t: Union[float, jax.Array],
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      time_step_calculator_state: State
  ) -> Union[jax.Array, bool]:
    """Returns True until the whole array has been visited, then False."""
    del t, dynamic_config_slice  # Unused for this type of TimeStepCalculator.
    idx = time_step_calculator_state
    return idx < self.arr.shape[0] - 1

  def next_dt(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      sim_state: state_module.State,
      time_step_calculator_state: State,
      transport_model: transport_model_lib.TransportModel,
  ) -> tuple[jax.Array, State]:
    """Returns the next diff between consecutive array entries."""
    del dynamic_config_slice, geo, sim_state, transport_model  # Unused.
    idx = time_step_calculator_state
    idx += 1
    return self.arr[idx] - self.arr[idx - 1], idx
