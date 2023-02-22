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

"""The TimeStepCalculator class.

Abstract base class defining time stepping interface.
"""

import abc
from typing import Protocol, TypeVar, Union

import jax
from jax import numpy as jnp
from torax import config_slice
from torax import geometry
from torax import state as state_module
from torax.transport_model import transport_model as transport_model_lib

# Subclasses override with their own state type
State = TypeVar('State')


class TimeStepCalculator(Protocol[State]):
  """Iterates over time during simulation.

  Usage follows this pattern:

  ts = <TimeStepCalculator subclass constructor>
  ts_state = ts.initial_state()
  t = 0.
  while ts.not_done(t):
    dt, ts_state = ts.next_dt(geo, time_step_calculator_state)
    t += dt
    sim_state = <update sim_state with step of size dt>
  """

  @abc.abstractmethod
  def initial_state(self) -> State:
    """Returns the initial internal state of the time stepper."""

  @abc.abstractmethod
  def not_done(
      self,
      t: Union[float, jax.Array],
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      state: State,
  ) -> Union[bool, jax.Array]:
    """If True, next_dt may be called again."""

  @abc.abstractmethod
  def next_dt(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      sim_state: state_module.State,
      time_step_calculator_state: State,
      transport_coeffs: transport_model_lib.TransportCoeffs,
  ) -> tuple[jnp.ndarray, State]:
    """Returns the next time step duration and internal time stepper state.

    Args:
      dynamic_config_slice: Input config parameters that can change without
        triggering a JAX recompilation.
      geo: Geometry for the Tokamak.
      sim_state: Main state of the simulator with temp_el, etc.
      time_step_calculator_state: Internal state of the time stepper.
      transport_coeffs: Transport coefficients.
    """
