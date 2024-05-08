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
from torax import geometry
from torax import state as state_module
from torax.config import runtime_params_slice

# Subclasses override with their own state type
State = TypeVar('State')


class TimeStepCalculator(Protocol[State]):
  """Iterates over time during simulation.

  Usage follows this pattern:

  .. code-block: python

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
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      state: State,
  ) -> Union[bool, jax.Array]:
    """If True, next_dt may be called again."""

  @abc.abstractmethod
  def next_dt(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state_module.CoreProfiles,
      time_step_calculator_state: State,
      core_transport: state_module.CoreTransport,
  ) -> tuple[jnp.ndarray, State]:
    """Returns the next time step duration and internal time stepper state.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry for the Tokamak.
      core_profiles: Core plasma profiles in the tokamak.
      time_step_calculator_state: Internal state of the time stepper.
      core_transport: Transport coefficients.
    """
