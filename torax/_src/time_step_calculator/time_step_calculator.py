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

"""Time step calculator base class."""

import abc

import jax
from jax import numpy as jnp
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.time_step_calculator import time_step_calculator_state as time_step_calculator_state_lib


class TimeStepCalculator(abc.ABC):
  """Iterates over time during simulation.

  Usage follows this pattern:

  .. code-block: python

    ts = <TimeStepCalculator subclass constructor>
    ts_state = ts.initial_state()
    t = 0.
    while not ts.is_done(t):
      dt, ts_state = ts.next_dt(geo, time_step_calculator_state)
      t += dt
      sim_state = <update sim_state with step of size dt>
  """

  def initial_state(
      self, runtime_params: runtime_params_lib.RuntimeParams
  ) -> time_step_calculator_state_lib.TimeStepCalculatorState:
    """Returns the initial state for the time step calculator."""
    del runtime_params
    return time_step_calculator_state_lib.TimeStepCalculatorState()

  def get_updated_state(
      self,
      sim_state: sim_state_lib.SimState,
  ) -> time_step_calculator_state_lib.TimeStepCalculatorState:
    """Returns the updated time step calculator state.

    By default, the state is not updated.

    Args:
      sim_state: State of the simulation.

    Returns:
      Updated time step calculator state.
    """
    return sim_state.time_step_calculator_state

  def is_done(
      self, t: float | jax.Array, t_final: float, tolerance: float
  ) -> bool | jax.Array:
    return t >= (t_final - tolerance)

  @jax.jit(
      static_argnames=['self'],
  )
  def next_dt(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      sim_state: sim_state_lib.SimState,
  ) -> jax.Array:
    """Returns the next time step duration."""
    dt = self._next_dt(
        runtime_params,
        sim_state,
    )
    crosses_t_final = (sim_state.t < runtime_params.numerics.t_final) * (
        sim_state.t + dt > runtime_params.numerics.t_final
    )
    dt = jax.lax.select(
        jnp.logical_and(
            runtime_params.numerics.exact_t_final,
            crosses_t_final,
        ),
        runtime_params.numerics.t_final - sim_state.t,
        dt,
    )
    return dt

  @abc.abstractmethod
  def _next_dt(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      sim_state: sim_state_lib.SimState,
  ) -> jax.Array:
    """Returns the next time step duration."""

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    """Equality for the TimeStepCalculator, needed for JAX."""

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Hash for the TimeStepCalculator, needed for JAX."""
