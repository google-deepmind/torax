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
import functools

import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.time_step_calculator import runtime_params as time_runtime_params


class TimeStepCalculator(abc.ABC):
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

  def not_done(
      self,
      t: float | jax.Array,
      t_final: float,
      time_calculator_params: time_runtime_params.RuntimeParams,
  ) -> bool | jax.Array:
    return t < (t_final - time_calculator_params.tolerance)

  @functools.partial(
      jax_utils.jit,
      static_argnames=['self'],
  )
  def next_dt(
      self,
      t: jax.Array,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_transport: state.CoreTransport,
  ) -> jax.Array:
    """Returns the next time step duration."""
    dt = self._next_dt(
        runtime_params,
        geo,
        core_profiles,
        core_transport,
    )
    crosses_t_final = (t < runtime_params.numerics.t_final) * (
        t + dt > runtime_params.numerics.t_final
    )
    dt = jax.lax.select(
        jnp.logical_and(
            runtime_params.numerics.exact_t_final,
            crosses_t_final,
        ),
        runtime_params.numerics.t_final - t,
        dt,
    )
    return dt

  @abc.abstractmethod
  def _next_dt(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_transport: state.CoreTransport,
  ) -> jax.Array:
    """Returns the next time step duration."""

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    """Equality for the TimeStepCalculator, needed for JAX."""

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Hash for the TimeStepCalculator, needed for JAX."""
