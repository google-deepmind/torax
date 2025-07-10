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
from torax._src.time_step_calculator import runtime_params


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
      dynamic_params: runtime_params.DynamicRuntimeParams,
  ) -> bool | jax.Array:
    return t < (t_final - dynamic_params.tolerance)

  @functools.partial(
      jax_utils.jit,
      static_argnames=[
          'self',
          'static_runtime_params_slice',
      ],
  )
  def next_dt(
      self,
      t: jax.Array,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_transport: state.CoreTransport,
  ) -> jax.Array:
    """Returns the next time step duration."""
    dt = self._next_dt(
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        core_transport,
    )
    crosses_t_final = (t < dynamic_runtime_params_slice.numerics.t_final) * (
        t + dt > dynamic_runtime_params_slice.numerics.t_final
    )
    dt = jax.lax.select(
        jnp.logical_and(
            static_runtime_params_slice.numerics.exact_t_final,
            crosses_t_final,
        ),
        dynamic_runtime_params_slice.numerics.t_final - t,
        dt,
    )
    return dt

  @abc.abstractmethod
  def _next_dt(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_transport: state.CoreTransport,
  ) -> jax.Array:
    """Returns the next time step duration.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry for the Tokamak.
      core_profiles: Core plasma profiles in the tokamak.
      core_transport: Transport coefficients.
    """

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    """Equality for the TimeStepCalculator, needed for JAX."""

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Hash for the TimeStepCalculator, needed for JAX."""
