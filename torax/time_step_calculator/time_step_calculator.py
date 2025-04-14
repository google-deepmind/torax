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

import jax
from torax import state as state_module
from torax.config import runtime_params_slice
from torax.geometry import geometry


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

  def __init__(self, tolerance: float = 1e-7):
    self.tolerance = tolerance

  def not_done(
      self,
      t: float | jax.Array,
      t_final: float,
  ) -> bool | jax.Array:
    return t < (t_final - self.tolerance)

  @abc.abstractmethod
  def next_dt(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state_module.CoreProfiles,
      core_transport: state_module.CoreTransport,
  ) -> jax.Array:
    """Returns the next time step duration and internal time stepper state.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry for the Tokamak.
      core_profiles: Core plasma profiles in the tokamak.
      core_transport: Transport coefficients.
    """
