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

"""The ChiTimeStepCalculator class.

Steps through time using a heuristic based on chi_max.
"""
import jax
from jax import numpy as jnp
from torax._src import state as state_module
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.time_step_calculator import time_step_calculator


class ChiTimeStepCalculator(time_step_calculator.TimeStepCalculator):
  """TimeStepCalculator based on chi_max heuristic."""

  def _next_dt(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state_module.CoreProfiles,
      core_transport: state_module.CoreTransport,
  ) -> jax.Array:
    """Calculates the next time step duration.

    This calculation is a heuristic scaling of the maximum stable step
    size for the explicit method, and is therefore a function of chi_max.

    Args:
      runtime_params: Input runtime parameters for the current timestep.
      geo: Geometry for the tokamak being simulated for the current timestep.
      core_profiles: Current core plasma profiles.
      core_transport: Used to calculate maximum step size.

    Returns:
      dt: Scalar time step duration.
    """

    chi_max = core_transport.chi_max(geo)

    basic_dt = (3.0 / 4.0) * (geo.drho_norm**2) / chi_max

    dt = jnp.minimum(
        runtime_params.numerics.chi_timestep_prefactor * basic_dt,
        runtime_params.numerics.max_dt,
    )

    return dt

  def __eq__(self, other) -> bool:
    return isinstance(other, type(self))

  def __hash__(self) -> int:
    return hash(type(self))
