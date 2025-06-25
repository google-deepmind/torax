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

import functools

import jax
from jax import numpy as jnp

from torax._src import jax_utils
from torax._src import state as state_module
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.time_step_calculator import time_step_calculator


class ChiTimeStepCalculator(time_step_calculator.TimeStepCalculator):
  """TimeStepCalculator based on chi_max heuristic.

  Attributes:
    config: General configuration parameters.
  """

  @functools.partial(jax_utils.jit, static_argnames=['self'])
  def next_dt(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state_module.CoreProfiles,
      core_transport: state_module.CoreTransport,
  ) -> jax.Array:
    """Calculates the next time step duration.

    This calculation is a heuristic scaling of the maximum stable step
    size for the explicit method, and is therefore a function of chi_max.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry for the tokamak being simulated.
      core_profiles: Current core plasma profiles.
      core_transport: Used to calculate maximum step size.

    Returns:
      dt: Scalar time step duration.
    """

    chi_max = core_transport.chi_max(geo)

    basic_dt = (3.0 / 4.0) * (geo.drho_norm**2) / chi_max

    dt = jnp.minimum(
        dynamic_runtime_params_slice.numerics.chi_timestep_prefactor * basic_dt,
        dynamic_runtime_params_slice.numerics.max_dt,
    )

    return dt
