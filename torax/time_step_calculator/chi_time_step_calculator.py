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
from typing import Union

import jax
from jax import numpy as jnp
from torax import config_slice
from torax import geometry
from torax import jax_utils
from torax import state as state_module
from torax.time_step_calculator import time_step_calculator
from torax.transport_model import transport_model as transport_model_lib

# Dummy state and type for compatibility with time_step_calculator base class
STATE = None
State = type(STATE)


class ChiTimeStepCalculator(time_step_calculator.TimeStepCalculator[State]):
  """TimeStepCalculator based on chi_max heuristic.

  Reproduces the behavior from PINT.

  Attributes:
    config: General configuration parameters.
  """

  def initial_state(self):
    return STATE

  def not_done(
      self,
      t: Union[float, jax.Array],
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      state: State,
  ) -> Union[bool, jax.Array]:
    """Returns True if iteration not done (t < config.t_final)."""
    return t < dynamic_config_slice.t_final

  @functools.partial(jax_utils.jit, static_argnames=['self'])
  def next_dt(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      sim_state: state_module.State,
      time_step_calculator_state: State,
      transport_coeffs: transport_model_lib.TransportCoeffs,
  ) -> tuple[jax.Array, State]:
    """Calculates the next time step duration.

    This calculation is a heuristic scaling of the maximum stable step
    size for the explicit method, and is therefore a function of chi_max.

    Args:
      dynamic_config_slice: Input config parameters that can change without
        triggering a JAX recompilation.
      geo: Geometry for the tokamak being simulated.
      sim_state: Current state of the tokamak.
      time_step_calculator_state: None, for compatibility with
        TimeStepCalculator base class.
      transport_coeffs: Used to calculate maximum step size.

    Returns:
      dt: Scalar time step duration.
    """

    chi_max = transport_coeffs.chi_max(geo)

    basic_dt = (3.0 / 4.0) * (geo.dr_norm**2) / chi_max * geo.rmax**2

    dt = jnp.minimum(
        dynamic_config_slice.dtmult * basic_dt,
        dynamic_config_slice.maxdt,
    )

    return dt, STATE
