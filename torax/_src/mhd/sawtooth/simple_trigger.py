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

"""Simple trigger model for sawteeth."""

import dataclasses
from typing import Annotated, Literal

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.mhd.sawtooth import trigger_base
from torax._src.torax_pydantic import torax_pydantic


@dataclasses.dataclass(frozen=True, eq=False)
class SimpleTrigger(
    trigger_base.TriggerModel, static_dataclass.StaticDataclass
):
  """Simple trigger model."""

  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tuple[array_typing.BoolScalar, array_typing.FloatScalar]:
    """Checks if the simple critical shear condition for a sawtooth is met.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry object.
      core_profiles: Core plasma profiles.

    Returns:
      tuple of (True if sawtooth crash is triggered, False otherwise,
        radius of q=1 surface (set to 0.0 if no surface exists))
    """
    # Extract necessary parameters
    sawtooth_params = runtime_params.mhd.sawtooth
    assert isinstance(sawtooth_params, sawtooth_runtime_params.RuntimeParams)
    assert isinstance(sawtooth_params.trigger_params, RuntimeParams)
    minimum_radius = sawtooth_params.trigger_params.minimum_radius
    s_critical = sawtooth_params.trigger_params.s_critical

    q_face = core_profiles.q_face
    s_face = core_profiles.s_face
    rho_face_norm = geo.rho_face_norm
    eps = constants.CONSTANTS.eps

    # Find the rightmost location where q crosses 1.0.
    # Allows for non-monotonic q profiles.

    # Calculate interpolated rho_q1 for all intervals
    q1 = q_face[:-1]
    q2 = q_face[1:]
    rho_norm1 = rho_face_norm[:-1]
    rho_norm2 = rho_face_norm[1:]
    # Handle division by zero if q1 == q2
    dq = jnp.where(jnp.abs(q2 - q1) > eps, q2 - q1, eps)
    # Calculate value of rho for q=1 by linearization of the q-profile
    # in the interval. If the calculated rho is within the interval, then the
    # q=1 surface exists for that interval.
    potential_rho_norm_q1 = (
        rho_norm1 + (rho_norm2 - rho_norm1) * (1.0 - q1) / dq
    )
    valid_interval_mask = jnp.logical_and(
        potential_rho_norm_q1 >= rho_norm1, potential_rho_norm_q1 <= rho_norm2
    )

    # Mask out intervals where there isn't actually a crossing
    # We set invalid rhos to 0.0, as rho_q1 must be > 0 for a valid surface
    valid_rho_norm_q1 = jnp.where(
        valid_interval_mask, potential_rho_norm_q1, 0.0
    )

    # The rightmost crossing corresponds to the maximum valid rho_q1
    # For no crossings, rho_q1 = 0.0 which will not trigger the sawtooth.
    rho_norm_q1 = jnp.max(valid_rho_norm_q1)
    s_at_q1 = jnp.interp(rho_norm_q1, rho_face_norm, s_face)

    # Check trigger conditions
    rho_norm_above_minimum = rho_norm_q1 > minimum_radius
    s_above_critical = s_at_q1 > s_critical

    return (
        jnp.logical_and(rho_norm_above_minimum, s_above_critical),
        rho_norm_q1,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sawtooth_runtime_params.TriggerRuntimeParams):
  """Runtime params for simple trigger model.

  Attributes:
    s_critical: Critical shear value at q=1 for sawtooth triggering.
  """

  s_critical: array_typing.FloatScalar


class SimpleTriggerConfig(trigger_base.TriggerConfig):
  """Pydantic model for simple trigger configuration.

  Attributes:
    s_critical: Critical shear value.
  """

  s_critical: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )
  model_name: Annotated[Literal['simple'], torax_pydantic.JAX_STATIC] = 'simple'

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return RuntimeParams(
        **base_kwargs,
        s_critical=self.s_critical.get_value(t),
    )

  def build_trigger_model(self) -> SimpleTrigger:
    return SimpleTrigger()
